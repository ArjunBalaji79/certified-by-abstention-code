"""Conformal risk control for CoT verifiers — score generation (Modal, CPU).

Two stages, both resumable and cached on the `jspace-results` volume:

  cell_meta()      metadata + label audit for every non-degenerate cell
  emit_scores()    per model: rebuild residual-probe features, fit each signal's
                   score function ONCE on a grouped GSM8K train split (per outer
                   seed), then emit per-trace scores for held-out GSM8K and for
                   every other benchmark of the same model (shift evaluation).

Score functions are frozen before any calibration happens; all conformal
calibration/test resplitting is done downstream (locally, on the emitted
scores), which is what the exchangeability argument requires.

Run:
  modal run conformal_scores.py::meta_entry
  modal run --detach conformal_scores.py::scores_entry
"""

import json
import os
import time

import modal

JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
RESULTS = "/results"
HF = "/hf"
SEED = 0
MIN_STEPS = 3
OUTER_SEEDS = [0, 1, 2, 3, 4]
TRAIN_FRAC = 0.30          # of GSM8K problems, used only to fit score functions
                           # (all traces of those problems; cal/test use 1/problem)
BUDGET_CAP_USD = 10.0

# (slug, benchmark) -> trace root on the volume; from results_v*_cross.json
CELLS = {
    ("olmo-3-7b", "arc"): "v3/olmo-3-7b/",
    ("olmo-3-7b", "bbhlogic"): "v3/olmo-3-7b/",
    ("olmo-3-7b", "csqa"): "v4/olmo-3-7b/",
    ("olmo-3-7b", "gsm8k"): "v3/olmo-3-7b/",
    ("qwen2.5-7b-it", "arc"): "v3/qwen2.5-7b-it/",
    ("qwen2.5-7b-it", "bbhlogic"): "v3/qwen2.5-7b-it/",
    ("qwen2.5-7b-it", "csqa"): "v4/qwen2.5-7b-it/",
    ("qwen2.5-7b-it", "gsm8k"): "v3/qwen2.5-7b-it/",
    ("qwen2.5-7b-it", "strategyqa"): "v3/qwen2.5-7b-it/",
    ("qwen3-1.7b", "arc"): "v4/qwen3-1.7b/",
    ("qwen3-1.7b", "csqa"): "v4/qwen3-1.7b/",
    ("qwen3-1.7b", "gsm8k"): "v4/qwen3-1.7b/",
    ("qwen3-1.7b", "strategyqa"): "v4/qwen3-1.7b/",
    ("qwen3-4b", "arc"): "v3/qwen3-4b/",
    ("qwen3-4b", "bbhlogic"): "v3/qwen3-4b/",
    ("qwen3-4b", "gsm8k"): "v2/qwen3-4b/",
    ("qwen3-4b", "strategyqa"): "v3/qwen3-4b/",
    ("qwen3-8b", "arc"): "v4/qwen3-8b/",
    ("qwen3-8b", "csqa"): "v4/qwen3-8b/",
    ("qwen3-8b", "gsm8k"): "v4/qwen3-8b/",
    ("qwen3-8b", "strategyqa"): "v4/qwen3-8b/",
    ("qwen3.5-0.8b", "arc"): "v3/qwen3.5-0.8b/",
    ("qwen3.5-0.8b", "bbhlogic"): "v3/qwen3.5-0.8b/",
    ("qwen3.5-0.8b", "csqa"): "v4/qwen3.5-0.8b/",
    ("qwen3.5-0.8b", "gsm8k"): "v2/qwen3.5-0.8b/",
    ("qwen3.5-0.8b", "strategyqa"): "v3/qwen3.5-0.8b/",
    ("qwen3.5-4b", "arc"): "v3/qwen3.5-4b/",
    ("qwen3.5-4b", "bbhlogic"): "v3/qwen3.5-4b/",
    ("qwen3.5-4b", "csqa"): "v4/qwen3.5-4b/",
    ("qwen3.5-4b", "gsm8k"): "",
    ("qwen3.5-4b", "strategyqa"): "v3/qwen3.5-4b/",
}
MODELS = sorted({s for s, _ in CELLS})

# signal name -> features.npz condition key ("__probe__" = residual probe)
SIGNALS = {
    "confidence": "X_confidence_only",
    "jspace": "X_jspace_only",
    "logitlens": "X_logitlens_only",
    "surface": "X_surface_only",
    "probe": "__probe__",
    "jspace_plus_confidence": "X_jspace_plus_confidence",
}
PROBE_C = 0.01      # matches analyze_v3's high-dim regularization
DEFAULT_C = 1.0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.13.0", "transformers==5.14.1", "accelerate==1.14.0",
        "datasets==5.0.0", "sentence-transformers==5.6.1", "scikit-learn==1.9.0",
        "numpy==2.5.1", "pandas==3.0.5", "matplotlib==3.11.1", "pyarrow==25.0.0",
        "huggingface-hub==1.24.0",
        f"jlens @ git+https://github.com/anthropics/jacobian-lens@{JLENS_COMMIT}",
    )
    .env({"HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"})
)

app = modal.App("jspace-conformal")
results_vol = modal.Volume.from_name("jspace-results", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
VOLS = {RESULTS: results_vol, HF: hf_vol}
cpu_kw = dict(image=image, volumes=VOLS, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=4, initial_delay=15.0))

VALID_LABELS = {"mc": set("abcdefg"), "yesno": {"yes", "no"}}


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(m):
    print(f"[{now_iso()}] {m}", flush=True)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def append_decision(text):
    try:
        results_vol.reload()
    except Exception:
        pass
    with open(f"{RESULTS}/DECISIONS.md", "a") as f:
        f.write(f"- [{now_iso()}] {text}\n")
    results_vol.commit()
    log(f"DECISION: {text}")


def load_cell_traces(root, benchmark):
    """Usable traces for one cell, indexed by uid (same filter as analyze_v3)."""
    import pandas as pd

    man = read_json(f"{RESULTS}/{root}traces/manifest.json", {"shards": []})
    dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(man["shards"]))
           if os.path.exists(f"{RESULTS}/{s}")]
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"] & (df["dataset"] == benchmark)]
    return df.set_index("uid")


def rebuild_uid_order(root, benchmark):
    """Reproduce analyze_v3's row order exactly (it is a deterministic loop).

    Returns (uids, y, groups). Callers MUST verify y/groups against
    features.npz before trusting the alignment.
    """
    import numpy as np

    df = load_cell_traces(root, benchmark)
    if df is None:
        return None
    rman = read_json(f"{RESULTS}/{root}readouts/manifest.json", {"done": []})
    cand = [u for u in rman["done"] if u in df.index
            and os.path.exists(f"{RESULTS}/{root}readouts/{u}.npz")
            and os.path.exists(f"{RESULTS}/{root}confidence/{u}.json")]
    uids, ys, gs = [], [], []
    for u in cand:
        r = df.loc[u]
        with np.load(f"{RESULTS}/{root}readouts/{u}.npz") as npz:
            n_steps_ext = npz["j_topk_ids"].shape[0] - 1
        if n_steps_ext < MIN_STEPS:
            continue
        if not read_json(f"{RESULTS}/{root}confidence/{u}.json"):
            continue
        n_json_steps = len(json.loads(r["steps_json"]))
        if min(n_steps_ext, n_json_steps) < MIN_STEPS:  # len(E) < MIN_STEPS
            continue
        uids.append(u)
        ys.append(int(bool(r["correct"])))
        gs.append(str(r["problem_id"]))
    return uids, np.array(ys, dtype=int), np.array(gs, dtype=object)


@app.function(timeout=7200, **cpu_kw)
def cell_meta():
    """Metadata + label audit for every cell. No modelling, no results."""
    import numpy as np

    results_vol.reload()
    out = {}
    audit = {}
    for (slug, bench), root in sorted(CELLS.items()):
        key = f"{slug}|{bench}"
        fpath = f"{RESULTS}/v3/{slug}/{bench}/features.npz"
        if not os.path.exists(fpath):
            out[key] = {"error": "no features.npz"}
            continue
        with np.load(fpath, allow_pickle=True) as z:
            y = z["y"]
            groups = z["groups"]
            dims = {k: int(z[k].shape[1]) for k in z.files if k.startswith("X_")}
        gsize = {}
        for g in groups:
            gsize[g] = gsize.get(g, 0) + 1
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        # group-level nulls: problems with >=1 incorrect trace
        gneg = {}
        for g, yy in zip(groups, y):
            gneg[g] = gneg.get(g, 0) or int(yy == 0)
        out[key] = {
            "n_traces": int(len(y)), "n_pos": n_pos, "n_neg": n_neg,
            "accuracy": round(float(y.mean()), 4),
            "n_groups": int(len(gsize)),
            "traces_per_group_mean": round(float(np.mean(list(gsize.values()))), 2),
            "n_groups_with_error": int(sum(gneg.values())),
            "feature_dims": dims,
        }
        # --- label audit (style-anchored parsing gate) ---
        # Audit is PER ROW: bbhlogic legitimately mixes mc (logical_deduction,
        # tracking) and yesno (web_of_lies) styles within one cell, and each row
        # was graded against its own style.
        df = load_cell_traces(root, bench)
        if df is not None and "style" in df.columns and len(df):
            styles = df["style"].tolist()
            preds = df["pred"].tolist()
            golds = df["gold"].tolist()
            n_checked = n_bad = n_gold_bad = 0
            examples = []
            for st, p, g in zip(styles, preds, golds):
                if st not in VALID_LABELS:
                    continue
                n_checked += 1
                ps = str(p).strip().lower() if p is not None else None
                gs = str(g).strip().lower() if g is not None else None
                if ps is not None and ps not in VALID_LABELS[st]:
                    n_bad += 1
                    if len(examples) < 3:
                        examples.append({"style": st, "pred": ps})
                if gs is not None and gs not in VALID_LABELS[st]:
                    n_gold_bad += 1
            audit[key] = {"styles": sorted(set(styles)), "n_checked": n_checked,
                          "n_invalid_pred": n_bad, "n_invalid_gold": n_gold_bad,
                          "examples": examples}
    write_json(f"{RESULTS}/conformal/cell_meta.json",
               {"generated_at": now_iso(), "cells": out, "label_audit": audit})
    results_vol.commit()
    log(f"cell_meta: {len(out)} cells")
    return {"cells": out, "label_audit": audit}


@app.function(timeout=14400, **cpu_kw)
def build_probe(slug: str, bench: str):
    """Rebuild residual-probe features for one cell, aligned to features.npz.

    Saves v3/{slug}/{bench}/probe_X.npz with X (float16), uids, and the
    verification status. Idempotent: skips if already present.
    """
    import numpy as np

    results_vol.reload()
    root = CELLS[(slug, bench)]
    outp = f"{RESULTS}/v3/{slug}/{bench}/probe_X.npz"
    if os.path.exists(outp):
        with np.load(outp, allow_pickle=True) as z:
            return {"cell": f"{slug}|{bench}", "cached": True,
                    "verified": bool(z["verified"]), "n": int(z["X"].shape[0])}

    fpath = f"{RESULTS}/v3/{slug}/{bench}/features.npz"
    with np.load(fpath, allow_pickle=True) as z:
        y_ref, g_ref = z["y"], np.array([str(g) for g in z["groups"]])

    rebuilt = rebuild_uid_order(root, bench)
    if rebuilt is None:
        return {"cell": f"{slug}|{bench}", "error": "no traces"}
    uids, y_new, g_new = rebuilt
    verified = (len(y_new) == len(y_ref) and np.array_equal(y_new, y_ref)
                and np.array_equal(g_new.astype(str), g_ref))
    if not verified:
        log(f"WARNING {slug}|{bench}: row-order rebuild MISMATCH "
            f"(n_new={len(y_new)} n_ref={len(y_ref)})")

    X, missing = [], 0
    dim = None
    for u in uids:
        p = f"{RESULTS}/{root}residuals/{u}.npz"
        if not os.path.exists(p):
            X.append(None)
            missing += 1
            continue
        with np.load(p) as z:
            R = z["resid"].astype(np.float32)      # [P, n_depths, d_model]
        mid = min(1, R.shape[1] - 1)
        v = np.concatenate([R[:-1, mid].mean(0), R[-1, mid]])
        dim = v.shape[0]
        X.append(v)
    if dim is None:
        return {"cell": f"{slug}|{bench}", "error": "no residuals"}
    X = np.stack([v if v is not None else np.zeros(dim, dtype=np.float32)
                  for v in X]).astype(np.float16)
    np.savez_compressed(outp, X=X, uids=np.array(uids), verified=verified,
                        missing=missing)
    results_vol.commit()
    log(f"probe {slug}|{bench}: X{X.shape} verified={verified} missing={missing}")
    return {"cell": f"{slug}|{bench}", "verified": bool(verified),
            "n": int(X.shape[0]), "dim": int(dim), "missing": int(missing)}


@app.function(timeout=14400, **cpu_kw)
def build_probe_depths(slug: str, bench: str):
    """Probe features at EARLY (depth idx 0) and LATE (idx 2) as well as the
    step-mean / answer-position split, for the probe-gap experiment."""
    import numpy as np

    results_vol.reload()
    root = CELLS[(slug, bench)]
    outp = f"{RESULTS}/v3/{slug}/{bench}/probe_depths.npz"
    if os.path.exists(outp):
        return {"cell": f"{slug}|{bench}", "cached": True}
    rebuilt = rebuild_uid_order(root, bench)
    if rebuilt is None:
        return {"cell": f"{slug}|{bench}", "error": "no traces"}
    uids, y_new, _ = rebuilt
    fpath = f"{RESULTS}/v3/{slug}/{bench}/features.npz"
    with np.load(fpath, allow_pickle=True) as z:
        y_ref = z["y"]
    if not (len(y_new) == len(y_ref) and np.array_equal(y_new, y_ref)):
        return {"cell": f"{slug}|{bench}", "error": "row-order mismatch"}

    cols = {k: [] for k in ("early_step", "early_ans", "mid_step", "mid_ans",
                            "late_step", "late_ans")}
    for u in uids:
        p = f"{RESULTS}/{root}residuals/{u}.npz"
        if not os.path.exists(p):
            for k in cols:
                cols[k].append(None)
            continue
        with np.load(p) as z:
            R = z["resid"].astype(np.float32)      # [P, n_depths, d_model]
        D = R.shape[1]
        for name, di in (("early", 0), ("mid", min(1, D - 1)), ("late", D - 1)):
            cols[f"{name}_step"].append(R[:-1, di].mean(0))
            cols[f"{name}_ans"].append(R[-1, di])
    dim = next(v.shape[0] for v in cols["mid_step"] if v is not None)
    out = {k: np.stack([v if v is not None else np.zeros(dim, dtype=np.float32)
                        for v in vs]).astype(np.float16)
           for k, vs in cols.items()}
    np.savez_compressed(outp, uids=np.array(uids), **out)
    results_vol.commit()
    log(f"probe_depths {slug}|{bench}: {dim}d x6 blocks, n={len(uids)}")
    return {"cell": f"{slug}|{bench}", "n": len(uids), "dim": int(dim)}


@app.local_entrypoint()
def depths_entry():
    todo = [(s, b) for (s, b) in sorted(CELLS) if b == "gsm8k"]
    res = list(build_probe_depths.starmap(todo))
    print(json.dumps(res, indent=2, default=str))


def _load_cell(slug, bench):
    import numpy as np

    f = f"{RESULTS}/v3/{slug}/{bench}/features.npz"
    p = f"{RESULTS}/v3/{slug}/{bench}/probe_X.npz"
    with np.load(f, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    out = {"y": data["y"].astype(int),
           "groups": np.array([str(g) for g in data["groups"]]),
           "X": {}, "cols": {}}
    for sig, key in SIGNALS.items():
        if key == "__probe__":
            continue
        if key in data:
            out["X"][sig] = np.nan_to_num(data[key].astype(np.float64))
            out["cols"][sig] = [str(c) for c in data[f"cols_{key[2:]}"]]
    if os.path.exists(p):
        with np.load(p, allow_pickle=True) as z:
            if bool(z["verified"]) and z["X"].shape[0] == len(out["y"]):
                out["X"]["probe"] = z["X"].astype(np.float64)
                out["cols"]["probe"] = [f"r{i}" for i in range(z["X"].shape[1])]
    return out


@app.function(timeout=21600, **cpu_kw)
def emit_scores(slug: str):
    """Fit each signal's score function on a grouped GSM8K train split (one fit
    per outer seed), then emit per-trace scores for held-out GSM8K and for every
    other benchmark of this model."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    results_vol.reload()
    benches = sorted(b for (s, b) in CELLS if s == slug)
    if "gsm8k" not in benches:
        return {"slug": slug, "error": "no gsm8k cell"}
    cells = {b: _load_cell(slug, b) for b in benches}
    src = cells["gsm8k"]

    out = {"slug": slug, "generated_at": now_iso(), "seeds": OUTER_SEEDS,
           "train_frac": TRAIN_FRAC, "signals": {}, "cells": {}}
    arrays = {}
    for b in benches:
        arrays[f"y__{b}"] = cells[b]["y"]
        arrays[f"groups__{b}"] = cells[b]["groups"].astype(str)
        out["cells"][b] = {"n": int(len(cells[b]["y"])),
                           "signals_available": sorted(cells[b]["X"])}

    for seed in OUTER_SEEDS:
        rng = np.random.default_rng(1000 + seed)
        ug = np.unique(src["groups"])
        perm = rng.permutation(len(ug))
        n_tr = int(round(TRAIN_FRAC * len(ug)))
        train_g = set(ug[perm[:n_tr]])
        tr_mask = np.array([g in train_g for g in src["groups"]])
        arrays[f"trainmask__gsm8k__s{seed}"] = tr_mask

        for sig in SIGNALS:
            if sig not in src["X"]:
                continue
            Xtr, ytr = src["X"][sig][tr_mask], src["y"][tr_mask]
            if len(np.unique(ytr)) < 2:
                continue
            C = PROBE_C if sig == "probe" else DEFAULT_C
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=C, max_iter=3000))
            model.fit(Xtr, ytr)
            # in-domain: score every gsm8k trace (train rows flagged, dropped later)
            arrays[f"s__gsm8k__{sig}__s{seed}"] = model.predict_proba(
                src["X"][sig])[:, 1]
            # shift: same frozen scorer applied to other benchmarks
            for b in benches:
                if b == "gsm8k" or sig not in cells[b]["X"]:
                    continue
                cs, ct = src["cols"][sig], cells[b]["cols"][sig]
                if cs == ct:
                    Xb = cells[b]["X"][sig]
                else:
                    common = [c for c in cs if c in ct]
                    if len(common) < len(cs):     # scorer needs all its columns
                        continue
                    idx = [ct.index(c) for c in cs]
                    Xb = cells[b]["X"][sig][:, idx]
                arrays[f"s__{b}__{sig}__s{seed}"] = model.predict_proba(Xb)[:, 1]
            out["signals"].setdefault(sig, []).append(seed)

    os.makedirs(f"{RESULTS}/conformal", exist_ok=True)
    np.savez_compressed(f"{RESULTS}/conformal/scores_{slug}.npz", **arrays)
    write_json(f"{RESULTS}/conformal/scores_{slug}_meta.json", out)
    results_vol.commit()
    log(f"emit_scores {slug}: {len(arrays)} arrays, benches={benches}")
    return out


@app.local_entrypoint()
def meta_entry():
    r = cell_meta.remote()
    print(json.dumps(r, indent=2, default=str))


@app.local_entrypoint()
def scores_entry():
    todo = sorted(CELLS)
    res = list(build_probe.starmap(todo))
    bad = [r for r in res if r.get("error") or r.get("verified") is False]
    print(json.dumps({"probe_cells": len(res), "problem_cells": bad}, indent=2,
                     default=str))
    out = list(emit_scores.map(MODELS))
    print(json.dumps([{k: o.get(k) for k in ("slug", "error", "cells")}
                      for o in out], indent=2, default=str)[:3000])
