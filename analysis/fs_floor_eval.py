"""FS-floor evaluation (Section 5):
  (a) sweep the start constant c in {1,1.5,2,3,4};
  (b) cross-fitted threshold grid: quantiles of OUT-OF-FOLD scores on the training
      problems (5-fold refit of the same scorer recipe, locally), instead of
      in-sample training scores;
  (c) c chosen BEFORE calibration, per cell, by a Gaussian simulation at the
      training split's out-of-fold AUC, its base error and n_cal (so validity is
      untouched; the choice uses no calibration data).
Cached cal/test scores are used; a local full-train refit is compared to the
cached scores as a sanity check (Spearman 1.0 expected).
Writes outputs/json/FS_FLOOR2.json.
"""
import json, math, os, sys
from functools import lru_cache
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src import paths  # noqa: E402
from src import conformal_common as ca  # noqa: E402
from src import fs_floor as fs  # noqa: E402

CS = (1.0, 1.5, 2.0, 3.0, 4.0)
FEAT = {"confidence": "X_confidence_only", "jspace": "X_jspace_only", "probe": None}
CREG = {"confidence": 1.0, "jspace": 1.0, "probe": 0.01}
DEPTHS = paths.FEATURES


def features(slug, sig):
    F = np.load(os.path.join(DEPTHS, f"features_{slug}.npz"))
    if sig == "probe":
        D = np.load(os.path.join(DEPTHS, f"probe_depths_{slug}.npz"))
        return np.concatenate([D["mid_step"], D["mid_ans"]], axis=1).astype(np.float32)
    return np.nan_to_num(F[FEAT[sig]].astype(np.float64))


def oof_scores(X, y, groups, train, C):
    """Out-of-fold scores on training rows (5-fold by problem) and a full-train fit."""
    idx = np.where(train)[0]
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X[idx], y[idx], groups[idx]):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[idx[tr]], y[idx[tr]])
        oof[idx[te]] = m.predict_proba(X[idx[te]])[:, 1]
    full = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[idx], y[idx])
    return oof, full.predict_proba(X)[:, 1]


_SIM = {}
def c_from_sim(auc_, pi0, n_cal, rho, n_rep=1500):
    """Pre-calibration choice of c: Gaussian model at (AUC, pi0, n_cal), pick the c
    with the highest simulated mean coverage at target rho*pi0."""
    key = (round(auc_, 2), round(pi0, 2), int(n_cal), rho)
    if key in _SIM:
        return _SIM[key]
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    mu = math.sqrt(2) * stats.norm.ppf(max(min(auc_, 0.99), 0.51))
    alpha = rho * pi0
    def Cov(lam):
        return pi0 * stats.norm.sf(lam) + (1 - pi0) * stats.norm.sf(lam - mu)
    tot = {c: 0.0 for c in CS}
    for _ in range(n_rep):
        ytr = (rng.random(n_cal) >= pi0).astype(int)
        lams = np.quantile(rng.standard_normal(n_cal) + mu * ytr, 1 - fs.Q_GRID); lams[-1] = -np.inf
        yy = (rng.random(n_cal) >= pi0).astype(int); ss = rng.standard_normal(n_cal) + mu * yy
        for c in CS:
            lam = fs.fs_floor(lams, ss, yy, alpha, c)
            if lam != np.inf:
                tot[c] += Cov(lam) if np.isfinite(lam) else 1.0
    best = max(CS, key=lambda c: tot[c])
    _SIM[key] = (best, {c: round(tot[c] / n_rep, 4) for c in CS})
    return _SIM[key]


def real():
    out, sanity = {}, {}
    for fn in sorted(os.listdir(ca.SCORES)):
        if not (fn.startswith("scores_") and fn.endswith(".npz")):
            continue
        slug = fn[len("scores_"):-4]
        md = ca.load_model(slug)
        if md is None or "y__gsm8k" not in md["data"]:
            continue
        d = md["data"]; y = d["y__gsm8k"].astype(int); groups = d["groups__gsm8k"].astype(str)
        for sig in ca.CORE_SIGNALS:
            X = features(slug, sig)
            per_seed = {}
            for seed in ca.OUTER_SEEDS:
                tm, key = f"trainmask__gsm8k__s{seed}", f"s__gsm8k__{sig}__s{seed}"
                if tm not in d or key not in d:
                    continue
                train = d[tm].astype(bool)
                oof, full_local = oof_scores(X, y, groups, train, CREG[sig])
                sanity[f"{slug}|{sig}|s{seed}"] = round(float(stats.spearmanr(full_local[~train], d[key][~train])[0]), 4)
                lam_in = np.quantile(d[key][train], 1 - fs.Q_GRID); lam_in[-1] = -np.inf
                lam_x = np.quantile(oof[train], 1 - fs.Q_GRID); lam_x[-1] = -np.inf
                auc_oof = ca.auc(y[train], oof[train]); pi0_tr = 1 - float(y[train].mean())
                gidx = ca.group_index(groups, ~train); gl = [gidx[g] for g in sorted(gidx)]
                n_cal = int(round(ca.CAL_SHARE_OF_POOL * len(gl)))
                per_seed[seed] = (train, key, lam_in, lam_x, auc_oof, pi0_tr, gl, n_cal)
            for rho in (1.0, 0.75, 0.5):
                alphas = []
                names = ["cp_bonf"] + [f"in_c{c}" for c in CS] + [f"x_c{c}" for c in CS] + ["in_csim", "x_csim"]
                A = {m: dict(cov=[], nacc=[], nerr=[]) for m in names}
                csim_log = {}
                for seed, (train, key, lam_in, lam_x, auc_oof, pi0_tr, gl, n_cal) in per_seed.items():
                    alpha = rho * pi0_tr; alphas.append(alpha)
                    c_sim, sim_cov = c_from_sim(auc_oof, pi0_tr, n_cal, rho)
                    csim_log[seed] = {"auc_oof": round(auc_oof, 3), "pi0_train": round(pi0_tr, 3), "c": c_sim, "sim_cov": sim_cov}
                    for r in range(ca.N_REPL):
                        rng = np.random.default_rng(10_000 + 1_000_000 * seed + r)
                        rows = ca.one_per_problem(rng, gl); perm = rng.permutation(len(gl))
                        cal, tst = rows[perm[:n_cal]], rows[perm[n_cal:]]
                        cs, cy, ts, ty = d[key][cal], y[cal], d[key][tst], y[tst]
                        if (cy == 0).sum() == 0 or len(np.unique(ty)) < 2:
                            continue
                        lam = {"cp_bonf": fs.cp_bonf(cs, cy, alpha)}
                        for c in CS:
                            lam[f"in_c{c}"] = fs.fs_floor(lam_in, cs, cy, alpha, c)
                            lam[f"x_c{c}"] = fs.fs_floor(lam_x, cs, cy, alpha, c)
                        lam["in_csim"] = lam[f"in_c{c_sim}"]; lam["x_csim"] = lam[f"x_c{c_sim}"]
                        for m, l in lam.items():
                            sel = ts >= l
                            A[m]["cov"].append(float(sel.mean())); A[m]["nacc"].append(int(sel.sum())); A[m]["nerr"].append(int((ty[sel] == 0).sum()))
                out[f"{slug}|rho={rho}|{sig}"] = {"alpha": round(float(np.mean(alphas)), 4), "c_sim": csim_log, **{
                    m: {"coverage": round(float(np.mean(a["cov"])), 4),
                        "p_fire": round(float(np.mean(np.array(a["nacc"]) > 0)), 4),
                        "R_over_alpha": (round(sum(a["nerr"]) / sum(a["nacc"]) / alpha, 3) if sum(a["nacc"]) else None)}
                    for m, a in A.items()}}
            print(slug, sig, {k: v for k, v in sanity.items() if k.startswith(f"{slug}|{sig}")}, flush=True)
    return out, sanity


def synthetic(n_rep=20000, seed=3):
    rng = np.random.default_rng(seed); out = {}
    for pi0, auc_, n_cal in ((0.16, 0.82, 154), (0.16, 0.71, 154), (0.22, 0.76, 78), (0.09, 0.66, 153)):
        mu = math.sqrt(2) * stats.norm.ppf(auc_)
        def R(lam):
            a0 = pi0 * stats.norm.sf(lam); a1 = (1 - pi0) * stats.norm.sf(lam - mu); return a0 / (a0 + a1)
        def Cov(lam):
            return pi0 * stats.norm.sf(lam) + (1 - pi0) * stats.norm.sf(lam - mu)
        for rho in (0.75, 0.5):
            alpha = rho * pi0; T = {}
            for _ in range(n_rep):
                ytr = (rng.random(n_cal) >= pi0).astype(int)
                lams = np.quantile(rng.standard_normal(n_cal) + mu * ytr, 1 - fs.Q_GRID); lams[-1] = -np.inf
                yy = (rng.random(n_cal) >= pi0).astype(int); ss = rng.standard_normal(n_cal) + mu * yy
                lam = {"cp_bonf": fs.cp_bonf(ss, yy, alpha)}
                for c in CS:
                    lam[f"fs_c{c}"] = fs.fs_floor(lams, ss, yy, alpha, c)
                for m, l in lam.items():
                    t = T.setdefault(m, [0, 0, 0.0])
                    if l != np.inf:
                        t[0] += 1; t[1] += int(R(l) > alpha) if np.isfinite(l) else int(pi0 > alpha); t[2] += Cov(l) if np.isfinite(l) else 1.0
            out[f"pi0={pi0}|auc={auc_}|ncal={n_cal}|rho={rho}"] = {
                "c_sim_rule": c_from_sim(auc_, pi0, n_cal, rho)[0],
                **{m: {"p_fire": round(t[0] / n_rep, 4), "p_fail": round(t[1] / n_rep, 4),
                       "p_fail_given_fire": round(t[1] / t[0], 4) if t[0] else None,
                       "coverage": round(t[2] / n_rep, 4)} for m, t in T.items()}}
            print("syn", pi0, auc_, n_cal, rho, flush=True)
    return out


if __name__ == "__main__":
    syn = synthetic()
    real_out, sanity = real()
    json.dump({"synthetic": syn, "real": real_out, "local_refit_spearman_vs_cached": sanity},
              open(os.path.join(paths.OUT_JSON, "FS_FLOOR2.json"), "w"), indent=1)
    for rho in (1.0, 0.75, 0.5):
        cells = {k: v for k, v in real_out.items() if f"|rho={rho}|" in k}
        print(f"\nREAL rho={rho}")
        for m in ["cp_bonf"] + [f"in_c{c}" for c in CS] + [f"x_c{c}" for c in CS] + ["in_csim", "x_csim"]:
            cov = [v[m]["coverage"] for v in cells.values()]; pf = [v[m]["p_fire"] for v in cells.values()]
            print(f"  {m:9s} cov {np.mean(cov):.3f} >=10%: {sum(c >= .1 for c in cov)}/21  P_fire {np.mean(pf):.3f}  <=delta: {sum(p <= .1 for p in pf)}/21  beatsCP {sum(v[m]['coverage'] > v['cp_bonf']['coverage'] for v in cells.values())}/21")
