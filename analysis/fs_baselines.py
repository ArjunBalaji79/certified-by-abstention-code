"""Two cheap baselines for FS-floor and a pre-calibration prediction of P_fire.
  cp_G5, cp_G10   rule (1) with a 5- or 10-point Bonferroni grid (G=20 is the paper's)
  fs_start        the floor start alone: test the single fixed threshold at j0 at level delta
  pred_pfire      P_fire predicted by the same Gaussian simulation that chooses c
Same cross-fitted grid, same c, same replicate stream as fs_floor_eval.py.  Writes FS_BASELINES.json."""
import json, math, os, sys
from functools import lru_cache
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from src import paths  # noqa: E402
from src import conformal_common as ca  # noqa: E402
from src import fs_floor as fs  # noqa: E402
import fs_floor_eval as f2  # noqa: E402
from src import cert  # noqa: E402

CS = f2.CS

def cp_exact_G(cs, cy, alpha, n_grid, delta=cert.DELTA):
    o = np.argsort(-cs, kind="stable"); s = cs[o]; cum = np.cumsum((cy[o] == 0).astype(int)); n = len(s)
    if n < cert.MIN_K: return np.inf
    ks = np.unique(np.linspace(cert.MIN_K, n, n_grid).astype(int)); d = delta / len(ks); lam = np.inf
    for k in ks:
        e = int(cum[k - 2]) if k >= 2 else 0
        if cert.cp_upper(e, int(k - 1), d) <= alpha: lam = np.nextafter(s[k - 1], np.inf)
    return lam

def fs_start(lams, cs, cy, alpha, c, delta=fs.DELTA):
    n = len(cs); need = c * fs.floor0(alpha, delta)
    j0 = next((j for j, q in enumerate(fs.Q_GRID) if q * n >= need), None)
    if j0 is None: return np.inf
    acc = cs >= lams[j0]
    return lams[j0] if fs.binom_p(int((cy[acc] == 0).sum()), int(acc.sum()), alpha) <= delta else np.inf

_SIM = {}
def sim_choose(auc_, pi0, n_cal, rho, n_rep=1500):
    key = (round(auc_, 2), round(pi0, 2), int(n_cal), rho)
    if key in _SIM: return _SIM[key]
    rng = np.random.default_rng(abs(hash(key)) % (2**32)); mu = math.sqrt(2) * stats.norm.ppf(max(min(auc_, 0.99), 0.51)); alpha = rho * pi0
    def Cov(lam): return pi0 * stats.norm.sf(lam) + (1 - pi0) * stats.norm.sf(lam - mu)
    cov = {c: 0.0 for c in CS}; fire = {c: 0 for c in CS}
    for _ in range(n_rep):
        ytr = (rng.random(n_cal) >= pi0).astype(int); lams = np.quantile(rng.standard_normal(n_cal) + mu * ytr, 1 - fs.Q_GRID); lams[-1] = -np.inf
        yy = (rng.random(n_cal) >= pi0).astype(int); ss = rng.standard_normal(n_cal) + mu * yy
        for c in CS:
            lam = fs.fs_floor(lams, ss, yy, alpha, c)
            if lam != np.inf: fire[c] += 1; cov[c] += Cov(lam) if np.isfinite(lam) else 1.0
    best = max(CS, key=lambda c: cov[c]); _SIM[key] = (best, fire[best] / n_rep); return _SIM[key]

def main():
    out = {}
    for fn in sorted(os.listdir(ca.SCORES)):
        if not (fn.startswith("scores_") and fn.endswith(".npz")): continue
        slug = fn[len("scores_"):-4]; md = ca.load_model(slug)
        if md is None or "y__gsm8k" not in md["data"]: continue
        d = md["data"]; y = d["y__gsm8k"].astype(int); groups = d["groups__gsm8k"].astype(str)
        for sig in ca.CORE_SIGNALS:
            X = f2.features(slug, sig); per_seed = {}
            for seed in ca.OUTER_SEEDS:
                tm, key = f"trainmask__gsm8k__s{seed}", f"s__gsm8k__{sig}__s{seed}"
                if tm not in d or key not in d: continue
                train = d[tm].astype(bool); oof, _ = f2.oof_scores(X, y, groups, train, f2.CREG[sig])
                lam_x = np.quantile(oof[train], 1 - fs.Q_GRID); lam_x[-1] = -np.inf
                gidx = ca.group_index(groups, ~train); gl = [gidx[g] for g in sorted(gidx)]
                per_seed[seed] = (key, lam_x, ca.auc(y[train], oof[train]), 1 - float(y[train].mean()), gl, int(round(ca.CAL_SHARE_OF_POOL * len(gl))))
            for rho in (1.0, 0.75, 0.5):
                A = {m: dict(cov=[], nacc=[], nerr=[]) for m in ("cp_G20", "cp_G10", "cp_G5", "fs_start")}; preds = []
                for seed, (key, lam_x, auc_oof, pi0_tr, gl, n_cal) in per_seed.items():
                    alpha = rho * pi0_tr; c_sim, pf_pred = sim_choose(auc_oof, pi0_tr, n_cal, rho); preds.append(pf_pred)
                    for r in range(ca.N_REPL):
                        rng = np.random.default_rng(10_000 + 1_000_000 * seed + r)
                        rows = ca.one_per_problem(rng, gl); perm = rng.permutation(len(gl))
                        cal, tst = rows[perm[:n_cal]], rows[perm[n_cal:]]
                        cs, cy, ts, ty = d[key][cal], y[cal], d[key][tst], y[tst]
                        if (cy == 0).sum() == 0 or len(np.unique(ty)) < 2: continue
                        lam = {"cp_G20": cp_exact_G(cs, cy, alpha, 20), "cp_G10": cp_exact_G(cs, cy, alpha, 10), "cp_G5": cp_exact_G(cs, cy, alpha, 5),
                               "fs_start": fs_start(lam_x, cs, cy, alpha, c_sim)}
                        for m, l in lam.items():
                            sel = ts >= l; A[m]["cov"].append(float(sel.mean())); A[m]["nacc"].append(int(sel.sum())); A[m]["nerr"].append(int((ty[sel] == 0).sum()))
                out[f"{slug}|rho={rho}|{sig}"] = {"pred_pfire_fs": round(float(np.mean(preds)), 4), **{
                    m: {"coverage": round(float(np.mean(a["cov"])), 4), "p_fire": round(float(np.mean(np.array(a["nacc"]) > 0)), 4),
                        "R_over_alpha": (round(sum(a["nerr"]) / sum(a["nacc"]) / (rho * np.mean([v[3] for v in per_seed.values()])), 3) if sum(a["nacc"]) else None)}
                    for m, a in A.items()}}
            print(slug, sig, flush=True)
    json.dump(out, open(os.path.join(paths.OUT_JSON, "FS_BASELINES.json"), "w"), indent=1)
    F = json.load(open(os.path.join(paths.OUT_JSON, "FS_FLOOR2.json")))["real"]
    for rho in (1.0, 0.75, 0.5):
        cells = {k: v for k, v in out.items() if f"|rho={rho}|" in k}
        print(f"rho={rho}: " + " | ".join(f"{m} cov {np.mean([v[m]['coverage'] for v in cells.values()]):.3f}" for m in ("cp_G20", "cp_G10", "cp_G5", "fs_start"))
              + f" | FS-floor {np.mean([F[k]['x_csim']['coverage'] for k in cells]):.3f} | pred vs real P_fire rho: {stats.spearmanr([v['pred_pfire_fs'] for v in cells.values()], [F[k]['x_csim']['p_fire'] for k in cells])[0]:.2f}")

if __name__ == "__main__":
    main()
