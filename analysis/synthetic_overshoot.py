"""How wrong are the wrong certificates?  Replays the Gaussian-model draws of
bounds_extras.synthetic() (same seed and order) and records R(lambda_hat)/alpha
for firing draws: mean over firing draws, and median / 90th percentile over
FAILING draws (R > alpha).  Writes SYNTH_OVERSHOOT.json."""
import json, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from src import paths  # noqa: E402
import bounds_extras as be  # noqa: E402

def main(n_rep=20000, seed=11):
    rng = np.random.default_rng(seed); out = {}
    for pi0, auc_, n_cal in ((0.16, 0.82, 154), (0.16, 0.71, 154), (0.22, 0.76, 78), (0.09, 0.66, 153)):
        mu = np.sqrt(2) * stats.norm.ppf(auc_)
        def R(lam):
            a0 = pi0 * stats.norm.sf(lam); a1 = (1 - pi0) * stats.norm.sf(lam - mu); return a0 / (a0 + a1)
        for rho in (0.75, 0.5):
            alpha = rho * pi0; rat = {}
            for _ in range(n_rep):
                yy = (rng.random(n_cal) >= pi0).astype(int); ss = rng.standard_normal(n_cal) + mu * yy
                for m, lam in be.thresholds(ss, yy, alpha).items():
                    if np.isfinite(lam):
                        rat.setdefault(m, []).append(R(lam) / alpha)
            res = {}
            for m, r in rat.items():
                r = np.array(r); bad = r[r > 1]
                res[m] = {"n_fire": int(len(r)), "mean_R_over_alpha_fire": round(float(r.mean()), 3),
                          "n_fail": int(len(bad)),
                          "median_R_over_alpha_fail": round(float(np.median(bad)), 3) if len(bad) else None,
                          "p90_R_over_alpha_fail": round(float(np.percentile(bad, 90)), 3) if len(bad) else None,
                          "max_R_over_alpha_fail": round(float(bad.max()), 3) if len(bad) else None}
            out[f"pi0={pi0}|auc={auc_}|ncal={n_cal}|rho={rho}"] = res
            print(pi0, auc_, n_cal, rho, {m: (v["n_fail"], v["median_R_over_alpha_fail"], v["p90_R_over_alpha_fail"]) for m, v in res.items() if m in ("cp", "walk_from_first_pass")}, flush=True)
    json.dump(out, open(os.path.join(paths.OUT_JSON, "SYNTH_OVERSHOOT.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
