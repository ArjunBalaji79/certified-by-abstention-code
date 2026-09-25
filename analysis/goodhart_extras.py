"""Best-of-n certificate statistics that do NOT average empty certificates in as
zero error. Same data, estimator, seeds and replicate stream as
the original pipeline (helpers in src/goodhart_common.py).

Per (model, signal, arm, n):
  coverage, p_empty, err_all_reps (original convention), err_nonempty,
  err_pooled = sum(accepted errors)/sum(accepted), viol_rate_all,
  policy_err = error rate of the BoN-selected test traces before any certificate.

Writes outputs/json/GOODHART_EXTRAS.json.
"""

import json
import os
import sys
from functools import lru_cache

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src import paths  # noqa: E402
from src import goodhart_common as ga  # noqa: E402
from src import cert  # noqa: E402
CONF = paths.SCORES

N_RESPLIT = 1000


@lru_cache(maxsize=None)
def cp_upper(n_err, n_acc, delta):
    if n_acc == 0 or n_err >= n_acc:
        return 1.0
    return float(stats.beta.ppf(1 - delta, n_err + 1, n_acc - n_err))


ga.cp_upper = cp_upper          # memoised drop-in; same values


def run_model(slug):
    d = ga.load(slug)
    y, K = d["y"], d["K"]
    P = len(d["problems"])
    rng = np.random.default_rng(777)
    idx0 = ga.draw_idx(d, rng)
    alpha, alpha_seeds = cert.pilot_pi0(slug, CONF)          # exogenous pilot: GSM8K training-partition base error
    res = {"alpha": round(alpha, 4), "alpha_pilot_by_seed": alpha_seeds, "bon_sample_base_err": round(1.0 - float(y[idx0].mean()), 4), "n_problems": int(P), "signals": {}}
    for sig in ga.SIGNALS:
        si = ga.per_problem_sorted(d, sig, idx0)
        S = d["S"][sig]
        entry = {}
        for arm in ("naive", "corrected"):
            per_n = {}
            for n in ga.N_LEVELS:
                cov, err, nacc, nerr, pol = [], [], [], [], []
                for r in range(N_RESPLIT):
                    rb = np.random.default_rng(50_000 + 97 * r)
                    pm = rb.permutation(P)
                    ncal = int(round(ga.CAL_SHARE * P))
                    cal_p, tst_p = pm[:ncal], pm[ncal:]
                    if arm == "naive":
                        cal_rows = np.array([si[p, rb.integers(K)] for p in cal_p])
                    else:
                        cal_rows = np.array([
                            si[p, rb.choice(K, size=n, replace=False).max()]
                            for p in cal_p])
                    tst_rows = np.array([
                        si[p, rb.choice(K, size=n, replace=False).max()]
                        for p in tst_p])
                    cs, cy = S[cal_rows], y[cal_rows]
                    ts, ty = S[tst_rows], y[tst_rows]
                    if int((cy == 0).sum()) == 0 or len(np.unique(ty)) < 2:
                        continue
                    lam = cert.cp_threshold_exact(cs, cy, alpha)
                    sel = ts >= lam
                    cov.append(float(sel.mean()))
                    err.append(ga.err_among(sel, ty))
                    nacc.append(int(sel.sum()))
                    nerr.append(int((ty[sel] == 0).sum()))
                    pol.append(float(1 - ty.mean()))
                cov, err = np.array(cov), np.array(err)
                nacc, nerr = np.array(nacc), np.array(nerr)
                ne = nacc > 0
                per_n[n] = {
                    "coverage": round(float(cov.mean()), 4),
                    "coverage_first200": round(float(cov[:200].mean()), 4),
                    "err_all_reps_first200": round(float(err[:200].mean()), 4),
                    "p_empty": round(float(1 - ne.mean()), 4),
                    "err_all_reps": round(float(err.mean()), 4),
                    "err_nonempty": round(float(err[ne].mean()), 4) if ne.any() else None,
                    "err_pooled": round(float(nerr.sum() / nacc.sum()), 4) if nacc.sum() else None,
                    "viol_rate_all": round(float((err > alpha).mean()), 4),
                    "policy_err": round(float(np.mean(pol)), 4),
                    "n_repl": int(len(cov)),
                }
            entry[arm] = per_n
        res["signals"][sig] = entry
        print(slug, sig, flush=True)
    return res


def main():
    out = {m: run_model(m) for m in ga.FULL_MODELS}
    with open(os.path.join(paths.OUT_JSON, "GOODHART_EXTRAS.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
