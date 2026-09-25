"""Shared certificate primitives for the paper's analyses.

cp_threshold_exact: Clopper-Pearson selective threshold on a Bonferroni grid of
accepted-set sizes k, testing the k-1 calibration points STRICTLY ABOVE the k-th
order statistic (Proposition 1 of the paper; valid with no assumption beyond
i.i.d. continuous scores).  Acceptance on test is s > s_(k); the returned value
is nextafter(s_(k)) so that callers may use s >= lam.  cp_threshold_boundary is
the older variant that also counts the boundary point (appendix comparison).
train_pi0: base error rate of the training partition (target alpha = rho * pi0
is fixed before any calibration data is seen).
"""
from functools import lru_cache
import numpy as np
from scipy import stats

DELTA, N_GRID, MIN_K = 0.10, 20, 10


@lru_cache(maxsize=None)
def cp_upper(e, k, delta):
    if k <= 0 or e >= k:
        return 1.0
    return float(stats.beta.ppf(1 - delta, e + 1, k - e))


def _grid(n):
    return np.unique(np.linspace(MIN_K, n, N_GRID).astype(int))


def cp_threshold_exact(cs, cy, alpha, delta=DELTA):
    o = np.argsort(-cs, kind="stable")
    s = cs[o]
    cum = np.cumsum((cy[o] == 0).astype(int))
    n = len(s)
    if n < MIN_K:
        return np.inf
    ks = _grid(n)
    d = delta / len(ks)
    lam = np.inf
    for k in ks:                                   # ascending: last pass = most permissive
        e = int(cum[k - 2]) if k >= 2 else 0       # errors among the k-1 points above s_(k)
        if cp_upper(e, int(k - 1), d) <= alpha:
            lam = np.nextafter(s[k - 1], np.inf)   # accept s > s_(k)
    return lam


def cp_threshold_boundary(cs, cy, alpha, delta=DELTA):
    o = np.argsort(-cs, kind="stable")
    s = cs[o]
    cum = np.cumsum((cy[o] == 0).astype(int))
    n = len(s)
    if n < MIN_K:
        return np.inf
    ks = _grid(n)
    d = delta / len(ks)
    lam = np.inf
    for k in ks:
        if cp_upper(int(cum[k - 1]), int(k), d) <= alpha:
            lam = s[k - 1]                          # accept s >= s_(k)
    return lam


def train_pi0(y, trainmask):
    return 1.0 - float(np.asarray(y)[np.asarray(trainmask).astype(bool)].mean())


def pilot_pi0(slug, scores_dir):
    """Mean over seeds of the GSM8K training-partition base error: an exogenous
    pilot estimate for experiments on other problems of the same model."""
    import os
    z = np.load(os.path.join(scores_dir, f"scores_{slug}.npz"), allow_pickle=True)
    y = z["y__gsm8k"].astype(int)
    vals = [train_pi0(y, z[k]) for k in z.files if k.startswith("trainmask__gsm8k__s")]
    return float(np.mean(vals)), [round(v, 4) for v in vals]
