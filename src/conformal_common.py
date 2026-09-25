"""Helpers copied verbatim (inspect.getsource) from the original analysis pipeline
so that every number in the paper reproduces bit-for-bit.  Only what the paper's
analyses use is kept."""
import os
import numpy as np
from scipy import stats
from .paths import SCORES

DELTA_CP = 0.1
N_REPL = 200
OUTER_SEEDS = [0, 1, 2, 3, 4]
CAL_SHARE_OF_POOL = 4.0 / 7.0
RHOS = [0.5, 0.75, 1.0]
ABS_ALPHAS = [0.05, 0.1, 0.2]
SIGNALS = ['confidence', 'jspace', 'probe', 'logitlens', 'surface']
CORE_SIGNALS = ['confidence', 'jspace', 'probe']


def bh_select(pvals, alpha):
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals, kind="stable")
    ok = pvals[order] <= alpha * np.arange(1, m + 1) / m
    sel = np.zeros(m, dtype=bool)
    if ok.any():
        sel[order[: np.max(np.where(ok)[0]) + 1]] = True
    return sel


def conformal_pvals(cal_scores, cal_y, test_scores):
    null = np.sort(cal_scores[cal_y == 0])
    n0 = len(null)
    if n0 == 0:
        return np.ones(len(test_scores)), 0
    ge = n0 - np.searchsorted(null, test_scores, side="left")
    return (1.0 + ge) / (n0 + 1.0), n0


def err_among_accepted(sel, y):
    n = int(sel.sum())
    return (float((y[sel] == 0).sum()) / n) if n else 0.0


def group_index(groups, mask=None):
    out = {}
    for i, g in enumerate(groups):
        if mask is not None and not mask[i]:
            continue
        out.setdefault(g, []).append(i)
    return {g: np.array(v) for g, v in out.items()}


def one_per_problem(rng, gl):
    return np.array([ix[rng.integers(len(ix))] for ix in gl])


def auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    r = stats.rankdata(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_model(slug):
    p = os.path.join(SCORES, f"scores_{slug}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    return {"data": d,
            "benches": sorted({k.split("__")[1] for k in d if k.startswith("y__")}),
            "signals": sorted({k.split("__")[2] for k in d if k.startswith("s__")})}

