"""Helpers copied verbatim (inspect.getsource) from the original analysis pipeline
so that every number in the paper reproduces bit-for-bit.  Only what the paper's
analyses use is kept."""
import os
import numpy as np
from scipy import stats
from .paths import SCORES, FEATURES as DEPTHS

READABLE = ['confidence', 'jspace', 'logitlens', 'surface']
SEEDS = [0, 1, 2, 3, 4]
TRAIN_FRAC = 0.3
CAL_SHARE = 4.0 / 7.0
PROBE_C, READ_C = 0.01, 1.0


def auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    r = stats.rankdata(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_index(groups, mask=None):
    out = {}
    for i, g in enumerate(groups):
        if mask is not None and not mask[i]:
            continue
        out.setdefault(g, []).append(i)
    return {g: np.array(v) for g, v in out.items()}


def load(slug):
    sp = os.path.join(SCORES, f"scores_{slug}.npz")
    dp = os.path.join(DEPTHS, f"probe_depths_{slug}.npz")
    fp = os.path.join(DEPTHS, f"features_{slug}.npz")
    if not (os.path.exists(sp) and os.path.exists(dp) and os.path.exists(fp)):
        return None
    S = dict(np.load(sp))
    D = dict(np.load(dp))
    F = dict(np.load(fp))
    y = F["y"].astype(int)
    groups = np.array([str(g) for g in F["groups"]])
    X = {}
    for sig in READABLE:
        k = {"confidence": "X_confidence_only", "jspace": "X_jspace_only",
             "logitlens": "X_logitlens_only", "surface": "X_surface_only"}[sig]
        if k in F:
            X[sig] = np.nan_to_num(F[k].astype(np.float64))
    if "X_length_only" in F:
        X["length"] = np.nan_to_num(F["X_length_only"].astype(np.float64))
    blocks = {k: D[k].astype(np.float64) for k in
              ("early_step", "early_ans", "mid_step", "mid_ans",
               "late_step", "late_ans") if k in D}
    if len(y) != next(iter(blocks.values())).shape[0]:
        return None
    return {"y": y, "groups": groups, "X": X, "blocks": blocks, "S": S}

