"""Helpers copied verbatim (inspect.getsource) from the original analysis pipeline
so that every number in the paper reproduces bit-for-bit.  Only what the paper's
analyses use is kept."""
import os
from math import comb
import numpy as np
from scipy import stats
from .paths import BON as GH

N_LEVELS = [1, 2, 4, 8, 16, 32]
K_FIXED = 32
CAL_SHARE = 0.5
FULL_MODELS = ['qwen3.5-4b', 'qwen3-4b']
SIGNALS = ['confidence', 'jspace', 'logitlens', 'surface', 'probe']


def bon_weights(K, n):
    denom = comb(K, n)
    return np.array([comb(j - 1, n - 1) / denom for j in range(1, K + 1)])


def cp_upper(n_err, n_acc, delta):
    if n_acc == 0 or n_err >= n_acc:
        return 1.0
    return float(stats.beta.ppf(1 - delta, n_err + 1, n_acc - n_err))


def err_among(sel, y):
    m = int(sel.sum())
    return (float((y[sel] == 0).sum()) / m) if m else 0.0


def load(slug):
    p = os.path.join(GH, f"scores_{slug}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    y = d["y"].astype(int)
    groups = np.array([str(g) for g in d["groups"]])
    sigs = sorted(k[3:] for k in d if k.startswith("s__"))
    # group traces by problem; require a common K across problems
    by_p = {}
    for i, g in enumerate(groups):
        by_p.setdefault(g, []).append(i)
    Ks = {len(v) for v in by_p.values()}
    probs = [g for g in sorted(by_p) if len(by_p[g]) >= K_FIXED]
    pools = [np.array(sorted(by_p[g])) for g in probs]
    rng = np.random.default_rng(4242)
    tiebreak = rng.random(len(y))
    return {"y": y, "groups": groups, "signals": sigs, "K": K_FIXED,
            "problems": np.array(probs), "pools": pools, "tiebreak": tiebreak,
            "S": {s: d[f"s__{s}"] for s in sigs},
            "K_spread": [int(min(Ks)), int(np.median(list(Ks))), int(max(Ks))],
            "n_problems_total": len(by_p), "n_problems_kept": len(probs)}


def draw_idx(d, rng):
    """[P, K] index matrix: K traces drawn without replacement per problem."""
    return np.array([p[rng.choice(len(p), size=d["K"], replace=False)]
                     for p in d["pools"]])


def per_problem_sorted(d, sig, idx):
    """Given a [P,K] index matrix, sort each row ASCENDING by signal score."""
    sc = d["S"][sig][idx]
    tb = d["tiebreak"][idx]
    order = np.argsort(sc + 1e-12 * tb, kind="stable")
    return np.take_along_axis(idx, order, axis=1)

