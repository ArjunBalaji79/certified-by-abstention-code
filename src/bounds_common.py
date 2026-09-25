"""Helpers copied verbatim (inspect.getsource) from the original analysis pipeline
so that every number in the paper reproduces bit-for-bit.  Only what the paper's
analyses use is kept."""
import numpy as np
from scipy import stats
from .paths import SCORES

DELTA = 0.1
N_REPL = 200
SEEDS = [0, 1, 2, 3, 4]
CAL_SHARE = 4.0 / 7.0
SIGNALS = ['confidence', 'jspace', 'probe']
MIN_K = 10
N_GRID = 20


def hb_pvalue(n_err, n_acc, alpha):
    """Hoeffding-Bentkus p-value for H0: risk > alpha (Bates et al. LTT)."""
    if n_acc == 0:
        return 1.0
    rhat = n_err / n_acc
    if rhat >= alpha:
        return 1.0
    h = (rhat * np.log(max(rhat, 1e-12) / alpha)
         + (1 - rhat) * np.log((1 - rhat) / (1 - alpha)))
    p_hoef = float(np.exp(-n_acc * h))
    p_bent = float(np.e * stats.binom.cdf(np.ceil(n_acc * rhat), n_acc, alpha))
    return min(1.0, min(p_hoef, p_bent))


def group_index(groups, mask=None):
    out = {}
    for i, g in enumerate(groups):
        if mask is not None and not mask[i]:
            continue
        out.setdefault(g, []).append(i)
    return {g: np.array(v) for g, v in out.items()}

