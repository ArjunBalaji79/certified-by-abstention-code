"""Floor-started fixed-sequence certificate (FS-floor), Section 5 of the paper."""
import math
from functools import lru_cache
import numpy as np
from scipy import stats
from . import cert

DELTA = 0.10
Q_GRID = np.round(np.arange(0.02, 1.0001, 0.02), 2)       # accept fractions of the fixed grid


@lru_cache(maxsize=None)
def binom_p(e, n, alpha):
    """p-value for H: risk > alpha, from e errors among n accepted."""
    return 1.0 if n == 0 else float(stats.binom.cdf(e, n, alpha))


def floor0(alpha, delta):
    return math.ceil(math.log(1 / delta) / -math.log(1 - alpha))


def fs_floor(lams, cs, cy, alpha, c, delta=DELTA):
    n = len(cs)
    need = c * floor0(alpha, delta)
    j0 = next((j for j, q in enumerate(Q_GRID) if q * n >= need), None)
    if j0 is None:
        return np.inf
    lam_hat = np.inf
    for j in range(j0, len(lams)):
        acc = cs >= lams[j]
        if binom_p(int((cy[acc] == 0).sum()), int(acc.sum()), alpha) > delta:
            break
        lam_hat = lams[j]
    return lam_hat


def cp_bonf(cs, cy, alpha, delta=DELTA):
    """Baseline: the Bonferroni Clopper-Pearson rule, Eq. (1) of the paper."""
    return cert.cp_threshold_exact(cs, cy, alpha, delta)
