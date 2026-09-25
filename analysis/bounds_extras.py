"""Bound comparison (Section 4.1 and the appendix): Clopper-Pearson, Hoeffding-Bentkus
and a mixture betting e-value in the same selective construction, plus the naive
fixed-sequence walks, on real data and in a Gaussian model with known selective risk.
The null check verifies each per-k test at its nominal level.
Writes outputs/json/BOUNDS_EXTRAS.json.
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
from src import bounds_common as ct  # noqa: E402

DELTA, MIN_K, N_GRID = ct.DELTA, ct.MIN_K, ct.N_GRID
C_GRID = np.linspace(0.05, 0.95, 19)


@lru_cache(maxsize=None)
def cp_upper(e, k, delta):
    if k <= 0 or e >= k:
        return 1.0
    return float(stats.beta.ppf(1 - delta, e + 1, k - e))


@lru_cache(maxsize=None)
def hb_p(e, k, alpha):
    return ct.hb_pvalue(e, k, alpha)


@lru_cache(maxsize=None)
def bet_log_e(e, k, alpha):
    """log of the mixture e-value for H0: risk >= alpha from e errors in k
    Bernoulli trials. Bets lambda = c/(1-alpha), c in (0,1), keep every factor
    1 + lambda (alpha - X) non-negative, so each product is a test supermartingale
    under H0 and their average is an e-value (Waudby-Smith & Ramdas)."""
    lam = C_GRID / (1.0 - alpha)
    logs = (k - e) * np.log1p(lam * alpha) + e * np.log1p(-lam * (1.0 - alpha))
    mx = logs.max()
    return float(mx + np.log(np.mean(np.exp(logs - mx))))


def passes(method, e, k, alpha, d):
    if method == "cp":
        return cp_upper(e, k, d) <= alpha
    if method == "hb":
        return hb_p(e, k, alpha) <= d
    if method == "bet":
        return bet_log_e(e, k, alpha) > np.log(1.0 / d)
    raise ValueError(method)


def thresholds(cs, cy, alpha):
    """All procedures on one calibration draw. Returns {name: lambda}."""
    o = np.argsort(-cs, kind="stable")
    s = cs[o]
    cum = np.cumsum((cy[o] == 0).astype(int))
    n = len(s)
    out = {m: np.inf for m in ("cp", "hb", "bet", "walk_strict", "walk_from_first_pass")}
    if n < MIN_K:
        return out
    ks = np.unique(np.linspace(MIN_K, n, N_GRID).astype(int))
    d = DELTA / len(ks)
    for m in ("cp", "hb", "bet"):
        for k in ks:
            if passes(m, int(cum[k - 2]), int(k - 1), alpha, d):
                out[m] = np.nextafter(s[k - 1], np.inf)
    # naive fixed-sequence walks: every test at the full delta, no split
    ok = [cp_upper(int(cum[k - 2]), int(k - 1), DELTA) <= alpha for k in ks]
    for k, p in zip(ks, ok):                       # strict -> permissive
        if not p:
            break
        out["walk_strict"] = np.nextafter(s[k - 1], np.inf)
    started = False
    for k, p in zip(ks, ok):                       # begin at first pass
        if p:
            started = True
            out["walk_from_first_pass"] = np.nextafter(s[k - 1], np.inf)
        elif started:
            break
    return out


def real_data():
    res = {}
    models = sorted(f[len("scores_"):-4] for f in os.listdir(ct.SCORES)
                    if f.startswith("scores_") and f.endswith(".npz"))
    for slug in models:
        d = dict(np.load(os.path.join(ct.SCORES, f"scores_{slug}.npz")))
        if "y__gsm8k" not in d:
            continue
        y = d["y__gsm8k"].astype(int)
        groups = d["groups__gsm8k"].astype(str)
        for sig in ct.SIGNALS:
            acc = {}
            for seed in ct.SEEDS:
                tm, ks_ = f"trainmask__gsm8k__s{seed}", f"s__gsm8k__{sig}__s{seed}"
                if tm not in d or ks_ not in d:
                    continue
                alpha = 1.0 - float(y[d[tm].astype(bool)].mean())
                gidx = ct.group_index(groups, ~d[tm].astype(bool))
                gl = [gidx[g] for g in sorted(gidx)]
                for r in range(ct.N_REPL):
                    rng = np.random.default_rng(10_000 + 1_000_000 * seed + r)
                    rows = np.array([ix[rng.integers(len(ix))] for ix in gl])
                    perm = rng.permutation(len(rows))
                    ncal = int(round(ct.CAL_SHARE * len(rows)))
                    cal, tst = rows[perm[:ncal]], rows[perm[ncal:]]
                    cs, cy, ts, ty = d[ks_][cal], y[cal], d[ks_][tst], y[tst]
                    if (cy == 0).sum() == 0 or len(np.unique(ty)) < 2:
                        continue
                    for m, lam in thresholds(cs, cy, alpha).items():
                        sel = ts >= lam
                        a = acc.setdefault(m, {"cov": [], "nacc": [], "nerr": [], "alpha": []})
                        a["cov"].append(float(sel.mean()))
                        a["nacc"].append(int(sel.sum()))
                        a["nerr"].append(int((ty[sel] == 0).sum()))
                        a["alpha"].append(alpha)
            res[f"{slug}|{sig}"] = {"alpha": round(float(np.mean([1.0 - float(y[d[f'trainmask__gsm8k__s{sd}'].astype(bool)].mean()) for sd in ct.SEEDS if f'trainmask__gsm8k__s{sd}' in d])), 4), **{
                m: {"coverage": round(float(np.mean(a["cov"])), 4),
                    "p_fire": round(float(np.mean(np.array(a["nacc"]) > 0)), 4),
                    "err_pooled_over_alpha": (round(float(sum(a["nerr"]) / sum(a["nacc"]) / np.mean(a["alpha"])), 3)
                                              if sum(a["nacc"]) else None)}
                for m, a in acc.items()}}
        print(slug, flush=True)
    agg = {}
    for m in ("cp", "hb", "bet", "walk_strict", "walk_from_first_pass"):
        covs = [v[m]["coverage"] for v in res.values()]
        agg[m] = {"mean_coverage": round(float(np.mean(covs)), 4),
                  "n_cov_ge_10pct": int(sum(c >= 0.10 for c in covs)), "n": len(covs)}
    wins = {m: int(sum(v[m]["coverage"] > v["cp"]["coverage"] + 1e-9 for v in res.values()))
            for m in ("hb", "bet")}
    return {"cells": res, "aggregate": agg, "n_cells_beating_cp": wins}


def synthetic(n_rep=20000, seed=11):
    """Gaussian score model with known selective risk R(lambda).
    Y=0 w.p. pi0; S|Y=0 ~ N(0,1), S|Y=1 ~ N(mu,1), mu set from the AUC."""
    rng = np.random.default_rng(seed)
    out = {}
    for pi0, auc_, n_cal in ((0.16, 0.82, 154), (0.16, 0.71, 154), (0.22, 0.76, 78), (0.09, 0.66, 153)):
        mu = np.sqrt(2) * stats.norm.ppf(auc_)

        def R(lam):
            a0 = pi0 * stats.norm.sf(lam)
            a1 = (1 - pi0) * stats.norm.sf(lam - mu)
            return a0 / (a0 + a1)

        def Cov(lam):
            return pi0 * stats.norm.sf(lam) + (1 - pi0) * stats.norm.sf(lam - mu)

        for rho in (0.75, 0.5):
            alpha = rho * pi0
            tally = {}
            for _ in range(n_rep):
                yy = (rng.random(n_cal) >= pi0).astype(int)
                ss = rng.standard_normal(n_cal) + mu * yy
                for m, lam in thresholds(ss, yy, alpha).items():
                    t = tally.setdefault(m, {"fire": 0, "bad": 0, "cov": 0.0})
                    if np.isfinite(lam):
                        t["fire"] += 1
                        t["bad"] += int(R(lam) > alpha)
                        t["cov"] += Cov(lam)
            out[f"pi0={pi0}|auc={auc_}|ncal={n_cal}|rho={rho}"] = {
                m: {"p_fire": round(t["fire"] / n_rep, 4),
                    "p_risk_gt_alpha": round(t["bad"] / n_rep, 4),
                    "p_risk_gt_alpha_given_fire": (round(t["bad"] / t["fire"], 4) if t["fire"] else None),
                    "mean_coverage": round(t["cov"] / n_rep, 4)}
                for m, t in tally.items()}
            print("synthetic", pi0, auc_, n_cal, rho, flush=True)
    return out


def null_check(n_rep=200000, seed=5):
    """False-certification rate of each per-k test at true risk exactly alpha."""
    rng = np.random.default_rng(seed)
    out = {}
    for alpha, k in ((0.0886, 60), (0.16, 100), (0.34, 50)):
        e = rng.binomial(k, alpha, size=n_rep)
        d = DELTA / N_GRID
        vals, cnt = np.unique(e, return_counts=True)
        row = {}
        for m in ("cp", "hb", "bet"):
            row[m] = round(float(sum(c for v, c in zip(vals, cnt) if passes(m, int(v), k, alpha, d)) / n_rep), 5)
        out[f"alpha={alpha}|k={k}|nominal={d}"] = row
    return out


if __name__ == "__main__":
    res = {"null_check": null_check()}
    print(json.dumps(res["null_check"], indent=1), flush=True)
    res["real_data"] = real_data()
    print(json.dumps(res["real_data"]["aggregate"], indent=1), res["real_data"]["n_cells_beating_cp"], flush=True)
    res["synthetic"] = synthetic()
    json.dump(res, open(os.path.join(paths.OUT_JSON, "BOUNDS_EXTRAS.json"), "w"), indent=1)
    print(json.dumps(res["synthetic"], indent=1))
