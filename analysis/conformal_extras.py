"""Certificate statistics on the cached GSM8K scores (same splits, seeds and
replicate stream as the original pipeline; helpers in src/conformal_common.py).

Rule: cp_threshold_exact (Prop. 1, boundary point excluded).  Targets alpha =
rho * pi0 use pi0 from the TRAINING partition of the same seed, fixed before
calibration; absolute targets are fixed constants.  Per (model, spec, signal):
coverage, p_empty, err_pooled (sum accepted errors / sum accepted over firing
replicates), coverage given firing, per-replicate violation rate, the
boundary-included variant's coverage, BH conformal selection statistics.
Also records the frozen scorer's test AUC per replicate (auc_test).
Writes outputs/json/CONFORMAL_EXTRAS.json.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src import paths  # noqa: E402
from src import conformal_common as ca  # noqa: E402
from src import cert  # noqa: E402


def run(md, spec):
    d = md["data"]
    y = d["y__gsm8k"].astype(int)
    groups = d["groups__gsm8k"].astype(str)
    A = {s: dict(cov=[], nacc=[], nerr=[], cov_b=[], nacc_b=[], nerr_b=[], bh_none=[], bh_rmin=[], n0=[], ncal=[], ntest=[], alpha=[], viol=[], auc=[])
         for s in ca.SIGNALS}
    nontrain_err = []
    for seed in ca.OUTER_SEEDS:
        tm = f"trainmask__gsm8k__s{seed}"
        if tm not in d:
            continue
        train = d[tm].astype(bool)
        alpha = spec[1] * cert.train_pi0(y, train) if spec[0] == "rho" else spec[1]
        nontrain_err.append(1 - float(y[~train].mean()))
        gidx = ca.group_index(groups, ~train)
        problems = sorted(gidx); gl = [gidx[g] for g in problems]
        for r in range(ca.N_REPL):
            rng = np.random.default_rng(10_000 + 1_000_000 * seed + r)
            rows = ca.one_per_problem(rng, gl)
            perm = rng.permutation(len(problems))
            cal_p = set(perm[: int(round(ca.CAL_SHARE_OF_POOL * len(problems)))].tolist())
            is_cal = np.array([u in cal_p for u in range(len(problems))])
            yy = y[rows]
            for s in ca.SIGNALS:
                key = f"s__gsm8k__{s}__s{seed}"
                if key not in d:
                    continue
                sc = d[key][rows]
                cs, cy, ts, ty = sc[is_cal], yy[is_cal], sc[~is_cal], yy[~is_cal]
                if len(ts) == 0 or (cy == 0).sum() == 0 or len(np.unique(ty)) < 2:
                    continue
                a = A[s]
                sel = ts >= cert.cp_threshold_exact(cs, cy, alpha)
                a["cov"].append(float(sel.mean())); a["nacc"].append(int(sel.sum())); a["nerr"].append(int((ty[sel] == 0).sum()))
                a["viol"].append(float(ca.err_among_accepted(sel, ty) > alpha))
                selb = ts >= cert.cp_threshold_boundary(cs, cy, alpha)
                a["cov_b"].append(float(selb.mean())); a["nacc_b"].append(int(selb.sum())); a["nerr_b"].append(int((ty[selb] == 0).sum()))
                p, n0 = ca.conformal_pvals(cs, cy, ts)
                a["bh_none"].append(float(not ca.bh_select(p, alpha).any()))
                a["bh_rmin"].append(int(np.ceil(len(ts) / (alpha * (n0 + 1)))))
                a["n0"].append(n0); a["ncal"].append(len(cs)); a["ntest"].append(len(ts)); a["alpha"].append(alpha); a["auc"].append(ca.auc(ty, ts))
    out = {}
    for s, a in A.items():
        if not a["cov"]:
            continue
        nacc, nerr, cov = np.array(a["nacc"]), np.array(a["nerr"]), np.array(a["cov"])
        ne = nacc > 0
        naccb, nerrb = np.array(a["nacc_b"]), np.array(a["nerr_b"])
        out[s] = {
            "coverage": round(float(cov.mean()), 4),
            "auc_test": round(float(np.mean(a["auc"])), 4),
            "p_empty": round(float(1 - ne.mean()), 4),
            "coverage_given_nonempty": round(float(cov[ne].mean()), 4) if ne.any() else None,
            "err_pooled": round(float(nerr.sum() / nacc.sum()), 4) if nacc.sum() else None,
            "viol_rate_all": round(float(np.mean(a["viol"])), 4),
            "coverage_boundary_variant": round(float(np.mean(a["cov_b"])), 4),
            "p_empty_boundary_variant": round(float(1 - (naccb > 0).mean()), 4),
            "err_pooled_boundary_variant": round(float(nerrb.sum() / naccb.sum()), 4) if naccb.sum() else None,
            "bh_frac_no_rejection": round(float(np.mean(a["bh_none"])), 4),
            "bh_lattice_min_rejections": int(np.median(a["bh_rmin"])),
            "n0_cal_mean": round(float(np.mean(a["n0"])), 2),
            "n_cal": int(np.median(a["ncal"])), "n_test": int(np.median(a["ntest"])),
            "n_repl": int(len(cov)),
        }
    alpha_seeds = A["confidence"]["alpha"]
    return {"alpha": round(float(np.mean(alpha_seeds)), 4), "alpha_by_seed": sorted({round(v, 4) for v in alpha_seeds}),
            "base_err_nontrain": round(float(np.mean(nontrain_err)), 4), "accuracy": round(float(y.mean()), 4), "signals": out}


def main():
    res = {}
    specs = [("rho", r) for r in ca.RHOS] + [("abs", a) for a in ca.ABS_ALPHAS]
    for fn in sorted(os.listdir(ca.SCORES)):
        if not (fn.startswith("scores_") and fn.endswith(".npz")):
            continue
        slug = fn[len("scores_"):-4]
        md = ca.load_model(slug)
        if md is None or "y__gsm8k" not in md["data"]:
            continue
        for spec in specs:
            res[f"{slug}|{spec[0]}={spec[1]}"] = run(md, spec)
            print(slug, spec, flush=True)
    json.dump(res, open(os.path.join(paths.OUT_JSON, "CONFORMAL_EXTRAS.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
