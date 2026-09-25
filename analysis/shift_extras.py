"""Shift cells: GSM8K-trained scorer and GSM8K-calibrated exact CP threshold
(alpha = training-partition base error of the same seed) applied to the same
model's traces on other benchmarks.  Same replicate stream as
code/conformal_analysis.py::run_shift.  Writes SHIFT_EXTRAS.json.
"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src import paths  # noqa: E402
from src import conformal_common as ca  # noqa: E402
from src import cert  # noqa: E402


def run(md):
    d = md["data"]
    y_s = d["y__gsm8k"].astype(int); g_s = d["groups__gsm8k"].astype(str)
    out = {}
    for bench in md["benches"]:
        if bench == "gsm8k":
            continue
        y_t = d[f"y__{bench}"].astype(int)
        gl_t = [v for _, v in sorted(ca.group_index(d[f"groups__{bench}"].astype(str)).items())]
        for s in ca.CORE_SIGNALS:
            cov, nacc, nerr, a_s, a_t, alphas = [], [], [], [], [], []
            for seed in ca.OUTER_SEEDS:
                ks, kt, tm = f"s__gsm8k__{s}__s{seed}", f"s__{bench}__{s}__s{seed}", f"trainmask__gsm8k__s{seed}"
                if not all(k in d for k in (ks, kt, tm)):
                    continue
                train = d[tm].astype(bool)
                alpha = cert.train_pi0(y_s, train)
                gl_s = [v for _, v in sorted(ca.group_index(g_s, ~train).items())]
                for r in range(ca.N_REPL):
                    rng = np.random.default_rng(77_000 + 1_000_000 * seed + r)
                    cal = ca.one_per_problem(rng, gl_s); tst = ca.one_per_problem(rng, gl_t)
                    cs, cy, ts, ty = d[ks][cal], y_s[cal], d[kt][tst], y_t[tst]
                    if (cy == 0).sum() == 0 or len(np.unique(ty)) < 2:
                        continue
                    sel = ts >= cert.cp_threshold_exact(cs, cy, alpha)
                    cov.append(float(sel.mean())); nacc.append(int(sel.sum())); nerr.append(int((ty[sel] == 0).sum()))
                    a_s.append(ca.auc(cy, cs)); a_t.append(ca.auc(ty, ts)); alphas.append(alpha)
            if not cov:
                continue
            nacc, nerr = np.array(nacc), np.array(nerr)
            out[f"{bench}|{s}"] = {
                "alpha": round(float(np.mean(alphas)), 4),
                "target_base_err": round(float(1 - y_t.mean()), 4),
                "coverage": round(float(np.mean(cov)), 4),
                "p_fire": round(float((nacc > 0).mean()), 4),
                "err_pooled": round(float(nerr.sum() / nacc.sum()), 4) if nacc.sum() else None,
                "auc_drop": round(float(np.mean(a_s) - np.mean(a_t)), 4),
            }
    return out


def main():
    res = {}
    for fn in sorted(os.listdir(ca.SCORES)):
        if fn.startswith("scores_") and fn.endswith(".npz"):
            slug = fn[len("scores_"):-4]
            md = ca.load_model(slug)
            if md is not None and "y__gsm8k" in md["data"]:
                res[slug] = run(md); print(slug, flush=True)
    cells = [v for blk in res.values() for v in blk.values()]
    fired = [v for v in cells if v["err_pooled"] is not None]
    viol = np.array([v["err_pooled"] - v["alpha"] for v in fired]); diff = np.array([v["target_base_err"] - v["alpha"] for v in fired])
    R = np.array([v["err_pooled"] for v in fired]); pin = np.array([v["target_base_err"] for v in fired]); cov = np.array([v["coverage"] for v in fired])
    big = cov > 0.02
    summ = {"n_cells": len(cells), "n_fired": len(fired), "n_violating": int((viol > 0).sum()), "max_violation": round(float(viol.max()), 4),
            "harder": {"n": int((diff > 0).sum()), "violating": int(((viol > 0) & (diff > 0)).sum())},
            "easier": {"n": int((diff <= 0).sum()), "violating": int(((viol > 0) & (diff <= 0)).sum())},
            "median_R_over_target_base": round(float(np.median(R / pin)), 3),
            "spearman_excess_vs_difficulty": [round(float(x), 3) for x in stats.spearmanr(viol, diff)],
            "spearman_excess_vs_difficulty_cov_gt_2pct": [round(float(x), 3) for x in stats.spearmanr(viol[big], diff[big])],
            "spearman_R_vs_target_base": [round(float(x), 3) for x in stats.spearmanr(R, pin)],
            "spearman_R_vs_target_base_cov_gt_2pct": [round(float(x), 3) for x in stats.spearmanr(R[big], pin[big])],
            "spearman_Rrel_vs_aucdrop": [round(float(x), 3) for x in stats.spearmanr(R / pin, [v["auc_drop"] for v in fired])],
            "n_cov_gt_2pct": int(big.sum())}
    res["_summary"] = summ; print(json.dumps(summ, indent=1))
    json.dump(res, open(os.path.join(paths.OUT_JSON, "SHIFT_EXTRAS.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
