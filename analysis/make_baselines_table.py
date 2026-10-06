"""Appendix table: FS-floor against two cheap baselines (rule (1) with G=10 and G=5, and the
floor start alone), plus predicted vs realized P_fire. Reads FS_BASELINES.json, FS_FLOOR2.json."""
import json, os, sys
import numpy as np
from scipy import stats
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from src import paths  # noqa: E402
OUT = paths.TABLES
B = json.load(open(os.path.join(paths.OUT_JSON, "FS_BASELINES.json"))); F = json.load(open(os.path.join(paths.OUT_JSON, "FS_FLOOR2.json")))["real"]
rows = []; summ = {}
for rho, lab in ((1.0, r"$\pi_0$"), (0.75, r"$0.75\pi_0$"), (0.5, r"$0.5\pi_0$")):
    keys = [k for k in B if f"|rho={rho}|" in k]
    cov = {m: np.mean([B[k][m]["coverage"] for k in keys]) for m in ("cp_G20", "cp_G10", "cp_G5", "fs_start")}
    cov["fs"] = np.mean([F[k]["x_csim"]["coverage"] for k in keys])
    pf = {m: np.mean([B[k][m]["p_fire"] for k in keys]) for m in ("cp_G20", "cp_G10", "cp_G5", "fs_start")}
    pf["fs"] = np.mean([F[k]["x_csim"]["p_fire"] for k in keys])
    wins = {m: sum(F[k]["x_csim"]["coverage"] > B[k][m]["coverage"] for k in keys) for m in ("cp_G20", "cp_G10", "cp_G5", "fs_start")}
    pred = np.array([B[k]["pred_pfire_fs"] for k in keys]); real = np.array([F[k]["x_csim"]["p_fire"] for k in keys])
    rho_s = stats.spearmanr(pred, real)[0]; mae = np.mean(np.abs(pred - real))
    summ[rho] = {"cov": {m: round(float(v), 3) for m, v in cov.items()}, "pfire": {m: round(float(v), 3) for m, v in pf.items()},
                 "wins": wins, "pred_spearman": round(float(rho_s), 2), "pred_mae": round(float(mae), 3), "pred_mean": round(float(pred.mean()), 3), "real_mean": round(float(real.mean()), 3)}
    rows.append(" & ".join([lab] + [f"{cov[m]:.3f} / {pf[m]:.2f}" for m in ("cp_G20", "cp_G10", "cp_G5", "fs_start", "fs")] + [f"{wins['cp_G5']}/21", f"{wins['fs_start']}/21", f"{rho_s:.2f}", f"{mae:.3f}"]) + r" \\")
s = ("\\begin{table}[H]\n\\centering\\scriptsize\n"
     "\\caption{FS-floor against two cheap baselines on the real cells (means over 7 models $\\times$ 3 signals, 1{,}000 replicates; each entry is mean coverage / $P_{\\mathrm{fire}}$). "
     "$G$: number of Bonferroni grid points in rule~\\eqref{eq:cert} ($G=20$ in the paper). Floor start: the single fixed threshold at $j_0$ tested at level $\\delta$, i.e.\\ FS-floor without the walk. "
     "Wins: pairs where FS-floor covers more than the baseline. The last two columns compare the $P_{\\mathrm{fire}}$ predicted before calibration by the simulation that chooses $c$ with the realized $P_{\\mathrm{fire}}$ across the 21 pairs (Spearman correlation and mean absolute error).}\n"
     "\\label{tab:baselines}\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{l ccccc cc cc}\n\\toprule\n"
     " & \\multicolumn{3}{c}{Rule~\\eqref{eq:cert}} & & & \\multicolumn{2}{c}{FS-floor wins vs.} & \\multicolumn{2}{c}{Predicted $P_{\\mathrm{fire}}$} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
     "Target & $G{=}20$ & $G{=}10$ & $G{=}5$ & Floor start & FS-floor & $G{=}5$ & Floor start & Spearman & MAE \\\\\n\\midrule\n"
     + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(OUT, "tab_baselines.tex"), "w").write(s)
json.dump(summ, open(os.path.join(paths.OUT_JSON, "BASELINES_PROSE.json"), "w"), indent=1); print(json.dumps(summ, indent=0))
