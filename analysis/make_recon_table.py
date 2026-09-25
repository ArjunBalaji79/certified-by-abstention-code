"""Appendix table for the cross-fitted probe reconstruction (PROBE_RECON.json)."""
import json, os
import numpy as np
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from src import paths  # noqa: E402
OUT = paths.TABLES
P = json.load(open(os.path.join(paths.OUT_JSON, "PROBE_RECON.json")))
NAME = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-4b": "Qwen3.5-4B", "olmo-3-7b": "OLMo-3-7B", "qwen3-1.7b": "Qwen3-1.7B",
        "qwen2.5-7b-it": "Qwen2.5-7B-It", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B"}
rows = []
for m in NAME:
    v = P[m]; a = v["auc"]; c = v["cert"]["1.0"]
    rows.append(" & ".join([NAME[m], f"{a['probe']:.3f}", f"{a['readable']:.3f}", f"{a['recon']:.3f}", f"{a['resid']:.3f}", f"{a['recon_insample_resid']:.3f}",
                            f"{c['probe']['coverage']:.3f}", f"{c['readable']['coverage']:.3f}", f"{c['recon']['coverage']:.3f}"]) + r" \\")
mean = {k: np.mean([P[m]["auc"][k] for m in NAME]) for k in ("probe", "readable", "recon", "resid", "recon_insample_resid")}
rows.append(r"\midrule"); rows.append(" & ".join(["Mean"] + [f"{mean[k]:.3f}" for k in ("probe", "readable", "recon", "resid", "recon_insample_resid")] + ["", "", ""]) + r" \\")
s = ("\\begin{table}[H]\n\\centering\\small\n"
     "\\caption{Cross-fitted probe reconstruction. Held-out AUC (mean over 5 train/held-out splits by problem) of the probe, the readable logistic regression, "
     "a ridge reconstruction of the probe score from the 119 readable features (fit on out-of-fold probe scores of the training problems), and the residual "
     "probe $-$ reconstruction; all regularization constants chosen by grouped cross-validation on the training problems. ``Resid.\\ (in-sample)'': the residual when "
     "the ridge is instead fit on in-sample probe scores, which are nearly separated. Right: mean coverage of the CP certificate at $\\alpha=\\pi_0$ on held-out problems "
     "with each score as the signal.}\n\\label{tab:recon}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l ccccc ccc}\n\\toprule\n"
     " & \\multicolumn{5}{c}{Held-out AUC} & \\multicolumn{3}{c}{Coverage, $\\alpha=\\pi_0$} \\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}\n"
     "Model & Probe & Readable & Recon. & Resid. & Resid.\\ (in-sample) & Probe & Readable & Recon. \\\\\n\\midrule\n"
     + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(OUT, "tab_recon.tex"), "w").write(s)
print({k: round(float(v), 3) for k, v in mean.items()})
