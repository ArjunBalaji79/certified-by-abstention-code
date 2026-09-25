"""Tables for FS-floor (Section 5) from FS_FLOOR2.json: tab_fs.tex (main), tab_fsfull.tex (appendix).
Headline variant: cross-fitted grid + simulation-chosen c ('x_csim')."""
import json, os
import numpy as np
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from src import paths  # noqa: E402
OUT = paths.TABLES
F = json.load(open(os.path.join(paths.OUT_JSON, "FS_FLOOR2.json")))
NAME = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-4b": "Qwen3.5-4B", "olmo-3-7b": "OLMo-3-7B", "qwen3-1.7b": "Qwen3-1.7B",
        "qwen2.5-7b-it": "Qwen2.5-7B-It", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B"}
SIG = {"confidence": "Conf.", "jspace": "J-lens", "probe": "Probe"}
CS = (1.0, 1.5, 2.0, 3.0, 4.0)

rows = []
for key in ["pi0=0.16|auc=0.82|ncal=154|rho=0.75", "pi0=0.16|auc=0.82|ncal=154|rho=0.5", "pi0=0.16|auc=0.71|ncal=154|rho=0.75",
            "pi0=0.16|auc=0.71|ncal=154|rho=0.5", "pi0=0.22|auc=0.76|ncal=78|rho=0.75", "pi0=0.09|auc=0.66|ncal=153|rho=0.75"]:
    kv = dict(x.split("=") for x in key.split("|")); v = F["synthetic"][key]
    c = v["c_sim_rule"]; a, b = v["cp_bonf"], v[f"fs_c{c}"]
    rows.append(f"{kv['auc']} & {kv['ncal']} & ${kv['rho']}\\pi_0$ & {a['coverage']:.3f} & {a['p_fire']:.2f} & {c:g} & {b['coverage']:.3f} & {b['p_fire']:.2f} & {b['p_fail']:.3f} \\\\")
rows.append(r"\midrule")
for rho, lab in ((1.0, r"$\pi_0$"), (0.75, r"$0.75\pi_0$"), (0.5, r"$0.5\pi_0$")):
    cells = [v for k, v in F["real"].items() if f"|rho={rho}|" in k]
    a = [c["cp_bonf"] for c in cells]; b = [c["x_csim"] for c in cells]
    cmed = np.median([info["c"] for c in cells for info in c["c_sim"].values()])
    rows.append(f"\\multicolumn{{2}}{{l}}{{real, 21 pairs}} & {lab} & {np.mean([x['coverage'] for x in a]):.3f} & {np.mean([x['p_fire'] for x in a]):.2f} & {cmed:g}$^{{\\ast}}$ & "
                f"{np.mean([x['coverage'] for x in b]):.3f} & {np.mean([x['p_fire'] for x in b]):.2f} & -- \\\\")
s_tab = ("\\begin{table}[t]\n\\centering\\scriptsize\n"
     "\\caption{FS-floor (cross-fitted grid, $c$ chosen before calibration by simulation) against the Bonferroni CP rule~\\eqref{eq:cert}, $\\delta=0.10$. "
     "Top: Gaussian score model with known risk at the given AUC and calibration size $n$, 20{,}000 draws; $P_{\\mathrm{fail}}=P(R(\\hat\\lambda)>\\alpha)$. "
     "Bottom: means over 7 models $\\times$ 3 signals, 1{,}000 replicates each; $^{\\ast}$median of the per-cell choices of $c$.}\n"
     "\\label{tab:fs}\n\\setlength{\\tabcolsep}{2.8pt}\n\\begin{tabular}{rrl cc cccc}\n\\toprule\n"
     " & & & \\multicolumn{2}{c}{CP, Eq.~\\eqref{eq:cert}} & \\multicolumn{4}{c}{FS-floor} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n"
     "AUC & $n$ & $\\alpha$ & Cov. & $P_{\\mathrm{fire}}$ & $c$ & Cov. & $P_{\\mathrm{fire}}$ & $P_{\\mathrm{fail}}$ \\\\\n\\midrule\n"
     + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(OUT, "tab_fs.tex"), "w").write(s_tab)

rows = []
for m in NAME:
    for i, sg in enumerate(SIG):
        cells = [NAME[m] if i == 0 else "", SIG[sg]]
        for rho in (1.0, 0.75):
            v = F["real"][f"{m}|rho={rho}|{sg}"]
            cells += [f"{v[k]['coverage']:.3f}" for k in ("cp_bonf", "in_c2.0")] + [f"{v[f'x_c{c}']['coverage']:.3f}" for c in CS] + [f"{v['x_csim']['coverage']:.3f}"]
        rows.append(" & ".join(cells) + r" \\")
    rows.append(r"\addlinespace")
s = ("\\begin{table}[H]\n\\centering\\scriptsize\n"
     "\\caption{FS-floor on real data, mean coverage for each (model, signal) pair; GSM8K, $\\delta=0.10$, 1{,}000 replicates. "
     "CP: rule~\\eqref{eq:cert}. In-s.: FS-floor with the grid from in-sample training scores, $c=2$. $c{=}1,\\dots,4$: cross-fitted grid at fixed $c$. "
     "Sim.: cross-fitted grid with $c$ chosen per cell by the pre-calibration simulation.}\n\\label{tab:fsfull}\n\\setlength{\\tabcolsep}{2.2pt}\n"
     "\\begin{tabular}{ll cccccccc cccccccc}\n\\toprule\n & & \\multicolumn{8}{c}{$\\alpha=\\pi_0$} & \\multicolumn{8}{c}{$\\alpha=0.75\\pi_0$} \\\\\n"
     "\\cmidrule(lr){3-10}\\cmidrule(lr){11-18}\n"
     "Model & Signal & CP & In-s. & $c{=}1$ & $1.5$ & $2$ & $3$ & $4$ & Sim. & CP & In-s. & $c{=}1$ & $1.5$ & $2$ & $3$ & $4$ & Sim. \\\\\n\\midrule\n"
     + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(OUT, "tab_fsfull.tex"), "w").write(s)

# prose numbers
P = {}
for rho in (1.0, 0.75, 0.5):
    cells = {k: v for k, v in F["real"].items() if f"|rho={rho}|" in k}
    a = np.mean([v["cp_bonf"]["coverage"] for v in cells.values()]); b = np.mean([v["x_csim"]["coverage"] for v in cells.values()])
    r = [v["x_csim"]["R_over_alpha"] for v in cells.values() if v["x_csim"]["R_over_alpha"] is not None and v["x_csim"]["p_fire"] >= 0.02]
    rc = [v["cp_bonf"]["R_over_alpha"] for v in cells.values() if v["cp_bonf"]["R_over_alpha"] is not None and v["cp_bonf"]["p_fire"] >= 0.02]
    P[f"rho={rho}"] = {"cp_cov": round(a, 3), "fs_cov": round(b, 3), "ratio": round(b / a, 2),
                       "cp_pfire": round(np.mean([v["cp_bonf"]["p_fire"] for v in cells.values()]), 3), "fs_pfire": round(np.mean([v["x_csim"]["p_fire"] for v in cells.values()]), 3),
                       "cp_le_delta": sum(v["cp_bonf"]["p_fire"] <= .1 for v in cells.values()), "fs_le_delta": sum(v["x_csim"]["p_fire"] <= .1 for v in cells.values()),
                       "cp_ge10": sum(v["cp_bonf"]["coverage"] >= .1 for v in cells.values()), "fs_ge10": sum(v["x_csim"]["coverage"] >= .1 for v in cells.values()),
                       "wins": sum(v["x_csim"]["coverage"] > v["cp_bonf"]["coverage"] for v in cells.values()),
                       "R_med_fs": round(float(np.median(r)), 3) if r else None, "R_med_cp": round(float(np.median(rc)), 3) if rc else None,
                       "x_c1_cov": round(np.mean([v["x_c1.0"]["coverage"] for v in cells.values()]), 3),
                       "in_c2_cov": round(np.mean([v["in_c2.0"]["coverage"] for v in cells.values()]), 3)}
syn = {}
for k, v in F["synthetic"].items():
    c = v["c_sim_rule"]; b = v[f"fs_c{c}"]; a = v["cp_bonf"]
    syn[k] = {"c": c, "ratio": round(b["coverage"] / a["coverage"], 2) if a["coverage"] > 0.005 else None, "p_fail": b["p_fail"], "pfail_fire": b["p_fail_given_fire"], "pfire": b["p_fire"], "cp_pfire": a["p_fire"]}
P["synthetic"] = syn
P["max_p_fail_any_c_any_setting"] = max(v[f"fs_c{c}"]["p_fail"] for v in F["synthetic"].values() for c in CS)
json.dump(P, open(os.path.join(paths.OUT_JSON, "FS_PROSE.json"), "w"), indent=1); print(json.dumps(P, indent=1))
B = json.load(open(os.path.join(paths.OUT_JSON, "BUDGET_CURVE.json")))
for a in (0.66, 0.76, 0.82, 0.90):
    print("AUC", a, " ".join(f"n={n}: fs {B[f'auc={a}|n={n}']['fs']['p_fire']:.2f}/{B[f'auc={a}|n={n}']['fs']['coverage']:.2f} cp {B[f'auc={a}|n={n}']['cp']['p_fire']:.2f}/{B[f'auc={a}|n={n}']['cp']['coverage']:.2f}" for n in (100, 200, 500, 1000, 3000)))
