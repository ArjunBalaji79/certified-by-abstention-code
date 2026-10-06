"""Generate the certificate tables from outputs/json (no number is hand-typed).
Writes paper/tables/*.tex and outputs/json/PROSE_NUMBERS.json (scalars quoted in the text)."""

import json
import os

import numpy as np
from scipy import stats

import sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from src import paths  # noqa: E402
OUT = paths.TABLES

E = json.load(open(os.path.join(paths.OUT_JSON, "CONFORMAL_EXTRAS.json")))
V3 = json.load(open(os.path.join(paths.PIPELINE_OUT, "results_v3_cross.json")))
V4 = json.load(open(os.path.join(paths.PIPELINE_OUT, "results_v4_cross.json")))

MODELS = ["qwen3.5-0.8b", "qwen3.5-4b", "olmo-3-7b", "qwen3-1.7b",
          "qwen2.5-7b-it", "qwen3-4b", "qwen3-8b"]
NAME = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-4b": "Qwen3.5-4B",
        "olmo-3-7b": "OLMo-3-7B", "qwen3-1.7b": "Qwen3-1.7B",
        "qwen2.5-7b-it": "Qwen2.5-7B-It", "qwen3-4b": "Qwen3-4B",
        "qwen3-8b": "Qwen3-8B"}
CORE = ["confidence", "jspace", "probe"]
SIGNAME = {"confidence": "Confidence", "jspace": "J-lens", "probe": "Probe",
           "logitlens": "Logit lens", "surface": "Surface"}


def f(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def ratio(r, alpha):
    ep = r["err_pooled"]
    if ep is None or (1 - r["p_empty"]) * r["n_repl"] < 50:
        return "--"          # fewer than 50 firing replicates: not reported
    return f"{ep / alpha:.2f}"


def tab_main():
    rows = []
    for m in MODELS:
        e1, e2 = E[f"{m}|rho=1.0"], E[f"{m}|rho=0.75"]
        cells = [NAME[m], str(e1["signals"]["confidence"]["n_cal"]), f(e1["alpha"]), f(e1["base_err_nontrain"])]
        for s in CORE:
            x1, x2 = e1["signals"][s], e2["signals"][s]
            cells += [f(x1["coverage"]), f(1 - x1["p_empty"], 3), ratio(x1, e1["alpha"]), f(x2["coverage"]), f(1 - x2["p_empty"], 3)]
        rows.append(" & ".join(cells) + r" \\")
    hdr = (r"\begin{table*}[t]" "\n" r"\centering\scriptsize" "\n"
           r"\caption{Rule~\eqref{eq:cert} on GSM8K, $\delta=0.10$, 1{,}000 replicates per cell. $n_{\mathrm{cal}}$: calibration problems; $\hat\pi_0$: training-partition base error (sets the targets); "
           r"$\pi_0^{\mathrm{held}}$: base error of the remaining problems. Per signal: mean coverage, $P_{\mathrm{fire}}$ and $\hat R/\alpha$ (error among accepted test traces pooled over firing "
           r"replicates, relative to the target; -- if fewer than 50 replicates fire) at $\alpha=\hat\pi_0$, and coverage and $P_{\mathrm{fire}}$ at $0.75\hat\pi_0$. Test AUCs: Appendix Table~\ref{tab:bh}.}" "\n"
           r"\label{tab:main}" "\n" r"\setlength{\tabcolsep}{2.6pt}" "\n"
           r"\begin{tabular}{lrrr rrrrr rrrrr rrrrr}" "\n" r"\toprule" "\n"
           r" & & & & \multicolumn{5}{c}{Confidence} & \multicolumn{5}{c}{J-lens} & \multicolumn{5}{c}{Probe} \\" "\n"
           r"\cmidrule(lr){5-9}\cmidrule(lr){10-14}\cmidrule(lr){15-19}" "\n"
           r" & & & & \multicolumn{3}{c}{$\alpha=\hat\pi_0$} & \multicolumn{2}{c}{$0.75\hat\pi_0$} & \multicolumn{3}{c}{$\alpha=\hat\pi_0$} & \multicolumn{2}{c}{$0.75\hat\pi_0$} & \multicolumn{3}{c}{$\alpha=\hat\pi_0$} & \multicolumn{2}{c}{$0.75\hat\pi_0$} \\" "\n"
           r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-12}\cmidrule(lr){13-14}\cmidrule(lr){15-17}\cmidrule(lr){18-19}" "\n"
           r"Model & $n_{\mathrm{cal}}$ & $\hat\pi_0$ & $\pi_0^{\mathrm{held}}$ & Cov. & $P_{\mathrm{fire}}$ & $\hat R/\alpha$ & Cov. & $P_{\mathrm{fire}}$"
           r" & Cov. & $P_{\mathrm{fire}}$ & $\hat R/\alpha$ & Cov. & $P_{\mathrm{fire}}$ & Cov. & $P_{\mathrm{fire}}$ & $\hat R/\alpha$ & Cov. & $P_{\mathrm{fire}}$ \\" "\n"
           r"\midrule" "\n")
    open(os.path.join(OUT, "tab_main.tex"), "w").write(
        hdr + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table*}" "\n")


def tab_bounds():
    B = json.load(open(os.path.join(paths.OUT_JSON, "BOUNDS_EXTRAS.json")))["real_data"]
    names = [("cp", "Clopper--Pearson"), ("hb", "Hoeffding--Bentkus"),
             ("bet", "Betting mixture")]
    rows = []
    for k, n in names:
        a = B["aggregate"][k]
        wins = "--" if k == "cp" else f"{B['n_cells_beating_cp'][k]}/21"
        rows.append(f"{n} & {a['mean_coverage']:.3f} & {a['n_cov_ge_10pct']}/21 & {wins} \\\\")
    a = B["aggregate"]["walk_from_first_pass"]
    rows.append(r"\midrule")
    rows.append(f"Data-dep.\\ walk$^\\dagger$ & {a['mean_coverage']:.3f} & {a['n_cov_ge_10pct']}/21 & -- \\\\")
    s = (r"\begin{table}[t]" "\n" r"\centering\small" "\n"
         r"\caption{Bounds in the same selective construction (same grid, Bonferroni split, data and replicates) at $\alpha=\pi_0$; "
         r"7 models $\times$ 3 signals. $^\dagger$No validity proof (see text).}"
         "\n" r"\label{tab:bounds}" "\n" r"\setlength{\tabcolsep}{4pt}" "\n" r"\begin{tabular}{lccc}" "\n" r"\toprule" "\n"
         r"Bound & Mean cov. & Cov.$\ge$10\% & Beats CP \\" "\n" r"\midrule" "\n"
         + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(OUT, "tab_bounds.tex"), "w").write(s)


def tab_synthetic():
    B = json.load(open(os.path.join(paths.OUT_JSON, "BOUNDS_EXTRAS.json")))["synthetic"]
    O = json.load(open(os.path.join(paths.OUT_JSON, "SYNTH_OVERSHOOT.json")))
    rows = []
    for key, v in B.items():
        kv = dict(x.split("=") for x in key.split("|"))
        for m, lab in (("cp", "CP, Eq.~(1)"), ("walk_from_first_pass", "Data-dep.\\ walk")):
            r = v[m]; o = O[key].get(m, {})
            cg = "--" if r["p_risk_gt_alpha_given_fire"] is None else f"{r['p_risk_gt_alpha_given_fire']:.3f}"
            med = "--" if o.get("median_R_over_alpha_fail") is None else f"{o['median_R_over_alpha_fail']:.2f}"
            p90 = "--" if o.get("p90_R_over_alpha_fail") is None else f"{o['p90_R_over_alpha_fail']:.2f}"
            rows.append(" & ".join([kv["pi0"], kv["auc"], kv["ncal"], f"${kv['rho']}\\pi_0$", lab,
                                    f"{r['p_fire']:.3f}", f"{r['p_risk_gt_alpha']:.4f}", cg, med, p90,
                                    f"{r['mean_coverage']:.3f}"]) + r" \\")
        rows.append(r"\addlinespace")
    s = (r"\begin{table}[H]" "\n" r"\centering\small" "\n"
         r"\caption{Gaussian score model with known selective risk; 20{,}000 calibration draws per row, $\delta=0.10$. "
         r"$P(R>\alpha)$ is the quantity bounded by $(\alpha,\delta)$-validity; $P(R>\alpha\mid\text{fire})$ is the probability that an issued certificate is wrong; "
         r"the next two columns give the median and 90th percentile of $R(\hat\lambda)/\alpha$ among failing draws, i.e.\ how far a wrong certificate overshoots its target.}"
         "\n" r"\label{tab:synthetic}" "\n" r"\setlength{\tabcolsep}{4pt}" "\n" r"\begin{tabular}{cccc l cccccc}" "\n" r"\toprule" "\n"
         r"$\pi_0$ & AUC & $n_{\mathrm{cal}}$ & $\alpha$ & Procedure & $P_{\mathrm{fire}}$ & $P(R>\alpha)$ & $P(R>\alpha\mid\text{fire})$ & med.\ $R/\alpha$ & p90 $R/\alpha$ & Mean cov. \\" "\n" r"\midrule" "\n"
         + "\n".join(rows[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(OUT, "tab_synthetic.tex"), "w").write(s)


def tab_full():
    specs = [("rho=0.5", r"$0.5\pi_0$"), ("rho=0.75", r"$0.75\pi_0$"), ("rho=1.0", r"$\pi_0$"),
             ("abs=0.05", "0.05"), ("abs=0.1", "0.10"), ("abs=0.2", "0.20")]
    sigs = ["confidence", "jspace", "logitlens", "surface", "probe"]
    rows = []
    for m in MODELS:
        for i, (sp, lab) in enumerate(specs):
            e = E[f"{m}|{sp}"]
            nv = e["alpha"] < e["base_err_nontrain"] - 1e-9
            cells = [NAME[m] if i == 0 else "", lab, f(e["alpha"], 3), "yes" if nv else "no"]
            for s in sigs:
                x = e["signals"].get(s)
                cells.append("--" if x is None else f"{x['coverage']:.3f} ({1 - x['p_empty']:.2f})")
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    s = (r"\begin{table}[H]" "\n" r"\centering\small" "\n"
         r"\caption{Mean coverage of the CP certificate (in parentheses: $P_{\mathrm{fire}}$) for every model, target and signal; GSM8K, $\delta=0.10$, 1{,}000 replicates. "
         r"``Non-vac.'': target below the base error rate of the non-training problems.}" "\n" r"\label{tab:full}" "\n"
         r"\begin{tabular}{llrc ccccc}" "\n" r"\toprule" "\n"
         r"Model & Target & $\alpha$ & Non-vac. & Confidence & J-lens & Logit lens & Surface & Probe \\" "\n" r"\midrule" "\n"
         + "\n".join(rows[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(OUT, "tab_full.tex"), "w").write(s)


def tab_exact_bh():
    rows = []
    for m in MODELS:
        e = E[f"{m}|rho=1.0"]
        x = e["signals"]
        cells = [NAME[m]]
        for s in CORE:
            cells += [f(x[s]["auc_test"]), f(x[s]["coverage"], 3), f(x[s]["coverage_boundary_variant"], 3)]
        p = x["probe"]
        cells += [f(p["n0_cal_mean"], 1), f"{p['bh_lattice_min_rejections']}/{p['n_test']}"]
        cells += [f(x[s]["bh_frac_no_rejection"]) for s in CORE]
        rows.append(" & ".join(cells) + r" \\")
    s = (r"\begin{table}[H]" "\n" r"\centering\small" "\n"
         r"\caption{Left: test AUC of the frozen scorer, and coverage of rule~\eqref{eq:cert} (``excl.'', boundary trace excluded) and of the boundary-included variant (``incl.''), $\alpha=\pi_0$. "
         r"Right: BH conformal selection at level $\alpha=\pi_0$: mean number of calibration errors $n_0$, the lattice minimum $\lceil m/(\alpha(n_0+1))\rceil$ of Proposition~\ref{prop:lattice} relative to the test size $m$, and the fraction of replicates with no rejection.}"
         "\n" r"\label{tab:bh}" "\n" r"\setlength{\tabcolsep}{2.6pt}" "\n"
         r"\begin{tabular}{l ccc ccc ccc rr ccc}" "\n" r"\toprule" "\n"
         r" & \multicolumn{3}{c}{Confidence} & \multicolumn{3}{c}{J-lens} & \multicolumn{3}{c}{Probe} & & & \multicolumn{3}{c}{BH: no rejection} \\" "\n"
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){13-15}" "\n"
         r"Model & AUC & excl. & incl. & AUC & excl. & incl. & AUC & excl. & incl. & $n_0$ & min.\ rej. & Conf. & J-lens & Probe \\" "\n" r"\midrule" "\n"
         + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(OUT, "tab_exact_bh.tex"), "w").write(s)


def tab_auc():
    rows = []
    src = {**V3, **V4}
    for m in MODELS:
        e = src[f"{m}|gsm8k"]
        a, pd = e["aucs"], e["paired_deltas"]

        def d(k):
            x = pd[k]
            star = "$^{*}$" if x["p_two_sided"] < 0.05 else ""
            return f"{x['delta']:+.3f}{star}"
        rows.append(" & ".join([
            NAME[m], str(e["n_analyzed"]), f(e["accuracy"]),
            f(a["confidence_only"]["auc"], 3), f(a["jspace_only"]["auc"], 3),
            f(a["logitlens_only"]["auc"], 3), f(a["surface_only"]["auc"], 3),
            f(a["random_readout_only"]["auc"], 3), f(a["resid_probe_mid"]["auc"], 3),
            d("jspace_vs_confidence"), d("jspace_vs_surface"), d("probe_vs_jspace")]) + r" \\")
    s = (r"\begin{table}[H]" "\n" r"\centering\small" "\n"
         r"\caption{GSM8K out-of-fold AUC (5-fold, grouped by problem, logistic regression) of each signal, and paired bootstrap differences ($^{*}$: two-sided $p<0.05$, uncorrected). "
         r"``Random'': J-lens features with a norm-matched random transport. These are cross-validated AUCs on all traces and differ from the frozen-scorer test AUCs of Table~\ref{tab:main}.}"
         "\n" r"\label{tab:auc}" "\n" r"\setlength{\tabcolsep}{4pt}" "\n"
         r"\begin{tabular}{lrr cccccc ccc}" "\n" r"\toprule" "\n"
         r"Model & Traces & Acc. & Conf. & J-lens & Logit lens & Surface & Random & Probe & J$-$Conf. & J$-$Surf. & Probe$-$J \\" "\n" r"\midrule" "\n"
         + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(OUT, "tab_auc.tex"), "w").write(s)


def prose():
    P = {}
    # strictly non-vacuous attainment
    hits = []
    for k, e in E.items():
        if e["alpha"] < e["base_err_nontrain"] - 1e-9:
            for s in CORE:
                if e["signals"][s]["coverage"] >= 0.10:
                    hits.append((k, s, e["signals"][s]["coverage"]))
    P["nonvacuous_cov_ge_10pct"] = hits
    P["pairs_attaining"] = sorted({(k.split("|")[0], s) for k, s, _ in hits})
    for spec in ("rho=1.0", "rho=0.75"):
        P[f"pfire_le_delta_{spec}"] = sum(
            (1 - E[f"{m}|{spec}"]["signals"][s]["p_empty"]) <= 0.10 for m in MODELS for s in CORE)
    # coverage vs AUC / accuracy (rho=1)
    for s in ("probe", "confidence"):
        cov = [E[f"{m}|rho=1.0"]["signals"][s]["coverage"] for m in MODELS]
        au = [E[f"{m}|rho=1.0"]["signals"][s]["auc_test"] for m in MODELS]
        ac = [E[f"{m}|rho=1.0"]["accuracy"] for m in MODELS]
        P[f"spearman_cov_auc_{s}"] = [round(float(v), 4) for v in stats.spearmanr(cov, au)]
        P[f"spearman_cov_acc_{s}"] = [round(float(v), 4) for v in stats.spearmanr(cov, ac)]
    # pooled conditional error range among cells with >=20 firing replicates
    rat = {}
    for m in MODELS:
        e = E[f"{m}|rho=1.0"]
        for s in CORE:
            x = e["signals"][s]
            if x["err_pooled"] is not None and (1 - x["p_empty"]) * x["n_repl"] >= 20:
                rat[f"{m}|{s}"] = round(x["err_pooled"] / e["alpha"], 3)
    P["pooled_ratio_rho1"] = rat
    P["pooled_ratio_rho1_range"] = [min(rat.values()), max(rat.values())]
    # geometric-mean coverage ratios
    def gm(a, b):
        r = [E[f"{m}|rho=1.0"]["signals"][a]["coverage"] / E[f"{m}|rho=1.0"]["signals"][b]["coverage"] for m in MODELS]
        return round(float(np.exp(np.mean(np.log(r)))), 3), int(sum(x > 1 for x in r))
    P["gm_probe_over_jspace"] = gm("probe", "jspace")
    P["gm_jspace_over_conf"] = gm("jspace", "confidence")
    d = [abs(x["coverage"] - x["coverage_boundary_variant"]) for e in E.values() for x in e["signals"].values()]
    P["boundary_variant_cov_absdiff_mean_max"] = [round(float(np.mean(d)), 4), round(float(np.max(d)), 4)]
    bh = [x["bh_frac_no_rejection"] for k, e in E.items() if k.endswith("rho=1.0") and not k.startswith("qwen3-8b")
          for s, x in e["signals"].items() if s in CORE]
    P["bh_frac_no_rejection_rho1_mean"] = round(float(np.mean(bh)), 3)
    bh_all = [x["bh_frac_no_rejection"] for k, e in E.items() if k.endswith("rho=1.0")
              for s_, x in e["signals"].items() if s_ in CORE]
    P["bh_frac_no_rejection_rho1_mean_all7"] = round(float(np.mean(bh_all)), 3)
    import math
    P["floor_qwen3_8b"] = math.ceil(math.log(200) / -math.log(1 - 0.0886))
    n_gsm = sum({**V3, **V4}[f"{m}|gsm8k"]["n_analyzed"] for m in MODELS)
    n_shift = sum(e["n_analyzed"] for k, e in {**V3, **V4}.items() if not k.endswith("|gsm8k"))
    P["traces"] = {"gsm8k": n_gsm, "other_benchmarks_all_cells": n_shift, "bon": 24793,
                   "total": n_gsm + n_shift + 24793}
    cgn = [E[f"{m}|rho=1.0"]["signals"][s_]["coverage_given_nonempty"] for m in MODELS for s_ in CORE]
    P["cov_given_fire_range_rho1"] = [min(cgn), max(cgn)]
    for spec in ("rho=1.0", "rho=0.75"):
        for s_ in ("probe", "confidence"):
            cov = [E[f"{m}|{spec}"]["signals"][s_]["coverage"] for m in MODELS]
            au = [E[f"{m}|rho=1.0"]["signals"][s_]["auc_test"] for m in MODELS]
            ac = [E[f"{m}|rho=1.0"]["accuracy"] for m in MODELS]
            P[f"spearman_{spec}_{s_}"] = {"auc": [round(float(v), 3) for v in stats.spearmanr(cov, au)], "acc": [round(float(v), 3) for v in stats.spearmanr(cov, ac)]}
    rat50 = {f"{m}|{s_}": round(E[f"{m}|rho=1.0"]["signals"][s_]["err_pooled"] / E[f"{m}|rho=1.0"]["alpha"], 3) for m in MODELS for s_ in CORE
             if E[f"{m}|rho=1.0"]["signals"][s_]["err_pooled"] is not None and (1 - E[f"{m}|rho=1.0"]["signals"][s_]["p_empty"]) * 1000 >= 50}
    P["pooled_ratio_rho1_ge50"] = [min(rat50.values()), max(rat50.values()), len(rat50)]
    P["vacuous_by_noise_rho1"] = [m for m in MODELS if E[f"{m}|rho=1.0"]["alpha"] > E[f"{m}|rho=1.0"]["base_err_nontrain"]]
    P["nonvac_strict_hits"] = [(k, s_, e["signals"][s_]["coverage"]) for k, e in E.items() if not k.endswith("rho=1.0") and e["alpha"] < e["base_err_nontrain"] - 1e-9 for s_ in ("confidence", "jspace", "logitlens", "surface", "probe") if e["signals"][s_]["coverage"] >= 0.10]
    P["rho075_ge10"] = sum(E[f"{m}|rho=0.75"]["signals"][s_]["coverage"] >= 0.10 for m in MODELS for s_ in CORE)
    P["rho075_gm"] = {}
    for a_, b_ in (("probe", "jspace"), ("probe", "confidence"), ("confidence", "jspace")):
        r = [max(E[f"{m}|rho=0.75"]["signals"][a_]["coverage"], 1e-4) / max(E[f"{m}|rho=0.75"]["signals"][b_]["coverage"], 1e-4) for m in MODELS]
        P["rho075_gm"][f"{a_}/{b_}"] = [round(float(np.exp(np.mean(np.log(r)))), 2), int(sum(x > 1 for x in r))]
    json.dump(P, open(os.path.join(paths.OUT_JSON, "PROSE_NUMBERS.json"), "w"), indent=1)
    print(json.dumps({k: P[k] for k in ("pfire_le_delta_rho=1.0", "pfire_le_delta_rho=0.75", "gm_probe_over_jspace", "gm_jspace_over_conf", "pooled_ratio_rho1_ge50", "vacuous_by_noise_rho1", "nonvac_strict_hits", "rho075_ge10", "rho075_gm", "spearman_rho=1.0_probe", "spearman_rho=1.0_confidence", "spearman_rho=0.75_probe", "spearman_rho=0.75_confidence", "cov_given_fire_range_rho1", "bh_frac_no_rejection_rho1_mean_all7")}, indent=1))


if __name__ == "__main__":
    tab_main(); tab_bounds(); tab_synthetic(); tab_full(); tab_exact_bh(); tab_auc(); prose()
