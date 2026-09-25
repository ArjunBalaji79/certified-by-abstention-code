"""Appendix figures and tables regenerated from the current result files.
  figs/fig_attain.pdf      per-model coverage vs target, CP (dashed) vs FS-floor (solid), 3 signals
  figs/fig_abstention.pdf  P(fail | fire) vs P_fire in the Gaussian model, with the delta/P_fire bound
  figs/fig_corrected.pdf   best-of-n coverage, naive vs selection-conditional arm
  tables/tab_shift.tex     every shift cell: target base error, alpha, coverage and R-hat per signal
  tables/tab_bon.tex       best-of-n per model and signal
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from src import paths  # noqa: E402
import make_figures as mf   # noqa: E402  (style, SIG colours, rerank_curves)
FIGS, TABS = paths.FIGS, paths.TABLES

F = json.load(open(os.path.join(paths.OUT_JSON, "FS_FLOOR2.json")))
B = json.load(open(os.path.join(paths.OUT_JSON, "BOUNDS_EXTRAS.json")))
G = json.load(open(os.path.join(paths.OUT_JSON, "GOODHART_EXTRAS.json")))
S = json.load(open(os.path.join(paths.OUT_JSON, "SHIFT_EXTRAS.json")))
MODELS = ["qwen3.5-0.8b", "qwen3.5-4b", "olmo-3-7b", "qwen3-1.7b", "qwen2.5-7b-it", "qwen3-4b", "qwen3-8b"]
NAME = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-4b": "Qwen3.5-4B", "olmo-3-7b": "OLMo-3-7B", "qwen3-1.7b": "Qwen3-1.7B",
        "qwen2.5-7b-it": "Qwen2.5-7B-It", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B"}
BENCH = {"arc": "ARC-C", "strategyqa": "StrategyQA", "csqa": "CSQA", "bbhlogic": "BBH-logic"}
CORE = ["confidence", "jspace", "probe"]
INK, MUTED = mf.INK, mf.MUTED


def fig_attain():
    rhos = [0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.12, top=0.9, wspace=0.12, hspace=0.35)
    for ax, m in zip(axes.flat, MODELS):
        for sig in CORE:
            c, mk, lab = mf.SIG[sig]
            cp = [F["real"][f"{m}|rho={r}|{sig}"]["cp_bonf"]["coverage"] for r in rhos]
            fs = [F["real"][f"{m}|rho={r}|{sig}"]["x_csim"]["coverage"] for r in rhos]
            ax.plot(rhos, cp, "--", color=c, marker=mk, ms=3.5, lw=1.0, markerfacecolor="white", markeredgewidth=0.8)
            ax.plot(rhos, fs, "-", color=c, marker=mk, ms=3.5, lw=1.3, label=lab, markeredgecolor="white", markeredgewidth=0.4)
        ax.set_title(NAME[m], fontsize=7.5, pad=2)
        ax.set_xticks(rhos); ax.set_xticklabels([r"$0.5\pi_0$", r"$0.75\pi_0$", r"$\pi_0$"])
        ax.set_ylim(0, 1); ax.grid(True, axis="y")
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=INK, lw=1.3, ls="-"), plt.Line2D([], [], color=INK, lw=1.0, ls="--")]
    l += ["FS-floor", "CP, Eq. (1)"]
    axes.flat[-1].legend(h, l, frameon=False, loc="center", handlelength=2.0, labelspacing=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean coverage")
    for ax in axes[1, :3]:
        ax.set_xlabel(r"target $\alpha$")
    fig.savefig(os.path.join(FIGS, "fig_attain.pdf"), metadata={"CreationDate": None}); plt.close(fig)


def fig_abstention():
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.2, top=0.97)
    pf = np.logspace(-3, 0, 200)
    ax.plot(pf, np.minimum(1, 0.10 / pf), color=INK, lw=0.9, ls=":", label=r"bound $\min\{1,\delta/P_{\mathrm{fire}}\}$")
    ax.axhline(0.10, color=MUTED, lw=0.7, ls="--")
    ax.text(0.0012, 0.115, r"$\delta$", fontsize=7, color=MUTED)
    cp = [(v["cp"]["p_fire"], v["cp"]["p_risk_gt_alpha_given_fire"]) for v in B["synthetic"].values() if v["cp"]["p_risk_gt_alpha_given_fire"] is not None]
    wk = [(v["walk_from_first_pass"]["p_fire"], v["walk_from_first_pass"]["p_risk_gt_alpha_given_fire"]) for v in B["synthetic"].values()
          if v["walk_from_first_pass"]["p_risk_gt_alpha_given_fire"] is not None]
    fs = [(v[f"fs_c{c}"]["p_fire"], v[f"fs_c{c}"]["p_fail_given_fire"]) for v in F["synthetic"].values() for c in (1.0, 1.5, 2.0, 3.0, 4.0)
          if v[f"fs_c{c}"]["p_fail_given_fire"] is not None and v[f"fs_c{c}"]["p_fire"] > 0]
    ax.scatter(*zip(*cp), s=18, marker="o", color="#2a78d6", edgecolor="white", lw=0.5, label="CP, Eq. (1)", zorder=3)
    ax.scatter(*zip(*fs), s=16, marker="s", color="#eb6834", edgecolor="white", lw=0.5, label=r"FS-floor, $c\in\{1,\dots,4\}$", zorder=3)
    ax.scatter(*zip(*wk), s=18, marker="^", facecolor="none", edgecolor=MUTED, lw=0.8, label="data-dependent walk", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-3, 1.05); ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel(r"$P_{\mathrm{fire}}$"); ax.set_ylabel(r"$P(R(\hat\lambda)>\alpha \mid \mathrm{fire})$")
    ax.grid(True, axis="y"); ax.legend(frameon=False, loc="lower left", fontsize=6.2, handletextpad=0.3)
    fig.savefig(os.path.join(FIGS, "fig_abstention.pdf"), metadata={"CreationDate": None}); plt.close(fig)


def fig_corrected():
    names = {"qwen3.5-4b": "Qwen3.5-4B", "qwen3-4b": "Qwen3-4B"}
    ns = [1, 2, 4, 8, 16, 32]; xs = np.arange(len(ns))
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.9))
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.24, top=0.74, wspace=0.35)
    for ax, m in zip(axes, names):
        for sig, (c, mk, lab) in mf.SIG.items():
            nv, cr = G[m]["signals"][sig]["naive"], G[m]["signals"][sig]["corrected"]
            if 1 - nv["1"]["p_empty"] < 0.02:
                continue
            ax.plot(xs, [nv[str(n)]["coverage"] for n in ns], "-", color=c, marker=mk, ms=3.2, lw=1.2, label=lab, markeredgecolor="white", markeredgewidth=0.4)
            ax.plot(xs, [cr[str(n)]["coverage"] for n in ns], "--", color=c, marker=mk, ms=3.2, lw=1.0, markerfacecolor="white", markeredgewidth=0.8)
        ax.set_xticks(xs); ax.set_xticklabels([str(n) for n in ns]); ax.set_xlabel("best-of-$n$", labelpad=1)
        ax.set_title(names[m], fontsize=7.5, pad=2); ax.grid(True, axis="y"); ax.set_ylim(0, None)
    axes[0].set_ylabel("mean coverage")
    h, l = [], []
    for ax in axes:
        hh, ll = ax.get_legend_handles_labels()
        for a, b in zip(hh, ll):
            if b not in l:
                h.append(a); l.append(b)
    h += [plt.Line2D([], [], color=INK, lw=1.2, ls="-"), plt.Line2D([], [], color=INK, lw=1.0, ls="--")]; l += ["naive arm", "corrected arm"]
    fig.legend(h, l, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(os.path.join(FIGS, "fig_corrected.pdf"), metadata={"CreationDate": None}); plt.close(fig)


def tab_shift():
    rows = []
    for m in MODELS:
        blk = S[m]
        benches = sorted({k.split("|")[0] for k in blk})
        for i, b in enumerate(benches):
            v0 = blk[f"{b}|confidence"]
            cells = [NAME[m] if i == 0 else "", BENCH.get(b, b), f"{v0['target_base_err']:.2f}", f"{v0['alpha']:.2f}"]
            for s in CORE:
                v = blk.get(f"{b}|{s}")
                if v is None or v["err_pooled"] is None:
                    cells += ["--", "--"]; continue
                r = f"{v['err_pooled']:.2f}"
                cells += [f"{v['coverage']:.2f}", (r"\textbf{" + r + "}") if v["err_pooled"] > v["alpha"] else r]
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    s = ("\\begin{table}[H]\n\\centering\\scriptsize\n"
         "\\caption{Every shift cell (Section~\\ref{sec:shift}): GSM8K-trained scorer and GSM8K-calibrated threshold ($\\alpha$: training-partition base error on GSM8K) applied to the "
         "same model's traces on another benchmark. $\\pi_{\\mathrm{new}}$: base error rate on the new benchmark. Per signal: mean coverage and $\\hat R$, the error among accepted "
         "traces pooled over firing replicates; bold where $\\hat R>\\alpha$. 1{,}000 replicates per cell.}\n\\label{tab:shift}\n\\setlength{\\tabcolsep}{4pt}\n"
         "\\begin{tabular}{ll cc cc cc cc}\n\\toprule\n & & & & \\multicolumn{2}{c}{Confidence} & \\multicolumn{2}{c}{J-lens} & \\multicolumn{2}{c}{Probe} \\\\\n"
         "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\nModel & Benchmark & $\\pi_{\\mathrm{new}}$ & $\\alpha$ & Cov. & $\\hat R$ & Cov. & $\\hat R$ & Cov. & $\\hat R$ \\\\\n\\midrule\n"
         + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TABS, "tab_shift.tex"), "w").write(s)


def tab_bon():
    names = {"qwen3.5-4b": "Qwen3.5-4B", "qwen3-4b": "Qwen3-4B"}
    rows = []
    for m in names:
        cur = mf.rerank_curves(m)
        for i, (sig, (c, mk, lab)) in enumerate(mf.SIG.items()):
            nv, cr = G[m]["signals"][sig]["naive"], G[m]["signals"][sig]["corrected"]; a = G[m]["alpha"]
            pf = 1 - nv["1"]["p_empty"]
            def rr(n):
                e = nv[str(n)]["err_pooled"]
                return "--" if (e is None or pf < 0.02) else f"{e / a:.2f}"
            mu = cur[sig][0]; peak = max(mu)
            cells = [names[m] if i == 0 else "", lab, f"{pf:.3f}", rr(1), rr(8), rr(32), f"{cr['1']['coverage']:.3f}", f"{cr['32']['coverage']:.3f}",
                     f"{mu[-1]:+.3f}", f"{mu[-1] - peak:+.3f}"]
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    s = ("\\begin{table}[H]\n\\centering\\scriptsize\n"
         "\\caption{Best-of-$n$ selection (Section~\\ref{sec:pressure}) per model and signal. $P_{\\mathrm{fire}}$: firing probability of the naive arm (independent of $n$). "
         "$\\hat R/\\alpha$: error among accepted BoN-selected test traces relative to the target, naive arm (-- where fewer than 2\\% of replicates fire). "
         "Corrected: mean coverage of the selection-conditional arm at $n=1$ and $n=32$. $\\Delta$acc: change in accuracy of the selected trace from $n=1$ to $n=32$, and from its peak over $n$ to $n=32$.}\n"
         "\\label{tab:bon}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{ll c ccc cc cc}\n\\toprule\n"
         " & & & \\multicolumn{3}{c}{$\\hat R/\\alpha$, naive} & \\multicolumn{2}{c}{Corrected cov.} & \\multicolumn{2}{c}{$\\Delta$acc} \\\\\n"
         "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\nModel & Signal & $P_{\\mathrm{fire}}$ & $n{=}1$ & $n{=}8$ & $n{=}32$ & $n{=}1$ & $n{=}32$ & $n{=}32$ & from peak \\\\\n\\midrule\n"
         + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TABS, "tab_bon.tex"), "w").write(s)


if __name__ == "__main__":
    fig_attain(); fig_abstention(); fig_corrected(); tab_shift(); tab_bon(); print("appendix extras ok")
