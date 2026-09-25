"""Draw paper/figs/fig_budget.pdf from outputs/json/BUDGET_CURVE.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import budget_curve as bc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import paths  # noqa: E402
res = json.load(open(os.path.join(paths.OUT_JSON, "BUDGET_CURVE.json")))
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42})
cols = {0.66: "#86b6ef", 0.76: "#3987e5", 0.82: "#1c5cab", 0.90: "#0d366b"}
fig, axes = plt.subplots(1, 2, figsize=(3.25, 2.05), sharex=True)
fig.subplots_adjust(left=0.14, right=0.985, bottom=0.2, top=0.76, wspace=0.5)
for a in bc.AUCS:
    for ax, key in zip(axes, ("p_fire", "coverage")):
        ax.plot(bc.NS, [res[f"auc={a}|n={n}"]["fs"][key] for n in bc.NS], "-", color=cols[a], lw=1.3, label=f"AUC {a:.2f}")
        ax.plot(bc.NS, [res[f"auc={a}|n={n}"]["cp"][key] for n in bc.NS], "--", color=cols[a], lw=0.9)
for ax, lab in zip(axes, (r"$P_{\mathrm{fire}}$", "mean coverage")):
    ax.set_xscale("log"); ax.set_xlabel("calibration problems"); ax.set_ylabel(lab); ax.grid(True, axis="y")
    ax.axvspan(67, 158, color="#eda100", alpha=0.16, lw=0); ax.set_ylim(0, 1); ax.set_xticks([100, 1000]); ax.set_xticklabels(["100", "1000"])
h, l = axes[0].get_legend_handles_labels()
h += [plt.Line2D([], [], color=INK, lw=1.3, ls="-"), plt.Line2D([], [], color=INK, lw=0.9, ls="--")]; l += ["FS-floor", "CP, Eq. (1)"]
fig.legend(h, l, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
fig.savefig(os.path.join(bc.OUT, "fig_budget.pdf"), metadata={"CreationDate": None})
