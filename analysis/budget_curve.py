"""Calibration budget curves: P_fire and mean coverage of the CP rule (Eq. 1) and
FS-floor (c=2) against the number of calibration problems, Gaussian score model,
pi0=0.16, alpha=0.75*pi0, several AUCs.  Writes outputs/json/BUDGET_CURVE.json (redraw_budget.py draws the figure)."""
import json, math, os, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from src import paths  # noqa: E402
from src import fs_floor as fs  # noqa: E402

NS = [50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
AUCS = [0.66, 0.76, 0.82, 0.90]
PI0, RHO, C, N_REP = 0.16, 0.75, 2.0, 2000
OUT = paths.FIGS


def run():
    rng = np.random.default_rng(21); res = {}
    alpha = RHO * PI0
    for auc_ in AUCS:
        mu = math.sqrt(2) * stats.norm.ppf(auc_)
        def Cov(lam):
            return PI0 * stats.norm.sf(lam) + (1 - PI0) * stats.norm.sf(lam - mu)
        for n in NS:
            T = {"cp": [0, 0.0], "fs": [0, 0.0]}
            for _ in range(N_REP):
                ytr = (rng.random(n) >= PI0).astype(int)
                lams = np.quantile(rng.standard_normal(n) + mu * ytr, 1 - fs.Q_GRID); lams[-1] = -np.inf
                yy = (rng.random(n) >= PI0).astype(int); ss = rng.standard_normal(n) + mu * yy
                for m, l in (("cp", fs.cp_bonf(ss, yy, alpha)), ("fs", fs.fs_floor(lams, ss, yy, alpha, C))):
                    if l != np.inf:
                        T[m][0] += 1; T[m][1] += Cov(l) if np.isfinite(l) else 1.0
            res[f"auc={auc_}|n={n}"] = {m: {"p_fire": T[m][0] / N_REP, "coverage": T[m][1] / N_REP} for m in T}
            print(auc_, n, res[f"auc={auc_}|n={n}"], flush=True)
    json.dump(res, open(os.path.join(paths.OUT_JSON, "BUDGET_CURVE.json"), "w"), indent=1)
    return res


def fig(res):
    INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
                         "pdf.fonttype": 42})
    cols = {0.66: "#cde2fb", 0.76: "#86b6ef", 0.82: "#2a78d6", 0.90: "#104281"}
    cols = {0.66: "#86b6ef", 0.76: "#3987e5", 0.82: "#1c5cab", 0.90: "#0d366b"}
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.9), sharex=True)
    for a in AUCS:
        for ax, key in zip(axes, ("p_fire", "coverage")):
            ax.plot(NS, [res[f"auc={a}|n={n}"]["fs"][key] for n in NS], "-", color=cols[a], lw=1.3, label=f"AUC {a}")
            ax.plot(NS, [res[f"auc={a}|n={n}"]["cp"][key] for n in NS], "--", color=cols[a], lw=0.9)
    for ax, lab in zip(axes, (r"$P_{\mathrm{fire}}$", "mean coverage")):
        ax.set_xscale("log"); ax.set_xlabel("calibration problems"); ax.set_ylabel(lab); ax.grid(True, axis="y")
        ax.axvspan(67, 158, color="#eda100", alpha=0.16, lw=0); ax.set_ylim(0, 1)
    axes[0].plot([], [], "-", color=INK, lw=1.3, label="FS-floor"); axes[0].plot([], [], "--", color=INK, lw=0.9, label=r"CP, Eq.~(1)")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.5, labelspacing=0.2)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_budget.pdf"), metadata={"CreationDate": None})


if __name__ == "__main__":
    fig(run())
