"""Main-text figures. Local, CPU, deterministic.

  fig_floor.pdf     analytic certification floor for a Clopper-Pearson selective
                    certificate (no data; Lemma 1 of the paper)
  fig_shift.pdf     guarantee violation vs AUC drop under benchmark shift
  fig_pressure.pdf  best-of-n: realized error / alpha, coverage, selected accuracy

Reads outputs/json/*.json and data/bon/*.npz; writes paper/figs/.
"""

import json
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from src import paths  # noqa: E402
GH = paths.BON
OUT = paths.FIGS
EXTRA = paths.OUT_JSON

# fixed signal order -> fixed colour + marker (identity never colour-alone)
SIG = {
    "confidence": ("#2a78d6", "o", "confidence"),
    "jspace":     ("#eb6834", "s", "J-lens"),
    "probe":      ("#1baf7a", "^", "probe"),
    "logitlens":  ("#eda100", "D", "logit lens"),
    "surface":    ("#e87ba4", "v", "surface"),
}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42,
})


def cp_upper(k, a, delta):
    if a == 0 or k >= a:
        return 1.0
    return float(stats.beta.ppf(1 - delta, k + 1, a - k))


def min_accepted(alpha, delta, err_frac):
    """Smallest accepted-set size a whose CP upper bound is <= alpha when the
    accepted set contains floor(err_frac * alpha * a) errors."""
    for a in range(1, 20001):
        k = int(np.floor(err_frac * alpha * a))
        if cp_upper(k, a, delta) <= alpha:
            return a
    return np.nan


def fig_floor():
    alphas = np.linspace(0.03, 0.35, 65)
    fig, ax = plt.subplots(figsize=(3.25, 1.85))
    specs = [(0.0, "-", "#104281", r"$\varepsilon=0$"),
             (0.25, "-", "#2a78d6", r"$\varepsilon=\alpha/4$"),
             (0.5, "-", "#86b6ef", r"$\varepsilon=\alpha/2$")]
    for frac, ls, c, lab in specs:
        ys = [min_accepted(a, 0.10 / 20, frac) for a in alphas]
        ax.plot(alphas, ys, ls, color=c, lw=1.4, label=lab)
    ys = [min_accepted(a, 0.10, 0.0) for a in alphas]
    ax.plot(alphas, ys, "--", color=MUTED, lw=1.0,
            label=r"$\varepsilon=0$, no grid ($\delta'=\delta$)")
    ax.axhspan(67, 158, color="#eda100", alpha=0.16, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(5, 3000)
    ax.set_xlim(0.03, 0.35)
    ax.set_xlabel(r"target selective error $\alpha$")
    ax.set_ylabel(r"min. accepted calibration points $a^\star$")
    ax.grid(True, which="major", axis="y")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6,
              bbox_to_anchor=(1.0, 1.04))
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_floor.pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def fig_shift():
    S = json.load(open(os.path.join(EXTRA, "SHIFT_EXTRAS.json")))
    fig, ax = plt.subplots(figsize=(3.25, 1.95))
    ax.plot([-0.22, 0.31], [-0.22, 0.31], color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(0.30, 0.325, "no benefit from selection", fontsize=6.5, ha="right",
            va="bottom", color=MUTED, rotation=0)
    for sig in ("confidence", "jspace", "probe"):
        c, mk, lab = SIG[sig]
        rows = [v for m, blk in S.items() if m != "_summary"
                for k, v in blk.items() if k.endswith("|" + sig) and v["err_pooled"] is not None]
        big = np.array([v["coverage"] > 0.02 for v in rows])
        xs = np.array([v["target_base_err"] - v["alpha"] for v in rows])
        ys = np.array([v["err_pooled"] - v["alpha"] for v in rows])
        ax.scatter(xs[big], ys[big], s=15, marker=mk, color=c, lw=0.5,
                   edgecolor="white", label=lab, zorder=3)
        ax.scatter(xs[~big], ys[~big], s=13, marker=mk, facecolor="none",
                   edgecolor=c, lw=0.7, zorder=2)
    ax.axhline(0, color=INK, lw=0.7, ls="--")
    ax.set_ylim(-0.6, 0.4)
    ax.set_xlabel(r"target base error $-\ \alpha$")
    ax.set_ylabel(r"error among accepted $-\ \alpha$")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    ax.grid(True, axis="y")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_shift.pdf"), metadata={"CreationDate": None})
    plt.close(fig)


# ---------------------------------------------------------------- pressure
K_FIXED, N_LEVELS = 32, [1, 2, 4, 8, 16, 32]


def bon_weights(K, n):
    d = comb(K, n)
    return np.array([comb(j - 1, n - 1) / d for j in range(1, K + 1)])


def rerank_curves(slug, n_boot=1000, seed=7):
    """Selected-trace accuracy under exact pooled best-of-n, as change from n=1,
    with a grouped (by-problem) bootstrap band. Same K=32 draw convention as
    the best-of-n analysis; bands average over 20 draws of the K-subset."""
    z = np.load(os.path.join(GH, f"scores_{slug}.npz"))
    y = z["y"].astype(int)
    groups = np.array([str(g) for g in z["groups"]])
    by_p = {}
    for i, g in enumerate(groups):
        by_p.setdefault(g, []).append(i)
    pools = [np.array(v) for g, v in sorted(by_p.items()) if len(v) >= K_FIXED]
    rng = np.random.default_rng(seed)
    tb = rng.random(len(y))
    W = np.stack([bon_weights(K_FIXED, n) for n in N_LEVELS])      # [N, K]
    out = {}
    for sig in SIG:
        s = z[f"s__{sig}"]
        acc = np.zeros((len(pools), len(N_LEVELS)))
        for _ in range(20):
            idx = np.array([p[rng.choice(len(p), K_FIXED, replace=False)]
                            for p in pools])
            order = np.argsort(s[idx] + 1e-12 * tb[idx], axis=1, kind="stable")
            ys = np.take_along_axis(y[idx], order, axis=1)          # [P, K]
            acc += ys @ W.T
        acc /= 20
        d = acc - acc[:, :1]
        bs = rng.integers(0, len(pools), size=(n_boot, len(pools)))
        boot = d[bs].mean(axis=1)                                   # [B, N]
        out[sig] = (d.mean(0), np.percentile(boot, 2.5, 0),
                    np.percentile(boot, 97.5, 0))
    return out


def fig_pressure():
    X = json.load(open(os.path.join(EXTRA, "GOODHART_EXTRAS.json")))
    models = ["qwen3.5-4b", "qwen3-4b"]
    names = {"qwen3.5-4b": "Qwen3.5-4B", "qwen3-4b": "Qwen3-4B"}
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.8))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.25, top=0.72, wspace=0.55)
    xs = np.arange(len(N_LEVELS))
    curves = {m: rerank_curves(m) for m in models}
    for j, m in enumerate(models):
        alpha = X[m]["alpha"]
        ax = axes[2 * j]
        for sig, (c, mk, lab) in SIG.items():
            by = X[m]["signals"][sig]["naive"]
            if 1 - by["1"]["p_empty"] < 0.02:      # < 20 firing replicates: too noisy
                ax.plot([], [], color=c, marker=mk, ms=3.2, lw=1.2, label=lab,
                        markeredgecolor="white", markeredgewidth=0.4)
                continue
            r = [by[str(n)]["err_pooled"] / alpha for n in N_LEVELS]
            ax.plot(xs, r, color=c, marker=mk, ms=3.2, lw=1.2, label=lab,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.axhline(1.0, color=INK, lw=0.7, ls="--")
        ax.set_ylim(0, 1.6)
        ax.set_title(f"{names[m]}\ncertificate", fontsize=7.5, pad=2)
        ax.set_ylabel(r"$\hat R/\alpha$")
        ax = axes[2 * j + 1]
        for sig, (c, mk, lab) in SIG.items():
            mu, lo, hi = curves[m][sig]
            ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
            ax.plot(xs, mu, color=c, marker=mk, ms=3.2, lw=1.2,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(f"{names[m]}\nselected accuracy", fontsize=7.5, pad=2)
        ax.set_ylabel(r"$\Delta$ accuracy")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in N_LEVELS])
        ax.set_xlabel("best-of-$n$", labelpad=1)
        ax.grid(True, axis="y")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
    fig.savefig(os.path.join(OUT, "fig_pressure.pdf"), metadata={"CreationDate": None})
    plt.close(fig)
    return {m: {s: [round(float(v), 4) for v in curves[m][s][0]] for s in SIG} for m in models}


if __name__ == "__main__":
    fig_floor()
    fig_shift()
    print(json.dumps(fig_pressure(), indent=1))
    for a in (0.05, 0.089, 0.10, 0.16, 0.20, 0.34):
        print(a, [min_accepted(a, 0.005, f) for f in (0, .25, .5)],
              min_accepted(a, 0.10, 0))
