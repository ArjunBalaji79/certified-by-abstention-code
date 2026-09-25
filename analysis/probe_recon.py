"""Cross-fitted probe reconstruction (replaces the in-sample residual test).

Question: is the probe's edge over the readable signals information the readable
features cannot express, or better estimation of the same information?
Test: reconstruct the probe score from the 119 readable features with a ridge
regression fit on OUT-OF-FOLD probe scores of the training problems, then on
held-out problems compare AUC(probe), AUC(readable logreg), AUC(reconstruction)
and AUC(probe - reconstruction).  All regularization constants chosen by grouped
CV on the training problems.  Also runs the CP certificate on held-out problems
with the reconstruction as the signal.  Writes outputs/json/PROBE_RECON.json.
"""
import json, os, sys
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src import paths  # noqa: E402
from src import probegap_common as pg  # noqa: E402  (load, auc, group_index)
from src import cert  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
C_READ = [0.01, 0.1, 1.0, 10.0]
C_PROBE = [0.001, 0.01, 0.1]
A_RIDGE = [0.1, 1.0, 10.0, 100.0]
N_REPL = 200


def cv_pick(X, y, groups, Cs):
    best, best_auc = None, -1
    for C in Cs:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        a = pg.auc(y, oof)
        if a > best_auc:
            best, best_auc, best_oof = C, a, oof.copy()
    return best, best_auc, best_oof


def cert_stats(s, y, groups, alpha, rng, n_repl=N_REPL):
    gl = [v for _, v in sorted(pg.group_index(groups).items())]
    cov, nacc, nerr = [], [], []
    for _ in range(n_repl):
        rows = np.array([ix[rng.integers(len(ix))] for ix in gl])
        perm = rng.permutation(len(rows)); ncal = int(round(pg.CAL_SHARE * len(rows)))
        cal, tst = rows[perm[:ncal]], rows[perm[ncal:]]
        if (y[cal] == 0).sum() == 0 or len(np.unique(y[tst])) < 2:
            continue
        sel = s[tst] >= cert.cp_threshold_exact(s[cal], y[cal], alpha)
        cov.append(sel.mean()); nacc.append(int(sel.sum())); nerr.append(int((y[tst][sel] == 0).sum()))
    return cov, nacc, nerr


def main():
    models = sorted(f[len("probe_depths_"):-4] for f in os.listdir(pg.DEPTHS) if f.startswith("probe_depths_"))
    out = {}
    for slug in models:
        d = pg.load(slug)
        if d is None:
            continue
        y, groups = d["y"], d["groups"]
        readable = np.concatenate([d["X"][k] for k in sorted(d["X"])], axis=1)
        probe = np.concatenate([d["blocks"]["mid_step"], d["blocks"]["mid_ans"]], axis=1).astype(np.float32)
        pi0 = 1 - y.mean()
        R = {k: [] for k in ("probe", "readable", "recon", "resid", "recon_insample_resid", "readable_C", "probe_C", "ridge_a", "r2_recon")}
        cert = {rho: {k: {"cov": [], "nacc": [], "nerr": []} for k in ("probe", "readable", "recon")} for rho in (1.0, 0.75)}
        for seed in SEEDS:
            rng = np.random.default_rng(3000 + seed)
            gidx = pg.group_index(groups); problems = sorted(gidx)
            perm = rng.permutation(len(problems)); ntr = int(round(pg.TRAIN_FRAC * len(problems)))
            train_g = {problems[i] for i in perm[:ntr]}
            tr = np.array([g in train_g for g in groups]); held = ~tr
            if len(np.unique(y[tr])) < 2:
                continue
            Cr, _, _ = cv_pick(readable[tr], y[tr], groups[tr], C_READ)
            Cp, _, probe_oof_tr = cv_pick(probe[tr], y[tr], groups[tr], C_PROBE)
            m_read = make_pipeline(StandardScaler(), LogisticRegression(C=Cr, max_iter=2000)).fit(readable[tr], y[tr])
            m_probe = make_pipeline(StandardScaler(), LogisticRegression(C=Cp, max_iter=2000)).fit(probe[tr], y[tr])
            s_read = m_read.predict_proba(readable)[:, 1]
            s_probe = m_probe.predict_proba(probe)[:, 1]
            # ridge: readable -> OOF probe score, alpha by grouped CV R^2 on train
            best_a, best_r2 = None, -np.inf
            for a in A_RIDGE:
                pred = np.zeros(tr.sum())
                Xt, yt, gt = readable[tr], probe_oof_tr, groups[tr]
                for i1, i2 in GroupKFold(5).split(Xt, yt, gt):
                    rg = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xt[i1], yt[i1]); pred[i2] = rg.predict(Xt[i2])
                r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
                if r2 > best_r2:
                    best_a, best_r2 = a, r2
            rg = make_pipeline(StandardScaler(), Ridge(alpha=best_a)).fit(readable[tr], probe_oof_tr)
            recon = rg.predict(readable)
            resid = s_probe - recon
            # the old (in-sample target) residual, for reference
            rg_old = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(readable[tr], s_probe[tr])
            resid_old = s_probe - rg_old.predict(readable)
            yh, gh = y[held], groups[held]
            R["probe"].append(pg.auc(yh, s_probe[held])); R["readable"].append(pg.auc(yh, s_read[held]))
            R["recon"].append(pg.auc(yh, recon[held])); R["resid"].append(pg.auc(yh, resid[held]))
            R["recon_insample_resid"].append(pg.auc(yh, resid_old[held]))
            R["readable_C"].append(Cr); R["probe_C"].append(Cp); R["ridge_a"].append(best_a)
            R["r2_recon"].append(1 - ((s_probe[held] - recon[held]) ** 2).sum() / ((s_probe[held] - s_probe[held].mean()) ** 2).sum())
            pi0_tr = 1 - y[tr].mean()
            for rho in (1.0, 0.75):
                rc = np.random.default_rng(7000 + seed)
                for name, s in (("probe", s_probe), ("readable", s_read), ("recon", recon)):
                    c, na, ne = cert_stats(s[held], yh, gh, rho * pi0_tr, np.random.default_rng(rc.integers(1e9)))
                    cert[rho][name]["cov"] += c; cert[rho][name]["nacc"] += na; cert[rho][name]["nerr"] += ne
            print(slug, seed, {k: round(R[k][-1], 3) for k in ("probe", "readable", "recon", "resid")}, Cr, Cp, best_a, flush=True)
        out[slug] = {"n": int(len(y)), "pi0": round(float(pi0), 4),
                     "auc": {k: round(float(np.mean(R[k])), 4) for k in ("probe", "readable", "recon", "resid", "recon_insample_resid", "r2_recon")},
                     "auc_by_seed": {k: [round(float(v), 4) for v in R[k]] for k in ("probe", "readable", "recon", "resid")},
                     "chosen": {"readable_C": R["readable_C"], "probe_C": R["probe_C"], "ridge_alpha": R["ridge_a"]},
                     "cert": {str(rho): {k: {"coverage": round(float(np.mean(v["cov"])), 4),
                                            "p_fire": round(float(np.mean(np.array(v["nacc"]) > 0)), 4),
                                            "R_over_alpha": (round(sum(v["nerr"]) / sum(v["nacc"]) / (rho * pi0), 3) if sum(v["nacc"]) else None)}  # pi0: full-sample, descriptive
                                        for k, v in cert[rho].items()} for rho in (1.0, 0.75)}}
    json.dump(out, open(os.path.join(paths.OUT_JSON, "PROBE_RECON.json"), "w"), indent=1)
    print(json.dumps({m: v["auc"] for m, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
