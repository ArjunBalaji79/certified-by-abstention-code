# Certified by Abstention

Code and data for **Certified by Abstention: Distribution-Free Guarantees for Chain-of-Thought Verifiers at Small Calibration Budgets** (under review, AISTATS 2027).

Every number, table and figure in the paper regenerates from the cached per-trace scores in `data/` in about 70 minutes on a laptop CPU. No GPU, model weights or API access is needed; those were used only to produce `data/` (see `pipeline/`).

## Reproduce

```bash
pip install -r requirements.txt   # Python 3.9+, pinned versions
./reproduce.sh                    # ~70 min; writes outputs/json, outputs/tables, outputs/figs
```

All analyses are seeded and deterministic. `outputs/json/PROSE_NUMBERS.json` and `FS_PROSE.json` hold the scalars quoted in the text.

## Layout

```
src/        the certificate (cert.py), FS-floor (fs_floor.py) and the data-loading helpers
analysis/   one script per experiment, plus the table and figure generators
data/       cached inputs: frozen per-trace scores, labels, features (148 MB; see data/README.md)
pipeline/   the GPU pipeline that produced data/ (not needed to reproduce the paper)
```

## What each script produces

| Script | Output | In the paper |
|---|---|---|
| `analysis/conformal_extras.py` | `CONFORMAL_EXTRAS.json` | Table 1, Tables 5–6, Section 4 |
| `analysis/shift_extras.py` | `SHIFT_EXTRAS.json` | Section 6, Table 10, Figure 6 |
| `analysis/bounds_extras.py` | `BOUNDS_EXTRAS.json` | Sections 4.1–4.2, Tables 4 and 7, Figure 5 |
| `analysis/probe_recon.py` | `PROBE_RECON.json` | Section 4.3, Appendix C, Table 3 |
| `analysis/goodhart_extras.py` | `GOODHART_EXTRAS.json` | Section 7, Table 11, Figures 3 and 7 |
| `analysis/fs_floor_eval.py` | `FS_FLOOR2.json` | Section 5, Tables 2 and 8, Figure 4 |
| `analysis/budget_curve.py` | `BUDGET_CURVE.json` | Figure 2 |
| `analysis/synthetic_overshoot.py` | `SYNTH_OVERSHOOT.json` | Table 7 |
| `analysis/fs_baselines.py` | `FS_BASELINES.json` | Section 5, Table 9 |
| `analysis/make_tables.py`, `make_fs_tables.py`, `make_recon_table.py`, `make_baselines_table.py`, `make_appendix_extras.py` | `outputs/tables/*.tex`, Figures 4–7 | all tables |
| `analysis/make_figures.py`, `redraw_budget.py` | `outputs/figs/*.pdf` | Figures 1–3 |

## Method in one paragraph

A verifier signal scores a chain-of-thought trace; a certificate is a threshold with a finite-sample bound on the error rate among accepted traces. `src/cert.py` implements the Clopper–Pearson selective certificate of Eq. (1): a Bonferroni grid over accepted-set sizes, testing the traces strictly above each order statistic, valid for any continuous score distribution. `src/fs_floor.py` implements FS-floor: fixed-sequence testing over a threshold grid fixed from the training split, started at the certification floor of Lemma 1, with the start constant chosen before calibration by simulation. Targets α = ρ·π₀ use the training-partition estimate of the base error rate, so no calibration or test label enters the target.

## Data

`data/` holds, for seven open models on GSM8K and four shift benchmarks, the labels and the frozen scores of five verifier signals under five training seeds; for two models, 32 best-of-n samples per problem with the same frozen scores; and for seven models the readable feature matrices and mid-depth residual-stream vectors used by the probe analyses. Field-level documentation is in `data/README.md`.

## License

Code: MIT (see `LICENSE`). Data (`data/`): CC BY 4.0. The traces were generated with open-weight models under their own licenses (Qwen: Apache-2.0; OLMo: Apache-2.0) on GSM8K (MIT), ARC-Challenge (CC BY-SA 4.0), StrategyQA (MIT), CommonsenseQA (MIT) and BIG-Bench-Hard (MIT).

## Citation

Anonymized for review. A citation entry will be added on publication.
