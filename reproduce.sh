#!/usr/bin/env bash
# Regenerates every number, table and figure in the paper from data/.
# Total wall time on a laptop CPU: about 70 minutes. Each step is deterministic (seeds pinned).
set -euo pipefail
cd "$(dirname "$0")"
run() { echo "== $1"; python3 "analysis/$1"; }
run conformal_extras.py      # ~1 min   certificates on GSM8K (Tables 1, 5, 6; Section 4)
run shift_extras.py          # ~1 min   certificates under benchmark shift (Section 6, Table 10, Figure 6)
run bounds_extras.py         # ~10 min  bound comparison + Gaussian model (Section 4.1-4.2, Tables 4, 7)
run probe_recon.py           # ~10 min  cross-fitted probe reconstruction (Section 4.3, Appendix C, Table 3)
run goodhart_extras.py       # ~15 min  best-of-n selection (Section 7, Table 11, Figures 3, 7)
run fs_floor_eval.py         # ~25 min  FS-floor vs the Bonferroni rule (Section 5, Tables 2, 8, Figure 4)
run budget_curve.py          # ~5 min   calibration-budget curves (Figure 2)
run synthetic_overshoot.py   # <1 min   overshoot of failing certificates in the Gaussian model (Table 7)
run fs_baselines.py          # ~2 min   G=10/G=5 and floor-start baselines, predicted P_fire (Section 5, Table 9)
run redraw_budget.py
run make_figures.py
run make_tables.py
run make_fs_tables.py
run make_recon_table.py
run make_baselines_table.py
run make_appendix_extras.py
python3 verify.py
