# Pipeline (optional)

These two Modal apps produced everything in `data/`. They are included for transparency; **nothing in the paper needs them**, since their outputs are cached.

- `jspace_pipeline.py` — generation (sampling traces from each model), grading, step segmentation, feature extraction (Jacobian lens, logit lens, random transport, sentence embeddings, token log-probabilities, residual-stream vectors), and per-cell cross-validated AUCs (`data/pipeline_outputs/`). Also contains the best-of-n generation.
- `conformal_scores.py` — fits and freezes the per-seed scorers on the GSM8K training partitions and emits the per-trace scores and residual blocks in `data/scores/`, `data/bon/` and `data/features/`.

Requirements: a [Modal](https://modal.com) account, A100-40GB GPUs, Hugging Face access to the open-weight models named in the paper, and the Jacobian lens package `jlens` from `anthropics/jacobian-lens` pinned at commit `581d398` with the pre-fitted lenses from `neuronpedia/jacobian-lens` (revision `qwen-n1000`). The scripts reference two Modal volumes (`jspace-results` for outputs, `hf-cache` for weights); create them or rename. Expect several GPU-hours per model.
