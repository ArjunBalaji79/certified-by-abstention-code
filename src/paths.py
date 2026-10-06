"""Repository paths. Every script resolves data and outputs through this module."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SCORES = os.path.join(DATA, "scores")          # frozen per-trace scores, GSM8K + shift benchmarks (7 models)
BON = os.path.join(DATA, "bon")                # best-of-n samples, frozen scores (2 models)
FEATURES = os.path.join(DATA, "features")      # readable feature matrices and residual-stream blocks (7 models)
PIPELINE_OUT = os.path.join(DATA, "pipeline_outputs")
OUT_JSON = os.path.join(ROOT, "outputs", "json")
FIGS = os.path.join(ROOT, "outputs", "figs")
TABLES = os.path.join(ROOT, "outputs", "tables")
for _d in (OUT_JSON, FIGS, TABLES):
    os.makedirs(_d, exist_ok=True)
