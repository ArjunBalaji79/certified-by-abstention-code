"""J-space vs surface trajectory verification pipeline.

Research question: do J-lens latent concept trajectories predict CoT
correctness better than surface trajectories from sentence embeddings?

Stages (Modal functions):
  smoke()            - GPU sanity: canary readouts + 2 problems end-to-end + cost projection
  generate_traces()  - GPU: sampled CoT traces on GSM8K (escalate MATH), graded + segmented
  extract_readouts() - GPU: J-lens / logit-lens / random-lens readouts at step boundaries
  analyze()          - CPU: feature battery, classifiers, figures, results.json, SUMMARY.md
  ablate()           - GPU (optional Stage D): concept-direction ablation
  orchestrate()      - CPU driver: vertical slice (50 problems) -> scale -> escalate -> analyze

Commands:
  modal run jspace_pipeline.py::smoke_entry            # synchronous smoke (must pass first)
  modal run --detach jspace_pipeline.py::orchestrate   # detached full pipeline
  modal app logs jspace-pipeline
  modal volume get jspace-results / ./jspace_results
"""

import json
import math
import os
import re
import time

import modal

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

MODEL = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"

# v2: multi-model registry. "root" is the path prefix inside the volume; the
# primary model keeps the original flat layout so v1 artifacts are reused.
MODELS = {
    "qwen3.5-4b": {"hf": MODEL, "lens_file": LENS_FILE, "root": ""},
    "qwen3.5-0.8b": {
        "hf": "Qwen/Qwen3.5-0.8B",
        "lens_file": "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
        "root": "v2/qwen3.5-0.8b/",
    },
    "qwen3-4b": {
        "hf": "Qwen/Qwen3-4B",
        "lens_file": "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
        "root": "v2/qwen3-4b/",
    },
    # --- v3: cross-family (ungated), lenses pinned to lens-repo main ---
    "olmo-3-7b": {
        "hf": "allenai/Olmo-3-1025-7B",
        "lens_file": "olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt",
        "lens_revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
        "root": "v3/olmo-3-7b/",
    },
    "qwen2.5-7b-it": {
        "hf": "Qwen/Qwen2.5-7B-Instruct",
        "lens_file": "qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt",
        "lens_revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
        "root": "v3/qwen2.5-7b-it/",
    },
    "gpt-oss-20b": {
        "hf": "openai/gpt-oss-20b",
        "lens_file": "gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt",
        "lens_revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
        "root": "v3/gpt-oss-20b/",
        "gpu": "A100-80GB",  # MXFP4 dequantizes to bf16 on Ampere (~48GB)
    },
    # --- v4: pre-registered out-of-sample models (PREREGISTRATION.md) ---
    "qwen3-1.7b": {
        "hf": "Qwen/Qwen3-1.7B",
        "lens_file": "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
        "lens_revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
        "root": "v4/qwen3-1.7b/",
    },
    "qwen3-8b": {
        "hf": "Qwen/Qwen3-8B",
        "lens_file": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
        "lens_revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
        "root": "v4/qwen3-8b/",
    },
}
V4_BUDGET_USD = 12.0
V4_CELLS = (  # pre-registered in PREREGISTRATION.md — do not edit
    [("qwen3-1.7b", b) for b in ("gsm8k", "arc", "strategyqa", "csqa")]
    + [("qwen3-8b", b) for b in ("gsm8k", "arc", "strategyqa", "csqa")]
    + [(s, "csqa") for s in ("qwen3.5-4b", "qwen3.5-0.8b", "olmo-3-7b",
                             "qwen2.5-7b-it")]
)
PREREG_SHA = "f9def7299b50a635512d405b33122eb2dd264df60664b7fbe3c5e4de04c41f45"
PRIMARY = "qwen3.5-4b"
EXTRA_MODELS = ["qwen3.5-0.8b", "qwen3-4b"]
V3_MODELS = ["olmo-3-7b", "qwen2.5-7b-it", "gpt-oss-20b"]
V3_BENCHMARKS = ["gsm8k", "bbhlogic", "strategyqa", "arc"]
V3_BUDGET_USD = 12.0  # final-phase cap (2026-07-28): user approved +$30; keep 2.4x margin over the ~$5 estimate

DATASET_PRIMARY = "openai/gsm8k"
N_PROBLEMS = 400
N_SLICE = 50            # vertical slice size
K_SAMPLES = 4
TEMP = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 640
MIN_INCORRECT = 120
MATH_N_PROBLEMS = 200
TOP_K_CONCEPTS = 20
SEED = 0
BUDGET_USD = 60.0
MAX_SEQ = 1536          # tokenized prompt+trace cap for extraction
GEN_BATCH = 32
MIN_STEPS = 3

VOLUME = "jspace-results"
APP_NAME = "jspace-pipeline"
RESULTS = "/results"
HF = "/hf"

T_SMOKE, T_GEN, T_EXTRACT, T_ANALYZE = 900, 7200, 14400, 3600

A100_RATE = 0.000583            # $/s, modal.com/pricing 2026-07-26 (A100 40GB)
CPU_RATE = 0.000176             # $/s est: 8 cores + 32GiB on Modal CPU pricing

FEWSHOT = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Let's think step by step. In April, Natalia sold 48 clips. In May, she sold half as many, so she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. #### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: Let's think step by step. Weng earns 12 / 60 = $0.2 per minute. For 50 minutes, she earned 0.2 x 50 = $10. #### 10

Q: {q}
A: Let's think step by step."""

DIRECT_FEWSHOT = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: #### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: #### 10

Q: {q}
A: ####"""

MC_FEWSHOT = """Q: On a shelf there are three books: a red book, a green book, and a blue book. The green book is to the right of the red book. The blue book is to the left of the red book. Which book is in the middle?
Options:
(A) the red book
(B) the green book
(C) the blue book
A: Let's think step by step. The blue book is left of the red book, so the order starts blue, red. The green book is right of the red book, so the order is blue, red, green. The middle one is the red book. #### A

Q: Tom has a meeting on the day after Monday. On which day is the meeting?
Options:
(A) Sunday
(B) Tuesday
(C) Wednesday
A: Let's think step by step. The day after Monday is Tuesday. So the meeting is on Tuesday. #### B

Q: {q}
A: Let's think step by step."""

YESNO_FEWSHOT = """Q: Would a pet cat fit inside a standard shoebox?
A: Let's think step by step. A standard shoebox is about 35 cm long. An adult cat is longer than that and much taller when sitting. So a pet cat would not fit comfortably inside. #### no

Q: Is the number of legs on two spiders greater than the number of legs on three dogs?
A: Let's think step by step. Two spiders have 2 x 8 = 16 legs. Three dogs have 3 x 4 = 12 legs. 16 is greater than 12. #### yes

Q: {q}
A: Let's think step by step."""

STYLE_FEWSHOT = {"numeric": FEWSHOT, "mc": MC_FEWSHOT, "yesno": YESNO_FEWSHOT}


def build_prompt(problem):
    return STYLE_FEWSHOT[problem.get("style", "numeric")].format(q=problem["question"])


def norm_answer(s, style):
    if s is None:
        return None
    if style == "numeric":
        return parse_num(s)
    s = str(s).strip().lower()
    s = re.sub(r"[().\"']", "", s).strip()
    return s or None


def answer_equal(pred, gold, style):
    if style == "numeric":
        return nums_equal(pred, gold if isinstance(gold, float) else parse_num(gold))
    return pred is not None and gold is not None and str(pred) == str(gold)


CANARIES = [
    ("Fact: The currency used in the country shaped like a boot is",
     ["italy", "ital", "lira", "euro", "rome"]),
    ("Fact: The number of legs on the animal that spins webs is",
     ["spider", "eight", "8", "arachn"]),
    ("Fact: The capital city of the country famous for the Eiffel Tower is",
     ["paris", "france", "french"]),
]

# ----------------------------------------------------------------------------
# Modal app / image
# ----------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.14.1",
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "sentence-transformers==5.6.1",
        "scikit-learn==1.9.0",
        "numpy==2.5.1",
        "pandas==3.0.5",
        "matplotlib==3.11.1",
        "pyarrow==25.0.0",
        "huggingface-hub==1.24.0",
        f"jlens @ git+https://github.com/anthropics/jacobian-lens@{JLENS_COMMIT}",
    )
    .env({"HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"})
)

app = modal.App(APP_NAME)
results_vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
VOLS = {RESULTS: results_vol, HF: hf_vol}

gpu_kw = dict(image=image, gpu="A100-40GB", volumes=VOLS, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
gpu80_kw = dict(gpu_kw, gpu="A100-80GB", memory=65536)
A100_80_RATE = 0.000694  # $/s, modal.com/pricing 2026-07-26
cpu_kw = dict(image=image, volumes=VOLS, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))

# ----------------------------------------------------------------------------
# Pure-python helpers (no heavy imports)
# ----------------------------------------------------------------------------


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    print(f"[{now_iso()}] {msg}", flush=True)


def append_decision(text):
    """Append one line to /results/DECISIONS.md (call from inside a container)."""
    try:
        results_vol.reload()
    except Exception:
        pass
    with open(f"{RESULTS}/DECISIONS.md", "a") as f:
        f.write(f"- [{now_iso()}] {text}\n")
    results_vol.commit()
    log(f"DECISION: {text}")


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def add_cost(usd, label):
    """Accumulate estimated spend in /results/budget.json."""
    try:
        results_vol.reload()
    except Exception:
        pass
    b = read_json(f"{RESULTS}/budget.json", {"spent_usd": 0.0, "events": []})
    b["spent_usd"] = round(b["spent_usd"] + usd, 4)
    b["events"].append({"t": now_iso(), "label": label, "usd": round(usd, 4)})
    write_json(f"{RESULTS}/budget.json", b)
    results_vol.commit()
    return b["spent_usd"]


def get_spent():
    try:
        results_vol.reload()
    except Exception:
        pass
    return read_json(f"{RESULTS}/budget.json", {"spent_usd": 0.0})["spent_usd"]


NUM_RE = r"-?\$?[\d,]+(?:\.\d+)?"


def parse_num(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def nums_equal(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) <= 1e-4 * max(1.0, abs(b))


def truncate_gen(gen_text):
    """Cut generation at the next few-shot question or just after the #### answer line."""
    g = gen_text.split("\nQ:")[0]
    m = re.search(r"####[^\n]*", g)
    if m:
        g = g[: m.end()]
    return g


def extract_pred(gen_text):
    m = re.search(rf"####\s*({NUM_RE})", gen_text)
    return parse_num(m.group(1)) if m else None


def segment_steps(gen_text):
    """Split the reasoning (before ####) into sentence steps with char offsets.

    Returns list of (text, start, end) in gen_text coordinates.
    """
    m = re.search(r"####", gen_text)
    reasoning_end = m.start() if m else len(gen_text)
    steps = []
    pos = 0
    # split on newlines and sentence enders, keeping offsets
    for piece in re.split(r"(?<=[.!?;])\s+|\n+", gen_text[:reasoning_end]):
        if piece is None:
            continue
        start = gen_text.find(piece, pos)
        if start < 0:
            continue
        end = start + len(piece)
        pos = end
        stripped = piece.strip()
        if len(stripped) >= 8 and any(c.isalnum() for c in stripped):
            steps.append((stripped, start, end))
    return steps


def answer_char_end(gen_text):
    m = re.search(rf"####\s*{NUM_RE}", gen_text)
    return m.end() if m else len(gen_text)


def extract_boxed(sol):
    i = sol.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth = 1
    out = []
    while j < len(sol) and depth:
        c = sol[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        j += 1
    s = "".join(out)
    s = re.sub(r"\\(?:!|,|;| )", "", s).replace("$", "").replace(",", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).strip()
    frac = re.fullmatch(r"\\d?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if frac:
        denom = float(frac.group(2))
        return float(frac.group(1)) / denom if denom else None
    return parse_num(s)


def char_to_token(offsets, char_end):
    """Largest token index whose span starts before char_end."""
    idx = None
    for i, (s, e) in enumerate(offsets):
        if s == e == 0 and i > 0:  # special tokens
            continue
        if s < char_end:
            idx = i
        else:
            break
    return idx


# ----------------------------------------------------------------------------
# In-container heavy helpers
# ----------------------------------------------------------------------------


def load_model_tok(slug=PRIMARY):
    import torch
    import transformers

    hf_name = MODELS[slug]["hf"]
    tok = transformers.AutoTokenizer.from_pretrained(hf_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_name, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tok


def load_lenses(lens_model, slug=PRIMARY):
    """Returns (jlens_lens, random_lens) with jacobians pre-moved to cuda."""
    import torch
    from jlens import JacobianLens

    lens = JacobianLens.from_pretrained(
        LENS_REPO, filename=MODELS[slug]["lens_file"],
        revision=MODELS[slug].get("lens_revision", LENS_REVISION),
    )
    g = torch.Generator().manual_seed(SEED)
    rnd = {}
    for layer, J in lens.jacobians.items():
        R = torch.randn(J.shape, generator=g)
        rnd[layer] = R * (J.norm() / R.norm())
    random_lens = JacobianLens(jacobians=rnd, n_prompts=1, d_model=lens.d_model)
    for lobj in (lens, random_lens):
        lobj.jacobians = {k: v.cuda() for k, v in lobj.jacobians.items()}
    return lens, random_lens


def probe_layers_of(lens, n_layers, stride=2):
    layers = [l for l in lens.source_layers if 0 <= l < n_layers]
    return layers[::stride]


def sample_problem_ids(n_total, n_want):
    import random as _random

    rng = _random.Random(SEED)
    return rng.sample(range(n_total), min(n_want, n_total))


def load_gsm8k():
    import datasets

    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, row in enumerate(ds):
        gold = parse_num(row["answer"].split("####")[-1])
        if gold is not None:
            problems.append({"pid": f"gsm8k-{i}", "question": row["question"], "gold": gold})
    return problems


def load_math(levels=(3, 4, 5)):
    import datasets

    last_err = None
    for name in ("nlile/hendrycks-MATH-benchmark", "EleutherAI/hendrycks_math"):
        try:
            if name == "EleutherAI/hendrycks_math":
                parts = []
                for cfg in ("algebra", "counting_and_probability", "number_theory",
                            "prealgebra"):
                    parts.append(datasets.load_dataset(name, cfg, split="test"))
                ds = datasets.concatenate_datasets(parts)
            else:
                ds = datasets.load_dataset(name, split="test")
            cols = ds.column_names
            qcol = "problem" if "problem" in cols else "question"
            scol = "solution" if "solution" in cols else "answer"
            problems = []
            for i, row in enumerate(ds):
                lv = row.get("level")
                lvnum = None
                if isinstance(lv, int):
                    lvnum = lv
                elif lv is not None:
                    m = re.search(r"\d", str(lv))
                    lvnum = int(m.group()) if m else None
                if lvnum not in levels:
                    continue
                gold = extract_boxed(str(row[scol]))
                if gold is None:
                    continue
                problems.append({"pid": f"math-{i}", "question": row[qcol], "gold": gold})
            if problems:
                return problems, name
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"could not load a MATH dataset: {last_err}")


BENCHMARKS = {
    "gsm8k": {"n_default": N_PROBLEMS},
    "math": {"n_default": MATH_N_PROBLEMS},
    "bbhlogic": {"n_default": 200},
    "strategyqa": {"n_default": 200},
    "arc": {"n_default": 200},
    "csqa": {"n_default": 200},
}


def load_csqa():
    import datasets

    ds = datasets.load_dataset("tau/commonsense_qa", split="validation")
    problems = []
    for i, row in enumerate(ds):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        if row["answerKey"] not in labels:
            continue
        opts = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
        problems.append({
            "pid": f"csqa-{i}", "dataset": "csqa",
            "question": f"{row['question'].strip()}\nOptions:\n{opts}",
            "gold": row["answerKey"].lower(), "style": "mc",
        })
    return problems


def load_bbhlogic():
    import datasets

    configs = ["logical_deduction_five_objects",
               "tracking_shuffled_objects_three_objects", "web_of_lies"]
    problems = []
    last_err = None
    for repo in ("lukaemon/bbh", "maveriq/bigbenchhard"):
        try:
            for ci, cfg in enumerate(configs):
                ds = datasets.load_dataset(repo, cfg, split="test")
                for i, row in enumerate(ds):
                    tgt = str(row["target"]).strip()
                    style = "yesno" if tgt.lower() in ("yes", "no") else "mc"
                    gold = norm_answer(tgt, style)
                    if gold is None:
                        continue
                    problems.append({
                        "pid": f"bbhlogic-{ci}x{i}", "dataset": "bbhlogic",
                        "question": str(row["input"]).strip(),
                        "gold": gold, "style": style,
                    })
            if problems:
                return problems
        except Exception as e:  # noqa: BLE001
            last_err = e
            problems = []
    raise RuntimeError(f"could not load BBH: {last_err}")


def load_strategyqa():
    import datasets

    last_err = None
    for repo in ("ChilleD/StrategyQA", "wics/strategy-qa", "voidful/StrategyQA"):
        try:
            ds = datasets.load_dataset(repo, split="train")
            cols = ds.column_names
            qcol = "question" if "question" in cols else cols[0]
            acol = "answer" if "answer" in cols else None
            if acol is None:
                continue
            problems = []
            for i, row in enumerate(ds):
                ans = row[acol]
                if isinstance(ans, str):
                    ans = ans.strip().lower() in ("true", "yes")
                problems.append({
                    "pid": f"strategyqa-{i}", "dataset": "strategyqa",
                    "question": str(row[qcol]).strip(),
                    "gold": "yes" if ans else "no", "style": "yesno",
                })
            if problems:
                return problems
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"could not load StrategyQA: {last_err}")


def load_arc():
    import datasets

    ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    problems = []
    for i, row in enumerate(ds):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        # normalize numeric labels ("1".."4") to letters for a uniform format
        lmap = {l: chr(ord("A") + j) for j, l in enumerate(labels)}
        gold = lmap.get(row["answerKey"])
        if gold is None:
            continue
        opts = "\n".join(f"({lmap[l]}) {t}" for l, t in zip(labels, texts))
        problems.append({
            "pid": f"arc-{i}", "dataset": "arc",
            "question": f"{row['question'].strip()}\nOptions:\n{opts}",
            "gold": gold.lower(), "style": "mc",
        })
    return problems


def load_benchmark(name):
    if name == "gsm8k":
        return load_gsm8k()
    if name == "math":
        return load_math()[0]
    if name == "bbhlogic":
        return load_bbhlogic()
    if name == "strategyqa":
        return load_strategyqa()
    if name == "arc":
        return load_arc()
    if name == "csqa":
        return load_csqa()
    raise ValueError(f"unknown benchmark {name}")


def gen_batch(model, tok, full_prompts, seed_offset, k=None):
    """Generate k samples per prompt in one batch. Returns list of lists of texts.

    full_prompts are COMPLETE prompt strings (few-shot already applied)."""
    import torch

    k = k or K_SAMPLES
    prompts = [p for p in full_prompts for _ in range(k)]
    torch.manual_seed(SEED * 100003 + seed_offset)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    kwargs = dict(do_sample=True, temperature=TEMP, top_p=TOP_P,
                  max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
    with torch.inference_mode():
        try:
            out = model.generate(**enc, **kwargs, stop_strings=["\nQ:"], tokenizer=tok)
        except (TypeError, ValueError):
            out = model.generate(**enc, **kwargs)
    gen_ids = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
    return [texts[i * k:(i + 1) * k] for i in range(len(full_prompts))]


def extract_pred_text(gen_text):
    m = re.search(r"####\s*([^\n#]+)", gen_text)
    return m.group(1).strip() if m else None


def extract_pred_styled(gen_text, style):
    """Style-anchored answer parse. Base models often continue generating with
    no newline after the answer ("#### yespassage: ..."), so the answer must be
    matched at the START of the captured span, not taken wholesale."""
    if style == "numeric":
        return extract_pred(gen_text)
    raw = extract_pred_text(gen_text)
    if raw is None:
        return None
    if style == "yesno":
        m = re.match(r"\(?\s*(yes|no)\b", raw, re.IGNORECASE)
        return m.group(1).lower() if m else None
    if style == "mc":
        m = re.match(r"\(?\s*([A-Za-z])\)?(?![A-Za-z])", raw)
        return m.group(1).lower() if m else None
    return norm_answer(raw, style)


def make_trace_rows(problem, gens, tok):
    style = problem.get("style", "numeric")
    rows = []
    for sid, raw in enumerate(gens):
        gen = truncate_gen(raw)
        pred = extract_pred_styled(gen, style)
        steps = segment_steps(gen)
        prompt_text = build_prompt(problem)
        rows.append({
            "uid": f"{problem['pid']}-s{sid}",
            "problem_id": problem["pid"],
            "sample_id": sid,
            "dataset": problem.get("dataset", problem["pid"].split("-")[0]),
            "question": problem["question"],
            "gold": str(problem["gold"]),
            "style": style,
            "prompt_text": prompt_text,
            "gen_text": gen,
            "pred": str(pred) if pred is not None else None,
            "correct": bool(answer_equal(pred, problem["gold"], style)),
            "graded": pred is not None,
            "n_steps": len(steps),
            "usable": len(steps) >= MIN_STEPS and pred is not None,
            "steps_json": json.dumps([{"text": t, "start": s, "end": e} for t, s, e in steps]),
            "answer_char_end": answer_char_end(gen),
            "n_gen_tokens": len(tok(gen).input_ids),
        })
    return rows


# ----------------------------------------------------------------------------
# smoke
# ----------------------------------------------------------------------------


@app.function(timeout=T_SMOKE, **gpu_kw)
def smoke():
    import torch  # noqa: F401
    import jlens

    t_start = time.time()
    report = {"pass": False, "checks": {}, "timings": {}, "projection": {}}

    t0 = time.time()
    model_hf, tok = load_model_tok()
    lmodel = jlens.from_hf(model_hf, tok)
    lens, _ = load_lenses(lmodel)
    hf_vol.commit()  # persist downloaded weights
    report["timings"]["load_s"] = round(time.time() - t0, 1)
    n_layers = lmodel.n_layers
    probes = probe_layers_of(lens, n_layers)
    report["checks"]["model"] = f"{MODEL} n_layers={n_layers} d_model={lmodel.d_model}"
    report["checks"]["lens"] = repr(lens)
    report["checks"]["probe_layers"] = probes

    # --- canaries ---
    mid = [l for l in probes if 0.25 * n_layers <= l <= 0.95 * n_layers] or probes
    hits = 0
    canary_detail = []
    for prompt, expected in CANARIES:
        jl, _, _ = lens.apply(lmodel, prompt, layers=mid, positions=[-2, -1])
        found = None
        for layer in mid:
            for p in range(2):
                top = jl[layer][p].topk(TOP_K_CONCEPTS).indices.tolist()
                toks = [tok.decode([t]).strip().lower() for t in top]
                for exp in expected:
                    if any(exp in tk for tk in toks if tk):
                        found = (layer, p, exp)
                        break
                if found:
                    break
            if found:
                break
        hits += bool(found)
        canary_detail.append({"prompt": prompt, "hit": found})
    report["checks"]["canaries"] = canary_detail
    report["checks"]["canary_hits"] = f"{hits}/3"

    # --- 2 GSM8K problems end-to-end ---
    problems = load_gsm8k()
    ids = sample_problem_ids(len(problems), N_PROBLEMS)
    two = [problems[i] for i in ids[:2]]
    t0 = time.time()
    gens = gen_batch(model_hf, tok, [build_prompt(p) for p in two],
                     seed_offset=999999, k=K_SAMPLES)
    t_gen8 = time.time() - t0
    report["timings"]["gen_batch8_s"] = round(t_gen8, 1)

    rows = []
    for p, g in zip(two, gens):
        rows.extend(make_trace_rows(p, g, tok))
    usable = [r for r in rows if r["usable"]]
    report["checks"]["smoke_traces"] = {
        "total": len(rows), "usable": len(usable),
        "correct": sum(r["correct"] for r in rows),
    }

    t_extract = None
    if usable:
        r = usable[0]
        t0 = time.time()
        full = r["prompt_text"] + r["gen_text"]
        enc = tok(full, return_offsets_mapping=True, truncation=True, max_length=MAX_SEQ)
        offsets = enc["offset_mapping"]
        plen = len(r["prompt_text"])
        steps = json.loads(r["steps_json"])
        positions = []
        for st in steps:
            ti = char_to_token(offsets, plen + st["end"])
            if ti is not None:
                positions.append(ti)
        ans = char_to_token(offsets, plen + r["answer_char_end"])
        positions = sorted(set(positions + ([ans] if ans is not None else [])))
        for use_j in (True, False):
            lens.apply(lmodel, full, layers=probes, positions=positions,
                       max_seq_len=MAX_SEQ, use_jacobian=use_j)
        t_extract = (time.time() - t0) * 1.5  # 3 passes in prod vs 2 here
        report["timings"]["extract_per_trace_s"] = round(t_extract, 2)

    # --- projection ---
    def project(n_problems, k):
        n_traces = n_problems * k
        per_batch32 = 2.0 * t_gen8  # conservative: batch32 ~= 2x batch8 wall time
        gen_s = math.ceil(n_traces / GEN_BATCH) * per_batch32
        ext_s = n_traces * (t_extract or 3.0)
        overhead_s = 3 * report["timings"]["load_s"] + 600
        return gen_s + ext_s + overhead_s

    smoke_cost = (time.time() - t_start) * A100_RATE
    gpu_s_full = project(N_PROBLEMS, K_SAMPLES)
    gpu_s_min = project(100, 2)  # smallest config the shrink rules allow
    proj = {
        "full_gpu_hours": round(gpu_s_full / 3600, 2),
        "full_total_usd": round(gpu_s_full * A100_RATE + 1.0 + smoke_cost, 2),
        "min_config_total_usd": round(gpu_s_min * A100_RATE + 1.0 + smoke_cost, 2),
        "smoke_cost_usd": round(smoke_cost, 2),
        "budget_usd": BUDGET_USD,
    }
    report["projection"] = proj

    ok_canary = hits >= 2
    ok_e2e = len(usable) >= 1 and t_extract is not None
    # only fail on budget if even the minimum shrunk config cannot fit;
    # ordinary over-projection is handled by the orchestrator's shrink rules
    ok_budget = proj["min_config_total_usd"] <= BUDGET_USD
    report["checks"]["ok_canary"] = ok_canary
    report["checks"]["ok_e2e"] = ok_e2e
    report["checks"]["ok_budget"] = ok_budget
    report["pass"] = ok_canary and ok_e2e and ok_budget

    os.makedirs(RESULTS, exist_ok=True)
    write_json(f"{RESULTS}/smoke.json", report)
    results_vol.commit()
    add_cost(smoke_cost, "smoke")
    log(f"SMOKE {'PASS' if report['pass'] else 'FAIL'}: {json.dumps(report['checks'], default=str)[:800]}")
    log(f"PROJECTION: {json.dumps(proj)}")
    return report


# ----------------------------------------------------------------------------
# Stage A - generation
# ----------------------------------------------------------------------------


def _generate_traces_impl(dataset, want_pids, k_samples=K_SAMPLES, slug=PRIMARY,
                          root_override=None):
    """Generate traces for the given problem pids (subset semantics, resumable).

    Returns {"complete": bool, "n_done": int, "counts": {...}, "gpu_s": float}
    """
    import pandas as pd

    import random as _random

    t_start = time.time()
    deadline = t_start + T_GEN - 900
    root = root_override if root_override is not None else MODELS[slug]["root"]

    results_vol.reload()
    mpath = f"{RESULTS}/{root}traces/manifest.json"
    manifest = read_json(mpath, {"done": {}, "shards": []})

    problems = load_benchmark(dataset)
    by_pid = {p["pid"]: p for p in problems}
    log(f"benchmark {dataset}: {len(problems)} problems available")
    if want_pids is None:  # seeded sample chosen in-container
        n_want = BENCHMARKS[dataset]["n_default"]
        rng = _random.Random(SEED)
        sample = rng.sample(problems, min(n_want, len(problems)))
        want_pids = [p["pid"] for p in sample]

    todo_all = [p for p in want_pids if p not in manifest["done"]]
    if not todo_all:
        counts = count_traces(root)
        return {"complete": True, "n_done": 0, "counts": counts,
                "gpu_s": time.time() - t_start}

    model, tok = load_model_tok(slug)
    hf_vol.commit()
    os.makedirs(f"{RESULTS}/{root}traces", exist_ok=True)

    todo = [p for p in todo_all if p in by_pid]
    dropped = set(todo_all) - set(todo)
    if dropped:
        log(f"WARNING: {len(dropped)} pids not found in dataset, skipping")

    per_call = max(1, GEN_BATCH // k_samples)
    n_done = 0
    batch_i = 0
    while todo and time.time() < deadline:
        chunk = todo[:per_call]
        todo = todo[per_call:]
        probs = [by_pid[pid] for pid in chunk]
        seed_off = int(re.sub(r"\D", "", chunk[0]) or 0)
        gens = gen_batch(model, tok, [build_prompt(p) for p in probs],
                         seed_offset=seed_off, k=k_samples)
        rows = []
        for p, g in zip(probs, gens):
            rows.extend(make_trace_rows(p, g, tok))
        shard = f"{root}traces/{dataset}_{chunk[0]}.parquet"
        pd.DataFrame(rows).to_parquet(f"{RESULTS}/{shard}", index=False)
        for pid in chunk:
            manifest["done"][pid] = shard
        manifest["shards"].append(shard)
        write_json(mpath, manifest)
        results_vol.commit()
        n_done += len(chunk)
        batch_i += 1
        if batch_i % 5 == 0:
            log(f"gen progress: {n_done}/{len(todo_all)} problems, "
                f"{time.time() - t_start:.0f}s elapsed")

    gpu_s = time.time() - t_start
    add_cost(gpu_s * A100_RATE, f"generate:{slug}:{dataset}:{n_done}problems")
    counts = count_traces(root)
    return {"complete": not todo, "n_done": n_done, "counts": counts, "gpu_s": gpu_s}


@app.function(timeout=T_GEN, **gpu_kw)
def generate_traces(dataset: str, want_pids: list, k_samples: int = K_SAMPLES,
                    slug: str = PRIMARY, root_override: str = None):
    return _generate_traces_impl(dataset, want_pids, k_samples, slug, root_override)


@app.function(timeout=T_GEN, **gpu80_kw)
def generate_traces_80(dataset: str, want_pids: list, k_samples: int = K_SAMPLES,
                       slug: str = PRIMARY, root_override: str = None):
    return _generate_traces_impl(dataset, want_pids, k_samples, slug, root_override)


def count_traces(root=""):
    import pandas as pd

    manifest = read_json(f"{RESULTS}/{root}traces/manifest.json",
                         {"done": {}, "shards": []})
    counts = {"n_traces": 0, "n_usable": 0, "n_correct": 0, "n_incorrect": 0,
              "by_dataset": {}}
    for shard in sorted(set(manifest["shards"])):
        try:
            df = pd.read_parquet(f"{RESULTS}/{shard}")
        except Exception:
            continue
        counts["n_traces"] += len(df)
        u = df[df["usable"]]
        counts["n_usable"] += len(u)
        counts["n_correct"] += int(u["correct"].sum())
        counts["n_incorrect"] += int((~u["correct"]).sum())
        for ds, n in u.groupby("dataset").size().items():
            counts["by_dataset"][ds] = counts["by_dataset"].get(ds, 0) + int(n)
    return counts


# ----------------------------------------------------------------------------
# Stage B - extraction
# ----------------------------------------------------------------------------


def readout_positions(tok, row):
    """Token positions for step ends + answer. Returns (full_text, step_pos,
    ans_pos) or None if the trace is unextractable."""
    full = row["prompt_text"] + row["gen_text"]
    enc = tok(full, return_offsets_mapping=True, truncation=True, max_length=MAX_SEQ)
    offsets = enc["offset_mapping"]
    plen = len(row["prompt_text"])
    steps = json.loads(row["steps_json"])
    step_pos = []
    for st in steps:
        ti = char_to_token(offsets, plen + st["end"])
        if ti is not None and (not step_pos or ti > step_pos[-1]):
            step_pos.append(ti)
    ans_pos = char_to_token(offsets, plen + row["answer_char_end"])
    if ans_pos is None or len(step_pos) < MIN_STEPS:
        return None
    if step_pos and ans_pos <= step_pos[-1]:
        ans_pos = step_pos[-1] + 1 if step_pos[-1] + 1 < len(offsets) else step_pos[-1]
    return full, step_pos, ans_pos


def _extract_readouts_impl(pid_filter=None, probe_stride=2, slug=PRIMARY,
                           root_override=None):
    """J-lens / logit-lens / random-lens readouts for usable traces.

    Writes readouts/{uid}.npz. Resumable via readouts/manifest.json.
    """
    import numpy as np
    import pandas as pd
    import torch
    import jlens

    t_start = time.time()
    deadline = t_start + T_EXTRACT - 900
    root = root_override if root_override is not None else MODELS[slug]["root"]

    results_vol.reload()
    tman_path = f"{RESULTS}/{root}traces/manifest.json"
    tman = read_json(tman_path, {"shards": []})
    rman_path = f"{RESULTS}/{root}readouts/manifest.json"
    rman = read_json(rman_path, {"done": []})
    done = set(rman["done"])

    dfs = []
    for shard in sorted(set(tman.get("shards", []))):
        try:
            dfs.append(pd.read_parquet(f"{RESULTS}/{shard}"))
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: unreadable shard {shard}: {e}")
    if not dfs:
        return {"complete": True, "n_done": 0, "gpu_s": 0.0}
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"]]
    if pid_filter is not None:
        df = df[df["problem_id"].isin(set(pid_filter))]
    todo = df[~df["uid"].isin(done)]
    if todo.empty:
        return {"complete": True, "n_done": 0, "gpu_s": time.time() - t_start}

    model_hf, tok = load_model_tok(slug)
    lmodel = jlens.from_hf(model_hf, tok)
    lens, random_lens = load_lenses(lmodel, slug)
    hf_vol.commit()
    probes = probe_layers_of(lens, lmodel.n_layers, stride=probe_stride)
    L, K = len(probes), TOP_K_CONCEPTS
    os.makedirs(f"{RESULTS}/{root}readouts", exist_ok=True)

    def first_tok_id(text):
        ids = tok(text, add_special_tokens=False).input_ids
        return ids[0] if ids else -1

    n_done = 0
    for _, r in todo.iterrows():
        if time.time() > deadline:
            break
        rp = readout_positions(tok, r)
        if rp is None:
            rman["done"].append(r["uid"])  # unextractable; skip permanently
            continue
        full, step_pos, ans_pos = rp
        steps = json.loads(r["steps_json"])
        positions = step_pos + [ans_pos]
        P = len(positions)

        # target tokens for rank trajectories
        try:  # numeric golds normalize ("72.0"->"72"); text golds pass through
            gold_repr = format(float(r["gold"]), "g")
        except (TypeError, ValueError):
            gold_repr = str(r["gold"])
        gold_id = first_tok_id(" " + gold_repr)
        step_num_ids = []
        for st in steps[: len(step_pos)]:
            nums = re.findall(NUM_RE, st["text"])
            step_num_ids.append(first_tok_id(" " + nums[-1].replace("$", "").replace(",", ""))
                                if nums else -1)
        step_num_ids.append(-1)  # answer position row

        out = {}
        for tag, lobj, use_j in (("j", lens, True), ("ll", lens, False),
                                 ("rr", random_lens, True)):
            ll, _, _ = lobj.apply(lmodel, full, layers=probes, positions=positions,
                                  max_seq_len=MAX_SEQ, use_jacobian=use_j)
            topk_ids = np.zeros((P, L, K), dtype=np.int32)
            topk_p = np.zeros((P, L, K), dtype=np.float16)
            ent = np.zeros((P, L), dtype=np.float32)
            gold_rank = np.zeros((P, L), dtype=np.int32)
            num_rank = np.full((P, L), -1, dtype=np.int32)
            for li, layer in enumerate(probes):
                logits = ll[layer]  # [P, vocab] float32 cpu
                probs = torch.softmax(logits, dim=-1)
                tk = probs.topk(K, dim=-1)
                topk_ids[:, li] = tk.indices.numpy()
                topk_p[:, li] = tk.values.numpy().astype(np.float16)
                ent[:, li] = (-(probs * (probs + 1e-12).log()).sum(-1)).numpy()
                if gold_id >= 0:
                    gv = logits[:, gold_id].unsqueeze(1)
                    gold_rank[:, li] = (logits > gv).sum(-1).numpy().astype(np.int32)
                for pi, nid in enumerate(step_num_ids):
                    if nid >= 0:
                        num_rank[pi, li] = int((logits[pi] > logits[pi, nid]).sum())
            out.update({f"{tag}_topk_ids": topk_ids, f"{tag}_topk_p": topk_p,
                        f"{tag}_ent": ent, f"{tag}_gold_rank": gold_rank,
                        f"{tag}_num_rank": num_rank})
        out["positions"] = np.array(positions, dtype=np.int32)
        out["probe_layers"] = np.array(probes, dtype=np.int32)
        np.savez_compressed(f"{RESULTS}/{root}readouts/{r['uid']}.npz", **out)
        rman["done"].append(r["uid"])
        n_done += 1
        if n_done % 25 == 0:
            write_json(rman_path, rman)
            results_vol.commit()
            log(f"extract progress: {n_done}/{len(todo)}, "
                f"{time.time() - t_start:.0f}s elapsed")

    write_json(rman_path, rman)
    results_vol.commit()
    gpu_s = time.time() - t_start
    add_cost(gpu_s * A100_RATE, f"extract:{slug}:{n_done}traces")
    remaining = len(todo) - n_done
    return {"complete": remaining <= 0, "n_done": n_done, "remaining": remaining,
            "gpu_s": gpu_s, "probe_layers": probes}


@app.function(timeout=T_EXTRACT, **gpu_kw)
def extract_readouts(pid_filter: list = None, probe_stride: int = 2,
                     slug: str = PRIMARY, root_override: str = None):
    return _extract_readouts_impl(pid_filter, probe_stride, slug, root_override)


@app.function(timeout=T_EXTRACT, **gpu80_kw)
def extract_readouts_80(pid_filter: list = None, probe_stride: int = 2,
                        slug: str = PRIMARY, root_override: str = None):
    return _extract_readouts_impl(pid_filter, probe_stride, slug, root_override)


def _extract_residuals_impl(pid_filter=None, slug=PRIMARY, root_override=None):
    """Raw residual vectors at 25/50/75% depth for the probe ceiling baseline.

    Writes residuals/{uid}.npz ([P, 3, d_model] f16). Resumable via manifest.
    """
    import numpy as np
    import pandas as pd
    import jlens
    from jlens import ActivationRecorder

    t_start = time.time()
    deadline = t_start + T_EXTRACT - 900
    root = root_override if root_override is not None else MODELS[slug]["root"]

    results_vol.reload()
    tman = read_json(f"{RESULTS}/{root}traces/manifest.json", {"shards": []})
    sman_path = f"{RESULTS}/{root}residuals/manifest.json"
    sman = read_json(sman_path, {"done": []})
    done = set(sman["done"])

    dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(tman.get("shards", [])))
           if os.path.exists(f"{RESULTS}/{s}")]
    if not dfs:
        return {"complete": True, "n_done": 0, "gpu_s": 0.0}
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"]]
    if pid_filter is not None:
        df = df[df["problem_id"].isin(set(pid_filter))]
    todo = df[~df["uid"].isin(done)]
    if todo.empty:
        return {"complete": True, "n_done": 0, "gpu_s": time.time() - t_start}

    model_hf, tok = load_model_tok(slug)
    lmodel = jlens.from_hf(model_hf, tok)
    hf_vol.commit()
    n = lmodel.n_layers
    depth_layers = sorted({max(1, round(n * 0.25)), round(n * 0.5),
                           min(n - 2, round(n * 0.75))})
    os.makedirs(f"{RESULTS}/{root}residuals", exist_ok=True)

    n_done = 0
    for _, r in todo.iterrows():
        if time.time() > deadline:
            break
        rp = readout_positions(tok, r)
        if rp is None:
            sman["done"].append(r["uid"])
            continue
        full, step_pos, ans_pos = rp
        positions = step_pos + [ans_pos]
        try:
            input_ids = lmodel.encode(full, max_length=MAX_SEQ)
            with ActivationRecorder(lmodel.layers, at=depth_layers) as rec:
                lmodel.forward(input_ids)
                acts = {i: rec.activations[i].detach() for i in depth_layers}
            resid = np.stack(
                [acts[l][0][positions].float().cpu().numpy() for l in depth_layers],
                axis=1).astype(np.float16)  # [P, n_depths, d_model]
            np.savez_compressed(f"{RESULTS}/{root}residuals/{r['uid']}.npz",
                                resid=resid,
                                positions=np.array(positions, dtype=np.int32),
                                layers=np.array(depth_layers, dtype=np.int32))
        except Exception as e:  # noqa: BLE001
            log(f"residuals error on {r['uid']}: {e}")
        sman["done"].append(r["uid"])
        n_done += 1
        if n_done % 50 == 0:
            write_json(sman_path, sman)
            results_vol.commit()
            log(f"residuals progress: {n_done}/{len(todo)}")

    write_json(sman_path, sman)
    results_vol.commit()
    gpu_s = time.time() - t_start
    add_cost(gpu_s * A100_RATE, f"residuals:{slug}:{n_done}")
    remaining = len(todo) - n_done
    return {"complete": remaining <= 0, "n_done": n_done, "remaining": remaining,
            "gpu_s": gpu_s}


@app.function(timeout=T_EXTRACT, **gpu_kw)
def extract_residuals(pid_filter: list = None, slug: str = PRIMARY,
                      root_override: str = None):
    return _extract_residuals_impl(pid_filter, slug, root_override)


@app.function(timeout=T_EXTRACT, **gpu80_kw)
def extract_residuals_80(pid_filter: list = None, slug: str = PRIMARY,
                         root_override: str = None):
    return _extract_residuals_impl(pid_filter, slug, root_override)


# ----------------------------------------------------------------------------
# Stage C - features + analysis (CPU)
# ----------------------------------------------------------------------------


def battery(M, prefix):
    """Trajectory feature battery on dense step matrix M [n_steps, d]."""
    import numpy as np

    f = {}
    n = M.shape[0]
    eps = 1e-8
    norms = np.linalg.norm(M, axis=1) + eps
    U = M / norms[:, None]
    cos_next = (U[:-1] * U[1:]).sum(-1).clip(-1, 1)
    vel = 1.0 - cos_next
    f[f"{prefix}_vel_mean"] = float(vel.mean())
    f[f"{prefix}_vel_std"] = float(vel.std())
    D = M[1:] - M[:-1]
    if len(D) >= 2:
        dn = np.linalg.norm(D, axis=1) + eps
        Ud = D / dn[:, None]
        ang = np.arccos((Ud[:-1] * Ud[1:]).sum(-1).clip(-1, 1))
        f[f"{prefix}_curv_mean"] = float(ang.mean())
        f[f"{prefix}_curv_std"] = float(ang.std())
    else:
        f[f"{prefix}_curv_mean"] = 0.0
        f[f"{prefix}_curv_std"] = 0.0
    path = float(np.linalg.norm(D, axis=1).sum())
    net = float(np.linalg.norm(M[-1] - M[0]))
    f[f"{prefix}_path"] = path
    f[f"{prefix}_net"] = net
    f[f"{prefix}_tort"] = path / (net + eps)
    term = (U[:-1] @ U[-1]).clip(-1, 1)
    f[f"{prefix}_term_mean"] = float(term.mean())
    if len(term) >= 2:
        f[f"{prefix}_term_slope"] = float(np.polyfit(np.arange(len(term)), term, 1)[0])
    else:
        f[f"{prefix}_term_slope"] = 0.0
    return f


def densify(topk_ids, topk_p, layer_indices):
    """Dense per-step matrix from top-k sparse readouts, averaged over layer_indices.

    topk_ids/topk_p: [P, L, K]. Uses step rows only (caller slices).
    """
    import numpy as np

    ids = topk_ids[:, layer_indices, :]
    ps = topk_p[:, layer_indices, :].astype(np.float32)
    vocab_union = np.unique(ids)
    col = {int(t): i for i, t in enumerate(vocab_union)}
    M = np.zeros((ids.shape[0], len(vocab_union)), dtype=np.float32)
    for p in range(ids.shape[0]):
        for li in range(ids.shape[1]):
            for k in range(ids.shape[2]):
                M[p, col[int(ids[p, li, k])]] += ps[p, li, k]
    return M / max(1, len(layer_indices))


def concept_features(tag, npz, depth_idx, prefix, tok_decode, trace_text_lower):
    import numpy as np

    f = {}
    ids = npz[f"{tag}_topk_ids"][:-1, depth_idx, :]   # steps only
    ent = npz[f"{tag}_ent"][:-1, depth_idx]
    f[f"{prefix}_ent_mean"] = float(ent.mean())
    f[f"{prefix}_ent_std"] = float(ent.std())
    if len(ent) >= 2:
        f[f"{prefix}_ent_slope"] = float(np.polyfit(np.arange(len(ent)), ent, 1)[0])
    else:
        f[f"{prefix}_ent_slope"] = 0.0
    churn = []
    for i in range(len(ids) - 1):
        a, b = set(ids[i].tolist()), set(ids[i + 1].tolist())
        churn.append(1.0 - len(a & b) / max(1, len(a)))
    f[f"{prefix}_churn"] = float(np.mean(churn)) if churn else 0.0
    union = set()
    for row in ids:
        union.update(int(t) for t in row.tolist())
    silent = total = 0
    for t in union:
        s = tok_decode(t).strip().lower()
        if len(s) >= 3 and s.isalpha():
            total += 1
            if s not in trace_text_lower:
                silent += 1
    f[f"{prefix}_silent"] = silent / total if total else 0.0
    return f


@app.function(timeout=T_ANALYZE, **cpu_kw)
def analyze(tag: str = "final"):
    """Feature battery + classifiers + figures. Writes results under
    /results/ (tag=final) or /results/preliminary/ (tag=preliminary)."""
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    t_start = time.time()
    results_vol.reload()
    outdir = RESULTS if tag == "final" else f"{RESULTS}/{tag}"
    os.makedirs(f"{outdir}/figures", exist_ok=True)

    tman = read_json(f"{RESULTS}/traces/manifest.json", {"shards": []})
    dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(tman["shards"]))
           if os.path.exists(f"{RESULTS}/{s}")]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"]].set_index("uid")

    rman = read_json(f"{RESULTS}/readouts/manifest.json", {"done": []})
    uids = [u for u in rman["done"] if u in df.index
            and os.path.exists(f"{RESULTS}/readouts/{u}.npz")]
    if len(uids) < 40:
        raise RuntimeError(f"only {len(uids)} extracted traces; too few to analyze")

    tok = AutoTokenizer.from_pretrained(MODEL)
    decode_cache = {}

    def tok_decode(t):
        if t not in decode_cache:
            decode_cache[t] = tok.decode([t])
        return decode_cache[t]

    # --- surface embeddings ---
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                   device="cpu")
    hf_vol.commit()
    all_steps, spans = [], {}
    for u in uids:
        steps = [s["text"] for s in json.loads(df.loc[u, "steps_json"])]
        spans[u] = (len(all_steps), len(all_steps) + len(steps))
        all_steps.extend(steps)
    emb = st_model.encode(all_steps, batch_size=128, show_progress_bar=False,
                          convert_to_numpy=True)
    log(f"encoded {len(all_steps)} steps for {len(uids)} traces")

    # --- per-trace features ---
    feats = {c: {} for c in ("surface", "jspace", "logitlens", "randread", "length")}
    meta_rows = []
    emergence = {"j": {"correct": [], "incorrect": []},
                 "ll": {"correct": [], "incorrect": []}}
    example_traces = {}

    for u in uids:
        r = df.loc[u]
        npz = np.load(f"{RESULTS}/readouts/{u}.npz")
        probes = npz["probe_layers"]
        Lp = len(probes)
        n_steps_ext = npz["j_topk_ids"].shape[0] - 1  # last row = answer pos
        if n_steps_ext < MIN_STEPS:
            continue
        depth_ids = {"early": Lp // 5, "mid": Lp // 2, "late": (4 * Lp) // 5}
        trace_lower = r["gen_text"].lower()

        s0, s1 = spans[u]
        E = emb[s0:s0 + n_steps_ext]  # align with extracted step positions
        if len(E) < MIN_STEPS:
            continue
        fs = battery(E, "surf")
        feats["surface"][u] = fs

        row_len = {"len_nsteps": float(r["n_steps"]),
                   "len_ntok": float(r["n_gen_tokens"]),
                   "len_meanstep": float(np.mean([len(s["text"]) for s in
                                                  json.loads(r["steps_json"])]))}
        feats["length"][u] = row_len

        for cond, tag_ in (("jspace", "j"), ("logitlens", "ll"), ("randread", "rr")):
            f = {}
            for dname, di in depth_ids.items():
                M = densify(npz[f"{tag_}_topk_ids"][:-1], npz[f"{tag_}_topk_p"][:-1], [di])
                f.update(battery(M, f"{tag_}_{dname}"))
                f.update(concept_features(tag_, npz, di, f"{tag_}_{dname}",
                                          tok_decode, trace_lower))
            Mavg = densify(npz[f"{tag_}_topk_ids"][:-1], npz[f"{tag_}_topk_p"][:-1],
                           list(range(Lp)))
            f.update(battery(Mavg, f"{tag_}_avg"))
            feats[cond][u] = f

        # emergence depth: earliest probed layer where gold rank <= 10 at answer pos
        for tag_, key in (("j", "j"), ("ll", "ll")):
            granks = npz[f"{tag_}_gold_rank"][-1]  # [L]
            hit = np.where(granks <= 10)[0]
            depth = float(probes[hit[0]]) if len(hit) else float(probes[-1] + 2)
            emergence[key]["correct" if r["correct"] else "incorrect"].append(depth)

        meta_rows.append({"uid": u, "problem_id": r["problem_id"],
                          "dataset": r["dataset"], "correct": bool(r["correct"])})
        lab = "correct" if r["correct"] else "incorrect"
        if lab not in example_traces:
            example_traces[lab] = (u, E.copy(),
                                   [tok_decode(int(t)).strip() for t in
                                    npz["j_topk_ids"][:-1, depth_ids["mid"], 0]])

    meta = pd.DataFrame(meta_rows).set_index("uid")
    uids = list(meta.index)
    log(f"feature rows: {len(uids)} ({int(meta['correct'].sum())} correct)")

    def matrix(cond_list):
        cols = None
        rows = []
        for u in uids:
            d = {}
            for c in cond_list:
                d.update(feats[c][u])
            if cols is None:
                cols = sorted(d)
            rows.append([d.get(k, 0.0) for k in cols])
        return np.nan_to_num(np.array(rows, dtype=np.float64)), cols

    conditions = {
        "length_only": ["length"],
        "surface_only": ["surface"],
        "logitlens_only": ["logitlens"],
        "jspace_only": ["jspace"],
        "random_readout_only": ["randread"],
        "jspace_plus_surface": ["jspace", "surface"],
        "all_combined": ["jspace", "logitlens", "surface", "length"],
    }

    y = meta["correct"].values.astype(int)
    groups = meta["problem_id"].values
    is_gsm = (meta["dataset"] == "gsm8k").values
    have_math = bool((~is_gsm).sum() >= 40)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)

    def oof_auc(X, yv, gv, model_name, shuffle=False):
        yy = yv.copy()
        if shuffle:
            yy = rng.permutation(yy)
        if len(np.unique(gv)) < 5 or len(np.unique(yy)) < 2:
            return None, None
        preds = np.zeros(len(yy))
        for tr, te in GroupKFold(n_splits=5).split(X, yy, gv):
            if len(np.unique(yy[tr])) < 2:
                preds[te] = 0.5
                continue
            if model_name == "logreg":
                best, best_auc = None, -1
                for C in (0.01, 0.1, 1.0, 10.0):
                    m = make_pipeline(StandardScaler(),
                                      LogisticRegression(C=C, max_iter=2000))
                    inner_tr = tr[: int(0.8 * len(tr))]
                    inner_va = tr[int(0.8 * len(tr)):]
                    if len(np.unique(yy[inner_tr])) < 2 or len(np.unique(yy[inner_va])) < 2:
                        best = m
                        break
                    m.fit(X[inner_tr], yy[inner_tr])
                    a = roc_auc_score(yy[inner_va], m.predict_proba(X[inner_va])[:, 1])
                    if a > best_auc:
                        best, best_auc = make_pipeline(
                            StandardScaler(), LogisticRegression(C=C, max_iter=2000)), a
                best.fit(X[tr], yy[tr])
                preds[te] = best.predict_proba(X[te])[:, 1]
            else:
                m = HistGradientBoostingClassifier(random_state=SEED)
                m.fit(X[tr], yy[tr])
                preds[te] = m.predict_proba(X[te])[:, 1]
        return float(roc_auc_score(yy, preds)), preds

    def boot_ci(yv, preds, gv, n_boot=1000):
        gu = np.unique(gv)
        idx_by_g = {g: np.where(gv == g)[0] for g in gu}
        aucs = []
        for _ in range(n_boot):
            gs = rng.choice(gu, size=len(gu), replace=True)
            idx = np.concatenate([idx_by_g[g] for g in gs])
            if len(np.unique(yv[idx])) < 2:
                continue
            aucs.append(roc_auc_score(yv[idx], preds[idx]))
        if not aucs:
            return [None, None]
        return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]

    aucs = {}
    rocs = {}
    for cname, clist in conditions.items():
        X, _ = matrix(clist)
        aucs[cname] = {}
        for mname in ("logreg", "hgb"):
            a, preds = oof_auc(X, y, groups, mname)
            if a is None:
                aucs[cname][mname] = None
                continue
            aucs[cname][mname] = {"auc": round(a, 4),
                                  "ci95": boot_ci(y, preds, groups)}
            if mname == "logreg":
                fpr, tpr, _ = roc_curve(y, preds)
                rocs[cname] = (fpr.tolist(), tpr.tolist(), a)

    Xall, _ = matrix(conditions["all_combined"])
    a_shuf, _ = oof_auc(Xall, y, groups, "logreg", shuffle=True)

    # transfer: train GSM8K -> test MATH
    transfer = None
    if have_math:
        transfer = {}
        for cname, clist in conditions.items():
            X, _ = matrix(clist)
            if len(np.unique(y[is_gsm])) < 2 or len(np.unique(y[~is_gsm])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            m.fit(X[is_gsm], y[is_gsm])
            transfer[cname] = round(float(
                roc_auc_score(y[~is_gsm], m.predict_proba(X[~is_gsm])[:, 1])), 4)

    # emergence stats
    from scipy.stats import mannwhitneyu

    emergence_stats = {}
    for key in ("j", "ll"):
        c, i = emergence[key]["correct"], emergence[key]["incorrect"]
        st = {"n_correct": len(c), "n_incorrect": len(i),
              "mean_correct": round(float(np.mean(c)), 2) if c else None,
              "mean_incorrect": round(float(np.mean(i)), 2) if i else None}
        if len(c) >= 5 and len(i) >= 5:
            u_stat, p = mannwhitneyu(c, i, alternative="two-sided")
            st["mannwhitney_p"] = float(p)
        emergence_stats[key] = st

    # --- figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    for cname, (fpr, tpr, a) in rocs.items():
        plt.plot(fpr, tpr, label=f"{cname} (AUC={a:.3f})", lw=1.5)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"CoT correctness prediction ({tag}, n={len(uids)})")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"{outdir}/figures/roc_overlay.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    data, labels = [], []
    for key, name in (("j", "J-lens"), ("ll", "logit-lens")):
        for lab in ("correct", "incorrect"):
            if emergence[key][lab]:
                data.append(emergence[key][lab]); labels.append(f"{name}\n{lab}")
    if data:
        plt.violinplot(data, showmeans=True)
        plt.xticks(range(1, len(labels) + 1), labels, fontsize=8)
        plt.ylabel("answer-emergence layer")
        plt.tight_layout()
        plt.savefig(f"{outdir}/figures/emergence_depth.png", dpi=150)
    plt.close()

    if len(example_traces) == 2:
        from sklearn.decomposition import PCA
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (lab, (u, E, top1)) in zip(axes, sorted(example_traces.items())):
            P2 = PCA(n_components=2).fit_transform(E)
            ax.plot(P2[:, 0], P2[:, 1], "-o", ms=4)
            for i, (x0, y0) in enumerate(P2):
                c = top1[i] if i < len(top1) else ""
                ax.annotate(f"{i}:{c[:10]}", (x0, y0), fontsize=6)
            ax.set_title(f"{lab} trace {u}\n(surface PCA, labels = J-space top-1 @ mid)")
        plt.tight_layout()
        plt.savefig(f"{outdir}/figures/trajectories.png", dpi=150); plt.close()

    counts = count_traces()
    results = {
        "tag": tag, "generated_at": now_iso(), "n_analyzed": len(uids),
        "counts": counts, "aucs": aucs,
        "shuffled_label_auc": round(a_shuf, 4) if a_shuf else None,
        "transfer_gsm8k_to_math": transfer,
        "emergence": emergence_stats,
        "conditions_note": "AUC = out-of-fold GroupKFold(5) grouped by problem_id; "
                           "ci95 = 1000x grouped bootstrap",
    }
    write_json(f"{outdir}/results.json", results)

    def best(c):
        vals = [v["auc"] for v in (aucs.get(c) or {}).values() if v]
        return max(vals) if vals else float("nan")

    summary = f"""# J-space vs surface trajectories — {tag} results ({now_iso()})

**Question**: do J-lens concept trajectories predict CoT correctness better than
surface (sentence-embedding) trajectories?

**Data**: {counts['n_usable']} usable traces ({counts['n_correct']} correct /
{counts['n_incorrect']} incorrect) from {json.dumps(counts['by_dataset'])};
model {MODEL}, temp {TEMP}, K={K_SAMPLES}. Analyzed here: {len(uids)}.

## AUCs (best of logreg/HGB, grouped 5-fold OOF)

| condition | AUC |
|---|---|
""" + "\n".join(f"| {c} | {best(c):.3f} |" for c in conditions) + f"""

Full numbers incl. per-model CIs: `results.json`.

## Sanity checks
- shuffled-label AUC: {results['shuffled_label_auc']} (should be ~0.5)
- random-readout control: {best('random_readout_only'):.3f}
- trace-length-only baseline: {best('length_only'):.3f}

## Headline comparison
- surface_only: {best('surface_only'):.3f}
- jspace_only: {best('jspace_only'):.3f}
- logitlens_only: {best('logitlens_only'):.3f}
- jspace+surface over surface: {best('jspace_plus_surface') - best('surface_only'):+.3f}

## Answer-emergence depth (gold-answer rank<=10 at answer position)
{json.dumps(emergence_stats, indent=2)}

## Transfer (GSM8K -> MATH, logreg)
{json.dumps(transfer, indent=2) if transfer else 'MATH escalation did not run.'}

## Caveats
- Concepts are single-token readouts (top-{TOP_K_CONCEPTS}); multi-token concepts
  are only captured via their first token.
- One model ({MODEL}), one temperature ({TEMP}); results may not generalize.
- Correct/incorrect labels come from `####` numeric extraction; formatting
  failures are dropped (usable filter), which may bias trace selection.
- Steps are sentence-boundary segments; step counts differ across traces and
  length effects are only partially controlled by the length-only baseline.
"""
    with open(f"{outdir}/SUMMARY.md", "w") as f:
        f.write(summary)
    results_vol.commit()
    cpu_s = time.time() - t_start
    add_cost(cpu_s * CPU_RATE, f"analyze:{tag}")
    log(f"analyze({tag}) done in {cpu_s:.0f}s; n={len(uids)}")
    return {"tag": tag, "n_analyzed": len(uids), "aucs_best":
            {c: best(c) for c in conditions}, "cpu_s": cpu_s}


# ----------------------------------------------------------------------------
# Stage D - optional ablation
# ----------------------------------------------------------------------------


DIRECT_FEWSHOT_V2 = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: 10

Q: {q}
A:"""


@app.function(timeout=3600, **gpu_kw)
def ablate(n_problems: int = 100, v2: bool = False):
    """Zero top-k J-space concept directions at a mid layer during generation;
    measure accuracy drop for CoT vs direct answering.

    v2=True fixes the direct-answer condition: the prompt ends at "A:" and the
    prediction is the first number in the generation (v1's "####"-anchored
    grading could never match because "####" sat in the prompt)."""
    import numpy as np
    import pandas as pd
    import torch
    import jlens

    t_start = time.time()
    results_vol.reload()
    tman = read_json(f"{RESULTS}/traces/manifest.json", {"shards": []})
    dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(tman["shards"]))
           if os.path.exists(f"{RESULTS}/{s}")]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"] & (df["dataset"] == "gsm8k")]
    rman = read_json(f"{RESULTS}/readouts/manifest.json", {"done": []})

    pids = sorted(df["problem_id"].unique())[:n_problems]
    model_hf, tok = load_model_tok()
    lmodel = jlens.from_hf(model_hf, tok)
    lens, _ = load_lenses(lmodel)
    probes = probe_layers_of(lens, lmodel.n_layers)
    mid_layer = probes[len(probes) // 2]
    J = lens.jacobians[mid_layer]  # cuda float32
    W_U = lmodel._lm_head.weight  # [vocab, d]

    def build_Q(concept_ids):
        D = (J.T.float() @ W_U[concept_ids].T.float())  # [d, m]
        Q, _ = torch.linalg.qr(D)
        return Q  # [d, m] orthonormal

    hook_state = {"Q": None}

    def hook(module, args, output):
        # With KV cache, generated tokens arrive as seq_len==1 forwards (the
        # prefill covers only the prompt), so ablating decode steps == ablating
        # exactly the generated positions.
        if hook_state["Q"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] != 1:
            return output
        Q = hook_state["Q"].to(h.dtype)
        h = h - (h @ Q) @ Q.T
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = lmodel.layers[mid_layer].register_forward_hook(hook)

    def run(pid, prompt_tmpl, max_new, ablated, concept_ids, direct_grade=False):
        q = df[df["problem_id"] == pid].iloc[0]["question"]
        gold = df[df["problem_id"] == pid].iloc[0]["gold"]
        prompt = prompt_tmpl.format(q=q)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        hook_state["Q"] = build_Q(concept_ids) if (ablated and len(concept_ids)) else None
        torch.manual_seed(SEED)
        with torch.inference_mode():
            out = model_hf.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                    stop_strings=["\nQ:"], tokenizer=tok)
        gen = truncate_gen(tok.decode(out[0, enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True))
        if direct_grade:  # v2: first number in the generation is the answer
            m = re.search(NUM_RE, gen)
            pred = parse_num(m.group()) if m else None
        else:
            pred = extract_pred(gen)
        return bool(nums_equal(pred, gold))

    acc = {"cot": [], "cot_abl": [], "direct": [], "direct_abl": []}
    for pid in pids:
        if time.time() - t_start > 2700:
            break
        uid_rows = df[df["problem_id"] == pid]
        uid = uid_rows.iloc[0]["uid"]
        cids = []
        if uid in set(rman["done"]) and os.path.exists(f"{RESULTS}/readouts/{uid}.npz"):
            npz = np.load(f"{RESULTS}/readouts/{uid}.npz")
            li = list(npz["probe_layers"]).index(mid_layer)
            ids = npz["j_topk_ids"][:-1, li, :5]
            cids = torch.tensor(sorted(set(int(t) for t in ids.flatten())),
                                device="cuda")
        if len(cids) == 0:
            continue
        direct_tmpl = DIRECT_FEWSHOT_V2 if v2 else DIRECT_FEWSHOT
        try:
            acc["cot"].append(run(pid, FEWSHOT, MAX_NEW_TOKENS, False, cids))
            acc["cot_abl"].append(run(pid, FEWSHOT, MAX_NEW_TOKENS, True, cids))
            acc["direct"].append(run(pid, direct_tmpl, 12, False, cids,
                                     direct_grade=v2))
            acc["direct_abl"].append(run(pid, direct_tmpl, 12, True, cids,
                                         direct_grade=v2))
        except Exception as e:  # noqa: BLE001
            log(f"ablate error on {pid}: {e}")
    handle.remove()

    out = {c: {"n": len(v), "acc": round(float(np.mean(v)), 4) if v else None}
           for c, v in acc.items()}
    write_json(f"{RESULTS}/ablation_v2.json" if v2 else f"{RESULTS}/ablation.json", out)
    results_vol.commit()
    gpu_s = time.time() - t_start
    add_cost(gpu_s * A100_RATE, "ablate")
    return {"ablation": out, "gpu_s": gpu_s}


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------


def write_failure(msg, tb=""):
    try:
        results_vol.reload()
    except Exception:
        pass
    with open(f"{RESULTS}/FAILURE.md", "w") as f:
        f.write(f"""# FAILURE — {now_iso()}

## Diagnosis
{msg}

## Traceback
```
{tb}
```

## Repro
- `modal run jspace_pipeline.py::smoke_entry`
- `modal run --detach jspace_pipeline.py::orchestrate`
- inspect volume: `modal volume get {VOLUME} / ./jspace_results`
""")
    results_vol.commit()


@app.function(timeout=86400, image=image, volumes=VOLS,
              cpu=2, memory=8192,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def orchestrate():
    import subprocess
    import sys
    import traceback

    t_start = time.time()
    try:
        results_vol.reload()
        os.makedirs(RESULTS, exist_ok=True)
        write_json(f"{RESULTS}/config.json", {
            "model": MODEL, "lens_repo": LENS_REPO, "lens_revision": LENS_REVISION,
            "lens_file": LENS_FILE, "jlens_commit": JLENS_COMMIT,
            "n_problems": N_PROBLEMS, "k_samples": K_SAMPLES, "temp": TEMP,
            "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
            "min_incorrect": MIN_INCORRECT, "top_k_concepts": TOP_K_CONCEPTS,
            "seed": SEED, "budget_usd": BUDGET_USD, "gen_batch": GEN_BATCH,
            "started": now_iso(),
        })
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True).stdout
        with open(f"{RESULTS}/pip_freeze.txt", "w") as f:
            f.write(freeze)
        results_vol.commit()

        for d in [
            "Plain 2-shot text prompt (GSM8K train exemplars) instead of chat template: "
            "lens was fitted on raw text; avoids Qwen thinking-mode variance. No cost impact.",
            "Lens artifact: neuronpedia/jacobian-lens rev qwen-n1000 (n=1000 fit), "
            "per walkthrough.ipynb.",
            "Random-projection readout control = norm-matched random J per layer "
            "(3rd apply pass in extraction, ~+50% extraction time, ~$1).",
            "Terminal alignment computed against last step vector for all "
            "representations (answer-position vector reserved for emergence analysis).",
            "Reusing existing 'hf-cache' volume for model weights.",
        ]:
            append_decision(d)

        smoke_report = read_json(f"{RESULTS}/smoke.json")
        if not smoke_report or not smoke_report.get("pass"):
            raise RuntimeError("smoke.json missing or not passed; refusing to run")

        # helper loops -------------------------------------------------------
        def gen_all(dataset, pids, k=K_SAMPLES):
            total_gpu_s = 0.0
            for attempt in range(8):
                r = generate_traces.remote(dataset, pids, k)
                total_gpu_s += r["gpu_s"]
                log(f"generate({dataset}) attempt {attempt}: {r['n_done']} done, "
                    f"complete={r['complete']}")
                if r["complete"]:
                    r["gpu_s"] = total_gpu_s
                    return r
            raise RuntimeError(f"generation for {dataset} did not complete in 8 calls")

        def extract_all(pid_filter=None, stride=2):
            total_gpu_s = 0.0
            for attempt in range(8):
                r = extract_readouts.remote(pid_filter, stride)
                total_gpu_s += r["gpu_s"]
                log(f"extract attempt {attempt}: {r['n_done']} done, "
                    f"complete={r['complete']}")
                if r["complete"]:
                    r["gpu_s"] = total_gpu_s
                    return r
            raise RuntimeError("extraction did not complete in 8 calls")

        def check_budget(next_cost, label):
            spent = get_spent()
            if spent + next_cost > BUDGET_USD:
                raise RuntimeError(
                    f"budget stop before {label}: spent=${spent:.2f} + "
                    f"projected ${next_cost:.2f} > ${BUDGET_USD}")
            return spent

        # problem ids ---------------------------------------------------------
        # (dataset loading happens in GPU containers; here we mirror the id logic)
        n_gsm_total = 1319  # openai/gsm8k test size
        ids = sample_problem_ids(n_gsm_total, N_PROBLEMS)
        pids = [f"gsm8k-{i}" for i in ids]
        slice_pids, rest_pids = pids[:N_SLICE], pids[N_SLICE:]

        # ---- vertical slice ----
        log("=== vertical slice: 50 problems ===")
        r = gen_all("gsm8k", slice_pids)
        slice_gen_s = r["gpu_s"]
        r_ext = extract_all(pid_filter=slice_pids)
        slice_ext_s = r_ext["gpu_s"]
        try:
            analyze.remote("preliminary")
        except Exception as e:  # noqa: BLE001
            append_decision(f"preliminary analyze failed ({e}); continuing to full run.")

        # ---- budget projection & shrink ----
        per_prob_gpu_s = (slice_gen_s + slice_ext_s) / max(1, N_SLICE)
        n_target = N_PROBLEMS
        k_target = K_SAMPLES
        stride = 2
        spent = get_spent()

        def projected(n):
            rest = max(0, n - N_SLICE)
            return rest * per_prob_gpu_s * A100_RATE + 2.0  # +analysis/orchestrator

        while spent + projected(n_target) > BUDGET_USD and n_target > 100:
            n_target -= 50
        if spent + projected(n_target) > BUDGET_USD and k_target > 2:
            k_target = 2
        if spent + projected(n_target) > BUDGET_USD:
            stride = 4
        if (n_target, k_target, stride) != (N_PROBLEMS, K_SAMPLES, 2):
            append_decision(
                f"Budget shrink: N={n_target} K={k_target} probe_stride={stride} "
                f"(spent=${spent:.2f}, per-problem GPU {per_prob_gpu_s:.1f}s).")
        rest_pids = pids[N_SLICE:n_target]

        # ---- full GSM8K ----
        log(f"=== full run: {len(rest_pids)} more problems ===")
        check_budget(projected(n_target), "full generation")
        r = gen_all("gsm8k", rest_pids, k=k_target)
        counts = r["counts"]

        # ---- escalation ----
        if counts["n_incorrect"] < MIN_INCORRECT:
            append_decision(
                f"Escalating to MATH: only {counts['n_incorrect']} incorrect "
                f"(< {MIN_INCORRECT}).")
            math_cost = MATH_N_PROBLEMS * per_prob_gpu_s * A100_RATE
            check_budget(math_cost, "MATH escalation")
            r = gen_all("math", None, k=k_target)  # pids sampled in-container (seeded)
            log(f"math escalation: {r['counts']}")

        # ---- extraction (all) ----
        check_budget(counts["n_traces"] * 1.5 * A100_RATE, "extraction")
        extract_all(pid_filter=None, stride=stride)

        # ---- final analysis ----
        final = analyze.remote("final")
        log(f"final analysis: {json.dumps(final, default=str)[:500]}")

        # ---- Stage D ----
        spent = get_spent()
        if spent <= 0.7 * BUDGET_USD:
            try:
                abl = ablate.remote(100)
                append_decision(f"Stage D ablation ran: {json.dumps(abl['ablation'])}")
            except Exception as e:  # noqa: BLE001
                append_decision(f"Stage D skipped after error: {e}")
        else:
            append_decision(f"Stage D skipped: spent ${spent:.2f} > 70% of budget.")

        # ---- handoff ----
        spent = get_spent()
        results_vol.reload()
        with open(f"{RESULTS}/HANDOFF.md", "w") as f:
            f.write(f"""# HANDOFF — pipeline complete {now_iso()}

- Total estimated spend: ${spent:.2f} of ${BUDGET_USD} budget.
- Wall time: {(time.time() - t_start) / 3600:.1f} h (orchestrator).
- Fetch results: `modal volume get {VOLUME} / ./jspace_results`
- Key files: results.json, SUMMARY.md, figures/, preliminary/, DECISIONS.md,
  ablation.json (if Stage D ran), budget.json.
""")
        with open(f"{RESULTS}/DONE", "w") as f:
            f.write(now_iso())
        results_vol.commit()
        add_cost((time.time() - t_start) * 0.00003, "orchestrator-cpu")
        log("PIPELINE DONE")
        return {"status": "done", "spent_usd": spent}

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FATAL: {e}\n{tb}")
        write_failure(str(e), tb)
        return {"status": "failed", "error": str(e)}


# ----------------------------------------------------------------------------
# v2: confidence extraction (the standard correctness-prediction baseline)
# ----------------------------------------------------------------------------


def confidence_for_trace(model_hf, tok, row):
    """Teacher-forced logprob stats for one trace. Returns a feature dict."""
    import torch

    full = row["prompt_text"] + row["gen_text"]
    enc = tok(full, return_offsets_mapping=True, truncation=True,
              max_length=MAX_SEQ, return_tensors="pt")
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"][0].tolist()]
    ids = enc["input_ids"].to("cuda")
    S = ids.shape[1]
    plen = len(row["prompt_text"])
    p_end = char_to_token(offsets, plen)
    if p_end is None or p_end >= S - 2:
        return None
    with torch.inference_mode():
        logits = model_hf(input_ids=ids).logits[0].float()  # [S, V]
    lse = torch.logsumexp(logits, dim=-1)                   # [S]
    tgt = ids[0, 1:]
    lp = (logits[torch.arange(S - 1, device=ids.device), tgt]
          - lse[:-1]).cpu()                                  # lp[i] = logP(tok i+1)

    def span_lp(c0, c1):
        idxs = [i for i in range(p_end + 1, S)
                if offsets[i][0] >= c0 and offsets[i][0] < c1]
        vals = [float(lp[i - 1]) for i in idxs if i - 1 >= 0]
        return (sum(vals) / len(vals)) if vals else None

    gen_vals = [float(lp[i - 1]) for i in range(p_end + 1, S)]
    if not gen_vals:
        return None
    import statistics

    steps = json.loads(row["steps_json"])
    step_means = []
    for st in steps:
        m = span_lp(plen + st["start"], plen + st["end"])
        if m is not None:
            step_means.append(m)
    amatch = re.search(r"####\s*[^\n#]+", row["gen_text"])
    ans_lp = span_lp(plen + amatch.start(), plen + amatch.end()) if amatch else None
    mean_lp = sum(gen_vals) / len(gen_vals)
    return {
        "conf_mean_lp": mean_lp,
        "conf_std_lp": statistics.pstdev(gen_vals),
        "conf_min_step_lp": min(step_means) if step_means else mean_lp,
        "conf_ans_lp": ans_lp if ans_lp is not None else mean_lp,
        "conf_n_tok": float(len(gen_vals)),
    }


def _extract_confidence_impl(pid_filter=None, slug=PRIMARY, root_override=None):
    """Per-trace teacher-forced logprob features -> {root}confidence/{uid}.json."""
    import pandas as pd

    t_start = time.time()
    deadline = t_start + T_EXTRACT - 900
    root = root_override if root_override is not None else MODELS[slug]["root"]

    results_vol.reload()
    tman = read_json(f"{RESULTS}/{root}traces/manifest.json", {"shards": []})
    cman_path = f"{RESULTS}/{root}confidence/manifest.json"
    cman = read_json(cman_path, {"done": []})
    done = set(cman["done"])

    dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(tman.get("shards", [])))
           if os.path.exists(f"{RESULTS}/{s}")]
    if not dfs:
        return {"complete": True, "n_done": 0, "gpu_s": 0.0}
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"]]
    if pid_filter is not None:
        df = df[df["problem_id"].isin(set(pid_filter))]
    todo = df[~df["uid"].isin(done)]
    if todo.empty:
        return {"complete": True, "n_done": 0, "gpu_s": time.time() - t_start}

    model_hf, tok = load_model_tok(slug)
    hf_vol.commit()
    os.makedirs(f"{RESULTS}/{root}confidence", exist_ok=True)

    n_done = 0
    for _, r in todo.iterrows():
        if time.time() > deadline:
            break
        try:
            feats = confidence_for_trace(model_hf, tok, r)
        except Exception as e:  # noqa: BLE001
            log(f"confidence error on {r['uid']}: {e}")
            feats = None
        if feats is not None:
            write_json(f"{RESULTS}/{root}confidence/{r['uid']}.json", feats)
        cman["done"].append(r["uid"])  # failed traces marked done => skipped forever
        n_done += 1
        if n_done % 50 == 0:
            write_json(cman_path, cman)
            results_vol.commit()
            log(f"confidence progress: {n_done}/{len(todo)}")

    write_json(cman_path, cman)
    results_vol.commit()
    gpu_s = time.time() - t_start
    add_cost(gpu_s * A100_RATE, f"confidence:{slug}:{n_done}")
    remaining = len(todo) - n_done
    return {"complete": remaining <= 0, "n_done": n_done, "remaining": remaining,
            "gpu_s": gpu_s}


@app.function(timeout=T_EXTRACT, **gpu_kw)
def extract_confidence(pid_filter: list = None, slug: str = PRIMARY,
                       root_override: str = None):
    return _extract_confidence_impl(pid_filter, slug, root_override)


@app.function(timeout=T_EXTRACT, **gpu80_kw)
def extract_confidence_80(pid_filter: list = None, slug: str = PRIMARY,
                          root_override: str = None):
    return _extract_confidence_impl(pid_filter, slug, root_override)


# ----------------------------------------------------------------------------
# v2: analysis with confidence conditions + paired bootstrap deltas + transfer
# ----------------------------------------------------------------------------


@app.function(timeout=T_ANALYZE, **cpu_kw)
def analyze_v2(slug: str = PRIMARY):
    """Feature battery + classifiers incl. confidence baseline, paired AUC
    deltas, and GSM8K->MATH transfer. Writes /results/v2/{slug}/."""
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    t_start = time.time()
    results_vol.reload()
    root = MODELS[slug]["root"]
    outdir = f"{RESULTS}/v2/{slug}"
    os.makedirs(f"{outdir}/figures", exist_ok=True)

    tman = read_json(f"{RESULTS}/{root}traces/manifest.json", {"shards": []})
    dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(tman["shards"]))
           if os.path.exists(f"{RESULTS}/{s}")]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"]].set_index("uid")

    rman = read_json(f"{RESULTS}/{root}readouts/manifest.json", {"done": []})
    uids = [u for u in rman["done"] if u in df.index
            and os.path.exists(f"{RESULTS}/{root}readouts/{u}.npz")
            and os.path.exists(f"{RESULTS}/{root}confidence/{u}.json")]
    if len(uids) < 40:
        raise RuntimeError(f"only {len(uids)} traces with readouts+confidence")

    tok = AutoTokenizer.from_pretrained(MODELS[slug]["hf"])
    decode_cache = {}

    def tok_decode(t):
        if t not in decode_cache:
            decode_cache[t] = tok.decode([t])
        return decode_cache[t]

    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                   device="cpu")
    hf_vol.commit()
    all_steps, spans = [], {}
    for u in uids:
        steps = [s["text"] for s in json.loads(df.loc[u, "steps_json"])]
        spans[u] = (len(all_steps), len(all_steps) + len(steps))
        all_steps.extend(steps)
    emb = st_model.encode(all_steps, batch_size=128, show_progress_bar=False,
                          convert_to_numpy=True)
    log(f"encoded {len(all_steps)} steps for {len(uids)} traces [{slug}]")

    feats = {c: {} for c in ("surface", "jspace", "logitlens", "randread",
                             "length", "confidence")}
    meta_rows = []
    emergence = {"j": {"correct": [], "incorrect": []},
                 "ll": {"correct": [], "incorrect": []}}

    for u in uids:
        r = df.loc[u]
        npz = np.load(f"{RESULTS}/{root}readouts/{u}.npz")
        probes = npz["probe_layers"]
        Lp = len(probes)
        n_steps_ext = npz["j_topk_ids"].shape[0] - 1
        if n_steps_ext < MIN_STEPS:
            continue
        conf = read_json(f"{RESULTS}/{root}confidence/{u}.json")
        if not conf:
            continue
        depth_ids = {"early": Lp // 5, "mid": Lp // 2, "late": (4 * Lp) // 5}
        trace_lower = r["gen_text"].lower()
        s0, _ = spans[u]
        E = emb[s0:s0 + n_steps_ext]
        if len(E) < MIN_STEPS:
            continue

        feats["surface"][u] = battery(E, "surf")
        feats["confidence"][u] = conf
        feats["length"][u] = {
            "len_nsteps": float(r["n_steps"]),
            "len_ntok": float(r["n_gen_tokens"]),
            "len_meanstep": float(np.mean([len(s["text"]) for s in
                                           json.loads(r["steps_json"])]))}
        for cond, tag_ in (("jspace", "j"), ("logitlens", "ll"), ("randread", "rr")):
            f = {}
            for dname, di in depth_ids.items():
                M = densify(npz[f"{tag_}_topk_ids"][:-1], npz[f"{tag_}_topk_p"][:-1], [di])
                f.update(battery(M, f"{tag_}_{dname}"))
                f.update(concept_features(tag_, npz, di, f"{tag_}_{dname}",
                                          tok_decode, trace_lower))
            Mavg = densify(npz[f"{tag_}_topk_ids"][:-1], npz[f"{tag_}_topk_p"][:-1],
                           list(range(Lp)))
            f.update(battery(Mavg, f"{tag_}_avg"))
            feats[cond][u] = f

        for tag_, key in (("j", "j"), ("ll", "ll")):
            granks = npz[f"{tag_}_gold_rank"][-1]
            hit = np.where(granks <= 10)[0]
            depth = float(probes[hit[0]]) if len(hit) else float(probes[-1] + 2)
            emergence[key]["correct" if r["correct"] else "incorrect"].append(depth)

        meta_rows.append({"uid": u, "problem_id": r["problem_id"],
                          "dataset": r["dataset"], "correct": bool(r["correct"])})

    meta = pd.DataFrame(meta_rows).set_index("uid")
    uids = list(meta.index)
    log(f"feature rows [{slug}]: {len(uids)} ({int(meta['correct'].sum())} correct)")

    def matrix(cond_list):
        cols = None
        rows = []
        for u in uids:
            d = {}
            for c in cond_list:
                d.update(feats[c][u])
            if cols is None:
                cols = sorted(d)
            rows.append([d.get(k, 0.0) for k in cols])
        return np.nan_to_num(np.array(rows, dtype=np.float64)), cols

    conditions = {
        "length_only": ["length"],
        "confidence_only": ["confidence"],
        "surface_only": ["surface"],
        "logitlens_only": ["logitlens"],
        "jspace_only": ["jspace"],
        "random_readout_only": ["randread"],
        "jspace_plus_surface": ["jspace", "surface"],
        "jspace_plus_confidence": ["jspace", "confidence"],
        "confidence_plus_length": ["confidence", "length"],
        "all_combined": ["jspace", "logitlens", "surface", "length", "confidence"],
    }

    y = meta["correct"].values.astype(int)
    groups = meta["problem_id"].values
    is_gsm = (meta["dataset"] == "gsm8k").values
    have_math = bool((~is_gsm).sum() >= 40)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)

    def oof_preds(X, yv, gv, model_name):
        preds = np.zeros(len(yv))
        for tr, te in GroupKFold(n_splits=5).split(X, yv, gv):
            if len(np.unique(yv[tr])) < 2:
                preds[te] = 0.5
                continue
            if model_name == "logreg":
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=2000))
            else:
                m = HistGradientBoostingClassifier(random_state=SEED)
            m.fit(X[tr], yv[tr])
            preds[te] = m.predict_proba(X[te])[:, 1]
        return preds

    PAIRS = {
        "jspace_vs_logitlens": ("jspace_only", "logitlens_only"),
        "jspace_vs_confidence": ("jspace_only", "confidence_only"),
        "jspace_vs_surface": ("jspace_only", "surface_only"),
        "jspace_plus_conf_vs_conf": ("jspace_plus_confidence", "confidence_only"),
        "all_vs_confidence": ("all_combined", "confidence_only"),
    }

    def eval_block(sel):
        """AUCs + paired deltas on a row subset (avoids cross-dataset
        difficulty confound when sel restricts to one dataset)."""
        ys, gs = y[sel], groups[sel]
        gu = np.unique(gs)
        idx_by_g = {g: np.where(gs == g)[0] for g in gu}
        boot_groups = [rng.choice(gu, size=len(gu), replace=True)
                       for _ in range(1000)]

        def boot_aucs(preds):
            out = []
            for gsamp in boot_groups:
                idx = np.concatenate([idx_by_g[g] for g in gsamp])
                if len(np.unique(ys[idx])) < 2:
                    out.append(np.nan)
                    continue
                out.append(roc_auc_score(ys[idx], preds[idx]))
            return np.array(out)

        aucs_b, preds_store, rocs_b = {}, {}, {}
        for cname, clist in conditions.items():
            X, _ = matrix(clist)
            X = X[sel]
            aucs_b[cname] = {}
            for mname in ("logreg", "hgb"):
                p = oof_preds(X, ys, gs, mname)
                ba = boot_aucs(p)
                ba_ok = ba[~np.isnan(ba)]
                aucs_b[cname][mname] = {
                    "auc": round(float(roc_auc_score(ys, p)), 4),
                    "ci95": [round(float(np.percentile(ba_ok, 2.5)), 4),
                             round(float(np.percentile(ba_ok, 97.5)), 4)],
                }
                if mname == "logreg":
                    preds_store[cname] = p
                    fpr, tpr, _ = roc_curve(ys, p)
                    rocs_b[cname] = (fpr.tolist(), tpr.tolist(),
                                     aucs_b[cname][mname]["auc"])

        def paired_delta(a, b):
            ba, bb = boot_aucs(preds_store[a]), boot_aucs(preds_store[b])
            d = ba - bb
            d = d[~np.isnan(d)]
            point = float(roc_auc_score(ys, preds_store[a])
                          - roc_auc_score(ys, preds_store[b]))
            p_lo = float((d <= 0).mean())
            p_hi = float((d >= 0).mean())
            return {"delta": round(point, 4),
                    "ci95": [round(float(np.percentile(d, 2.5)), 4),
                             round(float(np.percentile(d, 97.5)), 4)],
                    "p_two_sided": round(min(1.0, 2 * min(p_lo, p_hi)), 4)}

        paired_b = {k: paired_delta(a, b) for k, (a, b) in PAIRS.items()}
        return aucs_b, paired_b, rocs_b

    aucs, paired, rocs = eval_block(np.ones(len(y), dtype=bool))
    gsm_block = None
    if have_math and is_gsm.sum() >= 100:
        a_g, p_g, _ = eval_block(is_gsm)
        gsm_block = {"aucs": a_g, "paired_deltas_logreg": p_g,
                     "n": int(is_gsm.sum())}

    Xall, _ = matrix(conditions["all_combined"])
    y_shuf = rng.permutation(y)
    p_shuf = oof_preds(Xall, y_shuf, groups, "logreg")
    a_shuf = round(float(roc_auc_score(y_shuf, p_shuf)), 4)

    transfer = None
    if have_math:
        transfer = {}
        for cname, clist in conditions.items():
            X, _ = matrix(clist)
            if len(np.unique(y[is_gsm])) < 2 or len(np.unique(y[~is_gsm])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            m.fit(X[is_gsm], y[is_gsm])
            transfer[cname] = round(float(
                roc_auc_score(y[~is_gsm], m.predict_proba(X[~is_gsm])[:, 1])), 4)

    from scipy.stats import mannwhitneyu

    emergence_stats = {}
    for key in ("j", "ll"):
        c, i = emergence[key]["correct"], emergence[key]["incorrect"]
        st = {"n_correct": len(c), "n_incorrect": len(i),
              "mean_correct": round(float(np.mean(c)), 2) if c else None,
              "mean_incorrect": round(float(np.mean(i)), 2) if i else None}
        if len(c) >= 5 and len(i) >= 5:
            _, p = mannwhitneyu(c, i, alternative="two-sided")
            st["mannwhitney_p"] = float(p)
        emergence_stats[key] = st

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    for cname, (fpr, tpr, a) in rocs.items():
        plt.plot(fpr, tpr, label=f"{cname} (AUC={a:.3f})", lw=1.5)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"CoT correctness prediction — {slug} (n={len(uids)})")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"{outdir}/figures/roc_overlay.png", dpi=150); plt.close()

    counts = count_traces(root)
    results = {
        "slug": slug, "hf_model": MODELS[slug]["hf"], "generated_at": now_iso(),
        "n_analyzed": len(uids), "counts": counts, "aucs": aucs,
        "paired_deltas_logreg": paired,
        "gsm8k_only": gsm_block,
        "shuffled_label_auc": a_shuf,
        "transfer_gsm8k_to_math": transfer,
        "emergence": emergence_stats,
        "note": "AUC = OOF GroupKFold(5) by problem_id; CI/deltas = 1000x paired "
                "grouped bootstrap; logreg C=1 fixed in v2 for paired comparability",
    }
    write_json(f"{outdir}/results_v2.json", results)
    results_vol.commit()
    cpu_s = time.time() - t_start
    add_cost(cpu_s * CPU_RATE, f"analyze_v2:{slug}")
    log(f"analyze_v2({slug}) done in {cpu_s:.0f}s; n={len(uids)}")
    return results


# ----------------------------------------------------------------------------
# v2: smoke + orchestrator
# ----------------------------------------------------------------------------


@app.function(timeout=1800, **gpu_kw)
def smoke_v2():
    """Load each extra model + its lens, run one canary readout, a tiny
    generation, and the confidence path. Must pass before detaching v2."""
    import jlens

    report = {"pass": False, "models": {}}
    t_all = time.time()
    for slug in EXTRA_MODELS:
        t0 = time.time()
        entry = {}
        try:
            model_hf, tok = load_model_tok(slug)
            lmodel = jlens.from_hf(model_hf, tok)
            lens, _ = load_lenses(lmodel, slug)
            probes = probe_layers_of(lens, lmodel.n_layers)
            jl, _, _ = lens.apply(lmodel, CANARIES[0][0],
                                  layers=probes[len(probes) // 2:len(probes) // 2 + 1],
                                  positions=[-2, -1])
            top = [tok.decode([t]).strip().lower()
                   for t in list(jl.values())[0][0].topk(TOP_K_CONCEPTS).indices.tolist()]
            entry["canary_top5"] = top[:5]
            entry["canary_hit"] = any(e in t for t in top if t
                                      for e in CANARIES[0][1])
            problems = load_gsm8k()
            two = [problems[sample_problem_ids(len(problems), 2)[0]]]
            gens = gen_batch(model_hf, tok, [build_prompt(two[0])], seed_offset=7, k=2)
            rows = make_trace_rows(two[0], gens[0], tok)
            entry["gen_ok"] = True
            entry["n_usable"] = sum(r["usable"] for r in rows)
            usable = [r for r in rows if r["usable"]] or rows
            conf = confidence_for_trace(model_hf, tok, usable[0])
            entry["confidence_ok"] = conf is not None
            entry["conf_sample"] = conf
            entry["load_s"] = round(time.time() - t0, 1)
            del model_hf, lmodel, lens
            import torch
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            import traceback
            entry["error"] = f"{e}\n{traceback.format_exc()[-800:]}"
        report["models"][slug] = entry
    hf_vol.commit()
    report["pass"] = all(
        m.get("gen_ok") and m.get("confidence_ok") and "error" not in m
        for m in report["models"].values())
    smoke_cost = (time.time() - t_all) * A100_RATE
    add_cost(smoke_cost, "smoke_v2")
    os.makedirs(RESULTS, exist_ok=True)
    write_json(f"{RESULTS}/smoke_v2.json", report)
    results_vol.commit()
    log(f"SMOKE_V2 {'PASS' if report['pass'] else 'FAIL'}")
    return report


@app.function(timeout=86400, image=image, volumes=VOLS, cpu=2, memory=8192,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def orchestrate_v2():
    import traceback

    t_start = time.time()
    try:
        results_vol.reload()
        smoke_report = read_json(f"{RESULTS}/smoke_v2.json")
        if not smoke_report or not smoke_report.get("pass"):
            raise RuntimeError("smoke_v2.json missing or failed; refusing to run")

        append_decision(
            "v2 run start: confidence baseline, paired bootstrap deltas, forced "
            "MATH transfer, models qwen3.5-0.8b + qwen3-4b (lenses early-stopped "
            "at n=233/479 prompts per Hub config.yaml), fixed Stage D grading.")
        append_decision(
            "smoke_v2 canary note: qwen3.5-0.8b mid-layer readout shows surface "
            "echo ('boots/heel'), qwen3-4b raw top-k punctuation-heavy (Qwen "
            "quirk noted in jlens walkthrough, which masks to word-like tokens "
            "for display only). Proceeding: pipeline treats all models "
            "identically; cross-model readout quality is itself a result.")

        def gen_all(dataset, pids, slug, k=K_SAMPLES):
            for _ in range(8):
                r = generate_traces.remote(dataset, pids, k, slug)
                if r["complete"]:
                    return r
            raise RuntimeError(f"generation {slug}/{dataset} incomplete after 8 calls")

        def extract_all(slug):
            for _ in range(8):
                r = extract_readouts.remote(None, 2, slug)
                if r["complete"]:
                    return r
            raise RuntimeError(f"extraction {slug} incomplete after 8 calls")

        def conf_all(slug):
            for _ in range(8):
                r = extract_confidence.remote(None, slug)
                if r["complete"]:
                    return r
            raise RuntimeError(f"confidence {slug} incomplete after 8 calls")

        def guard(next_usd, label):
            spent = get_spent()
            if spent + next_usd > BUDGET_USD:
                append_decision(f"SKIP {label}: spent ${spent:.2f} + est "
                                f"${next_usd:.2f} > ${BUDGET_USD}")
                return False
            return True

        n_gsm_total = 1319
        gsm_pids = [f"gsm8k-{i}" for i in sample_problem_ids(n_gsm_total, N_PROBLEMS)]
        cross = {}

        # ---- primary model: MATH + confidence + reanalysis + fixed ablation ----
        log("=== v2 primary: forced MATH escalation ===")
        gen_all("math", None, PRIMARY)
        extract_all(PRIMARY)
        conf_all(PRIMARY)
        cross[PRIMARY] = analyze_v2.remote(PRIMARY)
        if guard(1.0, "ablate_v2"):
            try:
                abl = ablate.remote(100, True)
                append_decision(f"ablate_v2: {json.dumps(abl['ablation'])}")
            except Exception as e:  # noqa: BLE001
                append_decision(f"ablate_v2 failed: {e}")

        # ---- extra models ----
        for slug in EXTRA_MODELS:
            est = 2.5 if slug == "qwen3.5-0.8b" else 4.0
            if not guard(est, f"model {slug}"):
                continue
            try:
                log(f"=== v2 model: {slug} ===")
                gen_all("gsm8k", gsm_pids, slug)
                gen_all("math", None, slug)
                extract_all(slug)
                conf_all(slug)
                cross[slug] = analyze_v2.remote(slug)
            except Exception as e:  # noqa: BLE001
                append_decision(f"model {slug} FAILED, continuing: {e}")
                log(traceback.format_exc())

        # ---- cross-model summary ----
        results_vol.reload()
        lines = [f"# Cross-model summary (v2) — {now_iso()}", "",
                 "| model | n | jspace | logitlens | confidence | surface | "
                 "J-LL delta (p) | J-conf delta (p) | J+conf-conf delta (p) | "
                 "transfer J / conf |", "|---|---|---|---|---|---|---|---|---|"]
        for slug, res in cross.items():
            if not res:
                continue
            a = res["aucs"]
            pd_ = res["paired_deltas_logreg"]
            tr = res.get("transfer_gsm8k_to_math") or {}
            lines.append(
                f"| {slug} | {res['n_analyzed']} "
                f"| {a['jspace_only']['logreg']['auc']} "
                f"| {a['logitlens_only']['logreg']['auc']} "
                f"| {a['confidence_only']['logreg']['auc']} "
                f"| {a['surface_only']['logreg']['auc']} "
                f"| {pd_['jspace_vs_logitlens']['delta']} "
                f"({pd_['jspace_vs_logitlens']['p_two_sided']}) "
                f"| {pd_['jspace_vs_confidence']['delta']} "
                f"({pd_['jspace_vs_confidence']['p_two_sided']}) "
                f"| {pd_['jspace_plus_conf_vs_conf']['delta']} "
                f"({pd_['jspace_plus_conf_vs_conf']['p_two_sided']}) "
                f"| {tr.get('jspace_only', 'n/a')} / {tr.get('confidence_only', 'n/a')} |")
        lines += ["", "Full numbers: `v2/<model>/results_v2.json`. "
                  "Shuffled sanity + emergence stats per model in the same files.",
                  "", f"Total est. spend: ${get_spent():.2f} of ${BUDGET_USD}."]
        with open(f"{RESULTS}/SUMMARY_V2.md", "w") as f:
            f.write("\n".join(lines) + "\n")
        write_json(f"{RESULTS}/results_v2_cross.json",
                   {k: v for k, v in cross.items() if v})
        with open(f"{RESULTS}/DONE_V2", "w") as f:
            f.write(now_iso())
        results_vol.commit()
        add_cost((time.time() - t_start) * 0.00003, "orchestrator_v2-cpu")
        log("V2 PIPELINE DONE")
        return {"status": "done", "spent_usd": get_spent(),
                "models_completed": sorted(k for k, v in cross.items() if v)}

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FATAL v2: {e}\n{tb}")
        try:
            results_vol.reload()
        except Exception:
            pass
        with open(f"{RESULTS}/FAILURE_V2.md", "w") as f:
            f.write(f"# FAILURE (v2) — {now_iso()}\n\n{e}\n\n```\n{tb}\n```\n\n"
                    "Repro: modal run jspace_pipeline.py::smoke_v2_entry then "
                    "modal run --detach jspace_pipeline.py::orchestrate_v2\n")
        results_vol.commit()
        return {"status": "failed", "error": str(e)}


# ----------------------------------------------------------------------------
# v3: per-cell analysis (model x benchmark) with residual-probe ceiling
# ----------------------------------------------------------------------------


@app.function(timeout=T_ANALYZE, **cpu_kw)
def analyze_v3(slug: str, benchmark: str, root: str):
    """One (model, benchmark) cell: full condition battery + residual probe.
    Writes /results/v3/{slug}/{benchmark}/results_v3.json + features.npz."""
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    t_start = time.time()
    results_vol.reload()
    outdir = f"{RESULTS}/v3/{slug}/{benchmark}"
    os.makedirs(f"{outdir}/figures", exist_ok=True)

    tman = read_json(f"{RESULTS}/{root}traces/manifest.json", {"shards": []})
    dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(tman["shards"]))
           if os.path.exists(f"{RESULTS}/{s}")]
    if not dfs:
        raise RuntimeError(f"no trace shards under {root}")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"] & (df["dataset"] == benchmark)].set_index("uid")

    rman = read_json(f"{RESULTS}/{root}readouts/manifest.json", {"done": []})
    uids = [u for u in rman["done"] if u in df.index
            and os.path.exists(f"{RESULTS}/{root}readouts/{u}.npz")
            and os.path.exists(f"{RESULTS}/{root}confidence/{u}.json")]
    if len(uids) < 40:
        raise RuntimeError(f"cell {slug}/{benchmark}: only {len(uids)} traces")

    tok = AutoTokenizer.from_pretrained(MODELS[slug]["hf"])
    decode_cache = {}

    def tok_decode(t):
        if t not in decode_cache:
            decode_cache[t] = tok.decode([t])
        return decode_cache[t]

    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                   device="cpu")
    all_steps, spans = [], {}
    for u in uids:
        steps = [s["text"] for s in json.loads(df.loc[u, "steps_json"])]
        spans[u] = (len(all_steps), len(all_steps) + len(steps))
        all_steps.extend(steps)
    emb = st_model.encode(all_steps, batch_size=128, show_progress_bar=False,
                          convert_to_numpy=True)

    feats = {c: {} for c in ("surface", "jspace", "logitlens", "randread",
                             "length", "confidence")}
    resid_rows = {}
    meta_rows = []
    for u in uids:
        r = df.loc[u]
        npz = np.load(f"{RESULTS}/{root}readouts/{u}.npz")
        probes = npz["probe_layers"]
        Lp = len(probes)
        n_steps_ext = npz["j_topk_ids"].shape[0] - 1
        if n_steps_ext < MIN_STEPS:
            continue
        conf = read_json(f"{RESULTS}/{root}confidence/{u}.json")
        if not conf:
            continue
        depth_ids = {"early": Lp // 5, "mid": Lp // 2, "late": (4 * Lp) // 5}
        trace_lower = r["gen_text"].lower()
        s0, _ = spans[u]
        E = emb[s0:s0 + n_steps_ext]
        if len(E) < MIN_STEPS:
            continue
        rpath = f"{RESULTS}/{root}residuals/{u}.npz"
        if os.path.exists(rpath):
            rz = np.load(rpath)
            R = rz["resid"].astype(np.float32)  # [P, n_depths, d_model]
            mid = min(1, R.shape[1] - 1)
            resid_rows[u] = np.concatenate([R[:-1, mid].mean(0), R[-1, mid]])
        feats["surface"][u] = battery(E, "surf")
        feats["confidence"][u] = conf
        feats["length"][u] = {
            "len_nsteps": float(r["n_steps"]),
            "len_ntok": float(r["n_gen_tokens"]),
            "len_meanstep": float(np.mean([len(s["text"]) for s in
                                           json.loads(r["steps_json"])]))}
        for cond, tag_ in (("jspace", "j"), ("logitlens", "ll"), ("randread", "rr")):
            f = {}
            for dname, di in depth_ids.items():
                M = densify(npz[f"{tag_}_topk_ids"][:-1], npz[f"{tag_}_topk_p"][:-1], [di])
                f.update(battery(M, f"{tag_}_{dname}"))
                f.update(concept_features(tag_, npz, di, f"{tag_}_{dname}",
                                          tok_decode, trace_lower))
            Mavg = densify(npz[f"{tag_}_topk_ids"][:-1], npz[f"{tag_}_topk_p"][:-1],
                           list(range(Lp)))
            f.update(battery(Mavg, f"{tag_}_avg"))
            feats[cond][u] = f
        meta_rows.append({"uid": u, "problem_id": r["problem_id"],
                          "correct": bool(r["correct"])})

    meta = pd.DataFrame(meta_rows).set_index("uid")
    uids = list(meta.index)
    y = meta["correct"].values.astype(int)
    groups = meta["problem_id"].values
    if len(np.unique(y)) < 2 or min(int(y.sum()), int(len(y) - y.sum())) < 15:
        write_json(f"{outdir}/results_v3.json", {
            "slug": slug, "benchmark": benchmark, "n_analyzed": len(uids),
            "skipped": f"degenerate labels: {int(y.sum())}/{len(y)} correct"})
        results_vol.commit()
        return {"slug": slug, "benchmark": benchmark, "skipped": True,
                "n_correct": int(y.sum()), "n": int(len(y))}

    def matrix(cond_list):
        cols = None
        rows = []
        for u in uids:
            d = {}
            for c in cond_list:
                d.update(feats[c][u])
            if cols is None:
                cols = sorted(d)
            rows.append([d.get(k, 0.0) for k in cols])
        return np.nan_to_num(np.array(rows, dtype=np.float64)), cols

    conditions = {
        "length_only": ["length"],
        "confidence_only": ["confidence"],
        "surface_only": ["surface"],
        "logitlens_only": ["logitlens"],
        "jspace_only": ["jspace"],
        "random_readout_only": ["randread"],
        "jspace_plus_confidence": ["jspace", "confidence"],
        "all_combined": ["jspace", "logitlens", "surface", "length", "confidence"],
    }

    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    have_probe = len(resid_rows) >= 0.9 * len(uids) and len(resid_rows) >= 40

    def oof_preds(X, yv, gv, C=1.0):
        preds = np.zeros(len(yv))
        for tr, te in GroupKFold(n_splits=5).split(X, yv, gv):
            if len(np.unique(yv[tr])) < 2:
                preds[te] = 0.5
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=C, max_iter=2000))
            m.fit(X[tr], yv[tr])
            preds[te] = m.predict_proba(X[te])[:, 1]
        return preds

    gu = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in gu}
    boot_groups = [rng.choice(gu, size=len(gu), replace=True) for _ in range(1000)]

    def boot_aucs(preds):
        out = []
        for gs in boot_groups:
            idx = np.concatenate([idx_by_g[g] for g in gs])
            if len(np.unique(y[idx])) < 2:
                out.append(np.nan)
                continue
            out.append(roc_auc_score(y[idx], preds[idx]))
        return np.array(out)

    aucs, preds_store = {}, {}
    feat_dump = {}
    for cname, clist in conditions.items():
        X, cols = matrix(clist)
        feat_dump[cname] = (X, cols)
        p = oof_preds(X, y, groups)
        ba = boot_aucs(p)
        ba_ok = ba[~np.isnan(ba)]
        aucs[cname] = {"auc": round(float(roc_auc_score(y, p)), 4),
                       "ci95": [round(float(np.percentile(ba_ok, 2.5)), 4),
                                round(float(np.percentile(ba_ok, 97.5)), 4)]}
        preds_store[cname] = p

    if have_probe:
        proto = next(iter(resid_rows.values()))
        Xp = np.stack([resid_rows.get(u, np.zeros_like(proto)) for u in uids])
        p = oof_preds(Xp, y, groups, C=0.01)  # high-dim: strong regularization
        ba = boot_aucs(p)
        ba_ok = ba[~np.isnan(ba)]
        aucs["resid_probe_mid"] = {"auc": round(float(roc_auc_score(y, p)), 4),
                                   "ci95": [round(float(np.percentile(ba_ok, 2.5)), 4),
                                            round(float(np.percentile(ba_ok, 97.5)), 4)]}
        preds_store["resid_probe_mid"] = p

    def paired_delta(a, b):
        if a not in preds_store or b not in preds_store:
            return None
        ba, bb = boot_aucs(preds_store[a]), boot_aucs(preds_store[b])
        d = ba - bb
        d = d[~np.isnan(d)]
        point = float(roc_auc_score(y, preds_store[a])
                      - roc_auc_score(y, preds_store[b]))
        p_lo, p_hi = float((d <= 0).mean()), float((d >= 0).mean())
        return {"delta": round(point, 4),
                "ci95": [round(float(np.percentile(d, 2.5)), 4),
                         round(float(np.percentile(d, 97.5)), 4)],
                "p_two_sided": round(min(1.0, 2 * min(p_lo, p_hi)), 4)}

    paired = {
        "jspace_vs_confidence": paired_delta("jspace_only", "confidence_only"),
        "jspace_vs_logitlens": paired_delta("jspace_only", "logitlens_only"),
        "jspace_vs_surface": paired_delta("jspace_only", "surface_only"),
        "jspace_plus_conf_vs_conf": paired_delta("jspace_plus_confidence",
                                                 "confidence_only"),
        "probe_vs_jspace": paired_delta("resid_probe_mid", "jspace_only"),
        "probe_vs_confidence": paired_delta("resid_probe_mid", "confidence_only"),
    }

    Xall, _ = matrix(conditions["all_combined"])
    y_shuf = rng.permutation(y)
    a_shuf = round(float(roc_auc_score(y_shuf, oof_preds(Xall, y_shuf, groups))), 4)

    # persist feature matrices for cross-benchmark transfer
    np.savez_compressed(
        f"{outdir}/features.npz",
        y=y, groups=groups.astype(str),
        **{f"X_{c}": feat_dump[c][0] for c in conditions},
        **{f"cols_{c}": np.array(feat_dump[c][1]) for c in conditions})

    results = {
        "slug": slug, "benchmark": benchmark, "root": root,
        "generated_at": now_iso(), "n_analyzed": len(uids),
        "n_correct": int(y.sum()), "n_incorrect": int(len(y) - y.sum()),
        "accuracy": round(float(y.mean()), 4),
        "aucs": aucs, "paired_deltas": paired,
        "shuffled_label_auc": a_shuf, "have_probe": bool(have_probe),
    }
    write_json(f"{outdir}/results_v3.json", results)
    results_vol.commit()
    add_cost((time.time() - t_start) * CPU_RATE, f"analyze_v3:{slug}:{benchmark}")
    log(f"analyze_v3({slug},{benchmark}) n={len(uids)} "
        f"J={aucs['jspace_only']['auc']} conf={aucs['confidence_only']['auc']}")
    return results


@app.function(timeout=T_ANALYZE, **cpu_kw)
def transfer_v3(slug: str):
    """Train on the model's gsm8k cell, test on its other benchmark cells."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    results_vol.reload()
    base = f"{RESULTS}/v3/{slug}"
    src = f"{base}/gsm8k/features.npz"
    if not os.path.exists(src):
        return {"slug": slug, "skipped": "no gsm8k features"}
    S = np.load(src, allow_pickle=True)
    out = {}
    for bench in V3_BENCHMARKS:
        if bench == "gsm8k":
            continue
        tpath = f"{base}/{bench}/features.npz"
        if not os.path.exists(tpath):
            continue
        T = np.load(tpath, allow_pickle=True)
        out[bench] = {}
        for cond in ("confidence_only", "jspace_only", "logitlens_only",
                     "surface_only", "jspace_plus_confidence"):
            cols_s = list(S[f"cols_{cond}"])
            cols_t = list(T[f"cols_{cond}"])
            common = [c for c in cols_s if c in cols_t]
            if not common or len(np.unique(T["y"])) < 2:
                continue
            Xs = S[f"X_{cond}"][:, [cols_s.index(c) for c in common]]
            Xt = T[f"X_{cond}"][:, [cols_t.index(c) for c in common]]
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000))
            m.fit(Xs, S["y"])
            out[bench][cond] = round(float(
                roc_auc_score(T["y"], m.predict_proba(Xt)[:, 1])), 4)
    write_json(f"{base}/transfer_v3.json", {"slug": slug, "transfer": out})
    results_vol.commit()
    return {"slug": slug, "transfer": out}


@app.function(timeout=3600, **cpu_kw)
def regrade_traces(roots: list):
    """Re-grade stored mc/yesno traces in place with style-anchored parsing
    (fixes run-on answers like '#### yespassage: ...'). Numeric rows untouched."""
    import pandas as pd

    results_vol.reload()
    stats = {}
    for root in roots:
        man = read_json(f"{RESULTS}/{root}traces/manifest.json", None)
        if not man:
            continue
        changed_total = 0
        for shard in sorted(set(man.get("shards", []))):
            path = f"{RESULTS}/{shard}"
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            if "style" not in df.columns:
                continue
            mask = df["style"].isin(["mc", "yesno"])
            if not mask.any():
                continue
            changed = 0
            for i in df[mask].index:
                new_pred = extract_pred_styled(df.at[i, "gen_text"],
                                               df.at[i, "style"])
                new_correct = bool(answer_equal(new_pred, df.at[i, "gold"],
                                                df.at[i, "style"]))
                new_usable = bool(df.at[i, "n_steps"] >= MIN_STEPS
                                  and new_pred is not None)
                if (str(new_pred) != str(df.at[i, "pred"])
                        or new_correct != bool(df.at[i, "correct"])
                        or new_usable != bool(df.at[i, "usable"])):
                    changed += 1
                df.at[i, "pred"] = str(new_pred) if new_pred is not None else None
                df.at[i, "correct"] = new_correct
                df.at[i, "graded"] = new_pred is not None
                df.at[i, "usable"] = new_usable
            if changed:
                df.to_parquet(path, index=False)
                changed_total += changed
        results_vol.commit()
        stats[root] = changed_total
        log(f"regrade {root}: {changed_total} rows changed")
    append_decision(f"Regraded mc/yesno traces with style-anchored parsing: "
                    f"{json.dumps(stats)}")
    return stats


# ----------------------------------------------------------------------------
# v3: smoke + orchestrator
# ----------------------------------------------------------------------------


def _smoke_v3_impl(slugs):
    import torch
    import jlens

    report = {}
    for slug in slugs:
        t0 = time.time()
        entry = {}
        try:
            model_hf, tok = load_model_tok(slug)
            lmodel = jlens.from_hf(model_hf, tok)
            lens, _ = load_lenses(lmodel, slug)
            probes = probe_layers_of(lens, lmodel.n_layers)
            mid = probes[len(probes) // 2]
            jl, _, _ = lens.apply(lmodel, CANARIES[0][0], layers=[mid],
                                  positions=[-1])
            entry["canary_top5"] = [tok.decode([t]).strip()
                                    for t in jl[mid][0].topk(5).indices.tolist()]
            styles = {
                "numeric": {"pid": "s-num", "question":
                            "A farmer has 3 pens with 12 chickens each. He buys "
                            "8 more chickens. How many chickens does he have?",
                            "gold": 44.0, "style": "numeric"},
                "mc": {"pid": "s-mc", "question":
                       "Which is heavier?\nOptions:\n(A) a feather\n(B) a brick",
                       "gold": "b", "style": "mc"},
                "yesno": {"pid": "s-yn", "question":
                          "Is the sum of 2 and 2 equal to 4?",
                          "gold": "yes", "style": "yesno"},
            }
            graded = {}
            for sname, prob in styles.items():
                gens = gen_batch(model_hf, tok, [build_prompt(prob)],
                                 seed_offset=13, k=1)
                rows = make_trace_rows(prob, gens[0], tok)
                graded[sname] = {"pred": rows[0]["pred"],
                                 "correct": rows[0]["correct"],
                                 "n_steps": rows[0]["n_steps"]}
            entry["graded"] = graded
            entry["parse_ok"] = sum(1 for g in graded.values()
                                    if g["pred"] is not None)
            # confidence/positions path check on a REAL problem (toy smoke
            # questions can legitimately produce <3-step traces)
            problems = load_gsm8k()
            real = problems[sample_problem_ids(len(problems), 1)[0]]
            gens = gen_batch(model_hf, tok, [build_prompt(real)],
                             seed_offset=17, k=2)
            rrows = sorted(make_trace_rows(real, gens[0], tok),
                           key=lambda r: -r["n_steps"])
            best = rrows[0]
            entry["real_n_steps"] = best["n_steps"]
            conf = confidence_for_trace(model_hf, tok, best)
            entry["confidence_ok"] = conf is not None
            if best["n_steps"] >= MIN_STEPS:
                entry["positions_ok"] = readout_positions(tok, best) is not None
            else:  # pipeline drops such traces anyway; path untestable here
                entry["positions_ok"] = True
                entry["positions_note"] = "real trace <3 steps; skipped"
            entry["load_s"] = round(time.time() - t0, 1)
            entry["ok"] = bool(entry["parse_ok"] >= 2 and entry["confidence_ok"]
                               and entry["positions_ok"])
            del model_hf, lmodel, lens
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            import traceback
            entry["error"] = f"{e}\n{traceback.format_exc()[-600:]}"
            entry["ok"] = False
        report[slug] = entry
    hf_vol.commit()
    return report


@app.function(timeout=1800, **gpu_kw)
def smoke_v3_40():
    return _smoke_v3_impl(["olmo-3-7b", "qwen2.5-7b-it"])


@app.function(timeout=1800, **gpu80_kw)
def smoke_v3_80():
    return _smoke_v3_impl(["gpt-oss-20b"])


@app.function(image=image, volumes=VOLS, cpu=1, memory=2048, timeout=300)
def smoke_save(report: dict):
    results_vol.reload()
    write_json(f"{RESULTS}/smoke_v3.json", report)
    results_vol.commit()


@app.function(timeout=86400, image=image, volumes=VOLS, cpu=2, memory=8192,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def orchestrate_v3():
    import traceback

    try:
        results_vol.reload()
        smoke = read_json(f"{RESULTS}/smoke_v3.json")
        if not smoke or not smoke.get("pass_40"):
            raise RuntimeError("smoke_v3.json missing or 40GB models failed")
        v3_base = read_json(f"{RESULTS}/v3_budget.json", None)
        if v3_base is None:
            v3_base = {"baseline_usd": get_spent()}
            write_json(f"{RESULTS}/v3_budget.json", v3_base)
            results_vol.commit()

        def v3_spent():
            return get_spent() - v3_base["baseline_usd"]

        def guard(next_usd, label):
            if v3_spent() + next_usd > V3_BUDGET_USD:
                append_decision(f"V3 SKIP {label}: v3 spend ${v3_spent():.2f} + "
                                f"est ${next_usd:.2f} > ${V3_BUDGET_USD}")
                return False
            return True

        def fns_for(slug):
            if MODELS[slug].get("gpu") == "A100-80GB":
                return (generate_traces_80, extract_readouts_80,
                        extract_confidence_80, extract_residuals_80)
            return (generate_traces, extract_readouts,
                    extract_confidence, extract_residuals)

        def loop(fn, *args, label=""):
            for _ in range(8):
                r = fn.remote(*args)
                if r["complete"]:
                    return r
            raise RuntimeError(f"{label} incomplete after 8 calls")

        gsm_pids_v3 = [f"gsm8k-{i}" for i in sample_problem_ids(1319, 200)]
        append_decision(
            "v3 start: cross-family (olmo-3-7b, qwen2.5-7b-it, gpt-oss-20b) x "
            "cross-benchmark (gsm8k/bbhlogic/strategyqa/arc), K=2 N=200 per "
            "cell, residual-probe ceiling everywhere + backfill on v1/v2 "
            "gsm8k data. Cap $40 on top of prior spend.")

        cell_results = {}

        def run_cell(slug, bench, root, gen_needed=True):
            genf, extf, conff, residf = fns_for(slug)
            est = 1.5 if MODELS[slug].get("gpu") == "A100-80GB" else 1.0
            if not guard(est, f"cell {slug}/{bench}"):
                return None
            log(f"=== v3 cell: {slug} x {bench} (root={root}) ===")
            if gen_needed:
                pids = gsm_pids_v3 if bench == "gsm8k" else None
                loop(genf, bench, pids, 2, slug, root, label=f"gen {slug}/{bench}")
            loop(extf, None, 2, slug, root, label=f"extract {slug}/{bench}")
            loop(conff, None, slug, root, label=f"conf {slug}/{bench}")
            loop(residf, None, slug, root, label=f"resid {slug}/{bench}")
            return analyze_v3.remote(slug, bench, root)

        # 1. residual backfill + probe analysis on existing v1/v2 gsm8k data
        for slug in [PRIMARY] + EXTRA_MODELS:
            old_root = MODELS[slug]["root"]
            if guard(1.0, f"resid backfill {slug}"):
                try:
                    loop(extract_residuals, None, slug, old_root,
                         label=f"resid backfill {slug}")
                    cell_results[(slug, "gsm8k")] = analyze_v3.remote(
                        slug, "gsm8k", old_root)
                except Exception as e:  # noqa: BLE001
                    append_decision(f"backfill {slug} failed, continuing: {e}")

        # 2. existing models x new benchmarks
        for slug in [PRIMARY] + EXTRA_MODELS:
            for bench in ("bbhlogic", "strategyqa", "arc"):
                try:
                    r = run_cell(slug, bench, f"v3/{slug}/")
                    if r:
                        cell_results[(slug, bench)] = r
                except Exception as e:  # noqa: BLE001
                    append_decision(f"cell {slug}/{bench} failed: {e}")
                    log(traceback.format_exc()[-800:])

        # 3. new families (gpt-oss last: pricier, gated by its own smoke)
        for slug in V3_MODELS:
            if slug == "gpt-oss-20b" and not smoke.get("pass_80"):
                append_decision("gpt-oss-20b skipped: smoke_v3 80GB failed.")
                continue
            for bench in V3_BENCHMARKS:
                try:
                    r = run_cell(slug, bench, MODELS[slug]["root"])
                    if r:
                        cell_results[(slug, bench)] = r
                except Exception as e:  # noqa: BLE001
                    append_decision(f"cell {slug}/{bench} failed: {e}")
                    log(traceback.format_exc()[-800:])
            try:
                transfer_v3.remote(slug)
            except Exception as e:  # noqa: BLE001
                append_decision(f"transfer_v3 {slug} failed: {e}")

        # 4. cross summary
        results_vol.reload()
        lines = [f"# v3 cross-family x cross-benchmark matrix — {now_iso()}", "",
                 "| model | benchmark | n | acc | J | conf | LL | surf | probe |"
                 " J-conf (p) | J+c-c (p) |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        serializable = {}
        for (slug, bench), res in sorted(cell_results.items()):
            if not res or res.get("skipped"):
                lines.append(f"| {slug} | {bench} | - | - | degenerate/skipped |"
                             " | | | | | |")
                continue
            a = res["aucs"]

            def A(c):
                return a.get(c, {}).get("auc", "n/a")

            pd_ = res["paired_deltas"]
            jc = pd_.get("jspace_vs_confidence") or {}
            jpc = pd_.get("jspace_plus_conf_vs_conf") or {}
            lines.append(
                f"| {slug} | {bench} | {res['n_analyzed']} | {res['accuracy']} "
                f"| {A('jspace_only')} | {A('confidence_only')} "
                f"| {A('logitlens_only')} | {A('surface_only')} "
                f"| {A('resid_probe_mid')} "
                f"| {jc.get('delta', 'n/a')} ({jc.get('p_two_sided', '-')}) "
                f"| {jpc.get('delta', 'n/a')} ({jpc.get('p_two_sided', '-')}) |")
            serializable[f"{slug}|{bench}"] = res
        lines += ["", f"v3 spend: ${v3_spent():.2f} of ${V3_BUDGET_USD} "
                  f"(total ${get_spent():.2f}).",
                  "Transfer: `v3/<model>/transfer_v3.json`. "
                  "Cells: `v3/<model>/<benchmark>/results_v3.json`."]
        with open(f"{RESULTS}/SUMMARY_V3.md", "w") as f:
            f.write("\n".join(lines) + "\n")
        write_json(f"{RESULTS}/results_v3_cross.json", serializable)
        with open(f"{RESULTS}/DONE_V3", "w") as f:
            f.write(now_iso())
        results_vol.commit()
        log("V3 PIPELINE DONE")
        return {"status": "done", "v3_spent": round(v3_spent(), 2),
                "cells": len(serializable)}

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FATAL v3: {e}\n{tb}")
        try:
            results_vol.reload()
        except Exception:
            pass
        with open(f"{RESULTS}/FAILURE_V3.md", "w") as f:
            f.write(f"# FAILURE (v3) — {now_iso()}\n\n{e}\n\n```\n{tb}\n```\n")
        results_vol.commit()
        return {"status": "failed", "error": str(e)}


# ----------------------------------------------------------------------------
# v4: pre-registered out-of-sample test (PREREGISTRATION.md, sha PREREG_SHA)
# ----------------------------------------------------------------------------


@app.function(timeout=1800, **gpu_kw)
def smoke_v4():
    return _smoke_v3_impl(["qwen3-1.7b", "qwen3-8b"])


@app.function(image=image, volumes=VOLS, cpu=1, memory=2048, timeout=300)
def smoke_v4_save(report: dict):
    results_vol.reload()
    write_json(f"{RESULTS}/smoke_v4.json", report)
    results_vol.commit()


@app.function(timeout=86400, image=image, volumes=VOLS, cpu=2, memory=8192,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def orchestrate_v4():
    import math as _math
    import traceback

    try:
        results_vol.reload()
        smoke = read_json(f"{RESULTS}/smoke_v4.json")
        if not smoke or not smoke.get("pass"):
            raise RuntimeError("smoke_v4.json missing or failed")
        v4_base = read_json(f"{RESULTS}/v4_budget.json", None)
        if v4_base is None:
            v4_base = {"baseline_usd": get_spent()}
            write_json(f"{RESULTS}/v4_budget.json", v4_base)
            results_vol.commit()

        def v4_spent():
            return get_spent() - v4_base["baseline_usd"]

        def guard(next_usd, label):
            if v4_spent() + next_usd > V4_BUDGET_USD:
                append_decision(f"V4 SKIP {label}: v4 spend ${v4_spent():.2f} + "
                                f"est ${next_usd:.2f} > ${V4_BUDGET_USD}")
                return False
            return True

        def loop(fn, *args, label=""):
            for _ in range(8):
                r = fn.remote(*args)
                if r["complete"]:
                    return r
            raise RuntimeError(f"{label} incomplete after 8 calls")

        append_decision(f"v4 start: pre-registered out-of-sample test, 12 cells, "
                        f"cap ${V4_BUDGET_USD}. PREREGISTRATION.md sha256="
                        f"{PREREG_SHA}")

        gsm_pids_v4 = [f"gsm8k-{i}" for i in sample_problem_ids(1319, 200)]
        cell_results = {}
        for slug, bench in V4_CELLS:
            root = f"v4/{slug}/"
            if not guard(1.0, f"cell {slug}/{bench}"):
                continue
            try:
                log(f"=== v4 cell: {slug} x {bench} ===")
                pids = gsm_pids_v4 if bench == "gsm8k" else None
                loop(generate_traces, bench, pids, 2, slug, root,
                     label=f"gen {slug}/{bench}")
                loop(extract_readouts, None, 2, slug, root,
                     label=f"extract {slug}/{bench}")
                loop(extract_confidence, None, slug, root,
                     label=f"conf {slug}/{bench}")
                loop(extract_residuals, None, slug, root,
                     label=f"resid {slug}/{bench}")
                cell_results[f"{slug}|{bench}"] = analyze_v3.remote(
                    slug, bench, root)
            except Exception as e:  # noqa: BLE001
                append_decision(f"v4 cell {slug}/{bench} failed: {e}")
                log(traceback.format_exc()[-600:])

        # ---- pre-registered verdict ----
        def norm_cdf(z):
            return 0.5 * (1 + _math.erf(z / _math.sqrt(2)))

        def norm_inv(p):
            # Beasley-Springer-Moro-lite via bisection (adequate here)
            lo, hi = -8.0, 8.0
            for _ in range(80):
                mid = (lo + hi) / 2
                if norm_cdf(mid) < p:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2

        included, excluded = {}, {}
        for key, res in cell_results.items():
            if not res or res.get("skipped"):
                excluded[key] = "degenerate/failed"
                continue
            included[key] = res
        qualifying, strong_ctrl = [], []
        for key, res in included.items():
            conf = res["aucs"]["confidence_only"]["auc"]
            d = res["paired_deltas"]["jspace_plus_conf_vs_conf"]
            entry = {"cell": key, "conf_auc": conf, "delta": d["delta"],
                     "p_two": d["p_two_sided"]}
            if conf < 0.55:
                qualifying.append(entry)
            if conf >= 0.65:
                strong_ctrl.append(entry)

        verdict = {"prereg_sha256": PREREG_SHA, "generated_at": now_iso(),
                   "n_cells_run": len(cell_results),
                   "n_included": len(included), "excluded": excluded,
                   "qualifying": qualifying, "strong_control": strong_ctrl}
        if len(qualifying) < 3:
            verdict["primary"] = {"result": "inconclusive",
                                  "reason": f"only {len(qualifying)} qualifying "
                                            "cells (<3, per prereg)"}
        else:
            zs = []
            for q in qualifying:
                p_one = (q["p_two"] / 2 if q["delta"] > 0
                         else 1 - q["p_two"] / 2)
                p_one = min(max(p_one, 1e-6), 1 - 1e-6)
                zs.append(norm_inv(1 - p_one))
            Z = sum(zs) / _math.sqrt(len(zs))
            p_comb = 1 - norm_cdf(Z)
            n_pos = sum(1 for q in qualifying if q["delta"] > 0)
            k = len(qualifying)
            p_sign = sum(_math.comb(k, i) for i in range(n_pos, k + 1)) / 2 ** k
            verdict["primary"] = {
                "stouffer_Z": round(Z, 3), "p_one_sided": round(p_comb, 4),
                "confirmed": bool(p_comb < 0.05),
            }
            verdict["secondary_sign_test"] = {
                "positive": n_pos, "of": k, "p_one_sided": round(p_sign, 4)}
        if strong_ctrl:
            mean_d = sum(e["delta"] for e in strong_ctrl) / len(strong_ctrl)
            verdict["negative_control"] = {
                "n": len(strong_ctrl), "mean_delta": round(mean_d, 4),
                "prediction_abs_lt": 0.03}

        results_vol.reload()
        write_json(f"{RESULTS}/results_v4_cross.json",
                   {k: v for k, v in cell_results.items() if v})
        write_json(f"{RESULTS}/VERDICT_V4.json", verdict)
        with open(f"{RESULTS}/DONE_V4", "w") as f:
            f.write(now_iso())
        results_vol.commit()
        log(f"V4 PIPELINE DONE: {json.dumps(verdict.get('primary'))}")
        return {"status": "done", "verdict": verdict.get("primary"),
                "v4_spent": round(v4_spent(), 2)}

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FATAL v4: {e}\n{tb}")
        try:
            results_vol.reload()
        except Exception:
            pass
        with open(f"{RESULTS}/FAILURE_V4.md", "w") as f:
            f.write(f"# FAILURE (v4) — {now_iso()}\n\n{e}\n\n```\n{tb}\n```\n")
        results_vol.commit()
        return {"status": "failed", "error": str(e)}


@app.local_entrypoint()
def smoke_v4_entry():
    r = smoke_v4.remote()
    print(json.dumps(r, indent=2, default=str))
    ok = all(v.get("ok") for v in r.values())
    smoke_v4_save.remote({"models": r, "pass": ok})
    if not ok:
        raise SystemExit("SMOKE_V4 FAILED — do not detach.")
    print("SMOKE_V4 PASSED — launch: "
          "modal run --detach jspace_pipeline.py::orchestrate_v4")


# ----------------------------------------------------------------------------
# Goodhart: best-of-n pressure (DESIGN_GOODHART.md / PREREG_GOODHART.md)
# ----------------------------------------------------------------------------

GOODHART_K = 64
GOODHART_FULL = ["qwen3.5-4b", "qwen3-4b"]          # full battery, N=100
GOODHART_CONF = ["qwen3.5-0.8b", "qwen2.5-7b-it"]   # confidence-deep, N=150
GOODHART_N_FULL, GOODHART_N_CONF = 100, 150
GOODHART_BUDGET_USD = 60.0
GOODHART_SHAS = ("adc0b3036cae302e05fad7e0c7e86585aec9ac1e2a08912dde67a10fdd529c8a",
                 "921c4f576f8294f17d0ef87a8836b36a484a33faeb78dc73f17ee4862ef04921")
V3_USABLE_RATE = 0.853   # v1-v4 GSM8K usable fraction; quality gate reference


def goodhart_pids(n_want):
    """Problems never used in v1-v4, so frozen scorers have never seen them."""
    import random as _random

    used = set()
    for n in (N_PROBLEMS, 200):
        used |= set(sample_problem_ids(1319, n))
    free = sorted(set(range(1319)) - used)
    rng = _random.Random(12345)
    return [f"gsm8k-{i}" for i in rng.sample(free, min(n_want, len(free)))]


@app.function(timeout=86400, image=image, volumes=VOLS, cpu=2, memory=8192,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def orchestrate_goodhart():
    import traceback

    import pandas as pd

    try:
        results_vol.reload()
        base = read_json(f"{RESULTS}/goodhart_budget.json", None)
        if base is None:
            base = {"baseline_usd": get_spent(), "checkpoints": []}
            write_json(f"{RESULTS}/goodhart_budget.json", base)
            results_vol.commit()

        def spent():
            return get_spent() - base["baseline_usd"]

        def checkpoint():
            for mark in (20.0, 40.0):
                if spent() >= mark and mark not in base["checkpoints"]:
                    base["checkpoints"].append(mark)
                    write_json(f"{RESULTS}/goodhart_budget.json", base)
                    append_decision(f"GOODHART checkpoint ${mark:.0f}: phase spend "
                                    f"${spent():.2f} of ${GOODHART_BUDGET_USD}, "
                                    f"total ${get_spent():.2f}.")

        def guard(est, label):
            checkpoint()
            if spent() + est > GOODHART_BUDGET_USD:
                append_decision(f"GOODHART SKIP {label}: ${spent():.2f} + est "
                                f"${est:.2f} > ${GOODHART_BUDGET_USD}")
                return False
            return True

        def loop(fn, *args, label=""):
            for _ in range(12):
                r = fn.remote(*args)
                checkpoint()
                if r["complete"]:
                    return r
            raise RuntimeError(f"{label} incomplete after 12 calls")

        append_decision(
            f"GOODHART start: K={GOODHART_K}, full-battery {GOODHART_FULL} "
            f"N={GOODHART_N_FULL}, confidence-deep {GOODHART_CONF} "
            f"N={GOODHART_N_CONF}, GSM8K problems unused in v1-v4. "
            f"DESIGN sha={GOODHART_SHAS[0][:16]} PREREG sha={GOODHART_SHAS[1][:16]}. "
            f"Cap ${GOODHART_BUDGET_USD}.")

        def quality_gate(slug, root):
            """Stop rather than grade through a format-quality regression."""
            man = read_json(f"{RESULTS}/{root}traces/manifest.json", {"shards": []})
            dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(man["shards"]))
                   if os.path.exists(f"{RESULTS}/{s}")]
            if not dfs:
                return True
            df = pd.concat(dfs, ignore_index=True)
            rate = float(df["usable"].mean())
            ok = rate >= V3_USABLE_RATE - 0.05
            if not ok:
                append_decision(f"GOODHART QUALITY GATE FAILED {slug}: usable rate "
                                f"{rate:.3f} vs v3 {V3_USABLE_RATE:.3f} "
                                f"(>5pp worse) — stopping this model.")
            return ok

        done = {}
        for slug, npro, full in ([(s, GOODHART_N_FULL, True) for s in GOODHART_FULL]
                                 + [(s, GOODHART_N_CONF, False) for s in GOODHART_CONF]):
            root = f"goodhart/{slug}/"
            est = 12.0 if full else 9.0
            if not guard(est, f"model {slug}"):
                continue
            try:
                pids = goodhart_pids(npro)
                log(f"=== goodhart: {slug} (full={full}) N={npro} K={GOODHART_K} ===")
                loop(generate_traces, "gsm8k", pids, GOODHART_K, slug, root,
                     label=f"gen {slug}")
                if not quality_gate(slug, root):
                    done[slug] = "quality_gate_failed"
                    continue
                loop(extract_confidence, None, slug, root, label=f"conf {slug}")
                if full:
                    loop(extract_readouts, None, 2, slug, root, label=f"ext {slug}")
                    loop(extract_residuals, None, slug, root, label=f"resid {slug}")
                done[slug] = "full" if full else "confidence"
                checkpoint()
            except Exception as e:  # noqa: BLE001
                append_decision(f"GOODHART model {slug} failed: {e}")
                log(traceback.format_exc()[-800:])
                done[slug] = f"failed: {e}"

        write_json(f"{RESULTS}/goodhart_manifest.json",
                   {"generated_at": now_iso(), "models": done,
                    "K": GOODHART_K, "phase_spend_usd": round(spent(), 2),
                    "design_sha": GOODHART_SHAS[0], "prereg_sha": GOODHART_SHAS[1]})
        with open(f"{RESULTS}/DONE_GOODHART_GEN", "w") as f:
            f.write(now_iso())
        results_vol.commit()
        append_decision(f"GOODHART generation complete: {json.dumps(done)}; "
                        f"phase spend ${spent():.2f}.")
        log(f"GOODHART GEN DONE: {json.dumps(done)}")
        return {"status": "done", "models": done, "phase_spend": round(spent(), 2)}

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FATAL goodhart: {e}\n{tb}")
        try:
            results_vol.reload()
        except Exception:
            pass
        with open(f"{RESULTS}/FAILURE_GOODHART.md", "w") as f:
            f.write(f"# FAILURE (goodhart) — {now_iso()}\n\n{e}\n\n```\n{tb}\n```\n")
        results_vol.commit()
        return {"status": "failed", "error": str(e)}


def assemble_cell_features(root, benchmark, slug, want_surface=True):
    """Feature matrices for every usable trace in a cell, in a deterministic
    uid order. Mirrors analyze_v3's assembly exactly (same primitives, same
    key order) so columns are comparable across cells."""
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    man = read_json(f"{RESULTS}/{root}traces/manifest.json", {"shards": []})
    dfs = [pd.read_parquet(f"{RESULTS}/{s}") for s in sorted(set(man["shards"]))
           if os.path.exists(f"{RESULTS}/{s}")]
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid")
    df = df[df["usable"] & (df["dataset"] == benchmark)].set_index("uid")
    rman = read_json(f"{RESULTS}/{root}readouts/manifest.json", {"done": []})
    uids = [u for u in rman["done"] if u in df.index
            and os.path.exists(f"{RESULTS}/{root}readouts/{u}.npz")
            and os.path.exists(f"{RESULTS}/{root}confidence/{u}.json")]
    if not uids:
        return None

    tok = AutoTokenizer.from_pretrained(MODELS[slug]["hf"])
    cache = {}

    def tok_decode(t):
        if t not in cache:
            cache[t] = tok.decode([t])
        return cache[t]

    emb, spans = None, {}
    if want_surface:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                 device="cpu")
        hf_vol.commit()
        steps_all = []
        for u in uids:
            ss = [s["text"] for s in json.loads(df.loc[u, "steps_json"])]
            spans[u] = (len(steps_all), len(steps_all) + len(ss))
            steps_all.extend(ss)
        emb = st.encode(steps_all, batch_size=128, show_progress_bar=False,
                        convert_to_numpy=True)

    rows = {c: [] for c in ("confidence", "length", "jspace", "logitlens",
                            "randread", "surface", "probe")}
    keep, ys, gs, sids = [], [], [], []
    for u in uids:
        r = df.loc[u]
        npz = np.load(f"{RESULTS}/{root}readouts/{u}.npz")
        probes = npz["probe_layers"]
        Lp = len(probes)
        n_ext = npz["j_topk_ids"].shape[0] - 1
        if n_ext < MIN_STEPS:
            continue
        conf = read_json(f"{RESULTS}/{root}confidence/{u}.json")
        if not conf:
            continue
        if want_surface:
            s0, _ = spans[u]
            E = emb[s0:s0 + n_ext]
            if len(E) < MIN_STEPS:
                continue
            rows["surface"].append(battery(E, "surf"))
        depth = {"early": Lp // 5, "mid": Lp // 2, "late": (4 * Lp) // 5}
        low = r["gen_text"].lower()
        rows["confidence"].append(conf)
        rows["length"].append({
            "len_nsteps": float(r["n_steps"]),
            "len_ntok": float(r["n_gen_tokens"]),
            "len_meanstep": float(np.mean([len(s["text"]) for s in
                                           json.loads(r["steps_json"])]))})
        for cond, tag in (("jspace", "j"), ("logitlens", "ll"), ("randread", "rr")):
            f = {}
            for dn, di in depth.items():
                M = densify(npz[f"{tag}_topk_ids"][:-1], npz[f"{tag}_topk_p"][:-1], [di])
                f.update(battery(M, f"{tag}_{dn}"))
                f.update(concept_features(tag, npz, di, f"{tag}_{dn}",
                                          tok_decode, low))
            Mavg = densify(npz[f"{tag}_topk_ids"][:-1], npz[f"{tag}_topk_p"][:-1],
                           list(range(Lp)))
            f.update(battery(Mavg, f"{tag}_avg"))
            rows[cond].append(f)
        rp = f"{RESULTS}/{root}residuals/{u}.npz"
        if os.path.exists(rp):
            with np.load(rp) as z:
                R = z["resid"].astype(np.float32)
            mid = min(1, R.shape[1] - 1)
            rows["probe"].append(np.concatenate([R[:-1, mid].mean(0), R[-1, mid]]))
        else:
            rows["probe"].append(None)
        keep.append(u)
        ys.append(int(bool(r["correct"])))
        gs.append(str(r["problem_id"]))
        sids.append(int(r["sample_id"]))

    out = {"uids": keep, "y": np.array(ys, int),
           "groups": np.array(gs, object), "sample_id": np.array(sids, int),
           "X": {}, "cols": {}}
    for cond in ("confidence", "length", "jspace", "logitlens", "randread",
                 "surface"):
        if not rows[cond]:
            continue
        cols = sorted(rows[cond][0])
        out["cols"][cond] = cols
        out["X"][cond] = np.nan_to_num(np.array(
            [[d.get(k, 0.0) for k in cols] for d in rows[cond]], dtype=np.float64))
    pv = [v for v in rows["probe"] if v is not None]
    if pv and len(pv) == len(keep):
        out["X"]["probe"] = np.array(pv, dtype=np.float64)
        out["cols"]["probe"] = [f"r{i}" for i in range(out["X"]["probe"].shape[1])]
    return out


@app.function(timeout=21600, **cpu_kw)
def goodhart_scores(slug: str, want_surface: bool = True):
    """Apply the FROZEN v3 scorers (refit deterministically on v3 train
    problems, seed 0) to the goodhart pressured traces. Nothing is fit on
    goodhart data."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    results_vol.reload()
    v3_root = MODELS[slug]["root"]
    gh_root = f"goodhart/{slug}/"

    src = assemble_cell_features(v3_root, "gsm8k", slug, want_surface)
    tgt = assemble_cell_features(gh_root, "gsm8k", slug, want_surface)
    if src is None or tgt is None:
        return {"slug": slug, "error": "missing features"}

    # identical train split to conformal_scores.emit_scores seed 0
    rng = np.random.default_rng(1000 + 0)
    ug = np.unique(src["groups"].astype(str))
    perm = rng.permutation(len(ug))
    n_tr = int(round(0.30 * len(ug)))
    train_g = set(ug[perm[:n_tr]])
    tr = np.array([str(g) in train_g for g in src["groups"]])

    arrays = {"y": tgt["y"], "groups": tgt["groups"].astype(str),
              "sample_id": tgt["sample_id"], "uids": np.array(tgt["uids"])}
    meta = {"slug": slug, "generated_at": now_iso(),
            "n_goodhart_traces": int(len(tgt["y"])),
            "n_problems": int(len(set(tgt["groups"].astype(str)))),
            "signals": [], "column_check": {}}
    for sig in ("confidence", "jspace", "logitlens", "randread", "surface",
                "probe"):
        if sig not in src["X"] or sig not in tgt["X"]:
            continue
        same = src["cols"][sig] == tgt["cols"][sig]
        meta["column_check"][sig] = bool(same)
        if not same:
            log(f"WARNING {slug}/{sig}: column mismatch, skipping")
            continue
        C = 0.01 if sig == "probe" else 1.0
        ytr = src["y"][tr]
        if len(np.unique(ytr)) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(src["X"][sig][tr], ytr)
        arrays[f"s__{sig}"] = m.predict_proba(tgt["X"][sig])[:, 1]
        # unpressured reference scores on v3 held-out, for sanity comparison
        arrays[f"v3heldout_s__{sig}"] = m.predict_proba(src["X"][sig][~tr])[:, 1]
        arrays[f"v3heldout_y"] = src["y"][~tr]
        arrays[f"v3heldout_groups"] = src["groups"].astype(str)[~tr]
        meta["signals"].append(sig)

    os.makedirs(f"{RESULTS}/goodhart", exist_ok=True)
    np.savez_compressed(f"{RESULTS}/goodhart/scores_{slug}.npz", **arrays)
    write_json(f"{RESULTS}/goodhart/scores_{slug}_meta.json", meta)
    results_vol.commit()
    log(f"goodhart_scores {slug}: {meta['n_goodhart_traces']} traces, "
        f"signals={meta['signals']}")
    return meta


#: Amendment 2: N sized by the calibration-feasibility gate (expected n0 >= 12
#: at alpha = base error rate), not by generation cost. See DECISIONS.md.
GOODHART_N_EXT = {"qwen3.5-4b": 175, "qwen3-4b": 285}


@app.function(timeout=86400, image=image, volumes=VOLS, cpu=2, memory=8192,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def goodhart_extend():
    """Extend the full-battery cells to calibration feasibility."""
    import traceback

    try:
        results_vol.reload()
        base = read_json(f"{RESULTS}/goodhart_budget.json",
                         {"baseline_usd": get_spent(), "checkpoints": []})

        def spent():
            return get_spent() - base["baseline_usd"]

        def loop(fn, *args, label=""):
            for _ in range(16):
                r = fn.remote(*args)
                if r["complete"]:
                    return r
            raise RuntimeError(f"{label} incomplete")

        append_decision(
            "GOODHART Amendment 2 (design error, disclosed): N=100 was sized by "
            "generation cost and failed the calibration-feasibility gate this "
            "project established in DESIGN_CONFORMAL sec 3 (expected n0 >= 12 at "
            "alpha = base error). Observed n0 = 4.1 (qwen3-4b) and 6.9 "
            "(qwen3.5-4b), so Clopper-Pearson certified nothing and coverage was "
            "0.00-0.14; the resulting beta = 0 is a VACUOUS-GUARANTEE ARTEFACT, "
            f"not evidence of a benign regime. Branch B is NOT claimed. Fix: "
            f"extend to N={GOODHART_N_EXT} (feasibility-sized) and re-run. "
            "Endpoint criteria unchanged.")

        out = {}
        for slug, npro in GOODHART_N_EXT.items():
            root = f"goodhart/{slug}/"
            if spent() + 10.0 > GOODHART_BUDGET_USD:
                append_decision(f"GOODHART extend SKIP {slug}: budget")
                continue
            pids = goodhart_pids(npro)
            log(f"=== goodhart extend: {slug} -> N={npro} ===")
            loop(generate_traces, "gsm8k", pids, GOODHART_K, slug, root,
                 label=f"gen {slug}")
            loop(extract_confidence, None, slug, root, label=f"conf {slug}")
            loop(extract_readouts, None, 2, slug, root, label=f"ext {slug}")
            loop(extract_residuals, None, slug, root, label=f"resid {slug}")
            out[slug] = npro
            log(f"extend {slug} done; phase spend ${spent():.2f}")
        with open(f"{RESULTS}/DONE_GOODHART_EXT", "w") as f:
            f.write(now_iso())
        results_vol.commit()
        append_decision(f"GOODHART extension complete: {json.dumps(out)}; "
                        f"phase spend ${spent():.2f}.")
        return {"status": "done", "models": out, "phase_spend": round(spent(), 2)}
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FATAL goodhart_extend: {e}\n{tb}")
        return {"status": "failed", "error": str(e)}


@app.local_entrypoint()
def goodhart_extend_entry():
    print(json.dumps(goodhart_extend.remote(), indent=2, default=str))


@app.local_entrypoint()
def goodhart_scores_full_entry():
    """Score only the full-battery models (the primary endpoint's scope)."""
    out = list(goodhart_scores.starmap([(s, True) for s in GOODHART_FULL]))
    print(json.dumps(out, indent=2, default=str)[:4000])


@app.local_entrypoint()
def goodhart_scores_entry():
    full = [(s, True) for s in GOODHART_FULL]
    conf = [(s, False) for s in GOODHART_CONF]
    out = list(goodhart_scores.starmap(full + conf))
    print(json.dumps(out, indent=2, default=str)[:4000])


@app.local_entrypoint()
def regrade_entry():
    roots = [f"v3/{s}/" for s in
             ["qwen3.5-4b", "qwen3.5-0.8b", "qwen3-4b", "olmo-3-7b",
              "qwen2.5-7b-it"]]
    print(json.dumps(regrade_traces.remote(roots), indent=2))


@app.local_entrypoint()
def smoke_v3_entry():
    r40 = smoke_v3_40.remote()
    print(json.dumps(r40, indent=2, default=str))
    pass_40 = all(v.get("ok") for v in r40.values())
    try:
        r80 = smoke_v3_80.remote()
        print(json.dumps(r80, indent=2, default=str))
        pass_80 = all(v.get("ok") for v in r80.values())
    except Exception as e:  # noqa: BLE001
        print("80GB smoke crashed:", e)
        r80, pass_80 = {"error": str(e)}, False
    report = {"models_40": r40, "models_80": r80,
              "pass_40": pass_40, "pass_80": pass_80}
    smoke_save.remote(report)
    if not pass_40:
        raise SystemExit("SMOKE_V3 FAILED (40GB models) — do not detach.")
    msg = "" if pass_80 else " (gpt-oss-20b failed smoke; v3 will skip it)"
    print(f"SMOKE_V3 PASSED{msg} — launch: "
          "modal run --detach jspace_pipeline.py::orchestrate_v3")


# ----------------------------------------------------------------------------
# Local entrypoints
# ----------------------------------------------------------------------------


@app.function(timeout=14400, image=image, volumes=VOLS, cpu=2, memory=4096,
              retries=modal.Retries(max_retries=2, initial_delay=30.0))
def reanalyze_remote():
    """Detach-safe reanalysis driver: reruns analyze_v2 for all models."""
    out = {}
    for slug in [PRIMARY] + EXTRA_MODELS:
        try:
            r = analyze_v2.remote(slug)
            out[slug] = r["n_analyzed"]
        except Exception as e:  # noqa: BLE001
            out[slug] = f"FAILED: {e}"
        log(f"reanalyze {slug}: {out[slug]}")
    log(f"REANALYZE DONE: {json.dumps(out, default=str)}")
    return out


@app.local_entrypoint()
def reanalyze_entry():
    """Rerun analyze_v2 (with the gsm8k_only block) for every model that has
    data on the volume. Run AFTER the detached v2 pipeline has finished."""
    for slug in [PRIMARY] + EXTRA_MODELS:
        try:
            res = analyze_v2.remote(slug)
            print(slug, "->", res["n_analyzed"], "traces reanalyzed")
        except Exception as e:  # noqa: BLE001
            print(slug, "-> FAILED:", e)


@app.local_entrypoint()
def smoke_v2_entry():
    report = smoke_v2.remote()
    print(json.dumps(report, indent=2, default=str))
    if not report.get("pass"):
        raise SystemExit("SMOKE_V2 FAILED — do not detach.")
    print("SMOKE_V2 PASSED — safe to launch: "
          "modal run --detach jspace_pipeline.py::orchestrate_v2")


@app.local_entrypoint()
def smoke_entry():
    report = smoke.remote()
    print(json.dumps(report, indent=2, default=str))
    if not report.get("pass"):
        raise SystemExit("SMOKE FAILED — do not detach. See report above.")
    print("SMOKE PASSED — safe to launch: "
          "modal run --detach jspace_pipeline.py::orchestrate")
