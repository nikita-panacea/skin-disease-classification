# Derm-1M Feature Extraction Pipeline

End-to-end pipeline: discover a clinical feature schema from captions, extract **0/1/2** encodings for every row with an LLM, and run analysis / SCIN comparison.

## Feature encoding

| Value | Meaning |
|-------|---------|
| **1** | Feature present / mentioned in the caption |
| **0** | Explicitly absent (e.g. “non-itchy”) |
| **2** | Unknown / not mentioned |

## Repository layout

| File | Role |
|------|------|
| `cleaned_caption_Derm1M.csv` | Input: `truncated_caption`, `label_name`, `disease_label`, … |
| `scin_feature_map.py` | Shared SCIN ↔ canonical feature names |
| `phase1_feature_discovery.py` | Build `feature_schema.json` (sampled or full captions) |
| `phase2_bulk_extraction.py` | LLM extraction → `derm1m_features.csv` + `checkpoints/` |
| `phase3_analysis.py` | Coverage, SCIN gap, MI + χ² + Cramér’s V, classwise OR, EDA |
| `phase3b_cooccurrence_analysis.py` | Per-disease signatures, SCIN reproducibility, confusion gaps |

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step-by-step pipeline (overview)

1. **Phase 1** — discovers and consolidates features; writes `feature_schema.json`.
2. **Phase 2** — loads the schema and runs **one LLM call per batch** of captions (checkpointed per `label_name`).
3. **Phase 3** — statistical analysis and plots under `analysis_outputs/`.
4. **Phase 3b** (optional) — co-occurrence / signature analysis under `analysis_outputs/cooccurrence/`.

Use the **same** `LLM_PROVIDER`, `CAPTION_COLUMN`, and (for Qwen) `QWEN_BASE_URL` / `QWEN_MODEL_NAME` for Phase 1 and Phase 2 so behavior stays consistent.

---

## Using **Qwen 3.5 9B** locally (recommended flow)

Qwen is **not** called with a cloud API key here. You run an **OpenAI-compatible server** (e.g. [SGLang](https://github.com/sgl-project/sglang) or [vLLM](https://github.com/vllm-project/vllm)) that loads **`Qwen/Qwen3.5-9B`** from Hugging Face, then the pipeline talks to it with the **`openai`** Python SDK (`base_url` + dummy key).

### Step 0 — Hardware and model

- You need a machine with enough GPU memory for **Qwen3.5-9B** (exact VRAM depends on quantisation and framework; see the [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)).
- The server must expose a **Chat Completions** API (OpenAI-compatible), typically at `http://localhost:8000/v1`.

### Step 1 — Start the local inference server

**Example (SGLang)** — adjust GPU count and flags per your setup and the model card.

For **this pipeline** (short captions + JSON), you do **not** need 262k context. A **smaller context** uses far less KV-cache GPU memory and starts more reliably:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-9B \
  --port 8000 \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --reasoning-parser qwen3
```

**Example (vLLM)** — **prefer a moderate `--max-model-len`** (e.g. `8192`–`32768`). Values like `262144` often leave almost **no KV cache** on a single consumer GPU and can trigger failures during CUDA-graph capture on Qwen3.5’s hybrid layers.

**Recommended first try** (caption / JSON extraction):

```bash
vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enforce-eager
```

- Drop `--language-model-only` unless you are sure you need it; if you use it and still crash, retry **without** it.
- If startup still fails, try `--max-model-len 8192` or add `--gpu-memory-utilization 0.92`.

Leave this process running. Check that `http://localhost:8000/v1` responds (e.g. health or a minimal chat request).

### Step 2 — Configure environment

Create or edit **`.env`** in the project root:

```env
# Use local Qwen for both phases
LLM_PROVIDER=qwen

# OpenAI SDK client points at your local server (no real API key)
QWEN_BASE_URL=http://localhost:8000/v1
QWEN_MODEL_NAME=Qwen/Qwen3.5-9B

# Captions column in cleaned_caption_Derm1M.csv
CAPTION_COLUMN=truncated_caption

# Phase 1: stratified = cheaper; full = every non-empty caption (very slow)
DISCOVERY_SAMPLING_MODE=stratified

# Must match vLLM --max-model-len (default here 8192). Phase 1 clamps completion tokens so
# prompt + max_tokens never exceeds this (avoids HTTP 400 from vLLM).
QWEN_MAX_MODEL_LEN=8192
```

You do **not** need `OPENAI_API_KEY` or `GEMINI_API_KEY` when `LLM_PROVIDER=qwen`.

**If vLLM returns HTTP 400** about “input … and requested … output tokens”: your **prompt + `max_tokens` exceeded `--max-model-len`**. Either **raise** `--max-model-len` (e.g. `32768`) and set **`QWEN_MAX_MODEL_LEN`** to the same value, or **lower** `DISCOVERY_BATCH_SIZE` so each request is shorter.

Optional (same shell, if you prefer not to use `.env` for the base URL):

```bash
# PowerShell
$env:LLM_PROVIDER="qwen"
$env:QWEN_BASE_URL="http://localhost:8000/v1"
$env:QWEN_MODEL_NAME="Qwen/Qwen3.5-9B"

# bash
export LLM_PROVIDER=qwen
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL_NAME=Qwen/Qwen3.5-9B
```

### Step 3 — Phase 1: feature discovery

```bash
python phase1_feature_discovery.py
```

**Outputs:**

- `feature_schema.json` — canonical features + SCIN alignment metadata  
- `discovery_outputs/` — per-label caches (`discovery_<label>_<mode>.json`), `sampled_captions.csv`

**Notes:**

- First run can take a long time (many LLM calls). Cached labels are skipped on rerun.
- Switching `DISCOVERY_SAMPLING_MODE` uses a **different** cache filename so stratified vs full do not overwrite each other.

### Step 4 — Phase 2: bulk extraction

```bash
python phase2_bulk_extraction.py
```

**Outputs:**

- `derm1m_features.csv` — original metadata columns + one column per schema feature  
- `checkpoints/llm_<label_name>.json` — resume if interrupted  
- `extraction_stats.json` — run statistics (includes `caption_column`)

**Notes:**

- This is the heaviest step (full dataset × batches). Qwen uses a short inter-batch sleep (`0.1s`) by default.
- Truncates each caption to **500 characters** before the LLM (`MAX_CAPTION_LEN` in script).

### Step 5 — Phase 3: analysis

```bash
python phase3_analysis.py
```

Requires `SCIN-dataset/dataset_scin_cases.csv` (paths are set at top of the script).

**Notable outputs:**

- `analysis_outputs/feature_importance_global.csv` — MI, χ², Cramér’s V  
- `analysis_outputs/feature_importance_with_scin_context.csv` — importance vs SCIN mapping / gap  
- `analysis_outputs/explainability_report.md`  
- `analysis_outputs/classwise_importance_all.csv`  
- `analysis_outputs/eda/` — plots and summary tables  

Optional: fine-grained label axis:

```env
LABEL_COL=disease_label
```

### Step 6 — Phase 3b (optional): co-occurrence

```bash
python phase3b_cooccurrence_analysis.py
```

Optional env:

```env
CONFUSION_PAIRS_CSV=path/to/pairs.csv
COOCCURRENCE_PHI_TOP_K=60
```

`pairs.csv` columns: `true_label`, `confused_with` (merged with built-in default pairs).

---

## Other LLM backends

| `LLM_PROVIDER` | Requirements |
|----------------|--------------|
| `gemini` | `GEMINI_API_KEY` in `.env` |
| `openai` | `OPENAI_API_KEY` in `.env` (cloud OpenAI API) |
| `qwen` | Local server at `QWEN_BASE_URL`; client uses `api_key="EMPTY"` |

Phase 1 only: set `OPENAI_JSON_RESPONSE=1` with `openai` to request JSON object mode (not used for Phase 2 array responses).

---

## Environment variables (quick reference)

| Variable | Default | Used in |
|----------|---------|---------|
| `LLM_PROVIDER` | phase1: `gemini`, phase2: `openai` | phase1, phase2 |
| `QWEN_BASE_URL` | `http://localhost:8000/v1` | phase1, phase2 (qwen) |
| `QWEN_MODEL_NAME` | `Qwen/Qwen3.5-9B` | phase1, phase2 (qwen) |
| `CAPTION_COLUMN` | `truncated_caption` | phase1, phase2 |
| `DISCOVERY_SAMPLING_MODE` | `stratified` | phase1 (`full` = all captions) |
| `LABEL_COL` | `label_name` | phase3 |
| `CONFUSION_PAIRS_CSV` | (empty) | phase3b |
| `COOCCURRENCE_PHI_TOP_K` | `60` (`0` = all features) | phase3b |

---

## Interpreting Derm-1M vs SCIN

| Observation | Possible implication |
|-------------|---------------------|
| Large gap in `derm_vs_scin_comparison.csv` | Captions carry more structured cues than SCIN forms collect → external validation may lack those cues. |
| High MI / Cramér’s V on features **not** in `SCIN_TO_CANONICAL` | Model may lean on signals that SCIN does not capture → questionnaire or retraining. |
| Phase 3b low **signature reproducibility** on SCIN | Diagnostic co-patterns in Derm-1M are hard to reproduce from SCIN fields alone. |

---

## Troubleshooting: vLLM, GPU, and Python env

### `pip install vllm` → protobuf / `google-ai-generativelanguage` conflicts

vLLM often pulls **`protobuf` 6.x**, while **`google-generativeai`** (Gemini) expects **`protobuf` &lt; 6**. Pip warns but both may break at runtime.

**Fix:** use **two environments** (recommended):

- **Env A (`Derm-serve` or similar):** only what you need to run **vLLM** + `openai` client → serve Qwen.
- **Env B (`Derm-pipeline`):** `pip install -r requirements.txt` for **phase1–3** (Gemini/OpenAI/cloud) **without** installing vLLM on the same env.

If everything runs on one machine: run vLLM in env A, run `python phase1_feature_discovery.py` in env B with `LLM_PROVIDER=qwen` and `QWEN_BASE_URL=http://localhost:8000/v1` (no vLLM import required in the pipeline).

### vLLM crashes after loading weights: `AssertionError` in `causal_conv1d` / `num_cache_lines >= batch`

Typical pattern in the log:

- **“Available KV cache memory: ~2–3 GiB”** and **“Maximum concurrency for 262144 tokens … 0.33x”**
- Failure during **“Capturing CUDA graphs”**

**Cause:** **`--max-model-len 262144`** reserves a huge KV budget; on one GPU there is little room left, and vLLM’s graph capture for **Qwen3.5** (GDN / linear-attention paths) can hit an internal assert.

**Fix (try in order):**

1. **Lower context** (enough for captions + JSON):  
   `--max-model-len 32768` or `8192`.
2. Add **`--enforce-eager`** (disables CUDA graphs; slower but stable).
3. Remove **`--language-model-only`** if you added it; retry.
4. Upgrade/downgrade vLLM to match [Qwen’s vLLM recipe](https://huggingface.co/Qwen/Qwen3.5-9B) or try **SGLang** instead.
5. As a last resort, set **`VLLM_USE_V1=0`** to force the legacy engine (if your vLLM build supports it).

### Pipeline note

Phase 1/2 use **small** `max_tokens` (e.g. 4096) and **short** prompts; the server only needs a **modest** `max_model_len`, not the model’s theoretical maximum.

---

## References

- [Qwen3.5-9B on Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B) — serving examples (SGLang, vLLM), thinking vs non-thinking modes.
