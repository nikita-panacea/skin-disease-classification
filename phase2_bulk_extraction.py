"""
PHASE 2: Bulk Feature Extraction (LLM) — Cost-Optimized
=======================================================
Strategy:
  - Deduplicate captions first: only process ~72K unique captions instead of ~182K rows.
  - Use numeric feature indices (0-N) with sparse output format to minimize output tokens.
  - Category-aware encoding: 0=absent/inferably absent, 1=present, 2=truly unknown.
  - Prompt caching: static system prompt (>1024 tokens) cached across all API calls.
  - Batch API: 50% cost discount on all token rates.
  - Replicate one-hot encoding back to all rows with duplicate captions.
  - Output: CSV with all original columns + one feature column per schema feature.

Prerequisites:
  pip install pandas numpy tqdm google-generativeai openai python-dotenv

For Qwen 3.5 (local):
  Serve the model locally via vLLM or SGLang first, e.g.:
    python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --port 8000 ...
  Then set LLM_PROVIDER=qwen in your .env or environment.
  Optionally set QWEN_BASE_URL (default: http://localhost:8000/v1).

Run after phase1_feature_discovery.py

Env (optional):
  CAPTION_COLUMN — must match phase1 (default: truncated_caption)
  LLM_BATCH_SIZE — captions per API call (default 25)
  OPENAI_MODEL_NAME — default gpt-4o-mini
  OPENAI_USE_BATCH — 1/true: enqueue all OpenAI extraction jobs on Batch API (~50% lower $, async ≤24h)
  OPENAI_PROMPT_CACHE_KEY — stable key for automatic prompt caching (default phase2_extraction_v1)
  OPENAI_PROMPT_CACHE_RETENTION — optional: in_memory | 24h
  OPENAI_BATCH_POLL_SEC — batch status poll interval (default 20)
  OPENAI_BATCH_MAX_REQUESTS — max lines per batch file (default 50000, API cap)
  OPENAI_BATCH_MAX_FILE_BYTES — max UTF-8 bytes per batch JSONL (default ~195 MiB; API cap 200 MB)
  OPENAI_LOG_USAGE — 1/true: print token usage when available

Cost optimizations applied:
  1. Caption deduplication: ~60% fewer API calls
  2. Sparse output format: ~97% fewer output tokens per caption
  3. Batch API: 50% discount on all rates
  4. Prompt caching: system prompt cached after 1st request (~41% of input tokens)
  Estimated cost: ~$1.50-2.00 for 72K unique captions with gpt-4o-mini
"""

import pandas as pd
import numpy as np
import json
import re
import os
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

from openai_batch_utils import (
    chunk_jobs_for_openai_batch,
    openai_batch_max_file_bytes,
    openai_batches_create_safe,
    write_openai_batch_jsonl,
)

# ── Load API keys from .env ───────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Paths / caption (before LLM init — used in startup logs) ─────────────────
CAPTION_COLUMN   = os.getenv("CAPTION_COLUMN", "truncated_caption")

# ── LLM Provider Selection ─────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "openai")

OPENAI_USE_BATCH = os.getenv("OPENAI_USE_BATCH", "").strip().lower() in ("1", "true", "yes")
OPENAI_PROMPT_CACHE_KEY = os.getenv("OPENAI_PROMPT_CACHE_KEY", "phase2_extraction_v1").strip()
OPENAI_PROMPT_CACHE_RETENTION = os.getenv("OPENAI_PROMPT_CACHE_RETENTION", "").strip()


def _safe_int_env(key: str, default: int, *, vmin: int | None = None, vmax: int | None = None) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        v = default
    else:
        try:
            v = int(raw)
        except ValueError:
            print(f"WARNING: invalid integer env {key}={raw!r}, using default {default}")
            v = default
    if vmin is not None:
        v = max(vmin, v)
    if vmax is not None:
        v = min(vmax, v)
    return v


OPENAI_BATCH_POLL_SEC = _safe_int_env("OPENAI_BATCH_POLL_SEC", 20, vmin=5)
OPENAI_BATCH_MAX_REQUESTS = _safe_int_env(
    "OPENAI_BATCH_MAX_REQUESTS", 50_000, vmin=1, vmax=50_000
)
OPENAI_LOG_USAGE = os.getenv("OPENAI_LOG_USAGE", "").strip().lower() in ("1", "true", "yes")

# Qwen local server config
QWEN_BASE_URL   = os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3.5-9B")

# Model configuration
if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Set GEMINI_API_KEY in your .env file for Gemini provider")
    import google.generativeai as genai

    openai_client = None
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-2.0-flash-thinking-exp"
    model = genai.GenerativeModel(MODEL_NAME)
elif LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        raise ValueError("Set OPENAI_API_KEY in your .env file for OpenAI provider")
    from openai import OpenAI

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip() or "gpt-4o-mini"
    model = None
elif LLM_PROVIDER == "qwen":
    from openai import OpenAI as QwenClient

    openai_client = None
    qwen_client = QwenClient(base_url=QWEN_BASE_URL, api_key="EMPTY")
    MODEL_NAME = QWEN_MODEL_NAME
    model = None
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'gemini', 'openai', or 'qwen'")

print(f"Using LLM Provider: {LLM_PROVIDER} with model: {MODEL_NAME}")
print(f"  Caption column: {CAPTION_COLUMN}")
if LLM_PROVIDER == "qwen":
    print(f"  Qwen base URL: {QWEN_BASE_URL}")
elif LLM_PROVIDER == "openai":
    if OPENAI_USE_BATCH:
        print(
            "  OpenAI extraction: Batch API (~50% lower token cost vs sync; async, within 24h typical window)"
        )
        print(
            f"  Batch file limits: ≤{OPENAI_BATCH_MAX_REQUESTS:,} requests/file, "
            f"≤{openai_batch_max_file_bytes() / (1024 * 1024):.0f} MiB UTF-8/file"
        )
    else:
        print("  OpenAI extraction: synchronous Chat Completions (OPENAI_USE_BATCH=1 for Batch API)")
    if OPENAI_PROMPT_CACHE_KEY:
        print(f"  Prompt cache key: {OPENAI_PROMPT_CACHE_KEY!r} (system prompt first → cache-friendly)")
    if OPENAI_PROMPT_CACHE_RETENTION in ("in_memory", "24h"):
        print(f"  prompt_cache_retention={OPENAI_PROMPT_CACHE_RETENTION!r}")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH         = "cleaned_caption_Derm1M.csv"
SCHEMA_PATH      = "feature_schema.json"
OUTPUT_CSV       = "derm1m_features.csv"
STATS_FILE       = "extraction_stats.json"
CHECKPOINT_DIR   = Path("checkpoints")
LLM_BATCH_SIZE   = _safe_int_env("LLM_BATCH_SIZE", 25, vmin=1)
MAX_CAPTION_LEN  = 3000
MAX_RETRIES      = 5
RATE_LIMIT_SLEEP = 0.1 if LLM_PROVIDER == "qwen" else 0.5
DEDUP_CKPT_FILE  = "dedup_caption_features.json"

CHECKPOINT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LLM API Wrapper Functions
# ══════════════════════════════════════════════════════════════════════════════

def _openai_apply_prompt_caching(create_kw: dict, cache_key: str) -> None:
    """
    OpenAI Prompt Caching: https://developers.openai.com/api/docs/guides/prompt-caching
    Use with static-then-variable messages (system schema first, user captions last).
    """
    if cache_key:
        create_kw["prompt_cache_key"] = cache_key
    if OPENAI_PROMPT_CACHE_RETENTION in ("in_memory", "24h"):
        create_kw["prompt_cache_retention"] = OPENAI_PROMPT_CACHE_RETENTION


def _openai_extraction_chat_body(system_prompt: str, user_prompt: str) -> dict:
    """Chat body for extraction: static system first, variable captions last (prompt caching)."""
    body: dict = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "stream": False,
    }
    _openai_apply_prompt_caching(body, OPENAI_PROMPT_CACHE_KEY)
    return body


def _maybe_log_openai_usage(resp) -> None:
    if not OPENAI_LOG_USAGE:
        return
    u = getattr(resp, "usage", None)
    if not u:
        return
    pt = int(getattr(u, "prompt_tokens", None) or 0)
    ct = int(getattr(u, "completion_tokens", None) or 0)
    details = getattr(u, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", None) or 0) if details else 0
    print(
        f"    OpenAI usage: prompt={pt} (cached={cached}, non_cached≈{max(0, pt - cached)}) "
        f"completion={ct}"
    )


def _openai_parse_batch_output_jsonl(raw: str) -> tuple[dict[str, str], dict[str, int]]:
    acc = {"prompt": 0, "completion": 0, "cached": 0}
    out: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = obj.get("custom_id")
        err = obj.get("error")
        if err:
            tqdm.write(f"    Batch item {cid!r} error: {err}")
            continue
        resp = obj.get("response") or {}
        if resp.get("status_code") != 200:
            tqdm.write(
                f"    Batch item {cid!r} HTTP {resp.get('status_code')}: "
                f"{str(resp.get('body'))[:160]}"
            )
            continue
        body = resp.get("body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                continue
        if not isinstance(body, dict):
            continue
        usage = body.get("usage") or {}
        acc["prompt"] += int(usage.get("prompt_tokens", 0))
        acc["completion"] += int(usage.get("completion_tokens", 0))
        details = usage.get("prompt_tokens_details") or {}
        acc["cached"] += int(details.get("cached_tokens", 0))
        choices = body.get("choices") or []
        if choices and cid and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            if not isinstance(msg, dict):
                msg = {}
            out[cid] = msg.get("content") or ""
    return out, acc


def _openai_download_batch_file_text(file_id: str | None) -> str:
    if not file_id:
        return ""
    file_resp = openai_client.files.content(file_id)
    return file_resp.text if hasattr(file_resp, "text") else file_resp.read().decode("utf-8")


def _openai_log_batch_error_file_summary(error_file_id: str | None) -> None:
    raw = _openai_download_batch_file_text(error_file_id)
    if not raw.strip():
        return
    n = sum(1 for line in raw.splitlines() if line.strip())
    tqdm.write(
        f"  Batch error_file: {n} line(s). Missing output lines fall back to sync extraction."
    )


def _estimate_openai_phase2_batch_cost_usd(acc: dict[str, int]) -> float:
    """gpt-4o-mini list rates with Batch API 50% discount."""
    pin, pcached, pout = 0.15 * 0.5, 0.075 * 0.5, 0.60 * 0.5
    non_cached = max(0, acc["prompt"] - acc["cached"])
    return (non_cached / 1e6) * pin + (acc["cached"] / 1e6) * pcached + (acc["completion"] / 1e6) * pout


def _run_openai_extraction_batch_chunk(
    jobs: list[dict], chunk_idx: int
) -> tuple[dict[str, str], dict[str, int], str]:
    input_path = CHECKPOINT_DIR / f"openai_batch_extraction_input_{chunk_idx}.jsonl"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    nbytes = write_openai_batch_jsonl(input_path, jobs)

    tqdm.write(
        f"  Uploading OpenAI batch chunk {chunk_idx}: {len(jobs):,} jobs, "
        f"{nbytes / (1024 * 1024):.2f} MiB → {input_path.name}"
    )
    with open(input_path, "rb") as fp:
        uploaded = openai_client.files.create(file=fp, purpose="batch")

    batch_job = openai_batches_create_safe(
        openai_client,
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"phase": "phase2_extraction", "chunk": str(chunk_idx)},
    )
    tqdm.write(
        f"  batch_id={batch_job.id}; polling every {OPENAI_BATCH_POLL_SEC}s "
        "(https://developers.openai.com/api/docs/guides/batch )"
    )

    terminal = {"completed", "failed", "expired", "cancelled"}
    while batch_job.status not in terminal:
        time.sleep(OPENAI_BATCH_POLL_SEC)
        batch_job = openai_client.batches.retrieve(batch_job.id)
        rc = batch_job.request_counts
        tqdm.write(
            f"    status={batch_job.status}  completed={rc.completed}/{rc.total}  failed={rc.failed}"
        )

    if batch_job.status == "expired" and batch_job.output_file_id:
        tqdm.write(
            "  NOTE: Batch status=expired — reading partial output_file (per Batch API rules)."
        )
    elif batch_job.status != "completed":
        raise RuntimeError(
            f"OpenAI batch ended with status={batch_job.status!r}. "
            "Check dashboard and error_file_id for per-request failures."
        )
    if not batch_job.output_file_id:
        raise RuntimeError(f"No output_file_id for batch status={batch_job.status!r}.")

    raw_text = _openai_download_batch_file_text(batch_job.output_file_id)
    _openai_log_batch_error_file_summary(getattr(batch_job, "error_file_id", None))
    mapping, acc = _openai_parse_batch_output_jsonl(raw_text)
    return mapping, acc, batch_job.id


def _run_openai_extraction_batch(jobs: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    if not jobs:
        return {}, {"prompt": 0, "completion": 0, "cached": 0}

    max_r = OPENAI_BATCH_MAX_REQUESTS
    file_cap = openai_batch_max_file_bytes()
    chunks = chunk_jobs_for_openai_batch(
        jobs, max_requests=max_r, max_file_bytes=file_cap
    )
    n_chunks = len(chunks)
    combined: dict[str, str] = {}
    acc_total = {"prompt": 0, "completion": 0, "cached": 0}

    if n_chunks > 1:
        tqdm.write(
            f"  Splitting {len(jobs):,} requests into {n_chunks} batch file(s) "
            f"(≤{max_r:,} lines each, ≤{file_cap / (1024 * 1024):.0f} MiB UTF-8 per file)."
        )

    for ci, chunk in enumerate(chunks):
        m, a, _bid = _run_openai_extraction_batch_chunk(chunk, ci)
        combined.update(m)
        for k in acc_total:
            acc_total[k] += a[k]

    tqdm.write(
        f"  Batch token totals (all chunks): prompt={acc_total['prompt']:,}, "
        f"cached_prompt={acc_total['cached']:,}, completion={acc_total['completion']:,}"
    )
    est = _estimate_openai_phase2_batch_cost_usd(acc_total)
    tqdm.write(f"  Approx. extraction batch cost (50% tier, list rates): ${est:.2f} USD")
    return combined, acc_total


def call_llm(
    prompt: str,
    system_prompt: str = None,
    retries: int = MAX_RETRIES,
    *,
    openai_prompt_cache_key: str | None = None,
) -> str:
    """
    Generic LLM caller that works for Gemini, OpenAI, and Qwen (local).
    For OpenAI, pass static instructions as system_prompt and variable text as prompt
    so prompt caching can reuse the system prefix across calls.
    """
    for attempt in range(retries):
        try:
            if LLM_PROVIDER == "gemini":
                full_prompt = f"{system_prompt or ''}\n\n{prompt}" if system_prompt else prompt
                response = model.generate_content(full_prompt)
                return response.text

            elif LLM_PROVIDER == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                create_kw = dict(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    stream=False,
                )
                ck = (
                    openai_prompt_cache_key
                    if openai_prompt_cache_key is not None
                    else OPENAI_PROMPT_CACHE_KEY
                )
                _openai_apply_prompt_caching(create_kw, ck)
                response = openai_client.chat.completions.create(**create_kw)
                _maybe_log_openai_usage(response)
                return response.choices[0].message.content

            elif LLM_PROVIDER == "qwen":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = qwen_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                return response.choices[0].message.content

        except Exception as e:
            err_msg = str(e)
            print(f"    API error (attempt {attempt+1}/{retries}): {err_msg[:120]}")

            if "429" in err_msg or "rate limit" in err_msg.lower() or "quota" in err_msg.lower():
                sleep_time = min(60, 10 * (2 ** attempt))
                print(f"    Rate limited. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                time.sleep(2 ** attempt)

    raise Exception(f"All {retries} LLM call attempts failed")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_schema(schema_path: str) -> dict:
    """
    Load the feature schema produced by Phase 1.
    New format: feature_categories is a list of {category, description, features: [values]}.
    Feature names are constructed as {category}_{value}.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    raw = schema.get("feature_categories")
    if not isinstance(raw, list):
        print(f"  WARNING: feature_categories is not a list (got {type(raw).__name__}); using [].")
        raw = []

    n_total = 0
    for entry in raw:
        if isinstance(entry, dict) and isinstance(entry.get("features"), list):
            n_total += len(entry["features"])

    print(f"  Loaded schema: {n_total} features across {len(raw)} subcategories")
    print(f"  Feature breakdown by subcategory:")
    for entry in raw:
        if isinstance(entry, dict):
            cat = entry.get("category", "?")
            feats = entry.get("features", [])
            print(f"    - {cat}: {len(feats)}")

    return schema


def get_all_feature_names(schema: dict) -> list[str]:
    """
    Build flat list of feature names as {category}_{value} from the schema.
    Order: sorted by category, then sorted values within each category.
    """
    names: list[str] = []
    for entry in schema.get("feature_categories", []):
        if not isinstance(entry, dict):
            continue
        cat = str(entry.get("category", "other"))
        for val in entry.get("features", []):
            v = str(val).strip()
            if v:
                names.append(f"{cat}_{v}")
    return names


def get_feature_categories(schema: dict) -> dict[str, str]:
    """Get mapping of full feature name -> subcategory."""
    m: dict[str, str] = {}
    for entry in schema.get("feature_categories", []):
        if not isinstance(entry, dict):
            continue
        cat = str(entry.get("category", "other"))
        for val in entry.get("features", []):
            v = str(val).strip()
            if v:
                m[f"{cat}_{v}"] = cat
    return m


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY GROUPING (for category-aware encoding inference)
# ══════════════════════════════════════════════════════════════════════════════

def build_category_groups(
    all_feature_names: list[str],
    feature_categories: dict[str, str],
) -> dict[str, list[int]]:
    """
    Build category groups for encoding inference rules.
    With the new schema, subcategories are already fine-grained (e.g. morphology_color,
    symptoms_dermatological), so we simply group by subcategory directly.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(all_feature_names):
        cat = feature_categories.get(name, "other")
        groups[cat].append(i)
    return dict(groups)


# ══════════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION — SYSTEM PROMPT (indices + sparse + category-aware encoding)
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_system_prompt(
    all_feature_names: list[str],
    feature_categories: dict[str, str],
) -> str:
    """
    Build the extraction system prompt with:
    - Numeric feature index map (0..N-1)
    - Category groups for encoding inference
    - Category-aware 0/1/2 encoding rules
    - Sparse output format: {"1":[indices], "2":[indices]}
    """
    n_feat = len(all_feature_names)

    index_lines = [f"{i}: {name}" for i, name in enumerate(all_feature_names)]
    index_map_str = "\n".join(index_lines)

    cat_groups = build_category_groups(all_feature_names, feature_categories)
    group_lines = []
    for gname, indices in sorted(cat_groups.items()):
        sample_names = [all_feature_names[i] for i in indices[:4]]
        suffix = ", ..." if len(indices) > 4 else ""
        group_lines.append(
            f"  {gname} (indices {indices[0]}-{indices[-1]}, {len(indices)} features): "
            f"{', '.join(sample_names)}{suffix}"
        )
    groups_str = "\n".join(group_lines)

    return f"""You are a clinical dermatology NLP specialist performing sparse one-hot feature extraction from skin disease captions.

FEATURE INDEX MAP ({n_feat} features — reference by index number only):
{index_map_str}

CATEGORY GROUPS (for encoding inference):
{groups_str}

=== ENCODING VALUES ===
  1 = PRESENT — feature is explicitly mentioned or clearly described in the caption
  0 = ABSENT  — feature is either explicitly negated OR inferably absent (DEFAULT for unlisted indices)
  2 = TRULY UNKNOWN — ONLY when the caption provides ZERO information about the feature's entire category group

=== CATEGORY-AWARE INFERENCE (CRITICAL) ===
When a caption mentions ANY feature or features within a category group encode them as 1, ALL OTHER features in that SAME group
must be 0 (inferably absent), NOT 2. Use 2 ONLY when the caption says NOTHING AT ALL about an
entire category group.

Examples:
- "red rash on face" → color_red=1, ALL other morphology_color features=0, location_face=1,
  ALL other body_location features=0. If no symptoms mentioned at all → all symptom features=2.
- "itchy raised lesion" → symptom_itching=1, ALL other symptoms_dermatological=0,
  texture_raised=1, ALL other morphology_texture=0. If no body location mentioned → all body_location=2.
- "non-itchy" → symptom_itching=0 (explicitly absent), other symptoms_dermatological=0.

=== OUTPUT FORMAT (SPARSE — minimizes tokens) ===
For EACH caption, output a JSON object with exactly two keys:
  "1": [list of feature INDICES that are PRESENT]
  "2": [list of feature INDICES that are TRULY UNKNOWN]
All indices NOT listed default to 0 (absent).

Return a JSON array with one object per input caption. Example for 2 captions:
[
  {{"1": [0, 8, 45, 88], "2": [150, 155, 160]}},
  {{"1": [3, 12], "2": []}}
]

RULES:
- Each index must be an integer in range 0 to {n_feat - 1}
- An index must NOT appear in both "1" and "2"
- If ALL features in a category group are unknown, list those indices in "2"
- No preamble, no markdown fences, no explanations — ONLY the JSON array
"""


def build_extraction_user_prompt(truncated_captions: list[str]) -> str:
    """Variable user message: numbered captions only (system holds full extraction rules)."""
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(truncated_captions))


# ══════════════════════════════════════════════════════════════════════════════
# SPARSE OUTPUT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _expand_sparse_to_dict(
    sparse: dict,
    all_feature_names: list[str],
) -> dict[str, int]:
    """
    Expand a sparse object {"1": [indices], "2": [indices]} into a full
    {feature_name: 0|1|2} dict.  Unlisted indices default to 0.
    """
    n = len(all_feature_names)
    result = {name: 0 for name in all_feature_names}

    present = sparse.get("1") or sparse.get(1) or []
    unknown = sparse.get("2") or sparse.get(2) or []

    if not isinstance(present, list):
        present = []
    if not isinstance(unknown, list):
        unknown = []

    for idx in present:
        idx = int(idx)
        if 0 <= idx < n:
            result[all_feature_names[idx]] = 1

    for idx in unknown:
        idx = int(idx)
        if 0 <= idx < n:
            result[all_feature_names[idx]] = 2

    return result


def parse_extraction_response_text(
    text: str,
    captions: list[str],
    all_feature_names: list[str],
) -> tuple[list[dict], dict]:
    """
    Parse sparse JSON array from LLM into per-caption feature dicts.
    Expected format: [{"1":[...], "2":[...]}, ...]
    Returns (results, stats).
    """
    stats: dict = {"success": False, "validation_fixes": 0}
    n_feat = len(all_feature_names)
    fallback_row = {k: 2 for k in all_feature_names}

    try:
        text = (text or "").strip()
        text = re.sub(r"```json\s*|```\s*", "", text).strip()
        parsed = json.loads(text)

        if isinstance(parsed, dict) and ("1" in parsed or "2" in parsed or 1 in parsed or 2 in parsed):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

        results: list[dict] = []
        fixes = 0
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                results.append(dict(fallback_row))
                fixes += 1
                continue

            present = item.get("1") or item.get(1) or []
            unknown = item.get("2") or item.get(2) or []
            if not isinstance(present, list):
                present = []
            if not isinstance(unknown, list):
                unknown = []

            overlap = set(int(x) for x in present if _is_valid_idx(x, n_feat)) & \
                       set(int(x) for x in unknown if _is_valid_idx(x, n_feat))
            if overlap:
                for idx in overlap:
                    unknown = [x for x in unknown if int(x) != idx]
                fixes += 1

            results.append(_expand_sparse_to_dict(item, all_feature_names))

        if len(results) < len(captions):
            results.extend([dict(fallback_row)] * (len(captions) - len(results)))
            fixes += len(captions) - len(parsed)
        results = results[:len(captions)]

        stats["validation_fixes"] = fixes
        stats["success"] = True
        return results, stats

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"    parse_extraction_response_text: {str(e)[:120]}")
        stats["success"] = False
        return [dict(fallback_row) for _ in captions], stats


def _is_valid_idx(val, n: int) -> bool:
    try:
        v = int(val)
        return 0 <= v < n
    except (ValueError, TypeError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SYNC EXTRACTION (single batch LLM call with retries)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_batch(
    captions: list[str],
    all_feature_names: list[str],
    system_prompt: str,
    retries: int = MAX_RETRIES,
) -> tuple[list[dict], dict]:
    """
    Batch LLM call for one chunk of captions.
    Returns (list of feature dicts aligned to input captions, stats dict).
    """
    truncated = [c[:MAX_CAPTION_LEN] for c in captions]
    user_prompt = build_extraction_user_prompt(truncated)

    stats = {
        "attempts": 0,
        "success": False,
        "validation_fixes": 0,
    }

    for attempt in range(retries):
        try:
            stats["attempts"] = attempt + 1
            text = call_llm(user_prompt, system_prompt, retries=1)
            valid_results, pst = parse_extraction_response_text(text, captions, all_feature_names)
            stats["validation_fixes"] = pst.get("validation_fixes", 0)
            if pst.get("success"):
                stats["success"] = True
                return valid_results, stats
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt+1}): {str(e)[:80]}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            err_str = str(e)
            print(f"    LLM attempt {attempt+1} failed: {err_str[:120]}")
            if attempt < retries - 1:
                if "429" in err_str or "quota" in err_str.lower():
                    time.sleep(10 * (attempt + 1))
                else:
                    time.sleep(2 ** attempt)

    print(f"    ERROR: All {retries} attempts failed. Returning unknown values.")
    fallback = [{k: 2 for k in all_feature_names} for _ in captions]
    return fallback, stats


# ══════════════════════════════════════════════════════════════════════════════
# DEDUP EXTRACTION PIPELINES (Batch API + Sync)
# ══════════════════════════════════════════════════════════════════════════════

def _run_dedup_openai_batch(
    unique_captions: list[str],
    all_feature_names: list[str],
    system_prompt: str,
    global_stats: dict,
) -> dict[str, dict]:
    """
    Process deduplicated captions via OpenAI Batch API.
    Returns {caption_text: {feature_name: value}}.
    """
    jobs: list[dict] = []
    job_batches: dict[str, list[str]] = {}

    for batch_idx, batch_start in enumerate(range(0, len(unique_captions), LLM_BATCH_SIZE)):
        batch = unique_captions[batch_start : batch_start + LLM_BATCH_SIZE]
        truncated = [c[:MAX_CAPTION_LEN] for c in batch]
        user_prompt = build_extraction_user_prompt(truncated)
        cid = f"p2_{batch_idx}"
        jobs.append(
            {"custom_id": cid, "body": _openai_extraction_chat_body(system_prompt, user_prompt)}
        )
        job_batches[cid] = batch

    global_stats["total_batches"] = len(jobs)
    global_stats["total_api_calls"] = len(jobs)

    mapping, _acc = _run_openai_extraction_batch(jobs)

    results: dict[str, dict] = {}
    for cid, batch in job_batches.items():
        text = mapping.get(cid) or ""
        batch_results, pst = parse_extraction_response_text(text, batch, all_feature_names)
        ok = pst.get("success", False)
        global_stats["total_validation_fixes"] += pst.get("validation_fixes", 0)

        if not ok:
            tqdm.write(f"    Sync fallback for batch {cid}")
            batch_results, st2 = extract_features_batch(batch, all_feature_names, system_prompt)
            ok = st2.get("success", False)
            global_stats["total_retries"] += 1
            global_stats["total_validation_fixes"] += st2.get("validation_fixes", 0)

        if ok:
            global_stats["successful_batches"] += 1
        else:
            global_stats["failed_batches"] += 1

        for cap, feats in zip(batch, batch_results):
            results[cap] = feats

    return results


def _run_dedup_sync(
    unique_captions: list[str],
    all_feature_names: list[str],
    system_prompt: str,
    global_stats: dict,
    ckpt_save_fn,
) -> dict[str, dict]:
    """
    Process deduplicated captions via synchronous LLM calls.
    Saves checkpoint after every 50 batches via ckpt_save_fn.
    Returns {caption_text: {feature_name: value}}.
    """
    results: dict[str, dict] = {}
    n_batches = (len(unique_captions) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
    batch_count = 0

    for batch_start in tqdm(
        range(0, len(unique_captions), LLM_BATCH_SIZE),
        desc="Extracting features (sync)",
        total=n_batches,
    ):
        batch = unique_captions[batch_start : batch_start + LLM_BATCH_SIZE]
        batch_results, stats = extract_features_batch(batch, all_feature_names, system_prompt)

        for cap, feats in zip(batch, batch_results):
            results[cap] = feats

        global_stats["total_batches"] += 1
        global_stats["total_api_calls"] += stats.get("attempts", 1)
        global_stats["total_retries"] += max(0, stats.get("attempts", 1) - 1)
        global_stats["total_validation_fixes"] += stats.get("validation_fixes", 0)
        if stats.get("success"):
            global_stats["successful_batches"] += 1
        else:
            global_stats["failed_batches"] += 1

        batch_count += 1
        if batch_count % 50 == 0:
            ckpt_save_fn(results)

        time.sleep(RATE_LIMIT_SLEEP)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_dedup_checkpoint(
    caption_features: dict[str, dict],
    all_feature_names: list[str],
    ckpt_path: Path,
) -> None:
    """Save caption->values checkpoint as compact {caption: [int, ...]} JSON."""
    compact: dict[str, list[int]] = {}
    for cap, feats in caption_features.items():
        compact[cap] = [feats.get(fn, 2) for fn in all_feature_names]
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False)


def _load_dedup_checkpoint(
    ckpt_path: Path,
    all_feature_names: list[str],
) -> dict[str, dict]:
    """Load checkpoint back to {caption: {feature_name: value}}."""
    with open(ckpt_path, "r", encoding="utf-8") as f:
        compact = json.load(f)
    result: dict[str, dict] = {}
    for cap, vals in compact.items():
        if len(vals) == len(all_feature_names):
            result[cap] = dict(zip(all_feature_names, vals))
        else:
            result[cap] = {fn: (vals[i] if i < len(vals) else 2) for i, fn in enumerate(all_feature_names)}
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(csv_path: str, schema_path: str):
    print("=" * 70)
    print("  PHASE 2: Bulk Feature Extraction (Cost-Optimized)")
    print("=" * 70 + "\n")

    # ── Load schema ───────────────────────────────────────────────────────────
    print("Loading feature schema...")
    schema = load_schema(schema_path)
    all_feature_names = get_all_feature_names(schema)
    feature_categories = get_feature_categories(schema)

    print(f"\n  Total features to extract: {len(all_feature_names)}")
    print(f"  Batch size: {LLM_BATCH_SIZE} captions per API call")
    print(f"  Output format: sparse JSON (indices only)\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if CAPTION_COLUMN not in df.columns:
        raise ValueError(
            f"Caption column {CAPTION_COLUMN!r} not found. "
            f"Available: {list(df.columns)}"
        )
    meta_needed = ["image", "label_name", "disease_label"]
    missing_meta = [c for c in meta_needed if c not in df.columns]
    if missing_meta:
        raise ValueError(
            f"CSV missing required column(s) {missing_meta}. Available: {list(df.columns)}"
        )
    df[CAPTION_COLUMN] = df[CAPTION_COLUMN].fillna("").astype(str)
    n = len(df)
    print(f"  {n:,} records loaded (caption={CAPTION_COLUMN!r})\n")

    # ── Deduplicate captions ──────────────────────────────────────────────────
    print("Deduplicating captions...")
    stripped = df[CAPTION_COLUMN].str.strip()
    unique_caps_series = stripped[stripped != ""].unique()
    unique_caps_list: list[str] = list(unique_caps_series)
    n_unique = len(unique_caps_list)
    n_empty = int((stripped == "").sum())
    n_dupes = n - n_unique - n_empty

    print(f"  Total rows: {n:,}")
    print(f"  Unique non-empty captions: {n_unique:,}")
    print(f"  Duplicate rows saved: {n_dupes:,} ({100 * n_dupes / max(1, n):.1f}%)")
    if n_empty:
        print(f"  Empty captions (all features=2): {n_empty:,}")
    print()

    # ── Build system prompt ───────────────────────────────────────────────────
    system_prompt = build_llm_system_prompt(all_feature_names, feature_categories)

    # ── Load checkpoint if exists ─────────────────────────────────────────────
    dedup_ckpt = CHECKPOINT_DIR / DEDUP_CKPT_FILE
    caption_features: dict[str, dict] = {}
    if dedup_ckpt.exists():
        try:
            caption_features = _load_dedup_checkpoint(dedup_ckpt, all_feature_names)
            print(f"  Loaded {len(caption_features):,} caption results from checkpoint")
        except Exception as e:
            print(f"  WARNING: checkpoint load failed ({e}), re-extracting all")
            caption_features = {}

    remaining = [c for c in unique_caps_list if c not in caption_features]

    # ── Statistics tracking ───────────────────────────────────────────────────
    global_stats = {
        "caption_column": CAPTION_COLUMN,
        "total_rows": n,
        "unique_captions": n_unique,
        "duplicates_saved": n_dupes,
        "empty_captions": n_empty,
        "captions_from_checkpoint": len(caption_features),
        "captions_to_process": len(remaining),
        "total_batches": 0,
        "successful_batches": 0,
        "failed_batches": 0,
        "total_api_calls": 0,
        "total_retries": 0,
        "total_validation_fixes": 0,
        "start_time": time.time(),
        "openai_use_batch": bool(LLM_PROVIDER == "openai" and OPENAI_USE_BATCH),
        "openai_prompt_cache_key": OPENAI_PROMPT_CACHE_KEY if LLM_PROVIDER == "openai" else "",
        "output_format": "sparse",
    }

    # ── Run extraction on remaining unique captions ───────────────────────────
    if remaining:
        n_api_calls_est = (len(remaining) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
        print(f"Processing {len(remaining):,} unique captions in ~{n_api_calls_est:,} API calls...")
        print(f"  Provider: {LLM_PROVIDER} | Model: {MODEL_NAME}\n")

        def _ckpt_saver(results):
            merged = dict(caption_features)
            merged.update(results)
            _save_dedup_checkpoint(merged, all_feature_names, dedup_ckpt)

        if LLM_PROVIDER == "openai" and OPENAI_USE_BATCH:
            new_results = _run_dedup_openai_batch(
                remaining, all_feature_names, system_prompt, global_stats
            )
        else:
            new_results = _run_dedup_sync(
                remaining, all_feature_names, system_prompt, global_stats, _ckpt_saver
            )

        caption_features.update(new_results)
        _save_dedup_checkpoint(caption_features, all_feature_names, dedup_ckpt)
        print(f"  Checkpoint saved: {len(caption_features):,} unique caption results\n")
    else:
        print("  All unique captions already in checkpoint — skipping extraction.\n")

    # ── Replicate to all rows ─────────────────────────────────────────────────
    elapsed_time = time.time() - global_stats["start_time"]
    global_stats["elapsed_seconds"] = elapsed_time
    global_stats["elapsed_formatted"] = f"{elapsed_time / 3600:.1f} hours"

    print(f"  Extraction complete!")
    print(f"  Time elapsed: {elapsed_time / 60:.1f} minutes ({elapsed_time / 3600:.2f} hours)")
    print(f"  Total API calls: {global_stats['total_api_calls']:,}")
    print(f"  Total retries: {global_stats['total_retries']:,}")
    print(f"  Validation fixes: {global_stats['total_validation_fixes']:,}")
    print(f"  Failed batches: {global_stats['failed_batches']:,}")

    print("\nReplicating features to all rows (including duplicates)...")
    fallback_row = {k: 2 for k in all_feature_names}
    feature_rows: list[dict] = []
    for cap in stripped:
        if cap and cap in caption_features:
            feature_rows.append(caption_features[cap])
        else:
            feature_rows.append(fallback_row)

    features_df = pd.DataFrame(feature_rows, columns=all_feature_names)

    # ── Combine with original columns and save ────────────────────────────────
    print("Creating output CSV...")
    desired_meta = ["image", "image_path", "label", "label_name", CAPTION_COLUMN, "disease_label"]
    meta_cols = [c for c in desired_meta if c in df.columns]
    final_df = pd.concat(
        [df[meta_cols].reset_index(drop=True), features_df.reset_index(drop=True)],
        axis=1,
    )

    for col in all_feature_names:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
            final_df[col] = final_df[col].fillna(2).astype(int)
            final_df[col] = final_df[col].clip(0, 2)

    final_df.to_csv(OUTPUT_CSV, index=False)

    with open(STATS_FILE, "w") as f:
        json.dump(global_stats, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  OUTPUT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Feature matrix saved: {OUTPUT_CSV}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Columns: {', '.join(meta_cols)} + {len(all_feature_names)} feature columns")
    print(f"  Total rows: {len(final_df):,} (all original rows, duplicates replicated)")
    print(f"  Unique captions processed: {n_unique:,}")
    print(f"  Statistics saved: {STATS_FILE}")

    print(f"\n  Feature value distribution (sample of first 10):")
    for col in all_feature_names[:10]:
        vc = final_df[col].value_counts().to_dict()
        present = vc.get(1, 0)
        absent = vc.get(0, 0)
        unknown = vc.get(2, 0)
        pct_present = 100 * present / len(final_df)
        print(
            f"    {col:40s}: present={present:6,} ({pct_present:4.1f}%), "
            f"absent={absent:6,}, unknown={unknown:6,}"
        )

    if len(all_feature_names) > 10:
        print(f"    ... and {len(all_feature_names) - 10} more features")

    return final_df, global_stats


if __name__ == "__main__":
    result_df, stats = run_extraction(CSV_PATH, SCHEMA_PATH)

    print("\n" + "=" * 70)
    print("Extraction complete!")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"Statistics saved to: {STATS_FILE}")
    print("=" * 70)
