"""
PHASE 2: Bulk Feature Extraction (LLM ONLY - SLOW LANE)
=======================================================
Strategy:
  - SLOW LANE ONLY: All features extracted via LLM (OpenAI GPT-4o-mini, Gemini,
    or Qwen 3.5 9B served locally) for every caption in the dataset.
  - No rule-based extraction - everything goes through the LLM for consistency.
  - Organized per label_name with checkpointing.
  - Encoding: 0 = explicitly absent, 1 = present, 2 = unknown/not mentioned.
  - Output: New CSV with all feature columns added (one-hot encoded).
  - Includes validation, statistics tracking, and robust error handling.

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
  OPENAI_BATCH_MAX_REQUESTS — max lines per batch file (default 50000, API cap; larger runs split into multiple batches)
  OPENAI_BATCH_MAX_FILE_BYTES — max UTF-8 bytes per batch JSONL (default ~195 MiB; API cap 200 MB)
  OPENAI_LOG_USAGE — 1/true: print token usage when available

Prompt caching: static system prompt (full schema) first, numbered captions in user message last.
  See https://developers.openai.com/api/docs/guides/prompt-caching (~1024+ token prefixes; OPENAI_PROMPT_CACHE_KEY).
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
# Options: "gemini", "openai", or "qwen"
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "openai")  # Default to openai for cost efficiency

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
LLM_BATCH_SIZE = _safe_int_env("LLM_BATCH_SIZE", 25, vmin=1)
MAX_CAPTION_LEN  = 500             # truncate before sending to LLM (chars)
MAX_RETRIES      = 5               # max retries per batch
RATE_LIMIT_SLEEP = 0.1 if LLM_PROVIDER == "qwen" else 0.5

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

def _normalize_schema_feature_categories(items: list | None) -> list[dict]:
    """
    Phase 1 may emit string entries in feature_categories (some local LLMs).
    Phase 2 requires dicts with string names.
    """
    if not items:
        return []
    out: list[dict] = []
    for feat in items:
        if isinstance(feat, dict):
            raw_name = feat.get("name")
            name_s = str(raw_name).strip() if raw_name is not None else ""
            if not name_s:
                continue
            d = dict(feat)
            d["name"] = name_s
            out.append(d)
        elif isinstance(feat, str):
            s = feat.strip()
            if s:
                out.append(
                    {
                        "name": s,
                        "category": "other",
                        "description": "",
                        "example_values": [],
                        "is_binary": True,
                    }
                )
    return out


def load_schema(schema_path: str) -> dict:
    """Load the feature schema produced by Phase 1."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    raw = schema.get("feature_categories")
    if not isinstance(raw, list):
        print(f"  WARNING: feature_categories is not a list (got {type(raw).__name__}); using [].")
        raw = []
    n_raw = len(raw)
    features = _normalize_schema_feature_categories(raw)
    schema["feature_categories"] = features
    if n_raw and len(features) < n_raw:
        print(
            f"  WARNING: dropped {n_raw - len(features)} invalid feature_categories entr(y/ies)"
        )
    print(f"  Loaded schema: {len(features)} features")

    # Show feature breakdown by category
    categories = defaultdict(int)
    for feat in features:
        cat = feat.get("category", "other") if isinstance(feat, dict) else "other"
        categories[str(cat)] += 1

    print(f"  Feature breakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count}")

    return schema


def get_all_feature_names(schema: dict) -> list[str]:
    """Get list of all feature names from the schema."""
    names: list[str] = []
    for f in schema.get("feature_categories", []):
        if isinstance(f, dict) and f.get("name"):
            names.append(str(f["name"]))
    return names


def get_feature_categories(schema: dict) -> dict[str, str]:
    """Get mapping of feature name to category."""
    m: dict[str, str] = {}
    for f in schema.get("feature_categories", []):
        if isinstance(f, dict) and f.get("name"):
            m[str(f["name"])] = str(f.get("category", "other"))
    return m


# ══════════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION - SLOW LANE (ALL FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_system_prompt(all_feature_names: list[str], feature_categories: dict) -> str:
    """Build the extraction prompt for ALL features from the schema."""
    feature_list_str = json.dumps(all_feature_names, indent=2)
    
    # Group features by category for better organization
    features_by_cat = defaultdict(list)
    for feat in all_feature_names:
        cat = feature_categories.get(feat, "other")
        features_by_cat[cat].append(feat)
    
    cat_summary = "\n".join([
        f"  {cat}: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}"
        for cat, features in sorted(features_by_cat.items())
    ])
    
    return f"""You are a clinical dermatology NLP specialist performing one-hot encoding feature extraction.

Your task: Extract ALL of the following features from each skin disease caption and return ONE-HOT ENCODED values.

FEATURE LIST ({len(all_feature_names)} total):
{feature_list_str}

FEATURES BY CATEGORY:
{cat_summary}

=== ONE-HOT ENCODING RULES (STRICT) ===
For EACH feature, you MUST return one of these exact values:

  1 = PRESENT/MENTIONED
      - Use when the caption explicitly states the feature is present
      - Examples: "itchy rash" → symptom_itching=1, "raised lesion" → texture_raised=1
      - "red patches" → color_red=1, "on the face" → location_face=1

  0 = EXPLICITLY ABSENT
      - Use ONLY when the caption explicitly states the feature is NOT present
      - Examples: "non-itchy", "no fever", "not raised", "absence of pain"
      - This is rare - most features will be 1 or 2

  2 = UNKNOWN/NOT MENTIONED (default)
      - Use when the caption says nothing about this feature
      - This is the DEFAULT for most features
      - If you're unsure, use 2

=== EXTRACTION GUIDELINES ===
1. EXTRACT ONLY what is ACTUALLY STATED - do not infer or guess
2. Look for EXACT phrases and keywords in the caption
3. For body locations: extract ALL locations mentioned (face AND arm → both=1)
4. For symptoms: extract ALL symptoms mentioned
5. For morphology: note color, texture, shape, size descriptors
6. For treatments: note any medications or therapies mentioned
7. For triggers: note sun, stress, allergens, infections, etc.
8. For demographics: extract age, sex, skin type if mentioned

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array with EXACTLY one object per caption.
Each object MUST contain ALL {len(all_feature_names)} feature keys with values 0, 1, or 2.

Example output for 2 captions:
[
  {{"symptom_itching": 1, "texture_raised": 1, "color_red": 1, "location_face": 1, ...}},
  {{"symptom_itching": 2, "texture_raised": 0, "color_red": 2, "location_face": 2, ...}}
]

No preamble, no markdown fences, no explanations - ONLY the JSON array.
"""


def build_extraction_user_prompt(truncated_captions: list[str]) -> str:
    """Variable user message: numbered captions only (system holds full extraction rules)."""
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(truncated_captions))


def parse_extraction_response_text(
    text: str,
    captions: list[str],
    all_feature_names: list[str],
) -> tuple[list[dict], dict]:
    """
    Parse LLM JSON array into per-caption feature dicts.
    Returns (results, stats) with stats keys: success, validation_fixes (mirrors extract_features_batch).
    """
    stats = {"success": False, "validation_fixes": 0}
    try:
        text = (text or "").strip()
        text = re.sub(r"```json\s*|```\s*", "", text).strip()
        parsed = json.loads(text)

        if not isinstance(parsed, list):
            if isinstance(parsed, dict) and "feature_categories" in parsed:
                parsed = [{} for _ in range(len(captions))]
            else:
                raise ValueError(f"Expected list, got {type(parsed).__name__}")

        valid_results, num_fixed = validate_batch_results(
            parsed, all_feature_names, len(captions)
        )
        stats["validation_fixes"] = num_fixed

        if len(valid_results) != len(captions):
            if len(valid_results) < len(captions):
                fallback = {k: 2 for k in all_feature_names}
                valid_results.extend([fallback] * (len(captions) - len(valid_results)))
            valid_results = valid_results[: len(captions)]

        stats["success"] = True
        return valid_results, stats
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"    parse_extraction_response_text: {str(e)[:100]}")
        stats["success"] = False
        fallback = [{k: 2 for k in all_feature_names} for _ in captions]
        return fallback, stats


def validate_batch_results(
    results: list[dict], 
    expected_features: list[str], 
    batch_size: int
) -> tuple[list[dict], int]:
    """
    Validate that batch results contain all expected features.
    Returns (valid_results, num_fixed).
    """
    valid_results = []
    num_fixed = 0
    
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            # Invalid result type - create fallback
            valid_results.append({k: 2 for k in expected_features})
            num_fixed += 1
            continue
        
        # Check for missing features
        missing_features = set(expected_features) - set(result.keys())
        
        if missing_features:
            # Add missing features with value 2 (unknown)
            for feat in missing_features:
                result[feat] = 2
            num_fixed += 1
        
        # Check for extra features (keep them but log if needed)
        extra_features = set(result.keys()) - set(expected_features)
        if extra_features:
            # Remove extra features not in schema
            for feat in list(extra_features):
                del result[feat]
        
        # Validate values are 0, 1, or 2 (coerce bool / float from JSON)
        for feat in expected_features:
            val = result.get(feat, 2)
            if isinstance(val, bool):
                val = int(val)
                result[feat] = val
            elif isinstance(val, (int, float)) and float(val).is_integer():
                val = int(val)
                result[feat] = val
            if val not in (0, 1, 2):
                result[feat] = 2  # Default to unknown for invalid values
        
        valid_results.append(result)
    
    return valid_results, num_fixed


def extract_features_batch(
    captions: list[str],
    all_feature_names: list[str],
    system_prompt: str,
    retries: int = MAX_RETRIES,
) -> tuple[list[dict], dict]:
    """
    Batch LLM call. Returns (list of dicts aligned to input captions, stats dict).
    OpenAI/Qwen: system = static schema/rules, user = numbered captions (prompt caching on OpenAI).
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


def _phase2_run_openai_batch_extraction(
    df: pd.DataFrame,
    label_names,
    all_feature_names: list[str],
    system_prompt: str,
    all_results_list: list,
    global_stats: dict,
) -> None:
    """
    Collect all non-checkpointed caption batches into one OpenAI Batch job, then
    write per-label checkpoints and fill all_results_list.
    """
    jobs: list[dict] = []
    job_meta: dict[str, dict] = {}
    job_seq = 0

    for label_idx, label_name in enumerate(tqdm(label_names, desc="Collecting OpenAI batch jobs"), 1):
        safe_name = re.sub(r"[^\w\-]", "_", str(label_name))[:60]
        ckpt_path = CHECKPOINT_DIR / f"llm_{safe_name}.json"

        label_mask = df["label_name"] == label_name
        label_indices = df.index[label_mask].tolist()
        label_captions = df.loc[label_indices, CAPTION_COLUMN].tolist()
        n_label = len(label_captions)
        n_batches_label = (n_label + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE

        if ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    label_results = json.load(f)
                if len(label_results) == len(label_indices):
                    print(
                        f"  [{label_idx}/{len(label_names)}] {label_name}: "
                        f"Loaded {len(label_results):,} results from checkpoint"
                    )
                    for idx, result in zip(label_indices, label_results):
                        all_results_list[idx] = result
                    global_stats["labels_skipped"].append(label_name)
                    continue
                print(
                    f"  [{label_idx}/{len(label_names)}] {label_name}: "
                    f"Checkpoint incomplete, re-extracting..."
                )
            except Exception as e:
                print(
                    f"  [{label_idx}/{len(label_names)}] {label_name}: "
                    f"Checkpoint error {e}, re-extracting..."
                )

        print(
            f"  [{label_idx}/{len(label_names)}] {label_name}: "
            f"queuing {n_label:,} captions in {n_batches_label} batch job(s)..."
        )

        for b_start in range(0, n_label, LLM_BATCH_SIZE):
            batch_end = min(b_start + LLM_BATCH_SIZE, n_label)
            batch = label_captions[b_start:batch_end]
            idx_slice = label_indices[b_start:batch_end]
            truncated = [c[:MAX_CAPTION_LEN] for c in batch]
            user_prompt = build_extraction_user_prompt(truncated)
            cid = f"p2_{job_seq}"
            job_seq += 1
            jobs.append(
                {"custom_id": cid, "body": _openai_extraction_chat_body(system_prompt, user_prompt)}
            )
            job_meta[cid] = {
                "label_name": label_name,
                "b_start": b_start,
                "batch": batch,
                "indices": idx_slice,
                "ckpt_path": ckpt_path,
                "label_indices": label_indices,
                "n_label": n_label,
            }

    if not jobs:
        tqdm.write("  No OpenAI batch jobs (all labels had valid checkpoints).")
        return

    mapping, _acc = _run_openai_extraction_batch(jobs)
    global_stats["total_batches"] = len(jobs)
    global_stats["total_api_calls"] = len(jobs)

    label_pending: dict[str, list] = defaultdict(list)
    for cid, meta in job_meta.items():
        label_pending[meta["label_name"]].append((meta["b_start"], cid, meta))

    for label_name in label_names:
        if label_name not in label_pending:
            continue
        entries = sorted(label_pending[label_name], key=lambda x: x[0])
        meta0 = entries[0][2]
        ckpt_path = meta0["ckpt_path"]
        label_indices = meta0["label_indices"]

        label_results: list = []
        sync_fallbacks = 0
        fixes_label = 0
        for _b_start, cid, meta in entries:
            text = mapping.get(cid) or ""
            batch_results, pst = parse_extraction_response_text(
                text, meta["batch"], all_feature_names
            )
            ok = bool(pst.get("success"))
            fixes_label += int(pst.get("validation_fixes", 0))
            if not ok:
                tqdm.write(
                    f"    Sync fallback: label={label_name!r} batch @ {meta['b_start']}"
                )
                batch_results, st2 = extract_features_batch(
                    meta["batch"], all_feature_names, system_prompt
                )
                sync_fallbacks += 1
                ok = bool(st2.get("success"))
                fixes_label += int(st2.get("validation_fixes", 0))
            label_results.extend(batch_results)
            if ok:
                global_stats["successful_batches"] += 1
            else:
                global_stats["failed_batches"] += 1

        global_stats["total_retries"] += sync_fallbacks
        global_stats["total_validation_fixes"] += fixes_label

        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(label_results, f)

        for idx, result in zip(label_indices, label_results):
            all_results_list[idx] = result

        global_stats["labels_processed"].append(
            {
                "label": label_name,
                "captions": meta0["n_label"],
                "batches": len(entries),
                "retries": sync_fallbacks,
                "fixes": fixes_label,
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(csv_path: str, schema_path: str):
    print("=" * 70)
    print("  PHASE 2: Bulk Feature Extraction (LLM ONLY - One-Hot Encoding)")
    print("=" * 70 + "\n")

    # ── Load schema ───────────────────────────────────────────────────────────
    print("Loading feature schema...")
    schema = load_schema(schema_path)
    all_feature_names = get_all_feature_names(schema)
    feature_categories = get_feature_categories(schema)
    
    print(f"\n  Total features to extract: {len(all_feature_names)}")
    print(f"  Batch size: {LLM_BATCH_SIZE} captions per API call\n")

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

    # ── LLM extraction for ALL features ──────────────────────────────────────────
    print(f"Running LLM extraction for ALL {len(all_feature_names)} features...")
    print(f"  Provider: {LLM_PROVIDER} | Model: {MODEL_NAME}")
    print(f"  This will process all {n:,} captions in batches of {LLM_BATCH_SIZE}\n")

    system_prompt = build_llm_system_prompt(all_feature_names, feature_categories)

    # Pre-allocate results array aligned to df index
    all_results_list = [None] * n
    
    # Statistics tracking
    global_stats = {
        "caption_column": CAPTION_COLUMN,
        "total_captions": n,
        "total_batches": 0,
        "successful_batches": 0,
        "failed_batches": 0,
        "total_api_calls": 0,
        "total_retries": 0,
        "total_validation_fixes": 0,
        "labels_processed": [],
        "labels_skipped": [],
        "start_time": time.time(),
        "openai_use_batch": bool(LLM_PROVIDER == "openai" and OPENAI_USE_BATCH),
        "openai_prompt_cache_key": OPENAI_PROMPT_CACHE_KEY if LLM_PROVIDER == "openai" else "",
    }

    # Process per label_name for checkpointing
    label_names = df["label_name"].unique()

    if LLM_PROVIDER == "openai" and OPENAI_USE_BATCH:
        _phase2_run_openai_batch_extraction(
            df,
            label_names,
            all_feature_names,
            system_prompt,
            all_results_list,
            global_stats,
        )
    else:
        for label_idx, label_name in enumerate(tqdm(label_names, desc="Processing labels"), 1):
            safe_name = re.sub(r"[^\w\-]", "_", str(label_name))[:60]
            ckpt_path = CHECKPOINT_DIR / f"llm_{safe_name}.json"

            label_mask = df["label_name"] == label_name
            label_indices = df.index[label_mask].tolist()
            label_captions = df.loc[label_indices, CAPTION_COLUMN].tolist()

            n_label = len(label_captions)
            n_batches_label = (n_label + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE

            if ckpt_path.exists():
                try:
                    with open(ckpt_path, "r", encoding="utf-8") as f:
                        label_results = json.load(f)
                    if len(label_results) == len(label_indices):
                        print(
                            f"  [{label_idx}/{len(label_names)}] {label_name}: "
                            f"Loaded {len(label_results):,} results from checkpoint"
                        )
                        for idx, result in zip(label_indices, label_results):
                            all_results_list[idx] = result
                        global_stats["labels_skipped"].append(label_name)
                        continue
                    print(
                        f"  [{label_idx}/{len(label_names)}] {label_name}: "
                        f"Checkpoint incomplete ({len(label_results)}/{len(label_indices)}), re-extracting..."
                    )
                except Exception as e:
                    print(
                        f"  [{label_idx}/{len(label_names)}] {label_name}: "
                        f"Error loading checkpoint: {e}, re-extracting..."
                    )

            print(
                f"  [{label_idx}/{len(label_names)}] {label_name}: "
                f"Processing {n_label:,} captions in {n_batches_label} batches..."
            )

            label_results = []
            label_stats = {
                "batches": 0,
                "retries": 0,
                "validation_fixes": 0,
            }

            batch_pbar = tqdm(
                range(0, len(label_captions), LLM_BATCH_SIZE),
                desc="  Batches",
                leave=False,
                total=n_batches_label,
            )

            for b_start in batch_pbar:
                batch_end = min(b_start + LLM_BATCH_SIZE, len(label_captions))
                batch = label_captions[b_start:batch_end]

                batch_results, stats = extract_features_batch(
                    batch, all_feature_names, system_prompt
                )

                label_results.extend(batch_results)
                label_stats["batches"] += 1
                label_stats["retries"] += stats["attempts"] - 1 if stats["attempts"] > 0 else 0
                label_stats["validation_fixes"] += stats["validation_fixes"]

                global_stats["total_batches"] += 1
                global_stats["total_api_calls"] += stats["attempts"]
                global_stats["total_retries"] += max(0, stats["attempts"] - 1)
                global_stats["total_validation_fixes"] += stats["validation_fixes"]

                if stats["success"]:
                    global_stats["successful_batches"] += 1
                else:
                    global_stats["failed_batches"] += 1

                batch_pbar.set_postfix(
                    {
                        "retries": label_stats["retries"],
                        "fixes": label_stats["validation_fixes"],
                    }
                )

                time.sleep(RATE_LIMIT_SLEEP)

            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(label_results, f)

            for idx, result in zip(label_indices, label_results):
                all_results_list[idx] = result

            global_stats["labels_processed"].append(
                {
                    "label": label_name,
                    "captions": n_label,
                    "batches": label_stats["batches"],
                    "retries": label_stats["retries"],
                    "fixes": label_stats["validation_fixes"],
                }
            )

    # Fill any remaining None entries with unknown
    fallback = {k: 2 for k in all_feature_names}
    none_count = sum(1 for r in all_results_list if r is None)
    if none_count > 0:
        print(f"\n  Warning: {none_count:,} records have no results, filling with unknown (2)")
        for i in range(n):
            if all_results_list[i] is None:
                all_results_list[i] = fallback

    # Normalize rows (checkpoints / edge cases may have non-dicts or bad values)
    all_results_list, row_fixes = validate_batch_results(
        all_results_list, all_feature_names, n
    )
    if row_fixes > 0:
        print(f"  Row validation: applied {row_fixes} fix(es) before building feature matrix")

    # Create features DataFrame
    features_df = pd.DataFrame(all_results_list)
    
    # Calculate elapsed time
    elapsed_time = time.time() - global_stats["start_time"]
    global_stats["elapsed_seconds"] = elapsed_time
    global_stats["elapsed_formatted"] = f"{elapsed_time/3600:.1f} hours"
    
    print(f"\n  Extraction complete!")
    print(f"  Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"  Total API calls: {global_stats['total_api_calls']:,}")
    print(f"  Total retries: {global_stats['total_retries']:,}")
    print(f"  Validation fixes: {global_stats['total_validation_fixes']:,}")
    print(f"  Failed batches: {global_stats['failed_batches']:,}")

    # ── Combine and save ───────────────────────────────────────────────────────
    print("\nCreating feature matrix...")
    
    # Keep original metadata columns
    meta_cols = df[["image", "label_name", "disease_label"]].reset_index(drop=True)
    
    # Combine with extracted features
    final_df = pd.concat(
        [meta_cols,
         features_df.reset_index(drop=True)],
        axis=1,
    )

    # Validate and clean feature columns
    print("  Validating feature columns...")
    for col in all_feature_names:
        if col in final_df.columns:
            # Convert to numeric, coerce errors to NaN
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
            # Fill NaN with 2 (unknown)
            final_df[col] = final_df[col].fillna(2).astype(int)
            # Ensure only 0, 1, 2 values
            final_df[col] = final_df[col].clip(0, 2)

    # Save the new CSV with all feature columns
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    # Save statistics
    with open(STATS_FILE, "w") as f:
        json.dump(global_stats, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"  OUTPUT SUMMARY")
    print(f"{'='*70}")
    print(f"  Feature matrix saved: {OUTPUT_CSV}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Total features: {len(all_feature_names)}")
    print(f"  Total rows: {len(final_df):,}")
    print(f"  Statistics saved: {STATS_FILE}")
    
    # Print feature value distribution summary
    print(f"\n  Feature value distribution (sample of first 10):")
    for col in all_feature_names[:10]:
        vc = final_df[col].value_counts().to_dict()
        present = vc.get(1, 0)
        absent = vc.get(0, 0)
        unknown = vc.get(2, 0)
        pct_present = 100 * present / len(final_df)
        print(f"    {col:40s}: present={present:6,} ({pct_present:4.1f}%), "
              f"absent={absent:6,}, unknown={unknown:6,}")
    
    if len(all_feature_names) > 10:
        print(f"    ... and {len(all_feature_names) - 10} more features")
    
    return final_df, global_stats


if __name__ == "__main__":
    result_df, stats = run_extraction(CSV_PATH, SCHEMA_PATH)

    print("\n" + "="*70)
    print("Extraction complete!")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"Statistics saved to: {STATS_FILE}")
    print("="*70)
