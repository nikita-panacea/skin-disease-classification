"""
PHASE 2: Bulk Feature Extraction (LLM) — Cost-Optimized
=======================================================
Strategy:
  - Deduplicate captions first: only process ~72K unique captions instead of ~182K rows.
  - Use per-category name-based JSON output (only categories the caption mentions).
    Omitted categories decode to 2 (unknown); listed values → 1; other values in a
    mentioned category → 0.
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
  OPENAI_BATCH_MAX_ENQUEUED_TOKENS — org-level enqueued token cap (default 1800000; gpt-4o-mini orgs often 2M).
                             Batches are split so each chunk's estimated tokens stay under this limit.
  OPENAI_BATCH_MAX_RETRIES — batch retry rounds for failed items (default 2; 0 to skip batch retry)
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
    estimate_job_enqueued_tokens,
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
OPENAI_JSON_RESPONSE = os.getenv("OPENAI_JSON_RESPONSE", "1").lower() in ("1", "true", "yes")
OPENAI_PROMPT_CACHE_KEY = os.getenv("OPENAI_PROMPT_CACHE_KEY", "phase2_extraction_v2_per_category").strip()
OPENAI_PROMPT_CACHE_RETENTION = os.getenv("OPENAI_PROMPT_CACHE_RETENTION", "").strip()

_EXTENDED_CACHE_MODELS = frozenset({
    "gpt-4.1", "gpt-5", "gpt-5-codex", "gpt-5.1", "gpt-5.1-codex",
    "gpt-5.1-codex-mini", "gpt-5.1-codex-max", "gpt-5.1-chat-latest",
    "gpt-5.2", "gpt-5.4",
})


def _warn_cache_retention_if_unsupported(model_name: str) -> None:
    if OPENAI_PROMPT_CACHE_RETENTION != "24h":
        return
    if model_name not in _EXTENDED_CACHE_MODELS:
        print(
            f"  WARNING: OPENAI_PROMPT_CACHE_RETENTION=24h is set but model {model_name!r} "
            "does not support extended retention (only gpt-4.1+ and gpt-5+ do). "
            "Falling back to in-memory caching (5-10 min active, max 1hr). "
            "Set OPENAI_MODEL_NAME to gpt-4.1 or newer for 24h retention."
        )


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
OPENAI_BATCH_MAX_ENQUEUED_TOKENS = _safe_int_env(
    "OPENAI_BATCH_MAX_ENQUEUED_TOKENS", 1_800_000, vmin=100_000
)
OPENAI_BATCH_MAX_RETRIES = _safe_int_env("OPENAI_BATCH_MAX_RETRIES", 2, vmin=0, vmax=5)
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
    _warn_cache_retention_if_unsupported(MODEL_NAME)
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
            f"≤{openai_batch_max_file_bytes() / (1024 * 1024):.0f} MiB UTF-8/file, "
            f"≤{OPENAI_BATCH_MAX_ENQUEUED_TOKENS:,} enqueued tokens/chunk"
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
MAX_RETRIES      = 5

TAGGED_CSV_PATH  = Path("discovery_outputs/caption_features_tagged.csv")
USE_TAGGED_FEATURES = os.getenv("USE_TAGGED_FEATURES", "1").strip().lower() in ("1", "true", "yes")
ESTIMATE_ONLY = os.getenv("ESTIMATE_ONLY", "").strip().lower() in ("1", "true", "yes")

# Approximate cost per 1M tokens (input/output) for common models
_MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int,
                   cached_fraction: float = 0.8) -> dict:
    """Estimate USD cost for a given model and token counts."""
    costs = _MODEL_COSTS.get(model, _MODEL_COSTS.get("gpt-4o-mini"))
    cached_input = int(input_tokens * cached_fraction)
    uncached_input = input_tokens - cached_input
    input_cost = (uncached_input / 1_000_000) * costs["input"]
    cached_cost = (cached_input / 1_000_000) * costs["cached_input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input,
        "uncached_input_tokens": uncached_input,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 4),
        "cached_cost_usd": round(cached_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(total, 4),
        "batch_api_cost_usd": round(total * 0.5, 4),
    }
RATE_LIMIT_SLEEP = 0.1 if LLM_PROVIDER == "qwen" else 0.5
DEDUP_CKPT_FILE  = "dedup_caption_features_v2.json"

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
    if OPENAI_JSON_RESPONSE:
        body["response_format"] = {"type": "json_object"}
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


def _openai_parse_batch_error_file(error_file_id: str | None) -> tuple[set[str], dict[str, int]]:
    """
    Download and parse the batch error file. Returns:
      - set of failed custom_ids
      - dict of error_code -> count (for diagnostic logging)
    """
    raw = _openai_download_batch_file_text(error_file_id)
    failed_ids: set[str] = set()
    error_counts: dict[str, int] = {}
    if not raw or not raw.strip():
        return failed_ids, error_counts
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = obj.get("custom_id")
        if cid:
            failed_ids.add(cid)
        err = obj.get("error") or {}
        code = err.get("code", "unknown") if isinstance(err, dict) else "unknown"
        error_counts[code] = error_counts.get(code, 0) + 1
    return failed_ids, error_counts


def _openai_log_batch_error_file_summary(error_file_id: str | None) -> None:
    failed_ids, error_counts = _openai_parse_batch_error_file(error_file_id)
    if not failed_ids:
        return
    err_breakdown = ", ".join(f"{code}={cnt}" for code, cnt in sorted(error_counts.items()))
    tqdm.write(
        f"  Batch error_file: {len(failed_ids)} failed request(s) [{err_breakdown}]."
    )


def _estimate_openai_phase2_batch_cost_usd(acc: dict[str, int]) -> float:
    """gpt-4o-mini list rates with Batch API 50% discount."""
    pin, pcached, pout = 0.15 * 0.5, 0.075 * 0.5, 0.60 * 0.5
    non_cached = max(0, acc["prompt"] - acc["cached"])
    return (non_cached / 1e6) * pin + (acc["cached"] / 1e6) * pcached + (acc["completion"] / 1e6) * pout


def _run_openai_extraction_batch_chunk(
    jobs: list[dict], chunk_idx: int
) -> tuple[dict[str, str], dict[str, int], str, set[str]]:
    """
    One Batch API chunk. Returns (mapping, usage_acc, batch_id, failed_custom_ids).
    Partial results are always collected when an output_file exists.
    """
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

    has_output = batch_job.output_file_id is not None

    if batch_job.status in ("expired", "completed") and has_output:
        if batch_job.status == "expired":
            tqdm.write(
                "  NOTE: Batch status=expired — reading partial output_file."
            )
    elif batch_job.status == "completed" and not has_output:
        raise RuntimeError(
            f"No output_file_id for batch status='completed' — nothing to parse."
        )
    elif not has_output:
        errors = getattr(batch_job, "errors", None)
        err_detail = ""
        if errors:
            err_data = getattr(errors, "data", None) or errors
            if isinstance(err_data, list):
                err_detail = "; ".join(
                    str(getattr(e, "message", e)) for e in err_data[:5]
                )
            else:
                err_detail = str(err_data)[:500]
        hint = ""
        if "enqueued token limit" in err_detail.lower() or "token limit" in err_detail.lower():
            hint = (
                f" Lower OPENAI_BATCH_MAX_ENQUEUED_TOKENS "
                f"(currently {OPENAI_BATCH_MAX_ENQUEUED_TOKENS:,}) to split into smaller chunks."
            )
        raise RuntimeError(
            f"OpenAI batch ended with status={batch_job.status!r} and no output file. "
            f"Inspect batch in dashboard."
            f"{' Errors: ' + err_detail if err_detail else ''}{hint}"
        )

    raw_text = _openai_download_batch_file_text(batch_job.output_file_id)
    err_id = getattr(batch_job, "error_file_id", None)
    _openai_log_batch_error_file_summary(err_id)

    mapping, acc = _openai_parse_batch_output_jsonl(raw_text)

    submitted_ids = {j["custom_id"] for j in jobs}
    failed_ids = submitted_ids - set(mapping.keys())
    if failed_ids:
        _, error_counts = _openai_parse_batch_error_file(err_id)
        err_breakdown = ", ".join(f"{c}={n}" for c, n in sorted(error_counts.items()))
        tqdm.write(
            f"  Chunk {chunk_idx}: {len(mapping)}/{len(jobs)} succeeded, "
            f"{len(failed_ids)} failed [{err_breakdown or 'see error_file'}]."
        )

    return mapping, acc, batch_job.id, failed_ids


def _run_openai_extraction_batch(jobs: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """
    Splits by request count, file size, AND enqueued-token limit.
    Chunks run sequentially. Failed items are retried in new batch rounds
    (up to OPENAI_BATCH_MAX_RETRIES) before the caller falls back to sync.
    """
    if not jobs:
        return {}, {"prompt": 0, "completion": 0, "cached": 0}

    max_r = OPENAI_BATCH_MAX_REQUESTS
    file_cap = openai_batch_max_file_bytes()
    token_cap = OPENAI_BATCH_MAX_ENQUEUED_TOKENS

    jobs_by_id = {j["custom_id"]: j for j in jobs}
    combined: dict[str, str] = {}
    acc_total = {"prompt": 0, "completion": 0, "cached": 0}
    pending_jobs = list(jobs)
    chunk_counter = 0

    for round_num in range(1 + OPENAI_BATCH_MAX_RETRIES):
        if not pending_jobs:
            break

        total_est_tokens = sum(estimate_job_enqueued_tokens(j) for j in pending_jobs)
        chunks = chunk_jobs_for_openai_batch(
            pending_jobs, max_requests=max_r, max_file_bytes=file_cap,
            max_enqueued_tokens=token_cap,
        )
        n_chunks = len(chunks)

        round_label = f"round {round_num}" if round_num > 0 else "initial"
        if n_chunks > 1 or round_num > 0:
            tqdm.write(
                f"  Batch {round_label}: {len(pending_jobs):,} requests "
                f"(~{total_est_tokens:,} est. tokens) → "
                f"{n_chunks} chunk(s) "
                f"(≤{max_r:,} lines, ≤{file_cap / (1024 * 1024):.0f} MiB, "
                f"≤{token_cap:,} enqueued tokens/chunk)."
            )

        all_failed_ids: set[str] = set()
        for ci, chunk in enumerate(chunks):
            m, a, _bid, failed = _run_openai_extraction_batch_chunk(chunk, chunk_counter)
            chunk_counter += 1
            combined.update(m)
            for k in acc_total:
                acc_total[k] += a[k]
            all_failed_ids.update(failed)

        if not all_failed_ids:
            break

        pending_jobs = [jobs_by_id[cid] for cid in all_failed_ids if cid in jobs_by_id]
        if round_num < OPENAI_BATCH_MAX_RETRIES and pending_jobs:
            wait = 30 * (round_num + 1)
            tqdm.write(
                f"  {len(pending_jobs)} job(s) failed — retrying in {wait}s "
                f"(batch retry {round_num + 1}/{OPENAI_BATCH_MAX_RETRIES})…"
            )
            time.sleep(wait)
        elif pending_jobs:
            tqdm.write(
                f"  {len(pending_jobs)} job(s) still failed after "
                f"{OPENAI_BATCH_MAX_RETRIES} batch retries — sync fallback will handle them."
            )

    tqdm.write(
        f"  Batch token totals (all rounds): prompt={acc_total['prompt']:,}, "
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
                if OPENAI_JSON_RESPONSE:
                    create_kw["response_format"] = {"type": "json_object"}
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
# LLM EXTRACTION — SYSTEM PROMPT (per-category, name-based, strict literal)
# ══════════════════════════════════════════════════════════════════════════════
#
# Design:
#   The LLM outputs ONLY the canonical values it EXPLICITLY READS in each caption,
#   grouped by category: {"morphology_color":["red"], "body_location":["forearm"]}.
#
#   Decoding rules (applied in `parse_extraction_response_text`):
#     • Category PRESENT in output → listed values = 1, other features in the
#       same category = 0 (inferably absent).
#     • Category OMITTED from output → every feature in that category = 2
#       (truly unknown — caption said nothing about it).
#
#   Benefits vs. the previous index-based scheme:
#     • No cross-category index leakage (was producing phantom body_location=1).
#     • "Only mark what is literally in the caption" becomes a single strict rule.
#     • Unknown-vs-absent semantics collapse to a single question per category:
#       "Is the category mentioned at all?"
# ══════════════════════════════════════════════════════════════════════════════


def _format_vocabulary_for_prompt(
    all_feature_names: list[str],
    feature_categories: dict[str, str],
) -> str:
    """Render the per-category canonical vocabulary block for the system prompt."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for name in all_feature_names:
        cat = feature_categories.get(name, "other")
        val = name[len(cat) + 1:] if name.startswith(cat + "_") else name
        by_cat[cat].append(val)

    lines = []
    for cat in sorted(by_cat):
        vals = sorted(by_cat[cat])
        lines.append(f"  {cat}: [{', '.join(vals)}]")
    return "\n".join(lines)


def build_llm_system_prompt(
    all_feature_names: list[str],
    feature_categories: dict[str, str],
) -> str:
    """
    Strict, name-based, per-category extraction prompt.

    Output shape per caption (sparse — only categories that the caption mentions):
        {"morphology_color":["red"], "symptoms_dermatological":["itching"]}
    Omitted categories decode to value 2 (unknown).
    """
    vocab_block = _format_vocabulary_for_prompt(all_feature_names, feature_categories)

    return f"""You are a clinical dermatology NLP specialist performing STRICT, LITERAL feature extraction from skin-condition image captions.

TASK
For EACH input caption, output ONE JSON OBJECT whose KEYS are CATEGORY names and whose VALUES are lists of CANONICAL FEATURE VALUES (from the vocabulary below) that are EXPLICITLY described in that caption. Return one object per caption inside a JSON array, in input order.

ABSOLUTE RULES — READ CAREFULLY

1. EXPLICIT MENTION ONLY. Include a feature value ONLY if the caption explicitly mentions it by name or an OBVIOUS clinical synonym. NEVER infer features from the disease name, the overall image concept, or prior medical knowledge. If the caption does not say WHERE on the body the lesion is, do NOT list any body_location feature — regardless of which disease is named.

2. OMIT categories the caption says nothing about. If the caption contains NO information about a given category, do NOT include that category key in the output object at all. Omitted categories decode to value 2 (truly unknown). This is the ONLY way to express "unknown".

3. A category key must appear in the output ONLY if you put AT LEAST ONE canonical value in its list. Do NOT output empty lists like `"body_location": []` — if no specific value is mentioned, OMIT the key entirely.

4. CANONICAL SYNONYM MAPPING. Always map wording in the caption to the canonical vocabulary name:
   - erythema / erythematous / reddened → morphology_color: "red"
   - itchy / itching / pruritus / pruritic → symptoms_dermatological: "itching"
   - elevated / bumpy / papular → morphology_texture: "raised"
   - scaling / flaky → morphology_texture: "scaly"
   - burning sensation → symptoms_dermatological: "burning"
   - tender / sore / painful → symptoms_dermatological: "pain" (or closest canonical)
   - hyperpigmentation / darkening of skin → morphology_color: "hyperpigmented"
   - "rash" / "eruption" alone → do NOT assume a color, texture, or body location
   - "lesion" / "spots" alone → do NOT assume a shape, color, texture, or body location

5. Demographics (age, sex, ethnicity, skin_type), duration, triggers, history, treatments, image_metadata, clinical_signs, severity — OMIT these categories UNLESS the caption literally states something mapping to them. "Photo of a 30-year-old man" → demographics_age: ["30-40"], demographics_sex: ["male"]. "Photo" alone → do not add image_metadata.

6. STRICT JSON output. Return ONLY the JSON array — no markdown fences, no comments, no preface.

OUTPUT EXAMPLES

Caption: "Red, raised, itchy patch on the left forearm"
→ {{"morphology_color":["red"],"morphology_texture":["raised"],"symptoms_dermatological":["itching"],"body_location":["forearm"]}}

Caption: "Scaly plaque"
→ {{"morphology_texture":["scaly"]}}

Caption: "Solitary lesion"
→ {{"lesion_count":["single"]}}

Caption: "Photo of suspected melanoma"
→ {{}}

Caption: "Non-itchy red macule on cheek"
→ {{"morphology_color":["red"],"morphology_texture":["flat"],"body_location":["cheek"]}}
(Explicit negation "non-itchy" is discussed, but we only list what is PRESENT; symptoms_dermatological is omitted so downstream marks it as unknown.)

VOCABULARY (valid canonical values per category — output MUST use only these names):
{vocab_block}

Return ONLY the JSON array.
"""


def build_extraction_user_prompt(captions: list[str]) -> str:
    """Variable user message: numbered captions only (system holds full extraction rules)."""
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(captions))


def build_tagged_system_prompt(
    all_feature_names: list[str],
    feature_categories: dict[str, str],
) -> str:
    """
    Optional variant for pre-tagged inputs. Input is a list of raw feature tag
    strings already extracted from each caption; the LLM maps each tag to a
    canonical value and groups them per category using the SAME output shape
    as `build_llm_system_prompt`.
    """
    vocab_block = _format_vocabulary_for_prompt(all_feature_names, feature_categories)

    return f"""You are a clinical dermatology NLP specialist converting pre-extracted feature tags into canonical per-category one-hot encoding.

For EACH input (a list of raw feature tag strings), output ONE JSON OBJECT mapping CATEGORY → list of CANONICAL VALUES taken from the vocabulary below. Return a JSON array with one object per input, in input order.

RULES
1. Include a category ONLY if at least one input tag maps to a canonical value in that category.
2. OMIT categories for which no input tag maps. Omitted categories decode to value 2 (unknown) downstream.
3. Map each tag to the closest canonical value (e.g. "erythematous" → morphology_color:"red", "pruritus" → symptoms_dermatological:"itching", "papule" → morphology_texture:"papular" or "raised"). Drop tags that do not map cleanly.
4. NEVER invent features that are not derivable from the input tag list.
5. Do NOT output empty lists like `"body_location": []` — omit the category key entirely instead.
6. Return ONLY the JSON array — no markdown, no commentary.

VOCABULARY (valid canonical values per category):
{vocab_block}
"""


def build_tagged_user_prompt(tag_lists: list[list[str]]) -> str:
    """User message when using pre-tagged feature lists instead of captions."""
    return "\n".join(
        f"[{i}] {', '.join(tags) if tags else '(empty)'}"
        for i, tags in enumerate(tag_lists)
    )


# ══════════════════════════════════════════════════════════════════════════════
# PER-CATEGORY OUTPUT PARSING
# ══════════════════════════════════════════════════════════════════════════════
#
# LLM output shape per caption (sparse, name-based):
#     {"morphology_color":["red"], "body_location":["forearm"], ...}
#
# Decoding:
#     category in output  →  listed values = 1, other features in that category = 0
#     category NOT in output →  every feature in that category = 2  (unknown)
# ══════════════════════════════════════════════════════════════════════════════


def _build_category_to_features_map(
    all_feature_names: list[str],
    feature_categories: dict[str, str],
) -> dict[str, set[str]]:
    """Map category -> set of canonical value suffixes (e.g. 'red', 'forearm')."""
    by_cat: dict[str, set[str]] = defaultdict(set)
    for name in all_feature_names:
        cat = feature_categories.get(name, "other")
        val = name[len(cat) + 1:] if name.startswith(cat + "_") else name
        by_cat[cat].add(val)
    return by_cat


def _expand_category_output_to_encoding(
    cat_output: dict,
    all_feature_names: list[str],
    feature_categories: dict[str, str],
    category_to_values: dict[str, set[str]],
) -> tuple[dict[str, int], int]:
    """
    Convert one caption's {"category": ["value1", ...]} dict into the full
    {feature_name: 0|1|2} encoding. Returns (encoding, num_fixes).

      • Categories present with at least one recognized value → listed features = 1,
        other features in the same category = 0.
      • Categories absent (or present with only unrecognized/empty values)
        → all features in that category = 2.
    """
    fixes = 0
    # Default every feature to 2 (unknown). We'll overwrite categories that are
    # present in the LLM's output.
    result = {name: 2 for name in all_feature_names}

    if not isinstance(cat_output, dict):
        return result, 1

    # Pre-compute feature name per (category, value) for fast writes.
    cat_val_to_name: dict[tuple[str, str], str] = {}
    for name in all_feature_names:
        cat = feature_categories.get(name, "other")
        val = name[len(cat) + 1:] if name.startswith(cat + "_") else name
        cat_val_to_name[(cat, val)] = name

    for cat_key, vals in cat_output.items():
        cat = str(cat_key).strip()
        if cat not in category_to_values:
            fixes += 1
            continue

        if not isinstance(vals, list):
            if isinstance(vals, str):
                vals = [vals]
            else:
                fixes += 1
                continue

        valid_set = category_to_values[cat]
        recognized: list[str] = []
        for v in vals:
            s = str(v).strip().lower().replace(" ", "_")
            # Tolerate "morphology_color_red" style → strip category prefix.
            prefix = cat + "_"
            if s.startswith(prefix) and s[len(prefix):] in valid_set:
                s = s[len(prefix):]
            if s in valid_set:
                recognized.append(s)
            else:
                fixes += 1

        if not recognized:
            # Category key was present but no recognized value → still treat
            # category as "considered": set everything in category to 0 (absent)
            # rather than 2. This matches the user's request: "only mark features
            # mentioned in the caption as 1 ... but if a category is considered,
            # others in it are 0 (inferably absent)".
            for val in valid_set:
                fname = cat_val_to_name.get((cat, val))
                if fname is not None:
                    result[fname] = 0
            continue

        # Mark present values = 1, all other features in this category = 0.
        present_set = set(recognized)
        for val in valid_set:
            fname = cat_val_to_name.get((cat, val))
            if fname is None:
                continue
            result[fname] = 1 if val in present_set else 0

    return result, fixes


def parse_extraction_response_text(
    text: str,
    captions: list[str],
    all_feature_names: list[str],
    feature_categories: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """
    Parse the LLM response (per-caption per-category JSON) into one-hot dicts.

    Expected text format:
        [
          {"morphology_color":["red"], "symptoms_dermatological":["itching"]},
          {"morphology_texture":["scaly"]},
          ...
        ]

    Returns (results, stats). On parse failure, returns a fallback row (all 2s)
    per input caption so the pipeline keeps making progress.
    """
    stats: dict = {"success": False, "validation_fixes": 0}
    fallback_row = {k: 2 for k in all_feature_names}

    if feature_categories is None:
        feature_categories = {name: name.split("_", 1)[0] for name in all_feature_names}

    category_to_values = _build_category_to_features_map(
        all_feature_names, feature_categories
    )

    try:
        txt = (text or "").strip()
        txt = re.sub(r"```json\s*|```\s*", "", txt).strip()
        parsed = json.loads(txt)

        # Tolerate the LLM wrapping the array in a container object.
        if isinstance(parsed, dict):
            for key in ("results", "data", "captions", "outputs", "items"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            else:
                # Single-caption object → wrap as a one-element array.
                parsed = [parsed]

        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

        results: list[dict] = []
        fixes = 0
        for item in parsed:
            encoding, f = _expand_category_output_to_encoding(
                item if isinstance(item, dict) else {},
                all_feature_names,
                feature_categories,
                category_to_values,
            )
            if not isinstance(item, dict):
                f += 1
            fixes += f
            results.append(encoding)

        if len(results) < len(captions):
            results.extend([dict(fallback_row)] * (len(captions) - len(results)))
            fixes += len(captions) - len(results) + (len(captions) - len(results))
        results = results[: len(captions)]

        stats["validation_fixes"] = fixes
        stats["success"] = True
        return results, stats

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"    parse_extraction_response_text: {str(e)[:160]}")
        stats["success"] = False
        return [dict(fallback_row) for _ in captions], stats


def _is_valid_idx(val, n: int) -> bool:
    """Kept for backward compatibility; no longer used in the new parser."""
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
    *,
    tag_lists: list[list[str]] | None = None,
    feature_categories: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """
    Batch LLM call for one chunk of captions (or pre-tagged feature lists).
    Returns (list of feature dicts aligned to input captions, stats dict).
    """
    if tag_lists is not None:
        user_prompt = build_tagged_user_prompt(tag_lists)
    else:
        user_prompt = build_extraction_user_prompt(captions)

    stats = {
        "attempts": 0,
        "success": False,
        "validation_fixes": 0,
    }

    for attempt in range(retries):
        try:
            stats["attempts"] = attempt + 1
            text = call_llm(user_prompt, system_prompt, retries=1)
            valid_results, pst = parse_extraction_response_text(
                text, captions, all_feature_names, feature_categories
            )
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
    *,
    caption_to_tags: dict[str, list[str]] | None = None,
    feature_categories: dict[str, str] | None = None,
) -> dict[str, dict]:
    """
    Process deduplicated captions via OpenAI Batch API.
    Returns {caption_text: {feature_name: value}}.
    """
    jobs: list[dict] = []
    job_batches: dict[str, list[str]] = {}

    for batch_idx, batch_start in enumerate(range(0, len(unique_captions), LLM_BATCH_SIZE)):
        batch = unique_captions[batch_start : batch_start + LLM_BATCH_SIZE]
        if caption_to_tags is not None:
            tag_lists = [caption_to_tags.get(c, []) for c in batch]
            user_prompt = build_tagged_user_prompt(tag_lists)
        else:
            user_prompt = build_extraction_user_prompt(batch)
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
        batch_results, pst = parse_extraction_response_text(
            text, batch, all_feature_names, feature_categories
        )
        ok = pst.get("success", False)
        global_stats["total_validation_fixes"] += pst.get("validation_fixes", 0)

        if not ok:
            tqdm.write(f"    Sync fallback for batch {cid}")
            batch_tags = (
                [caption_to_tags.get(c, []) for c in batch]
                if caption_to_tags is not None else None
            )
            batch_results, st2 = extract_features_batch(
                batch, all_feature_names, system_prompt, tag_lists=batch_tags,
                feature_categories=feature_categories,
            )
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
    *,
    caption_to_tags: dict[str, list[str]] | None = None,
    feature_categories: dict[str, str] | None = None,
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
        batch_tags = (
            [caption_to_tags.get(c, []) for c in batch]
            if caption_to_tags is not None else None
        )
        batch_results, stats = extract_features_batch(
            batch, all_feature_names, system_prompt, tag_lists=batch_tags,
            feature_categories=feature_categories,
        )

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
    print(f"  Output format: per-category JSON names (strict literal extraction)\n")

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

    # ── Detect pre-tagged features from Phase 1 ─────────────────────────────
    caption_to_tags: dict[str, list[str]] | None = None
    use_tags = False
    if USE_TAGGED_FEATURES and TAGGED_CSV_PATH.exists():
        try:
            tagged_df = pd.read_csv(TAGGED_CSV_PATH)
            if "extracted_features" in tagged_df.columns and CAPTION_COLUMN in tagged_df.columns:
                tagged_df[CAPTION_COLUMN] = tagged_df[CAPTION_COLUMN].fillna("").astype(str)
                tagged_df["extracted_features"] = tagged_df["extracted_features"].fillna("[]").astype(str)
                _tag_map: dict[str, list[str]] = {}
                for cap_val, tag_val in zip(
                    tagged_df[CAPTION_COLUMN].str.strip(),
                    tagged_df["extracted_features"],
                ):
                    if cap_val and cap_val not in _tag_map:
                        try:
                            parsed_tags = json.loads(tag_val)
                            if isinstance(parsed_tags, list):
                                _tag_map[cap_val] = [str(t) for t in parsed_tags if isinstance(t, str)]
                            else:
                                _tag_map[cap_val] = []
                        except (json.JSONDecodeError, TypeError):
                            _tag_map[cap_val] = []
                caption_to_tags = _tag_map
                use_tags = True
                n_with_tags = sum(1 for v in caption_to_tags.values() if v)
                print(f"  Loaded pre-tagged features for {len(caption_to_tags):,} unique captions "
                      f"({n_with_tags:,} with tags)")
                print(f"  Using TAGGED input mode (shorter prompts, ~60-80% input token reduction)")
        except Exception as e:
            print(f"  WARNING: Could not load tagged CSV ({e}), falling back to full captions")

    if not use_tags:
        print(f"  Using CAPTION input mode (full captions)")

    # ── Build system prompt ───────────────────────────────────────────────────
    if use_tags:
        system_prompt = build_tagged_system_prompt(all_feature_names, feature_categories)
    else:
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
        "output_format": "per_category_names",
        "using_tagged_features": use_tags,
    }

    # ── Cost estimation ────────────────────────────────────────────────────────
    if remaining:
        n_api_calls_est = (len(remaining) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
        sys_prompt_tokens = len(system_prompt.split()) * 1.5
        avg_cap_tokens = 80 if use_tags else 200
        est_input = int(n_api_calls_est * (sys_prompt_tokens + LLM_BATCH_SIZE * avg_cap_tokens))
        est_output = int(n_api_calls_est * 300)
        est = _estimate_cost(MODEL_NAME, est_input, est_output)
        print(f"\n  Cost estimate ({MODEL_NAME}):")
        print(f"    Est. input tokens:  {est['input_tokens']:>12,}")
        print(f"    Est. output tokens: {est['output_tokens']:>12,}")
        print(f"    Sync API cost:      ${est['total_cost_usd']:.4f}")
        print(f"    Batch API cost:     ${est['batch_api_cost_usd']:.4f}")
        if use_tags:
            print(f"    (using tagged features — ~60-80% fewer input tokens vs raw captions)")
        print()

        if ESTIMATE_ONLY:
            print("  ESTIMATE_ONLY=1 — skipping actual extraction. Exiting.")
            return

    # ── Run extraction on remaining unique captions ───────────────────────────
    if remaining:
        print(f"Processing {len(remaining):,} unique captions in ~{n_api_calls_est:,} API calls...")
        print(f"  Provider: {LLM_PROVIDER} | Model: {MODEL_NAME}\n")

        def _ckpt_saver(results):
            merged = dict(caption_features)
            merged.update(results)
            _save_dedup_checkpoint(merged, all_feature_names, dedup_ckpt)

        _tags_arg = caption_to_tags if use_tags else None
        if LLM_PROVIDER == "openai" and OPENAI_USE_BATCH:
            new_results = _run_dedup_openai_batch(
                remaining, all_feature_names, system_prompt, global_stats,
                caption_to_tags=_tags_arg,
                feature_categories=feature_categories,
            )
        else:
            new_results = _run_dedup_sync(
                remaining, all_feature_names, system_prompt, global_stats, _ckpt_saver,
                caption_to_tags=_tags_arg,
                feature_categories=feature_categories,
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
