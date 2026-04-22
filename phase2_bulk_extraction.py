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
  LLM_BATCH_SIZE — captions per API call (default 15)
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
# Strict JSON schema (response_format=json_schema, strict=True) sounds great
# on paper — the server refuses any wrong shape or unknown value — but on
# gpt-4o-mini with our 21-required-category shape it triggers a well-known
# early-stop pathology: because every inner object is dominated by "[]" keys,
# the constrained decoder puts high probability on the outer "]" closer and
# the model emits 1, 12, 13, or 14 items out of 15 with finish_reason='stop'.
# That cascades into huge numbers of sync-fallback retries and bisects.
#
# Default is OFF. The json_object mode + our few-shot-hardened prompt gives
# the same accuracy without the early-stop failure mode. Re-enable the strict
# schema only if you're on a model/version you've verified doesn't exhibit
# the short-stop behaviour.
OPENAI_USE_JSON_SCHEMA = os.getenv("OPENAI_USE_JSON_SCHEMA", "0").lower() in ("1", "true", "yes")
OPENAI_PROMPT_CACHE_KEY = os.getenv(
    "OPENAI_PROMPT_CACHE_KEY",
    "phase2_extraction_v6_jsonobj",
).strip()
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
# ── Concurrency vs. org-level enqueued-token budget ──────────────────────────
# OpenAI enforces a per-organisation "enqueued token" cap across ALL in-flight
# batches for the same model. For gpt-4o-mini the most common limit is 2M
# (Tier 1/2). If your concurrent chunks' token budgets sum above that cap,
# the 2nd/3rd chunk's `batches.create` call will succeed but the batch job
# immediately transitions to status='failed' with "Enqueued token limit
# reached for <model>".
#
# We now auto-derive the per-chunk cap from the org cap:
#     per_chunk_cap = floor((ORG_LIMIT - SAFETY_MARGIN) / CONCURRENCY)
# so the invariant  concurrency × per_chunk_cap  ≤  org_limit  always holds.
#
# If you want to pin a specific per-chunk cap, set OPENAI_BATCH_MAX_ENQUEUED_TOKENS
# explicitly; we still warn if it would breach the org cap at the chosen
# concurrency.
OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG = _safe_int_env(
    "OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG", 2_000_000, vmin=100_000
)
# Concurrency default dropped to 2 (was 3): with a 2M org budget, 2 parallel
# chunks at ~1M tokens each fits safely; 3 × ~700k works too but chunks too
# small to amortise Batch-API overhead. Users on higher-tier orgs (≥5M) can
# bump OPENAI_BATCH_CONCURRENCY to 3-5 for more parallelism.
OPENAI_BATCH_CONCURRENCY = _safe_int_env("OPENAI_BATCH_CONCURRENCY", 2, vmin=1, vmax=8)

# Safety margin covers two realities:
#   1) Our char→token estimator can drift a few percent vs OpenAI's real
#      tokenizer (especially on keyword-dense medical text).
#   2) OpenAI's accounting includes small per-request framing overhead that
#      may not be perfectly captured by estimate_job_enqueued_tokens.
# Use 15% of the org cap (minimum 200k) as headroom so two concurrent
# chunks packed to their per-chunk cap still leave a comfortable buffer
# before the real enqueue limit.
_safety_margin = max(200_000, OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG // 7)  # ≈14%
_auto_per_chunk = max(
    100_000,
    (OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG - _safety_margin) // OPENAI_BATCH_CONCURRENCY,
)
_explicit_per_chunk_env = os.getenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", "").strip()
if _explicit_per_chunk_env:
    OPENAI_BATCH_MAX_ENQUEUED_TOKENS = _safe_int_env(
        "OPENAI_BATCH_MAX_ENQUEUED_TOKENS", _auto_per_chunk, vmin=100_000
    )
    _budget = OPENAI_BATCH_CONCURRENCY * OPENAI_BATCH_MAX_ENQUEUED_TOKENS
    if _budget > OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG:
        print(
            f"  WARNING: OPENAI_BATCH_CONCURRENCY × OPENAI_BATCH_MAX_ENQUEUED_TOKENS "
            f"({OPENAI_BATCH_CONCURRENCY} × {OPENAI_BATCH_MAX_ENQUEUED_TOKENS:,} = "
            f"{_budget:,}) exceeds OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG "
            f"({OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG:,}). Expect 'Enqueued token "
            f"limit reached' failures on the 2nd+ concurrent chunk. "
            f"Recommended: unset OPENAI_BATCH_MAX_ENQUEUED_TOKENS to let the code "
            f"auto-derive a safe value."
        )
else:
    OPENAI_BATCH_MAX_ENQUEUED_TOKENS = _auto_per_chunk

OPENAI_BATCH_MAX_RETRIES = _safe_int_env("OPENAI_BATCH_MAX_RETRIES", 2, vmin=0, vmax=5)
# When a batch fails BECAUSE of the org enqueue-token cap (not a real error),
# wait this long before resubmitting. Gives any currently-running batch time
# to chew through its requests and free up tokens. 4 min is a good default;
# batches make steady progress once they leave 'validating'.
OPENAI_BATCH_ENQUEUE_WAIT_SEC = _safe_int_env(
    "OPENAI_BATCH_ENQUEUE_WAIT_SEC", 240, vmin=30
)
OPENAI_BATCH_ENQUEUE_MAX_WAITS = _safe_int_env(
    "OPENAI_BATCH_ENQUEUE_MAX_WAITS", 6, vmin=1, vmax=20
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
        _effective_budget = OPENAI_BATCH_CONCURRENCY * OPENAI_BATCH_MAX_ENQUEUED_TOKENS
        _source = "env-set" if _explicit_per_chunk_env else "auto-derived"
        print(
            f"  Batch concurrency: {OPENAI_BATCH_CONCURRENCY} chunk(s) in flight, "
            f"org enqueue cap={OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG:,}, "
            f"per-chunk={OPENAI_BATCH_MAX_ENQUEUED_TOKENS:,} ({_source}), "
            f"peak pending ≈ {_effective_budget:,} "
            f"({'OK' if _effective_budget <= OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG else 'EXCEEDS org cap — expect enqueue failures'})"
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
# LLM batch size: the number of captions packed into ONE LLM request.
# gpt-4o-mini has a persistent "close the array one item early" failure mode
# in json_object mode — we've seen ~5-10% of 10-caption batches emit exactly
# 9 objects and stop with finish_reason='stop'. Dropping to 6 roughly halves
# the incidence (fewer items → less counting drift) and, more importantly,
# when it DOES happen the wasted context is smaller and the sync-retry for
# the tail covers only 1 caption. Each batch costs a bit more per-caption
# at N=6 vs N=10 (prompt overhead amortises less), but the correctness win
# dominates for this model. Bump back to 10 only if you switch to a model
# that doesn't show the pathology (e.g. gpt-4.1-mini).
LLM_BATCH_SIZE   = _safe_int_env("LLM_BATCH_SIZE", 6, vmin=1)
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
# OpenAI rate limits are generous (gpt-4o-mini: 10k RPM, 200M TPM on tier 2);
# a blanket 0.5s sleep between every sync batch wastes ~8 min per 1000 batches.
# Qwen local needs a small pace to avoid overwhelming the server.
RATE_LIMIT_SLEEP = 0.1 if LLM_PROVIDER == "qwen" else 0.0
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


def _estimate_max_output_tokens(n_captions: int, *, schema_enforced: bool) -> int:
    """
    Budget ``max_tokens`` proportionally to the number of captions in the
    request. A fixed 8192 is wasteful for small batches AND lets a looping
    model burn the full budget. Typical per-caption output:
      • json_object (sparse)    ~= 40-80 tokens
      • strict schema (all keys) ~= 180-220 tokens (21 required categories)
    We add a flat safety pad of 200 tokens, and cap at 8192 (gpt-4o-mini
    allows 16384 but we never need near that for N<=25).
    """
    per_cap = 220 if schema_enforced else 90
    pad = 200
    est = per_cap * max(1, n_captions) + pad
    return max(512, min(8192, est))


def _openai_extraction_chat_body(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict | None = None,
    *,
    n_captions: int | None = None,
) -> dict:
    """Chat body for extraction: static system first, variable captions last (prompt caching).

    If `json_schema` is provided and OPENAI_USE_JSON_SCHEMA=1, the body uses
    `response_format={"type":"json_schema", "strict":True, ...}` so OpenAI
    enforces the output shape server-side. This eliminates short-response and
    wrong-shape failure modes that cause sync-fallback cascades.

    ``max_tokens`` is sized to the expected output for ``n_captions``. Passing
    None defaults to the current LLM_BATCH_SIZE, matching the old behaviour.
    """
    schema_enforced = bool(OPENAI_USE_JSON_SCHEMA and json_schema is not None)
    if n_captions is None:
        n_captions = LLM_BATCH_SIZE
    body: dict = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        # Sized to the batch. Prevents a looping model from burning the full
        # 8192-token budget for a 3-caption batch that only ever needed ~500.
        "max_tokens": _estimate_max_output_tokens(
            n_captions, schema_enforced=schema_enforced
        ),
        "stream": False,
    }
    if OPENAI_USE_JSON_SCHEMA and json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
    elif OPENAI_JSON_RESPONSE:
        body["response_format"] = {"type": "json_object"}
    _openai_apply_prompt_caching(body, OPENAI_PROMPT_CACHE_KEY)
    return body


# ──────────────────────────────────────────────────────────────────────────────
# Strict JSON-schema for structured outputs
# ──────────────────────────────────────────────────────────────────────────────
# OpenAI strict-mode schema rules (docs: structured-outputs):
#   - additionalProperties must be false on every object.
#   - Every property listed in `properties` must be in `required`.
#   - `enum` is supported; `minItems`/`maxItems` are NOT supported in strict mode.
# With strict mode, the server guarantees the model output parses as this
# schema exactly. That alone fixes:
#   • wrong top-level shape (model MUST emit {"captions":[...]}),
#   • unknown category keys (each inner object is a fixed dict of enum lists),
#   • unknown values (each list has an enum constraint).
# It does NOT fix count-skipping/looping (no min/max items), but combined with
# the count-enforcement prompt above and the partial-recovery parser, the
# pipeline now always makes forward progress instead of dropping whole batches.

_EXTRACTION_SCHEMA_NAME = "caption_features_v4"


def _build_extraction_json_schema(
    all_feature_names: list[str],
    feature_categories: dict[str, str],
) -> dict:
    """Build the strict JSON-schema for {"captions":[{<cat>: [<enum>, ...], ...}]}."""
    cat_to_vals = _build_category_to_features_map(all_feature_names, feature_categories)
    categories = sorted(cat_to_vals.keys())

    item_properties: dict = {}
    for cat in categories:
        values = sorted(cat_to_vals[cat])
        item_properties[cat] = {
            "type": "array",
            "items": {"type": "string", "enum": values},
        }

    item_schema = {
        "type": "object",
        "properties": item_properties,
        "required": categories,  # strict mode: all listed props required
        "additionalProperties": False,
    }

    return {
        "name": _EXTRACTION_SCHEMA_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "captions": {
                    "type": "array",
                    "items": item_schema,
                },
            },
            "required": ["captions"],
            "additionalProperties": False,
        },
    }


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


def _openai_parse_batch_output_jsonl(
    raw: str,
) -> tuple[dict[str, dict], dict[str, int]]:
    """
    Parse the JSONL output file from a Batch API job.

    Returns (mapping, usage_acc) where each mapping entry is a dict:
        {"content": <str>, "finish_reason": <str>, "completion_tokens": <int>}
    so downstream parsers can distinguish truncation (finish_reason='length')
    from shape/JSON errors (finish_reason='stop').
    """
    acc = {"prompt": 0, "completion": 0, "cached": 0}
    out: dict[str, dict] = {}
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
        ct = int(usage.get("completion_tokens", 0))
        acc["prompt"] += int(usage.get("prompt_tokens", 0))
        acc["completion"] += ct
        details = usage.get("prompt_tokens_details") or {}
        acc["cached"] += int(details.get("cached_tokens", 0))
        choices = body.get("choices") or []
        if choices and cid and isinstance(choices[0], dict):
            choice0 = choices[0]
            msg = choice0.get("message") or {}
            if not isinstance(msg, dict):
                msg = {}
            out[cid] = {
                "content": msg.get("content") or "",
                "finish_reason": choice0.get("finish_reason") or "",
                "completion_tokens": ct,
            }
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


def _batch_error_detail(batch_job) -> str:
    """Extract the best-effort error message(s) from a failed batch."""
    errors = getattr(batch_job, "errors", None)
    if not errors:
        return ""
    err_data = getattr(errors, "data", None) or errors
    if isinstance(err_data, list):
        return "; ".join(str(getattr(e, "message", e)) for e in err_data[:5])
    return str(err_data)[:500]


def _is_enqueue_limit_failure(err_detail: str) -> bool:
    """Recognise the 'Enqueued token limit reached for <model>' transient."""
    lo = err_detail.lower()
    return "enqueued token limit" in lo or "enqueue token limit" in lo


def _poll_until_terminal(batch_job, chunk_idx: int):
    """Poll a batch until it reaches a terminal state (or the 26h hard cap)."""
    terminal = {"completed", "failed", "expired", "cancelled"}
    poll_deadline = time.monotonic() + 26 * 3600
    last_progress_count = -1
    stall_start: float | None = None
    while batch_job.status not in terminal:
        time.sleep(OPENAI_BATCH_POLL_SEC)
        if time.monotonic() > poll_deadline:
            tqdm.write(
                f"  Batch chunk {chunk_idx} exceeded 26h wall-clock. Cancelling "
                f"and returning whatever is already in the output file."
            )
            try:
                openai_client.batches.cancel(batch_job.id)
            except Exception as cancel_err:
                tqdm.write(f"    (cancel failed: {cancel_err!r})")
            time.sleep(OPENAI_BATCH_POLL_SEC)
            batch_job = openai_client.batches.retrieve(batch_job.id)
            break
        batch_job = openai_client.batches.retrieve(batch_job.id)
        rc = batch_job.request_counts
        tqdm.write(
            f"    [chunk {chunk_idx}] status={batch_job.status}  "
            f"completed={rc.completed}/{rc.total}  failed={rc.failed}"
        )
        total_done = (rc.completed or 0) + (rc.failed or 0)
        now = time.monotonic()
        if total_done != last_progress_count:
            last_progress_count = total_done
            stall_start = now
        elif stall_start is not None and now - stall_start > 1800:
            tqdm.write(
                f"    (chunk {chunk_idx}: no progress in 30 min at "
                f"{total_done}/{rc.total}; continuing until 26h cap)"
            )
            stall_start = now
    return batch_job


def _run_openai_extraction_batch_chunk(
    jobs: list[dict], chunk_idx: int
) -> tuple[dict[str, str], dict[str, int], str, set[str]]:
    """
    One Batch API chunk. Returns (mapping, usage_acc, batch_id, failed_custom_ids).
    Partial results are always collected when an output_file exists.

    Handles the "Enqueued token limit reached" transient: when another
    concurrent chunk is still occupying the org-wide token budget, our
    newly-created batch immediately transitions to 'failed' with that error.
    We wait OPENAI_BATCH_ENQUEUE_WAIT_SEC and re-submit the SAME input file
    (no re-upload) up to OPENAI_BATCH_ENQUEUE_MAX_WAITS times.
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

    batch_job = None
    for submit_attempt in range(OPENAI_BATCH_ENQUEUE_MAX_WAITS):
        batch_job = openai_batches_create_safe(
            openai_client,
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"phase": "phase2_extraction", "chunk": str(chunk_idx)},
        )
        if submit_attempt == 0:
            tqdm.write(
                f"  batch_id={batch_job.id} (chunk {chunk_idx}); polling every "
                f"{OPENAI_BATCH_POLL_SEC}s"
            )
        else:
            tqdm.write(
                f"  chunk {chunk_idx} resubmitted as batch_id={batch_job.id} "
                f"(retry {submit_attempt}/{OPENAI_BATCH_ENQUEUE_MAX_WAITS - 1})"
            )

        batch_job = _poll_until_terminal(batch_job, chunk_idx)

        # If it failed specifically from the org-level enqueue cap, wait for
        # the concurrent batch(es) to chew through their tokens, then retry.
        if batch_job.status == "failed" and batch_job.output_file_id is None:
            err_detail = _batch_error_detail(batch_job)
            if _is_enqueue_limit_failure(err_detail):
                waited = OPENAI_BATCH_ENQUEUE_WAIT_SEC * (1 + submit_attempt // 2)
                tqdm.write(
                    f"  chunk {chunk_idx}: enqueue-token cap hit "
                    f"('{err_detail[:120]}'). Waiting {waited}s for other "
                    f"chunks to free budget, then resubmitting."
                )
                time.sleep(waited)
                continue  # resubmit the same uploaded file
        break  # non-enqueue failure OR success

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
        err_detail = _batch_error_detail(batch_job)
        hint = ""
        if _is_enqueue_limit_failure(err_detail):
            hint = (
                f" Still hitting the org enqueue cap after "
                f"{OPENAI_BATCH_ENQUEUE_MAX_WAITS} resubmits. "
                f"Lower OPENAI_BATCH_CONCURRENCY (currently "
                f"{OPENAI_BATCH_CONCURRENCY}) and/or set "
                f"OPENAI_BATCH_MAX_ENQUEUED_TOKENS_ORG to your actual org cap."
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

    # Clean up the local input JSONL: each chunk is 5-50 MiB and a 100-chunk
    # run was leaving multi-GB of stale files in checkpoints/. Keep the file
    # only when the batch didn't complete cleanly (aids post-mortem).
    if batch_job.status == "completed" and not failed_ids:
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass

    return mapping, acc, batch_job.id, failed_ids


def _run_openai_extraction_batch(
    jobs: list[dict],
    *,
    per_chunk_callback=None,
) -> tuple[dict[str, dict], dict[str, int]]:
    """
    Splits by request count, file size, AND enqueued-token limit.

    Chunks are dispatched CONCURRENTLY (up to ``OPENAI_BATCH_CONCURRENCY``
    at a time, default 3). Each chunk polls OpenAI independently — this is
    the key wall-clock fix: a 40-chunk run used to wait for each 4-8 hour
    chunk sequentially (>1 day); now 3 chunks share the 1.8M-token org
    budget and finish in ~1/3 the time.

    Failed items are retried in new rounds up to OPENAI_BATCH_MAX_RETRIES
    before the caller falls back to sync.

    If ``per_chunk_callback`` is provided, it is invoked after every chunk
    completes with that chunk's mapping. Called from the main thread under
    a lock so the callback's checkpoint writes stay safe.

    Returns the combined mapping (same per-id dict shape as each chunk) plus
    total usage accounting across all rounds.
    """
    if not jobs:
        return {}, {"prompt": 0, "completion": 0, "cached": 0}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    max_r = OPENAI_BATCH_MAX_REQUESTS
    file_cap = openai_batch_max_file_bytes()
    token_cap = OPENAI_BATCH_MAX_ENQUEUED_TOKENS
    concurrency = max(1, OPENAI_BATCH_CONCURRENCY)

    jobs_by_id = {j["custom_id"]: j for j in jobs}
    combined: dict[str, dict] = {}
    acc_total = {"prompt": 0, "completion": 0, "cached": 0}
    pending_jobs = list(jobs)
    chunk_counter = 0
    # Two locks, separate concerns:
    #   state_lock  — protects the small accumulator dicts (combined, acc_total,
    #                 all_failed_ids). Held for microseconds per chunk.
    #   callback_lock — serialises callback invocations (parse + sync retries +
    #                 checkpoint I/O). Separate from state_lock so slow
    #                 callbacks DO NOT block other chunks' state updates.
    # Previously a single lock wrapped both, which meant a 5-minute sync-retry
    # callback on chunk N blocked chunks N+1..N+k from recording their
    # results too. That alone cancelled most of the concurrency gain.
    state_lock = threading.Lock()
    callback_lock = threading.Lock()

    def _run_one_chunk(chunk: list[dict], ci: int) -> tuple[int, dict, dict, set[str], str | None]:
        """Execute one chunk; errors are converted to an 'all-failed' result."""
        try:
            m, a, _bid, failed = _run_openai_extraction_batch_chunk(chunk, ci)
            return ci, m, a, failed, None
        except Exception as chunk_err:  # noqa: BLE001
            return ci, {}, {"prompt": 0, "completion": 0, "cached": 0}, \
                   {j["custom_id"] for j in chunk}, \
                   f"{type(chunk_err).__name__}: {str(chunk_err)[:200]}"

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
        tqdm.write(
            f"  Batch {round_label}: {len(pending_jobs):,} requests "
            f"(~{total_est_tokens:,} est. tokens) → {n_chunks} chunk(s) "
            f"with concurrency={concurrency} "
            f"(≤{max_r:,} lines, ≤{file_cap / (1024 * 1024):.0f} MiB, "
            f"≤{token_cap:,} enqueued tokens/chunk)."
        )

        all_failed_ids: set[str] = set()

        # Pre-assign stable chunk indices so on-disk input files don't collide.
        chunk_tasks: list[tuple[int, list[dict]]] = []
        for chunk in chunks:
            chunk_tasks.append((chunk_counter, chunk))
            chunk_counter += 1

        # Fan chunks out on a thread pool. The API client (openai_client) is
        # safe to share across threads; each poll loop is independent.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_run_one_chunk, c, ci): ci for ci, c in chunk_tasks}
            for fut in as_completed(futures):
                ci, m, a, failed, err = fut.result()

                if err is not None:
                    tqdm.write(
                        f"  Batch chunk {ci} failed ({err}); "
                        f"{len(failed)} job(s) will be retried or routed to sync fallback."
                    )

                # Fast state update: hold state_lock for only a few operations.
                with state_lock:
                    combined.update(m)
                    for k in acc_total:
                        acc_total[k] += a.get(k, 0)
                    all_failed_ids.update(failed)

                # Callback (parse + sync retries + checkpoint I/O) runs OUTSIDE
                # state_lock under callback_lock. This lets other chunks keep
                # recording their state updates in parallel while one chunk's
                # slow sync-fallback work is in progress. Callbacks still
                # serialise between themselves (the callback mutates the
                # shared `results` dict and writes the checkpoint file).
                if per_chunk_callback is not None and m:
                    with callback_lock:
                        try:
                            per_chunk_callback(m)
                        except Exception as cb_err:  # noqa: BLE001
                            tqdm.write(
                                f"    per_chunk_callback raised: {cb_err!r} "
                                f"(continuing; data already in mapping)"
                            )

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
    json_schema: dict | None = None,
    n_captions: int | None = None,
) -> tuple[str, dict]:
    """
    Generic LLM caller that works for Gemini, OpenAI, and Qwen (local).
    For OpenAI, pass static instructions as system_prompt and variable text as
    prompt so prompt caching can reuse the system prefix across calls.

    Returns (text, meta) where ``meta`` contains ``finish_reason`` and
    ``completion_tokens`` when the provider exposes them. This lets callers
    distinguish truncation (finish_reason='length') from shape/parse errors.
    """
    for attempt in range(retries):
        try:
            if LLM_PROVIDER == "gemini":
                full_prompt = f"{system_prompt or ''}\n\n{prompt}" if system_prompt else prompt
                response = model.generate_content(full_prompt)
                return response.text, {"finish_reason": "", "completion_tokens": 0}

            elif LLM_PROVIDER == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                schema_enforced = bool(OPENAI_USE_JSON_SCHEMA and json_schema is not None)
                mt = _estimate_max_output_tokens(
                    n_captions if n_captions is not None else LLM_BATCH_SIZE,
                    schema_enforced=schema_enforced,
                )
                create_kw = dict(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    # Sized to request. Caps runaway-loop cost at
                    # ~0.22 * n_captions + 200 tokens vs. a flat 8192.
                    max_tokens=mt,
                    stream=False,
                )
                if OPENAI_USE_JSON_SCHEMA and json_schema is not None:
                    create_kw["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json_schema,
                    }
                elif OPENAI_JSON_RESPONSE:
                    create_kw["response_format"] = {"type": "json_object"}
                ck = (
                    openai_prompt_cache_key
                    if openai_prompt_cache_key is not None
                    else OPENAI_PROMPT_CACHE_KEY
                )
                _openai_apply_prompt_caching(create_kw, ck)
                response = openai_client.chat.completions.create(**create_kw)
                _maybe_log_openai_usage(response)
                choice0 = response.choices[0]
                meta = {
                    "finish_reason": getattr(choice0, "finish_reason", "") or "",
                    "completion_tokens": int(
                        getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
                    ),
                }
                return choice0.message.content, meta

            elif LLM_PROVIDER == "qwen":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = qwen_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                choice0 = response.choices[0]
                meta = {
                    "finish_reason": getattr(choice0, "finish_reason", "") or "",
                    "completion_tokens": int(
                        getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
                    ),
                }
                return choice0.message.content, meta

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
    *,
    schema_enforced: bool = False,
) -> str:
    """
    Strict, name-based, per-category extraction prompt.

    Output shape:
        {"captions": [
            {"<cat>": ["<value>", ...], ...},  # one object per input caption
            ...
        ]}

    Decoding contract (applied downstream in _expand_category_output_to_encoding):
      • Category has at least one recognised value → listed = 1, other values
        in that category = 0 (inferably absent).
      • Category is empty [] or has only unrecognised values → all values in
        that category stay 2 (unknown).
      • Category key missing from the object (only possible without strict
        schema) → all values stay 2 (unknown).

    When ``schema_enforced`` is True we assume the caller is sending a strict
    ``response_format=json_schema`` body that already pins:
      • the top-level ``{"captions":[...]}`` shape,
      • the full set of category keys,
      • the enum of valid values per category.
    Under that mode the prompt is much shorter — no vocabulary dump, no
    "omit categories" contradictions — which dramatically cuts prompt tokens
    and eliminates the "short-response" failure mode where the model tried
    to comply with both the schema (all keys required) and the prompt
    (omit unmentioned keys) at once.
    """
    if schema_enforced:
        return """You are a clinical dermatology NLP specialist performing STRICT, LITERAL feature extraction from skin-condition image captions.

The API will enforce the output shape via a strict JSON schema. Your only job is to FILL IN the correct values per caption.

FOR EACH INPUT CAPTION, produce ONE object containing EVERY category key. For each category:
  • Emit []  if the caption does NOT explicitly discuss that category. This is the DEFAULT — use it for almost every category.
  • Emit [<canonical_value>, ...] ONLY when the caption LITERALLY mentions one or more values for that category.

EXTRACTION RULES
1. EXPLICIT MENTION ONLY. Include a value ONLY when the caption states it (or an obvious clinical synonym). NEVER infer features from the disease name, the overall image concept, or prior medical knowledge. If the caption does not say WHERE on the body the lesion is, body_location MUST be [] — regardless of which disease is named.
2. CANONICAL SYNONYMS — map to the exact canonical values allowed by the schema:
   - erythema / erythematous / reddened → morphology_color: "red"
   - itchy / itching / pruritus / pruritic → symptoms_dermatological: "itching"
   - elevated / bumpy / papular → morphology_texture: "raised"
   - scaling / flaky → morphology_texture: "scaly"
   - burning sensation → symptoms_dermatological: "burning"
   - tender / sore / painful → symptoms_dermatological: "pain" (or closest canonical)
   - hyperpigmentation / darkening of skin → morphology_color: "hyperpigmented"
3. GENERIC WORDS DO NOT IMPLY CATEGORIES.
   - "rash" / "eruption" alone → no color, texture, or body_location value.
   - "lesion" / "spots" alone → no shape, color, texture, or body_location value.
   - "photo" / "image" alone → no image_metadata value.
4. DEMOGRAPHICS and HISTORY only when literally stated ("30-year-old man" → demographics_age, demographics_sex; "1-week rash" → duration).

COUNT / ORDER
- The user message starts with "INPUT: N caption(s)". Produce EXACTLY N caption objects, in input order.
- Emit every category key in every caption object (the schema requires it); use [] for unmentioned categories.
- Do not repeat, skip, or add extra caption objects. Once the N-th object is written, stop.

FEW-SHOT EXAMPLES (ONLY the categories that pick up values are shown below — in your actual output every other category MUST also be present with value []).

[E1] Multi-feature, clear anchors
    Input : "Red, raised, itchy patch on the left forearm"
    Non-empty values → body_location:["forearm"], morphology_color:["red"],
                       morphology_texture:["raised"], symptoms_dermatological:["itching"]
    (all other categories: [])

[E2] Disease name only — NO extractable features (common trap: do NOT infer from the diagnosis)
    Input : "Image of suspected melanoma"
    Non-empty values → (none)
    (every category: [])

[E3] Generic words alone — NO anchors (rash / eruption / lesion / photo tell you nothing specific)
    Input : "Skin eruption in an adult"
    Non-empty values → (none — "rash"/"eruption"/"adult-without-age" do not map)
    (every category: [])

[E4] Demographics stated literally, no lesion description
    Input : "Photo of a 30-year-old woman with a red bump on the cheek"
    Non-empty values → demographics_age:["30-40"], demographics_sex:["female"],
                       body_location:["cheek"], morphology_color:["red"],
                       morphology_texture:["raised"]
    (Note: "bump" maps to morphology_texture:"raised"; "photo" alone does NOT fill image_metadata.)

[E5] Explicit negation — list only what IS present; do NOT mark the negated value as "present"
    Input : "Non-itchy red macule on the chest"
    Non-empty values → body_location:["chest"], morphology_color:["red"],
                       morphology_texture:["flat"]
    (symptoms_dermatological stays [] — "non-itchy" means itching is absent, but our
     contract only marks mentioned-positive values.)

[E6] Synonym mapping in action
    Input : "Erythematous, scaly plaque with pruritus"
    Non-empty values → morphology_color:["red"], morphology_texture:["scaly"],
                       symptoms_dermatological:["itching"]
    (erythematous → red; pruritus → itching.)

Follow the examples above LITERALLY. When in doubt, prefer [] — unjustified 1s are worse than missing 1s.
"""

    # Non-schema-enforced fallback (json_object mode). Keep the vocabulary
    # block and the original rule set; the model needs the full spec here.
    vocab_block = _format_vocabulary_for_prompt(all_feature_names, feature_categories)

    return f"""You are a clinical dermatology NLP specialist performing STRICT, LITERAL feature extraction from skin-condition image captions.

TASK
Produce ONE top-level JSON OBJECT with a single key "captions" whose value is a JSON array containing ONE object per input caption, in input order. Each inner object maps CATEGORY name → list of CANONICAL FEATURE VALUES (from the vocabulary below) that are EXPLICITLY described in that caption.

REQUIRED OUTPUT SHAPE
{{
  "captions": [
    {{ <category>: [<canonical_value>, ...], ... }},
    ...
  ]
}}

COUNT RULES
- The user message starts with "INPUT: N caption(s)". Your "captions" array MUST have EXACTLY N elements.
- ONE object per caption, in the SAME ORDER as input. Empty object {{}} for captions with no extractable features.
- NEVER skip, repeat, or add extra entries. After the N-th object, close the JSON and stop.
- Return ONLY the JSON object above. No markdown, no commentary.

EXTRACTION RULES
1. EXPLICIT MENTION ONLY. Include a value ONLY when the caption literally mentions it (or an obvious clinical synonym). NEVER infer from the disease name or prior knowledge. If the caption does not say WHERE on the body, OMIT body_location — regardless of which disease is named.
2. OMIT categories the caption says nothing about. Omitted categories decode to 2 (unknown).
3. A category key should only appear if you put AT LEAST ONE canonical value in its list. Do NOT emit empty lists.
4. Canonical synonyms: erythema→red, itchy/pruritus→itching, raised/papular→raised, scaly/flaky→scaly, burning→burning, tender/sore→pain, hyperpigmentation→hyperpigmented.
5. Generic words ("rash", "eruption", "lesion", "spots", "photo") do NOT imply any category.

FEW-SHOT EXAMPLES (omit any category not shown in the inner object — omitted = unknown)

[E1] "Red, raised, itchy patch on the left forearm"
  → {{"body_location":["forearm"],"morphology_color":["red"],"morphology_texture":["raised"],"symptoms_dermatological":["itching"]}}

[E2] "Image of suspected melanoma"            (disease name only → NO anchors)
  → {{}}

[E3] "Skin eruption in an adult"              (generic words, no age → NO anchors)
  → {{}}

[E4] "Photo of a 30-year-old woman with a red bump on the cheek"
  → {{"demographics_age":["30-40"],"demographics_sex":["female"],"body_location":["cheek"],"morphology_color":["red"],"morphology_texture":["raised"]}}

[E5] "Non-itchy red macule on the chest"      (explicit negation: don't mark itching present)
  → {{"body_location":["chest"],"morphology_color":["red"],"morphology_texture":["flat"]}}

[E6] "Erythematous, scaly plaque with pruritus"   (synonym mapping)
  → {{"morphology_color":["red"],"morphology_texture":["scaly"],"symptoms_dermatological":["itching"]}}

Follow the examples above LITERALLY. When in doubt, OMIT the category — unjustified 1s are worse than missing 1s.

VOCABULARY (valid canonical values per category — output MUST use only these names):
{vocab_block}

Return ONLY the JSON object {{"captions":[...]}} — nothing else.
"""


def build_extraction_user_prompt(captions: list[str], *, schema_enforced: bool = False) -> str:
    """Variable user message with count-enforcement header AND footer.

    Why this prompt is written the way it is
    ─────────────────────────────────────────
    We observed a systematic off-by-one: with batch size N, gpt-4o-mini
    consistently emitted N-1 caption objects and stopped (`finish_reason='stop'`).
    Root causes identified:
      1. "numbered [1]..[N]" — the "dot-dot" range operator is **exclusive**
         in Rust/Python-slices (which the model has seen far more of than
         Ruby-style inclusive `..`), so the model reads it as "indices
         1 through N-1" → emits N-1.
      2. Count enforcement only at the TOP of the prompt. By the time the
         model has emitted N-1 objects, the "EXACTLY N" line is buried under
         thousands of tokens of input; closing the array `]}` becomes a
         stronger attractor than the count directive.
      3. Few-shot examples show `{}` as valid for hard captions. The model
         rationalises "the Nth would just be `{}` anyway — close now."

    Fixes applied here:
      • Replaced `[1]..[N]` with the unambiguous "1 through N inclusive" + an
        explicit "[N] is the last caption — it MUST be included in your output".
      • Added a TRAILING reminder right before the model starts generating,
        so the count is the MOST RECENT context (where gpt-4o-mini pays most
        attention).
      • Flagged the final caption inline as `[N] (LAST — MUST BE IN OUTPUT)`
        so the model can't miss it.
    """
    n = len(captions)
    count_clause = (
        f"INPUT: {n} caption(s), indexed 1 through {n} inclusive (so the last "
        f"caption is [{n}] and MUST be in your output)."
    )
    if schema_enforced:
        header = (
            f"{count_clause}\n"
            f"Emit EXACTLY {n} caption objects in the same order. For each "
            f"object, include every category key (use [] for unmentioned "
            f"categories). Never skip, repeat, or add extras."
        )
    else:
        header = (
            f"{count_clause}\n"
            f"Your 'captions' array MUST contain EXACTLY {n} objects in the same "
            f"order. Emit {{}} for any caption with no extractable features — "
            f"NEVER skip a caption, NEVER repeat a caption, NEVER emit extra items."
        )

    # Flag the final caption inline. Small but effective: the model can't
    # "round down to N-1" when the Nth caption line literally says LAST.
    body_lines = []
    for i, c in enumerate(captions):
        marker = f"[{i + 1}]" if i + 1 < n else f"[{n}] (LAST — MUST BE IN OUTPUT)"
        body_lines.append(f"{marker} {c}")
    body = "\n\n".join(body_lines)

    # Trailing reminder: the model pays more attention to the last few hundred
    # tokens than to anything earlier. Putting the count requirement here
    # (after the captions, right before generation begins) is the single most
    # effective change for suppressing the N-1 stop.
    footer = (
        f"\n\n---\n"
        f"REMEMBER: output MUST contain EXACTLY {n} objects — one for each of "
        f"captions [1] through [{n}]. Before emitting the closing `]}}`, verify "
        f"you have written object [{n}]. If not, emit it now."
    )
    return f"{header}\n\n{body}{footer}"


def build_tagged_system_prompt(
    all_feature_names: list[str],
    feature_categories: dict[str, str],
    *,
    schema_enforced: bool = False,
) -> str:
    """
    Optional variant for pre-tagged inputs. Input is a list of raw feature tag
    strings already extracted from each caption; the LLM maps each tag to a
    canonical value and groups them per category. Shares the same output
    contract as ``build_llm_system_prompt``.
    """
    if schema_enforced:
        return """You are a clinical dermatology NLP specialist converting pre-extracted feature tags into canonical per-category one-hot encoding.

The API enforces the output shape via a strict JSON schema. Your job is to FILL IN values per tag-list.

FOR EACH INPUT TAG-LIST, produce ONE object containing EVERY category key. For each category:
  • Emit []  if no input tag maps to that category (DEFAULT for most categories).
  • Emit [<canonical_value>, ...] containing every tag that maps to that category.

RULES
1. Map each raw tag to the CLOSEST canonical value allowed by the schema's enum (e.g. "erythematous" → morphology_color:"red", "pruritus" → symptoms_dermatological:"itching", "papule" → morphology_texture:"papular" or "raised").
2. Drop tags that do not map to any canonical value — do NOT invent categories or values.
3. NEVER infer features not present in the input tag list.

COUNT / ORDER
- The user message starts with "INPUT: N tag-list(s)". Produce EXACTLY N caption objects in input order.
- Emit every category key in every object (use [] for categories with no mapped tags).
- Do not repeat, skip, or add extra caption objects.

FEW-SHOT EXAMPLES (only non-empty categories shown; every other category MUST be present with value []):

[E1] Tags: ["red", "raised", "itchy", "forearm"]
    Non-empty → body_location:["forearm"], morphology_color:["red"],
                morphology_texture:["raised"], symptoms_dermatological:["itching"]

[E2] Tags: []                                   (no tags → everything [])
    Non-empty → (none)

[E3] Tags: ["melanoma"]                        (disease tag alone does NOT anchor anything)
    Non-empty → (none)

[E4] Tags: ["erythematous", "pruritus", "scaly"]   (synonym mapping)
    Non-empty → morphology_color:["red"], morphology_texture:["scaly"],
                symptoms_dermatological:["itching"]

[E5] Tags: ["30-year-old", "female", "red bump", "cheek"]
    Non-empty → demographics_age:["30-40"], demographics_sex:["female"],
                body_location:["cheek"], morphology_color:["red"],
                morphology_texture:["raised"]

Follow these examples literally. Drop any tag that has no clear canonical mapping.
"""

    vocab_block = _format_vocabulary_for_prompt(all_feature_names, feature_categories)

    return f"""You are a clinical dermatology NLP specialist converting pre-extracted feature tags into canonical per-category one-hot encoding.

TASK
Produce ONE top-level JSON OBJECT with a single key "captions" whose value is a JSON array containing ONE object per input (a list of raw feature tag strings), in input order. Each inner object maps CATEGORY → list of CANONICAL VALUES from the vocabulary below.

REQUIRED OUTPUT SHAPE
{{
  "captions": [
    {{ <category>: [<canonical_value>, ...], ... }},
    ...
  ]
}}

COUNT RULES
- The user message starts with "INPUT: N tag-list(s)". Your "captions" array MUST have EXACTLY N elements, in input order.
- Emit {{}} at any position where no tag maps — NEVER skip a position.
- DO NOT emit more than N objects. After the N-th object, close the array `]` and the outer object `}}` and STOP.
- DO NOT repeat objects. Return ONLY the JSON object — no markdown, no commentary.

RULES
1. Include a category ONLY if at least one input tag maps to a canonical value in that category.
2. OMIT categories for which no input tag maps. Omitted categories decode to value 2 (unknown) downstream.
3. Map each tag to the closest canonical value (e.g. "erythematous" → morphology_color:"red", "pruritus" → symptoms_dermatological:"itching", "papule" → morphology_texture:"papular" or "raised"). Drop tags that do not map cleanly.
4. NEVER invent features that are not derivable from the input tag list.
5. Do NOT output empty lists like `"body_location": []` — omit the category key entirely instead.

FEW-SHOT EXAMPLES (omit any category not shown — omitted = unknown)

[E1] Tags: ["red", "raised", "itchy", "forearm"]
  → {{"body_location":["forearm"],"morphology_color":["red"],"morphology_texture":["raised"],"symptoms_dermatological":["itching"]}}

[E2] Tags: []                                    (no tags)
  → {{}}

[E3] Tags: ["melanoma"]                         (disease-only tag — NO anchors)
  → {{}}

[E4] Tags: ["erythematous", "pruritus", "scaly"]   (synonym mapping)
  → {{"morphology_color":["red"],"morphology_texture":["scaly"],"symptoms_dermatological":["itching"]}}

[E5] Tags: ["30-year-old", "female", "red bump", "cheek"]
  → {{"demographics_age":["30-40"],"demographics_sex":["female"],"body_location":["cheek"],"morphology_color":["red"],"morphology_texture":["raised"]}}

VOCABULARY (valid canonical values per category):
{vocab_block}

Return ONLY the JSON object {{"captions":[...]}} — nothing else.
"""


def build_tagged_user_prompt(tag_lists: list[list[str]], *, schema_enforced: bool = False) -> str:
    """User message when using pre-tagged feature lists instead of captions.

    Same off-by-one defences as build_extraction_user_prompt — see that
    function's docstring for the rationale.
    """
    n = len(tag_lists)
    count_clause = (
        f"INPUT: {n} tag-list(s), indexed 1 through {n} inclusive (so the last "
        f"list is [{n}] and MUST be in your output)."
    )
    if schema_enforced:
        header = (
            f"{count_clause}\n"
            f"Emit EXACTLY {n} caption objects in the same order. Include every "
            f"category key in each object (use [] for categories no tag maps to)."
        )
    else:
        header = (
            f"{count_clause}\n"
            f"Your 'captions' array MUST contain EXACTLY {n} objects in the same "
            f"order. Emit {{}} for any input with no mappable tags — NEVER skip, "
            f"NEVER repeat, NEVER emit extra items."
        )
    body_lines = []
    for i, tags in enumerate(tag_lists):
        marker = f"[{i + 1}]" if i + 1 < n else f"[{n}] (LAST — MUST BE IN OUTPUT)"
        body_lines.append(f"{marker} {', '.join(tags) if tags else '(empty)'}")
    body = "\n".join(body_lines)
    footer = (
        f"\n\n---\n"
        f"REMEMBER: output MUST contain EXACTLY {n} objects — one for each of "
        f"inputs [1] through [{n}]. Before emitting the closing `]}}`, verify "
        f"you have written object [{n}]. If not, emit it now."
    )
    return f"{header}\n\n{body}{footer}"


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
    """Map category -> set of canonical value suffixes (e.g. 'red', 'forearm').

    Values are NOT lowercased here: downstream `_expand_category_output_to_encoding`
    does a case-normalised membership test against a lookup that matches by
    lowercased key, so preserving the original casing here is fine.
    """
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
    cat_val_to_name: dict[tuple[str, str], str] | None = None,
) -> tuple[dict[str, int], int]:
    """
    Convert one caption's {"category": ["value1", ...]} dict into the full
    {feature_name: 0|1|2} encoding. Returns (encoding, num_fixes).

    Tri-state semantics (matches the user's original spec):
      • Category present with ≥1 recognized value
          → listed values = 1 (present), OTHER values in that category = 0
            (inferably absent — we know it's X, so it's not Y or Z).
      • Category present with an empty list [] OR only unrecognized values
          → stays as 2 (unknown). The model couldn't anchor the category to
            any canonical value, so we make NO inference — neither present
            nor absent.
      • Category absent from the dict (only possible in json_object fallback)
          → stays as 2 (unknown).

    The empty-list=unknown rule is what keeps all three states alive under
    strict-schema mode (where every category is REQUIRED): most categories
    come back as ``[]`` for a typical caption, and they correctly decode to 2.
    """
    fixes = 0
    # Default every feature to 2 (unknown). Overwrite only when the LLM
    # EXPLICITLY anchors a category with at least one recognized value.
    result = {name: 2 for name in all_feature_names}

    if not isinstance(cat_output, dict):
        return result, 1

    # Pre-compute feature name per (category, value) — the hot path of the
    # pipeline. Cache across invocations when the caller provides the map.
    if cat_val_to_name is None:
        cat_val_to_name = {}
        for name in all_feature_names:
            cat = feature_categories.get(name, "other")
            val = name[len(cat) + 1:] if name.startswith(cat + "_") else name
            cat_val_to_name[(cat, val)] = name

    for cat_key, vals in cat_output.items():
        cat = str(cat_key).strip()
        if cat not in category_to_values:
            fixes += 1
            continue

        # Tolerate None (nullable schema) and scalar strings by normalising.
        if vals is None:
            continue  # unknown — stays 2
        if not isinstance(vals, list):
            if isinstance(vals, str):
                vals = [vals]
            else:
                fixes += 1
                continue

        valid_set = category_to_values[cat]
        # Case-insensitive lookup: build a lowercase→canonical map for the
        # category so models emitting "Forearm" match a schema "forearm" and
        # vice-versa. Tiny cost since valid_set is 5-50 values.
        canon_lc = {v.lower(): v for v in valid_set}
        recognized: list[str] = []
        for v in vals:
            s = str(v).strip().lower().replace(" ", "_")
            if not s:
                fixes += 1
                continue
            # Tolerate "morphology_color_red" style → strip category prefix.
            prefix = (cat + "_").lower()
            if s.startswith(prefix) and s[len(prefix):] in canon_lc:
                s = s[len(prefix):]
            canon = canon_lc.get(s)
            if canon is not None:
                recognized.append(canon)
            else:
                fixes += 1

        if not recognized:
            # Category present with empty list or only unrecognised values:
            # keep as 2 (unknown). The LLM didn't anchor the category, so we
            # make no claims about present/absent.
            continue

        # Mark present values = 1, all other features in this category = 0.
        present_set = set(recognized)
        for val in valid_set:
            fname = cat_val_to_name.get((cat, val))
            if fname is None:
                continue
            result[fname] = 1 if val in present_set else 0

    return result, fixes


_WRAPPED_ARRAY_KEYS = ("captions", "results", "data", "outputs", "items")


def _extract_captions_array(parsed) -> list | None:
    """
    Pull the inner per-caption array out of the LLM's top-level object.

    We insist on a wrapped object because we force `response_format=json_object`.
    This function returns None if the parsed JSON isn't a known wrapped shape,
    so the caller can fail fast instead of misinterpreting a shape mismatch as
    a one-caption response.
    """
    if isinstance(parsed, list):
        # If JSON mode is off, the model may emit a bare array — accept it.
        return parsed
    if isinstance(parsed, dict):
        for key in _WRAPPED_ARRAY_KEYS:
            val = parsed.get(key)
            if isinstance(val, list):
                return val
        # Some models emit {"0": {...}, "1": {...}, ...}. Accept that shape too.
        numeric_dict = all(
            isinstance(k, str) and k.isdigit() and isinstance(v, dict)
            for k, v in parsed.items()
        ) and len(parsed) > 0
        if numeric_dict:
            return [parsed[k] for k in sorted(parsed.keys(), key=int)]
    return None


def _salvage_truncated_captions_array(txt: str) -> list[dict] | None:
    """
    When the LLM response was cut off mid-string (finish_reason=length),
    `json.loads` fails. Locate the opening of the "captions" array (or any
    bare array start) and walk it incrementally with `raw_decode`, returning
    every top-level object that WAS fully emitted before the truncation point.
    Returns None if nothing can be salvaged.
    """
    if not txt:
        return None

    # Prefer the "captions" array start.
    idx = -1
    m = re.search(r'"captions"\s*:\s*\[', txt)
    if m:
        idx = m.end() - 1  # position of "["
    if idx < 0:
        idx = txt.find("[")
    if idx < 0:
        return None

    dec = json.JSONDecoder()
    i = idx + 1  # skip "["
    objects: list[dict] = []
    n = len(txt)
    while i < n:
        while i < n and txt[i] in " \t\r\n,":
            i += 1
        if i >= n or txt[i] == "]":
            break
        try:
            obj, end = dec.raw_decode(txt, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            objects.append(obj)
        i = end

    return objects if objects else None


def _response_tail_hint(text: str) -> str:
    if not text:
        return "empty response"
    tail = text[-80:].replace("\n", " ")
    return f"last 80 chars: {tail!r}"


def parse_extraction_response_text(
    text: str,
    captions: list[str],
    all_feature_names: list[str],
    feature_categories: dict[str, str] | None = None,
    *,
    finish_reason: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Parse the LLM response into per-caption one-hot dicts.

    Expected top-level shape (matches the system prompt):
        {"captions": [ {<cat>:[...], ...}, {...}, ... ]}

    Also tolerates alternate wrapper keys (``results``, ``data``, ``outputs``,
    ``items``), numeric-keyed dicts, and plain arrays when JSON mode is off.

    Returns (results, stats). The caller uses ``stats`` to decide how to retry:
      - stats["success"]    True iff ALL captions were parsed cleanly.
      - stats["salvaged"]   True iff we recovered a prefix from truncated JSON.
      - stats["truncated"]  True iff the parse failure looks like mid-string
                            truncation (either ``finish_reason == 'length'`` or
                            salvage recovered some objects). Drives bisect.
      - stats["shape_mismatch"]  True iff the response parsed fully but the
                            shape didn't match what we asked for. No point in
                            bisecting — just mark the batch failed.
    """
    stats: dict = {
        "success": False,
        "validation_fixes": 0,
        "salvaged": False,
        "truncated": False,
        "shape_mismatch": False,
    }
    fallback_row = {k: 2 for k in all_feature_names}

    if feature_categories is None:
        # Safer fallback than .split("_", 1)[0]: use everything before the last
        # "_" so compound categories (symptoms_dermatological_itching →
        # "symptoms_dermatological") survive. If a name has no "_", use itself.
        feature_categories = {}
        for name in all_feature_names:
            idx = name.rfind("_")
            feature_categories[name] = name[:idx] if idx > 0 else name

    category_to_values = _build_category_to_features_map(
        all_feature_names, feature_categories
    )
    # Build (cat, val) → feature_name ONCE for all captions in this batch.
    cat_val_to_name: dict[tuple[str, str], str] = {}
    for name in all_feature_names:
        cat = feature_categories.get(name, "other")
        val = name[len(cat) + 1:] if name.startswith(cat + "_") else name
        cat_val_to_name[(cat, val)] = name

    def _finalize(parsed_list: list, *, salvaged: bool) -> tuple[list[dict], dict]:
        results: list[dict] = []
        fixes = 0
        nontrivial_prefix = 0  # items in the LLM's prefix with ≥1 extracted value
        for item in parsed_list:
            if not isinstance(item, dict):
                fixes += 1
                item = {}
            encoding, f = _expand_category_output_to_encoding(
                item,
                all_feature_names,
                feature_categories,
                category_to_values,
                cat_val_to_name=cat_val_to_name,
            )
            fixes += f
            results.append(encoding)
            # Count any 0 or 1 (anchored present/absent) — all-2 rows are
            # either legitimately uninformative captions OR the "model gave
            # up and emitted {}" failure mode. We can't tell them apart for
            # any single caption, but the PATTERN (every prefix item all-2)
            # is a reliable signal of the latter.
            if isinstance(item, dict) and any(v != 2 for v in encoding.values()):
                nontrivial_prefix += 1

        num_llm_items = len(results)
        short = len(captions) - num_llm_items
        if short > 0:
            # CRITICAL: use a list comprehension — NOT `[dict(fallback_row)] * short`.
            # The `* short` form puts `short` references to the SAME dict in the
            # list, so any later mutation of one row silently corrupts all
            # padded rows. A comprehension gives independent dicts.
            results.extend(dict(fallback_row) for _ in range(short))
            fixes += short
        results = results[: len(captions)]
        # Record how many rows actually came from the LLM (pre-padding) so
        # callers can slice precisely without sniffing all-2 rows (which can
        # be a legitimate "everything unknown" caption in the new semantics).
        stats["num_llm_items"] = num_llm_items
        stats["num_nontrivial_prefix"] = nontrivial_prefix

        stats["validation_fixes"] = fixes
        stats["salvaged"] = salvaged
        # Partial salvage is not a full success — caller will retry the tail
        # (or accept the partial + unknown rows depending on policy).
        stats["success"] = not salvaged and short == 0
        return results, stats

    txt = (text or "").strip()
    txt = re.sub(r"```json\s*|```\s*", "", txt).strip()

    n = len(captions)

    # Path 1: JSON parses — figure out the shape.
    try:
        parsed = json.loads(txt)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Path 2: JSON broken — try salvage (typical for truncation).
        salvaged = _salvage_truncated_captions_array(txt)
        if salvaged:
            # Loop case: model repeated items past our requested count. Take
            # the first N as a full success (they were well-formed).
            if len(salvaged) >= n:
                tqdm.write(
                    f"    parse_extraction_response_text: {str(e)[:120]} — "
                    f"model looped (emitted {len(salvaged)}/{n} items); keeping "
                    f"first {n} as a clean recovery."
                )
                results, _ = _finalize(salvaged[:n], salvaged=False)
                stats["truncated"] = False
                stats["salvaged"] = False
                stats["success"] = True
                return results, stats
            # True truncation — partial prefix recovered.
            stats["truncated"] = True
            tqdm.write(
                f"    parse_extraction_response_text: {str(e)[:120]} — "
                f"salvaged {len(salvaged)}/{n} caption(s) from truncated JSON "
                f"(finish_reason={finish_reason!r})"
            )
            return _finalize(salvaged, salvaged=True)
        # No salvage possible. Tag as truncated only if API says so.
        stats["truncated"] = finish_reason == "length"
        tqdm.write(
            f"    parse_extraction_response_text: {str(e)[:160]} "
            f"(finish_reason={finish_reason!r}, {_response_tail_hint(txt)})"
        )
        return [dict(fallback_row) for _ in captions], stats

    inner = _extract_captions_array(parsed)
    if inner is None:
        stats["shape_mismatch"] = True
        top_keys = list(parsed.keys())[:6] if isinstance(parsed, dict) else None
        tqdm.write(
            f"    parse_extraction_response_text: unexpected JSON shape "
            f"(type={type(parsed).__name__}, top_keys={top_keys}). "
            f"Expected {{'captions': [...]}}."
        )
        return [dict(fallback_row) for _ in captions], stats

    # Loop case #2: JSON parsed AND array has more items than we asked for
    # (model repeated outputs). First N items are still correct; take them.
    if len(inner) > n:
        tqdm.write(
            f"    parse_extraction_response_text: model looped "
            f"(emitted {len(inner)}/{n} items); trimming to first {n}."
        )
        return _finalize(inner[:n], salvaged=False)

    # Short response case: JSON parsed but array has fewer items than asked.
    # This happens for two reasons:
    #   (a) finish_reason='length' → real truncation,
    #   (b) finish_reason='stop'   → model skipped captions (rare under strict
    #       schema; common under json_object).
    # Either way, keep the good prefix and retry only the missing tail.
    if len(inner) < n:
        stats["truncated"] = True  # use truncation path → tail-only sync retry
        tqdm.write(
            f"    parse_extraction_response_text: short response "
            f"({len(inner)}/{n} items, finish_reason={finish_reason!r}); "
            f"keeping prefix, will retry missing tail."
        )
        return _finalize(inner, salvaged=True)

    return _finalize(inner, salvaged=False)


# ══════════════════════════════════════════════════════════════════════════════
# SYNC EXTRACTION (single batch LLM call with retries)
# ══════════════════════════════════════════════════════════════════════════════

MAX_BISECT_DEPTH = _safe_int_env("MAX_BISECT_DEPTH", 4, vmin=0, vmax=8)


def extract_features_batch(
    captions: list[str],
    all_feature_names: list[str],
    system_prompt: str,
    retries: int = MAX_RETRIES,
    *,
    tag_lists: list[list[str]] | None = None,
    feature_categories: dict[str, str] | None = None,
    json_schema: dict | None = None,
    _bisect_depth: int = 0,
    retry_reminder: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Sync LLM call for one batch of captions (or pre-tagged feature lists), with
    retry + bisect-on-truncation.

    Failure policy:
      - finish_reason == "length" (truncated JSON): retry, then bisect smaller.
      - All other failures (shape mismatch, unparseable garbage, runaway loops):
        retry up to ``retries`` times. Do NOT bisect — smaller batches would
        not help because the LLM is misbehaving regardless of size. Return the
        best salvaged partial if any, else all-unknown rows.

    Bisect depth is capped at ``MAX_BISECT_DEPTH`` (default 4 → at most 16×
    the original batch count in API calls) so a pathological truncation loop
    can't burn through thousands of API calls before failing out.

    Returns (list of feature dicts aligned to input captions, stats dict).
    """
    schema_enforced = json_schema is not None and OPENAI_USE_JSON_SCHEMA
    if tag_lists is not None:
        user_prompt = build_tagged_user_prompt(tag_lists, schema_enforced=schema_enforced)
    else:
        user_prompt = build_extraction_user_prompt(captions, schema_enforced=schema_enforced)

    # retry_reminder: when a previous tail/truncation retry came back all-2s
    # (likely model-gave-up on single-caption retry emitting `{"captions":[{}]}`),
    # the caller passes an explicit reminder that prepends onto the user prompt
    # forbidding the `{}` shortcut. This is the ONLY case where we modify the
    # cached-prefix-friendly layout, so keep it as a short prepended note that
    # doesn't disturb the system prompt cache.
    if retry_reminder:
        user_prompt = f"{retry_reminder.strip()}\n\n{user_prompt}"

    stats = {
        "attempts": 0,
        "success": False,
        "validation_fixes": 0,
        "num_nontrivial_prefix": 0,
        "num_llm_items": 0,
    }

    best_partial: list[dict] | None = None
    saw_truncation = False
    last_failure_kind = "unknown"
    last_short_n: int | None = None
    short_repeat_count = 0

    for attempt in range(retries):
        try:
            stats["attempts"] = attempt + 1
            text, meta = call_llm(
                user_prompt, system_prompt, retries=1, json_schema=json_schema,
                n_captions=len(captions),
            )
            finish_reason = meta.get("finish_reason", "")
            valid_results, pst = parse_extraction_response_text(
                text, captions, all_feature_names, feature_categories,
                finish_reason=finish_reason,
            )
            stats["validation_fixes"] = pst.get("validation_fixes", 0)
            stats["num_nontrivial_prefix"] = pst.get("num_nontrivial_prefix", 0)
            stats["num_llm_items"] = pst.get("num_llm_items", 0)
            if pst.get("success"):
                stats["success"] = True
                return valid_results, stats

            if pst.get("truncated") or finish_reason == "length":
                saw_truncation = True
                last_failure_kind = "truncated"
            elif pst.get("shape_mismatch"):
                last_failure_kind = "shape_mismatch"
            else:
                last_failure_kind = "parse_error"

            # Keep the longest successful prefix across attempts so a consistently
            # truncating batch still returns real data for the recovered captions.
            if pst.get("salvaged"):
                best_partial = valid_results

            # Early-bisect heuristic: when the model keeps stopping at the same
            # prefix length across attempts (e.g. 14/15, then 14/15 again), a
            # third retry will almost certainly produce the same short output.
            # Break out of the retry loop and go straight to bisection — this
            # saves ~1 API call per truncated batch and cuts end-to-end time
            # noticeably when hundreds of batches truncate the same way.
            cur_short_n = pst.get("num_llm_items")
            if (
                saw_truncation
                and cur_short_n is not None
                and cur_short_n == last_short_n
                and len(captions) > 1
                and tag_lists is None
            ):
                short_repeat_count += 1
                if short_repeat_count >= 1:  # saw same short-count twice → bisect now
                    break
            else:
                short_repeat_count = 0
            last_short_n = cur_short_n

            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            tqdm.write(f"    JSON parse error (attempt {attempt+1}): {str(e)[:80]}")
            last_failure_kind = "parse_error"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            err_str = str(e)
            tqdm.write(f"    LLM attempt {attempt+1} failed: {err_str[:120]}")
            last_failure_kind = "api_error"
            if attempt < retries - 1:
                if "429" in err_str or "quota" in err_str.lower():
                    time.sleep(10 * (attempt + 1))
                else:
                    time.sleep(2 ** attempt)

    # All retries exhausted. Bisect ONLY when the root cause was truncation —
    # splitting a shape-mismatch batch just multiplies wasted calls because
    # every half will hit the same LLM misbehaviour. Cap recursion depth so a
    # pathological truncation loop can't spiral.
    if (
        saw_truncation
        and len(captions) > 1
        and tag_lists is None
        and _bisect_depth < MAX_BISECT_DEPTH
    ):
        mid = len(captions) // 2
        left_caps, right_caps = captions[:mid], captions[mid:]
        # Decay retries on bisect. The first bisect already used `retries`
        # attempts on the parent; each child is half the size so far less
        # likely to truncate again. Using the same `retries` at every depth
        # makes worst-case `retries * 2^MAX_BISECT_DEPTH` API calls for one
        # pathological batch (5 * 16 = 80 calls). Decay to 2 attempts per
        # half after the first split keeps it bounded at ~O(N) calls.
        child_retries = 2 if _bisect_depth >= 0 else retries
        tqdm.write(
            f"    Bisecting TRUNCATED batch ({len(captions)} → "
            f"{len(left_caps)} + {len(right_caps)}) after {retries} attempts "
            f"(depth {_bisect_depth + 1}/{MAX_BISECT_DEPTH}, "
            f"child_retries={child_retries})"
        )
        left, st_l = extract_features_batch(
            left_caps, all_feature_names, system_prompt,
            retries=child_retries, feature_categories=feature_categories,
            json_schema=json_schema, _bisect_depth=_bisect_depth + 1,
        )
        right, st_r = extract_features_batch(
            right_caps, all_feature_names, system_prompt,
            retries=child_retries, feature_categories=feature_categories,
            json_schema=json_schema, _bisect_depth=_bisect_depth + 1,
        )
        stats["validation_fixes"] += (
            st_l.get("validation_fixes", 0) + st_r.get("validation_fixes", 0)
        )
        stats["success"] = bool(st_l.get("success") and st_r.get("success"))
        stats["num_nontrivial_prefix"] = (
            st_l.get("num_nontrivial_prefix", 0) + st_r.get("num_nontrivial_prefix", 0)
        )
        stats["num_llm_items"] = (
            st_l.get("num_llm_items", 0) + st_r.get("num_llm_items", 0)
        )
        return left + right, stats

    if saw_truncation and _bisect_depth >= MAX_BISECT_DEPTH:
        tqdm.write(
            f"    Bisect depth cap ({MAX_BISECT_DEPTH}) reached for a "
            f"{len(captions)}-caption batch; giving up instead of subdividing further."
        )

    if best_partial is not None:
        unknown_count = sum(
            1 for row in best_partial if all(v == 2 for v in row.values())
        )
        tqdm.write(
            f"    All {retries} attempts failed ({last_failure_kind}). "
            f"Using salvaged partial: {len(captions) - unknown_count}/{len(captions)} "
            f"recovered, {unknown_count} left as unknown."
        )
        # Recompute nontrivial count from the partial we're actually returning,
        # so callers can detect "retry came back all-unknown" pathology.
        stats["num_nontrivial_prefix"] = sum(
            1 for row in best_partial if any(v != 2 for v in row.values())
        )
        stats["num_llm_items"] = len(best_partial)
        return best_partial, stats

    tqdm.write(
        f"    ERROR: All {retries} attempts failed ({last_failure_kind}). "
        f"Returning {len(captions)} unknown rows."
    )
    fallback = [{k: 2 for k in all_feature_names} for _ in captions]
    stats["num_nontrivial_prefix"] = 0
    stats["num_llm_items"] = 0
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
    ckpt_save_fn=None,
    seed_results: dict[str, dict] | None = None,
    json_schema: dict | None = None,
) -> dict[str, dict]:
    """
    Process deduplicated captions via OpenAI Batch API.

    Returns {caption_text: {feature_name: value}}.

    Incremental checkpointing: ``ckpt_save_fn(partial_results)`` (if provided)
    is called after every batch chunk finishes, so a mid-run crash loses at
    most one chunk's worth of captions.
    """
    jobs: list[dict] = []
    job_batches: dict[str, list[str]] = {}
    schema_enforced = json_schema is not None and OPENAI_USE_JSON_SCHEMA

    for batch_idx, batch_start in enumerate(range(0, len(unique_captions), LLM_BATCH_SIZE)):
        batch = unique_captions[batch_start : batch_start + LLM_BATCH_SIZE]
        if caption_to_tags is not None:
            tag_lists = [caption_to_tags.get(c, []) for c in batch]
            user_prompt = build_tagged_user_prompt(tag_lists, schema_enforced=schema_enforced)
        else:
            user_prompt = build_extraction_user_prompt(batch, schema_enforced=schema_enforced)
        cid = f"p2_{batch_idx}"
        jobs.append(
            {
                "custom_id": cid,
                "body": _openai_extraction_chat_body(
                    system_prompt, user_prompt, json_schema=json_schema,
                    n_captions=len(batch),
                ),
            }
        )
        job_batches[cid] = batch

    global_stats["total_batches"] = len(jobs)
    global_stats["total_api_calls"] = len(jobs)

    # Accumulator shared between the callback (per chunk) and the final sweep.
    results: dict[str, dict] = dict(seed_results) if seed_results else {}

    def _parse_and_accumulate(cid: str, item_meta: dict) -> bool:
        """Parse one batch item into per-caption one-hot dicts; returns ok."""
        batch = job_batches.get(cid, [])
        if not batch:
            return True
        text = item_meta.get("content") or ""
        finish_reason = item_meta.get("finish_reason") or ""
        batch_results, pst = parse_extraction_response_text(
            text, batch, all_feature_names, feature_categories,
            finish_reason=finish_reason,
        )
        ok = pst.get("success", False)
        global_stats["total_validation_fixes"] += pst.get("validation_fixes", 0)

        # If parse wasn't a clean success, recover what we can and only retry
        # the unsalvaged tail via sync. Shape-mismatch failures skip sync retry
        # entirely because a retry will hit the same misbehaviour.
        if not ok:
            # Exact salvage prefix from the parser's own count. Sniffing
            # "all-2 rows" is unreliable now that genuinely-unknown captions
            # legitimately produce all-2 output under strict-schema semantics.
            num_llm_items = pst.get("num_llm_items")
            if num_llm_items is None:
                # Fallback only if the parser version somehow didn't set it:
                # take a conservative zero so the whole batch is retried.
                salvaged_prefix = 0
            else:
                salvaged_prefix = min(int(num_llm_items), len(batch))

            is_truncation = pst.get("truncated") or finish_reason == "length"
            is_shape = pst.get("shape_mismatch")
            sync_retries = 2

            # "Model gave up" detection: the Batch API said finish_reason='stop'
            # (not 'length' → not real truncation) AND the short prefix it did
            # emit is mostly {} objects that decode to all-unknown. This is
            # the 9/10 early-stop pattern with gpt-4o-mini: the model closes
            # the array early with empty objects. Trusting those empty rows
            # means 9 out of 10 captions in the batch get written as all-2
            # (unknown) when real features were extractable. We must redo
            # the WHOLE batch synchronously, not just the tail.
            nontrivial_prefix = pst.get("num_nontrivial_prefix", salvaged_prefix)
            is_all_empty_salvage = (
                is_truncation
                and finish_reason == "stop"
                and salvaged_prefix > 0
                and salvaged_prefix < len(batch)
                and nontrivial_prefix == 0
            )

            if is_shape and not pst.get("salvaged"):
                tqdm.write(
                    f"    Shape mismatch for batch {cid} "
                    f"(finish_reason={finish_reason!r}) — marking failed, "
                    f"no sync retry (would repeat the same error)."
                )
                # batch_results is already full of unknown rows from the parser.
            elif is_all_empty_salvage:
                tqdm.write(
                    f"    Model-gave-up pattern for batch {cid}: "
                    f"{salvaged_prefix}/{len(batch)} emitted but ALL empty "
                    f"(finish_reason='stop'). Discarding prefix and "
                    f"retrying ENTIRE batch synchronously (sync retries={sync_retries})."
                )
                batch_tags = (
                    [caption_to_tags.get(c, []) for c in batch]
                    if caption_to_tags is not None else None
                )
                retry_results, st2 = extract_features_batch(
                    batch, all_feature_names, system_prompt,
                    retries=sync_retries,
                    tag_lists=batch_tags,
                    feature_categories=feature_categories,
                    json_schema=json_schema,
                )
                batch_results = retry_results
                ok = st2.get("success", False)
                global_stats["total_retries"] += 1
                global_stats["total_validation_fixes"] += st2.get("validation_fixes", 0)
            elif is_truncation:
                tail_batch = batch[salvaged_prefix:] if salvaged_prefix < len(batch) else []
                if tail_batch:
                    tqdm.write(
                        f"    Sync fallback for batch {cid} — truncation: "
                        f"salvaged {salvaged_prefix}/{len(batch)} "
                        f"(nontrivial={nontrivial_prefix}), "
                        f"retrying tail of {len(tail_batch)}."
                    )
                    batch_tags = (
                        [caption_to_tags.get(c, []) for c in tail_batch]
                        if caption_to_tags is not None else None
                    )
                    tail_results, st2 = extract_features_batch(
                        tail_batch, all_feature_names, system_prompt,
                        retries=sync_retries,
                        tag_lists=batch_tags,
                        feature_categories=feature_categories,
                        json_schema=json_schema,
                    )
                    global_stats["total_retries"] += 1
                    global_stats["total_validation_fixes"] += st2.get("validation_fixes", 0)

                    # GUARD AGAINST SILENT FAILURE: single-caption (or small)
                    # tail retries frequently come back as `{"captions":[{}]}`
                    # (completion ~6-40 tokens) — the model takes the prompt's
                    # `{}` shortcut and emits nothing. The parser accepts that
                    # as "success" but all features become 2 (unknown), silently
                    # losing data. Detect this and retry ONCE with an explicit
                    # reminder that forbids the `{}` shortcut.
                    tail_nontriv = st2.get("num_nontrivial_prefix", 0)
                    tail_has_substance = any(
                        len((c or "").strip()) >= 15 for c in tail_batch
                    )
                    if tail_nontriv == 0 and tail_has_substance:
                        reminder = (
                            "RETRY NOTICE: the previous pass for these captions "
                            "returned an empty `{}` object. That is ONLY valid if "
                            "the caption truly has NO extractable clinical content. "
                            "For every caption below, re-read it carefully and "
                            "emit every category whose value is literally mentioned "
                            "(e.g. body_location, morphology_color, morphology_texture, "
                            "demographics, symptoms, duration). Do NOT emit `{}` "
                            "unless the caption is genuinely devoid of clinical detail."
                        )
                        tqdm.write(
                            f"    Tail retry for {cid} returned all-unknown "
                            f"({len(tail_batch)} caption(s), likely model-gave-up "
                            f"on single-caption retry). Re-retrying with explicit "
                            f"reminder prompt."
                        )
                        tail_results2, st3 = extract_features_batch(
                            tail_batch, all_feature_names, system_prompt,
                            retries=1,
                            tag_lists=(
                                [caption_to_tags.get(c, []) for c in tail_batch]
                                if caption_to_tags is not None else None
                            ),
                            feature_categories=feature_categories,
                            json_schema=json_schema,
                            retry_reminder=reminder,
                        )
                        tail_nontriv2 = st3.get("num_nontrivial_prefix", 0)
                        global_stats["total_retries"] += 1
                        global_stats["total_validation_fixes"] += st3.get("validation_fixes", 0)
                        if tail_nontriv2 > tail_nontriv:
                            # Reminder pass produced real features — use it.
                            tail_results = tail_results2
                            st2 = st3
                            tail_nontriv = tail_nontriv2
                        else:
                            # Still nothing. Log the captions we're giving up on.
                            global_stats.setdefault("silent_empty_tail_captions", 0)
                            global_stats["silent_empty_tail_captions"] += len(tail_batch)
                            previews = " | ".join(
                                (c or "")[:60].replace("\n", " ") for c in tail_batch[:3]
                            )
                            tqdm.write(
                                f"    WARNING: tail captions for {cid} marked "
                                f"all-unknown after reminder retry (model "
                                f"persistently refuses to extract). "
                                f"Preview: {previews!r}"
                            )

                    batch_results = batch_results[:salvaged_prefix] + tail_results
                    ok = st2.get("success", False)
                else:
                    ok = True  # full batch salvaged
            else:
                # Generic parse error: retry once synchronously, no bisect.
                tqdm.write(
                    f"    Sync fallback for batch {cid} — parse error "
                    f"(finish_reason={finish_reason!r}), retrying full batch "
                    f"(sync retries={sync_retries})."
                )
                batch_tags = (
                    [caption_to_tags.get(c, []) for c in batch]
                    if caption_to_tags is not None else None
                )
                retry_results, st2 = extract_features_batch(
                    batch, all_feature_names, system_prompt,
                    retries=sync_retries,
                    tag_lists=batch_tags,
                    feature_categories=feature_categories,
                    json_schema=json_schema,
                )
                batch_results = retry_results
                ok = st2.get("success", False)
                global_stats["total_retries"] += 1
                global_stats["total_validation_fixes"] += st2.get("validation_fixes", 0)

        if ok:
            global_stats["successful_batches"] += 1
        else:
            global_stats["failed_batches"] += 1

        for cap, feats in zip(batch, batch_results):
            results[cap] = feats
        return ok

    # Checkpoint throttling: the checkpoint file is the full dedup result dict
    # (~20-50 MB JSON for 70K captions × 300 features). Writing it after
    # EVERY chunk made I/O a bottleneck — and since the callback is now
    # under callback_lock, a 5-second write also blocks the next chunk's
    # parse work. We throttle to "at most one save every N seconds" and
    # always do a final save at the end. Worst-case data loss: one interval.
    ckpt_min_interval_s = _safe_int_env("OPENAI_CKPT_MIN_INTERVAL_SEC", 120, vmin=0)
    _last_ckpt_time = [0.0]

    def _on_chunk_complete(chunk_mapping: dict[str, dict]) -> None:
        """Called after each Batch API chunk finishes — parse + checkpoint."""
        for cid, item_meta in chunk_mapping.items():
            _parse_and_accumulate(cid, item_meta)
        if ckpt_save_fn is None:
            return
        now = time.monotonic()
        if now - _last_ckpt_time[0] < ckpt_min_interval_s:
            return
        try:
            ckpt_save_fn(dict(results))
            _last_ckpt_time[0] = now
        except Exception as e:
            tqdm.write(f"    Checkpoint save failed: {e!r} (continuing)")

    # Parsing happens inside _on_chunk_complete as each chunk finishes, so we
    # don't use the aggregated mapping here — but it's still returned for
    # diagnostic parity with earlier versions.
    _mapping, _acc = _run_openai_extraction_batch(
        jobs, per_chunk_callback=_on_chunk_complete
    )

    # Catch any jobs that produced no output at all (batch error, not parsed in
    # the chunk callback because they never appeared in the chunk mapping).
    already_processed = {cap for cap in results.keys()}
    for cid, batch in job_batches.items():
        for cap in batch:
            if cap not in already_processed:
                # This id was never returned by any chunk — retry synchronously.
                tqdm.write(
                    f"    Batch {cid} produced no output for at least one caption "
                    f"— sync fallback for the full batch."
                )
                batch_tags = (
                    [caption_to_tags.get(c, []) for c in batch]
                    if caption_to_tags is not None else None
                )
                retry_results, st2 = extract_features_batch(
                    batch, all_feature_names, system_prompt,
                    tag_lists=batch_tags,
                    feature_categories=feature_categories,
                    json_schema=json_schema,
                )
                for c, feats in zip(batch, retry_results):
                    results[c] = feats
                if st2.get("success"):
                    global_stats["successful_batches"] += 1
                else:
                    global_stats["failed_batches"] += 1
                global_stats["total_retries"] += 1
                break  # one sync retry per batch, move on

    if ckpt_save_fn is not None:
        try:
            ckpt_save_fn(dict(results))
        except Exception as e:
            tqdm.write(f"    Final checkpoint save failed: {e!r}")

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
    json_schema: dict | None = None,
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
            json_schema=json_schema,
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

def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (tmp file + os.replace).

    Prevents partial / truncated files when the process is killed mid-write —
    a real risk for the dedup checkpoint which is overwritten after every
    Batch-API chunk during long (multi-hour) runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _save_dedup_checkpoint(
    caption_features: dict[str, dict],
    all_feature_names: list[str],
    ckpt_path: Path,
) -> None:
    """Save caption->values checkpoint as compact {caption: [int, ...]} JSON.

    Written atomically so a killed process can't leave a truncated JSON on
    disk that would fail to re-load on the next run.
    """
    compact: dict[str, list[int]] = {}
    for cap, feats in caption_features.items():
        compact[cap] = [feats.get(fn, 2) for fn in all_feature_names]
    _atomic_write_text(ckpt_path, json.dumps(compact, ensure_ascii=False))


# ── Failed-caption tracking (resume retries instead of freezing all-2 rows) ─
FAILED_CAPS_FILE = "dedup_failed_captions.json"


def _save_failed_captions(failed: set[str], path: Path) -> None:
    """Persist the set of captions that exhausted all retries (sync + batch).

    On resume, ``run_extraction`` deliberately EXCLUDES these from
    ``caption_features`` so they are re-attempted — otherwise they'd sit in
    the checkpoint as all-2 rows and never be retried."""
    try:
        _atomic_write_text(path, json.dumps(sorted(failed), ensure_ascii=False))
    except Exception as e:
        tqdm.write(f"  WARNING: failed to save failed-captions file: {e!r}")


def _load_failed_captions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return {str(x) for x in raw}
    except Exception as e:
        tqdm.write(f"  WARNING: failed to load failed-captions file: {e!r}")
    return set()


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

    # ── Build strict JSON schema FIRST so we can shrink the prompt when ────
    #    structured outputs will enforce shape + vocabulary server-side.
    extraction_json_schema: dict | None = None
    if LLM_PROVIDER == "openai" and OPENAI_USE_JSON_SCHEMA:
        try:
            extraction_json_schema = _build_extraction_json_schema(
                all_feature_names, feature_categories
            )
            n_cats = len(extraction_json_schema["schema"]["properties"]["captions"]
                         ["items"]["properties"])
            print(
                f"  Strict JSON schema enabled: {n_cats} categories, "
                f"{len(all_feature_names)} total values — server-side shape enforced."
            )
            # Cache-key rotation: include a short hash of the schema body so
            # prompt-cache entries are invalidated whenever the schema (and
            # therefore the effective system-level contract) changes.
            import hashlib
            schema_hash = hashlib.sha1(
                json.dumps(extraction_json_schema, sort_keys=True).encode("utf-8")
            ).hexdigest()[:10]
            global OPENAI_PROMPT_CACHE_KEY  # noqa: PLW0603
            # Guard against re-entrant runs double-appending the same hash.
            if OPENAI_PROMPT_CACHE_KEY and not OPENAI_PROMPT_CACHE_KEY.endswith(f"_{schema_hash}"):
                OPENAI_PROMPT_CACHE_KEY = f"{OPENAI_PROMPT_CACHE_KEY}_{schema_hash}"
                print(f"  Effective prompt-cache key: {OPENAI_PROMPT_CACHE_KEY!r}")
        except Exception as e:
            print(
                f"  WARNING: failed to build JSON schema ({e!r}); falling back "
                f"to json_object response_format."
            )
            extraction_json_schema = None

    schema_enforced = extraction_json_schema is not None

    # ── Build system prompt ───────────────────────────────────────────────────
    if use_tags:
        system_prompt = build_tagged_system_prompt(
            all_feature_names, feature_categories, schema_enforced=schema_enforced
        )
    else:
        system_prompt = build_llm_system_prompt(
            all_feature_names, feature_categories, schema_enforced=schema_enforced
        )

    # ── Load checkpoint if exists ─────────────────────────────────────────────
    dedup_ckpt = CHECKPOINT_DIR / DEDUP_CKPT_FILE
    failed_caps_path = CHECKPOINT_DIR / FAILED_CAPS_FILE
    caption_features: dict[str, dict] = {}
    if dedup_ckpt.exists():
        try:
            caption_features = _load_dedup_checkpoint(dedup_ckpt, all_feature_names)
            print(f"  Loaded {len(caption_features):,} caption results from checkpoint")
        except Exception as e:
            print(f"  WARNING: checkpoint load failed ({e}), re-extracting all")
            caption_features = {}

    # Captions that previously exhausted all retries are deliberately kicked
    # back out of the checkpoint so they get another shot this run. They are
    # re-added on retry failure via _save_failed_captions below.
    known_failed = _load_failed_captions(failed_caps_path)
    if known_failed:
        requeued = [c for c in known_failed if c in caption_features]
        for c in requeued:
            caption_features.pop(c, None)
        if requeued:
            print(
                f"  Requeueing {len(requeued):,} caption(s) that previously failed "
                f"all retries (from {failed_caps_path.name})."
            )

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
        "silent_empty_tail_captions": 0,
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
                ckpt_save_fn=_ckpt_saver,
                seed_results=dict(caption_features),
                json_schema=extraction_json_schema,
            )
        else:
            new_results = _run_dedup_sync(
                remaining, all_feature_names, system_prompt, global_stats, _ckpt_saver,
                caption_to_tags=_tags_arg,
                feature_categories=feature_categories,
                json_schema=extraction_json_schema,
            )

        caption_features.update(new_results)
        _save_dedup_checkpoint(caption_features, all_feature_names, dedup_ckpt)
        # Record any captions that still failed so they get retried next run.
        still_failed = {c for c in remaining if c not in new_results}
        _save_failed_captions(still_failed, failed_caps_path)
        if still_failed:
            print(
                f"  {len(still_failed):,} caption(s) failed all retries this run "
                f"→ saved to {failed_caps_path.name} for next-run retry."
            )
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
    silent_empty = global_stats.get("silent_empty_tail_captions", 0)
    if silent_empty:
        print(
            f"  Silent-empty tail captions: {silent_empty:,} "
            f"(model persistently refused to extract even after reminder retry — "
            f"these are recorded as all-unknown; check captions.jsonl for context)"
        )

    print("\nReplicating features to all rows (including duplicates)...")
    # Skip per-row dict construction entirely: build the matrix by direct
    # column-wise assembly. This is ~20× faster than pd.DataFrame(list[dict])
    # on 180K rows × 300 cols and avoids the `append(fallback_row)` shared-
    # reference footgun where every empty row pointed to the same dict.
    fallback_vec = np.full(len(all_feature_names), 2, dtype=np.int8)
    matrix = np.empty((len(stripped), len(all_feature_names)), dtype=np.int8)
    # Fast path: many rows share the same caption → cache the built vector.
    vec_cache: dict[str, np.ndarray] = {}
    for r, cap in enumerate(stripped):
        if not cap or cap not in caption_features:
            matrix[r] = fallback_vec
            continue
        vec = vec_cache.get(cap)
        if vec is None:
            feats = caption_features[cap]
            vec = np.array(
                [feats.get(name, 2) for name in all_feature_names], dtype=np.int8
            )
            vec_cache[cap] = vec
        matrix[r] = vec

    features_df = pd.DataFrame(matrix, columns=all_feature_names)

    # ── Combine with original columns and save ────────────────────────────────
    print("Creating output CSV...")
    desired_meta = ["image", "image_path", "label", "label_name", CAPTION_COLUMN, "disease_label"]
    meta_cols = [c for c in desired_meta if c in df.columns]
    final_df = pd.concat(
        [df[meta_cols].reset_index(drop=True), features_df.reset_index(drop=True)],
        axis=1,
    )

    # Values are already int8 in {0,1,2} from the matrix build above, so
    # the per-column coerce/fillna/clip pass is redundant — skipping it
    # saves ~10-15 s on a 180K×300 frame. Leave as int8 for CSV efficiency
    # (CSV serialisation writes the same digits regardless of dtype).

    # Write the final CSV atomically so a crash mid-serialisation can't leave
    # a partially-written 10 GB file that later tools would treat as valid.
    output_csv_path = Path(OUTPUT_CSV)
    tmp_out = output_csv_path.with_suffix(output_csv_path.suffix + f".tmp.{os.getpid()}")
    try:
        final_df.to_csv(tmp_out, index=False)
        os.replace(tmp_out, output_csv_path)
    finally:
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass

    _atomic_write_text(Path(STATS_FILE), json.dumps(global_stats, indent=2, default=str))

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
