"""
PHASE 1: Feature Schema Discovery (Bottom-Up)
===============================================
Discover ALL possible binary clinical features directly FROM the Derm-1M captions,
then consolidate into a canonical feature schema aligned with SCIN.

Output schema: {subcategory: [binary_feature_values]}  -- every feature is binary (0/1/2).
Feature names in Phase 2 are constructed as {subcategory}_{value} (e.g. body_location_nose).

Strategy:
  1. Load captions from cleaned_caption_Derm1M.csv (column configurable via CAPTION_COLUMN).
     Sampling: stratified (default) or full (all non-empty captions per label).
  2. Per-label-name LLM discovery via Gemini, OpenAI GPT-4o-mini, or Qwen 3.5 9B (local)
     Output: compact {subcategory: [values]} per batch (cost-optimized vs verbose JSON objects)
  3. Global consolidation to deduplicate synonym values within subcategories
  4. SCIN column alignment (after final feature list is established)

Run BEFORE phase2_bulk_extraction.py

Prerequisites:
  pip install pandas google-generativeai openai python-dotenv tqdm
  (OpenAI Batch also uses openai_batch_utils.py in this repo.)

OpenAI + gpt-4o-mini (discovery):
  .env: OPENAI_API_KEY=sk-...
  LLM_PROVIDER=openai
  Optional: OPENAI_MODEL_NAME=gpt-4o-mini (default)

For Qwen 3.5 (local):
  Serve the model locally via vLLM or SGLang first, e.g.:
    python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --port 8000 ...
  Then set LLM_PROVIDER=qwen in your .env or environment.
  Optionally set QWEN_BASE_URL (default: http://localhost:8000/v1).

Env (optional):
  CAPTION_COLUMN           — default truncated_caption (matches cleaned_caption_Derm1M.csv)
  DISCOVERY_SAMPLING_MODE  — stratified | full
  OPENAI_JSON_RESPONSE     — 1/true to use response_format json_object for OpenAI (discovery/consolidation)
  DISCOVERY_BATCH_SIZE     — captions per LLM batch (default 25; default 10 when LLM_PROVIDER=qwen for 8k context)
  QWEN_MAX_TOKENS          — max completion tokens (default scales with QWEN_MAX_MODEL_LEN if unset)
  QWEN_MAX_MODEL_LEN       — must match vLLM --max-model-len (default 8192); prompt+completion cannot exceed this
  QWEN_CONTEXT_BUFFER      — tokens reserved for chat template / special tokens (default 256)
  QWEN_CHARS_PER_TOKEN     — heuristic divisor when vLLM /tokenize fails (default 2.75)
  QWEN_TEMPLATE_TOKEN_OVERHEAD — extra tokens assumed for chat template (default 800)
  QWEN_COMPACT_DISCOVERY_PROMPT — force short system prompt (1/true) or disable auto-compact (0/false)
  LLM_PARSE_DEBUG          — 1/true to print a snippet when JSON parsing fails
  DISCOVERY_DEDUPE_CAPTIONS — 1/true (default): skip exact duplicate captions per label_name
                             (after .strip()) so repeated sentences are not sent to the LLM twice
  OPENAI_USE_BATCH         — 1/true: run discovery via Batch API (~50% lower $, async, ≤24h window)
  OPENAI_MODEL_NAME        — default gpt-4o-mini (must support Chat Completions + Batch per OpenAI model docs)
  OPENAI_PROMPT_CACHE_KEY  — stable key so identical system-prefix requests share prompt cache (discovery)
  OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY — same for consolidation call
  OPENAI_CONSOLIDATION_MAX_TOKENS — max completion tokens per consolidation LLM call (default 16384; gpt-4o-mini output cap)
  OPENAI_PROMPT_CACHE_RETENTION — optional: in_memory | 24h (if model supports extended cache)
  OPENAI_BATCH_POLL_SEC    — seconds between batch status polls (default 20)
  OPENAI_BATCH_MAX_REQUESTS — max lines per batch JSONL (default 50000, API cap 50k; larger jobs split into multiple batches)
  OPENAI_BATCH_MAX_FILE_BYTES — max UTF-8 bytes per batch JSONL (default ~195 MiB; API cap 200 MB per file)
  OPENAI_BATCH_MAX_ENQUEUED_TOKENS — org-level enqueued token cap (default 1800000; gpt-4o-mini orgs often 2M).
                             Batches are split so each chunk's estimated tokens stay under this limit;
                             chunks run sequentially, each completing before the next is submitted.
  OPENAI_BATCH_MAX_RETRIES — batch retry rounds for failed items (default 2; set 0 to skip batch retry)
  OPENAI_LOG_USAGE         — 1/true: print prompt/cached/output token stats when available

Prompt caching (OpenAI): static system message first, variable user message last; caching applies from
  ~1024+ identical prompt tokens (see https://developers.openai.com/api/docs/guides/prompt-caching ).
  Use OPENAI_PROMPT_CACHE_KEY consistently; OPENAI_PROMPT_CACHE_RETENTION=in_memory|24h (24h only on
  models that support extended retention per OpenAI docs).
"""

import pandas as pd
import json
import re
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm

from openai_batch_utils import (
    chunk_jobs_for_openai_batch,
    estimate_job_enqueued_tokens,
    openai_batch_max_file_bytes,
    openai_batches_create_safe,
    write_openai_batch_jsonl,
)
from scin_feature_map import SCIN_SCHEMA_FEATURES

# ── Load API keys from .env ───────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH        = "cleaned_caption_Derm1M.csv"
SCHEMA_OUT      = "feature_schema.json"
DISCOVERY_DIR   = Path("discovery_outputs")
DISCOVERY_DIR.mkdir(exist_ok=True)

CAPTION_COLUMN = os.getenv("CAPTION_COLUMN", "truncated_caption")
DISCOVERY_SAMPLING_MODE = os.getenv("DISCOVERY_SAMPLING_MODE", "stratified").strip().lower()
DISCOVERY_DEDUPE_CAPTIONS = os.getenv("DISCOVERY_DEDUPE_CAPTIONS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
OPENAI_JSON_RESPONSE = os.getenv("OPENAI_JSON_RESPONSE", "1").lower() in ("1", "true", "yes")
OPENAI_USE_BATCH = os.getenv("OPENAI_USE_BATCH", "").strip().lower() in ("1", "true", "yes")
OPENAI_PROMPT_CACHE_KEY = os.getenv("OPENAI_PROMPT_CACHE_KEY", "phase1_discovery_v1").strip()
OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY = os.getenv(
    "OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY", "phase1_consolidation_v1"
).strip()
OPENAI_PROMPT_CACHE_RETENTION = os.getenv("OPENAI_PROMPT_CACHE_RETENTION", "").strip()
ESTIMATE_ONLY = os.getenv("ESTIMATE_ONLY", "").strip().lower() in ("1", "true", "yes")

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


_EXTENDED_CACHE_MODELS = frozenset({
    "gpt-4.1", "gpt-5", "gpt-5-codex", "gpt-5.1", "gpt-5.1-codex",
    "gpt-5.1-codex-mini", "gpt-5.1-codex-max", "gpt-5.1-chat-latest",
    "gpt-5.2", "gpt-5.4",
})


def _warn_cache_retention_if_unsupported(model_name: str) -> None:
    """Warn if prompt_cache_retention=24h is set on a model that only supports in-memory."""
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


def _safe_float_env(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"WARNING: invalid float env {key}={raw!r}, using default {default}")
        return default


OPENAI_BATCH_POLL_SEC = _safe_int_env("OPENAI_BATCH_POLL_SEC", 20, vmin=5)
# OpenAI Batch: max 50,000 requests per batch file (https://developers.openai.com/api/docs/guides/batch)
OPENAI_BATCH_MAX_REQUESTS = _safe_int_env(
    "OPENAI_BATCH_MAX_REQUESTS", 50_000, vmin=1, vmax=50_000
)
# Org-level enqueued token limit (input + max_tokens across all in-progress batches).
# gpt-4o-mini default is 2M; we use 1.8M to leave headroom for concurrent usage.
OPENAI_BATCH_MAX_ENQUEUED_TOKENS = _safe_int_env(
    "OPENAI_BATCH_MAX_ENQUEUED_TOKENS", 1_800_000, vmin=100_000
)
OPENAI_CONSOLIDATION_MAX_TOKENS = _safe_int_env(
    "OPENAI_CONSOLIDATION_MAX_TOKENS", 16_384, vmin=4_096, vmax=65_536
)
OPENAI_LOG_USAGE = os.getenv("OPENAI_LOG_USAGE", "").strip().lower() in ("1", "true", "yes")
LLM_PARSE_DEBUG = os.getenv("LLM_PARSE_DEBUG", "").lower() in ("1", "true", "yes")
# Must match the server's --max-model-len; vLLM returns 400 if prompt_tokens + max_tokens exceeds this.
QWEN_MAX_MODEL_LEN = _safe_int_env("QWEN_MAX_MODEL_LEN", 8192, vmin=1)
# Full discovery JSON often needs >8k completion tokens when context allows; cap still clamped per-request.
_qwen_mt_env = os.getenv("QWEN_MAX_TOKENS", "").strip()
if _qwen_mt_env:
    try:
        QWEN_MAX_TOKENS = int(_qwen_mt_env)
    except ValueError:
        print(f"WARNING: invalid QWEN_MAX_TOKENS={_qwen_mt_env!r}, using heuristic from context")
        QWEN_MAX_TOKENS = min(16384, max(8192, QWEN_MAX_MODEL_LEN // 2))
else:
    QWEN_MAX_TOKENS = min(16384, max(8192, QWEN_MAX_MODEL_LEN // 2))
QWEN_CONTEXT_BUFFER = _safe_int_env("QWEN_CONTEXT_BUFFER", 256, vmin=0)
# Heuristic when /tokenize is unavailable (chars per token ~2.5–3 for English + medical terms)
QWEN_CHARS_PER_TOKEN = _safe_float_env("QWEN_CHARS_PER_TOKEN", 2.75)
QWEN_TEMPLATE_TOKEN_OVERHEAD = _safe_int_env("QWEN_TEMPLATE_TOKEN_OVERHEAD", 800, vmin=0)
_qwen_noted_low_cap: bool = False
_vllm_tokenize_warned: bool = False

# ── LLM Provider Selection ─────────────────────────────────────────────────
# Options: "gemini", "openai", or "qwen"
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "gemini")  # Set via env var or change here

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
    DISCOVERY_MODEL = "gemini-2.5-flash-lite"
    model = genai.GenerativeModel(DISCOVERY_MODEL)
elif LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        raise ValueError("Set OPENAI_API_KEY in your .env file for OpenAI provider")
    from openai import OpenAI

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    DISCOVERY_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip() or "gpt-4o-mini"
    _warn_cache_retention_if_unsupported(DISCOVERY_MODEL)
    model = None
elif LLM_PROVIDER == "qwen":
    from openai import OpenAI as QwenClient

    openai_client = None
    qwen_client = QwenClient(base_url=QWEN_BASE_URL, api_key="EMPTY")
    DISCOVERY_MODEL = QWEN_MODEL_NAME
    model = None
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'gemini', 'openai', or 'qwen'")

# Captions per LLM call — smaller batches for local Qwen + short max_model_len reduce truncated JSON
_bs_env = os.getenv("DISCOVERY_BATCH_SIZE", "").strip()
if _bs_env:
    BATCH_SIZE = _safe_int_env("DISCOVERY_BATCH_SIZE", 25, vmin=1)
elif LLM_PROVIDER == "qwen":
    # 8192 context + huge system prompt leaves little room for captions + JSON output
    BATCH_SIZE = 4 if QWEN_MAX_MODEL_LEN <= 8192 else 10
else:
    BATCH_SIZE = 25

print(f"Using LLM Provider: {LLM_PROVIDER} with model: {DISCOVERY_MODEL}")
if LLM_PROVIDER == "qwen":
    print(f"  Qwen base URL: {QWEN_BASE_URL}")
    if not os.getenv("QWEN_MAX_MODEL_LEN", "").strip():
        print(
            "  WARNING: QWEN_MAX_MODEL_LEN not set — using default 8192. "
            "Set QWEN_MAX_MODEL_LEN in .env to match vLLM --max-model-len (e.g. 16000), "
            "or token caps / compact-prompt logic will be wrong."
        )
    print(f"  Discovery batch size: {BATCH_SIZE} (set DISCOVERY_BATCH_SIZE to override)")
    print(
        f"  Qwen completion budget: up to {QWEN_MAX_TOKENS} tokens, "
        f"clamped to context {QWEN_MAX_MODEL_LEN} (set QWEN_MAX_MODEL_LEN=vLLM --max-model-len)"
    )
    _cp = os.getenv("QWEN_COMPACT_DISCOVERY_PROMPT", "").strip().lower()
    _use_compact = _cp in ("1", "true", "yes") or (
        _cp not in ("0", "false", "no") and QWEN_MAX_MODEL_LEN <= 8192
    )
    print(
        f"  Discovery system prompt: {'COMPACT (for short context)' if _use_compact else 'FULL'} "
        f"(QWEN_COMPACT_DISCOVERY_PROMPT / QWEN_MAX_MODEL_LEN)"
    )
elif LLM_PROVIDER == "openai":
    print(f"  Discovery batch size: {BATCH_SIZE} (set DISCOVERY_BATCH_SIZE to override)")
    if OPENAI_USE_BATCH:
        print(
            "  OpenAI discovery: Batch API (~50% lower token cost vs sync; async, typically within 24h). "
            "See https://developers.openai.com/api/docs/guides/batch"
        )
        print(
            f"  Batch file limits: ≤{OPENAI_BATCH_MAX_REQUESTS:,} requests/file, "
            f"≤{openai_batch_max_file_bytes() / (1024 * 1024):.0f} MiB UTF-8/file, "
            f"≤{OPENAI_BATCH_MAX_ENQUEUED_TOKENS:,} enqueued tokens/chunk"
        )
    else:
        print(
            "  OpenAI discovery: synchronous Chat Completions "
            "(set OPENAI_USE_BATCH=1 for Batch API + discount)"
        )
    if OPENAI_PROMPT_CACHE_KEY:
        print(
            f"  Prompt caching: discovery key={OPENAI_PROMPT_CACHE_KEY!r} "
            f"(static system prompt first → cache-friendly; "
            "https://developers.openai.com/api/docs/guides/prompt-caching )"
        )
    if OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY:
        print(f"  Prompt caching: consolidation key={OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY!r}")
    print(
        f"  Consolidation max_tokens={OPENAI_CONSOLIDATION_MAX_TOKENS:,} "
        f"(OPENAI_CONSOLIDATION_MAX_TOKENS)"
    )
    if OPENAI_PROMPT_CACHE_RETENTION in ("in_memory", "24h"):
        print(f"  prompt_cache_retention={OPENAI_PROMPT_CACHE_RETENTION!r}")

# ──────────────────────────────────────────────────────────────────────────────
# LLM API Wrapper Functions
# ──────────────────────────────────────────────────────────────────────────────

def strip_qwen_thinking(text: str) -> str:
    """
    Remove Qwen3-style thinking segments so JSON can be parsed.
    Handles raw model output and edge cases when vLLM does not fully strip reasoning.
    """
    if not text:
        return text
    t = text.strip()
    # Qwen3 templates use XML-style delimiters (avoid raw <think> in source for editors)
    _open = "\u003cthink\u003e"
    _close = "\u003c/think\u003e"
    t = re.sub(
        re.escape(_open) + r"[\s\S]*?" + re.escape(_close),
        "",
        t,
        flags=re.IGNORECASE,
    )
    # Optional backticks around closing tag
    t = re.sub(
        "`" + re.escape(_close) + "`",
        "",
        t,
        flags=re.IGNORECASE,
    )
    for marker in ("`" + _close + "`", _close):
        if marker in t:
            t = t[t.rfind(marker) + len(marker) :].strip()
    return t


def normalize_llm_json_text(text: str) -> str:
    """Strip fences and thinking wrappers before JSON parsing."""
    if not text:
        return ""
    t = str(text).strip()
    t = re.sub(r"```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```\s*", "", t).strip()
    t = strip_qwen_thinking(t)
    return t.strip()


def parse_llm_json(text: str, *, debug: bool = False) -> dict | list | None:
    """
    Parse JSON from an LLM reply: full document, or first object/array via raw_decode
    (handles leading/trailing prose and partially fenced output).
    """
    t = normalize_llm_json_text(text)
    if not t:
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for opener in ("{", "["):
        idx = t.find(opener)
        if idx < 0:
            continue
        try:
            obj, _end = dec.raw_decode(t, idx)
            return obj
        except json.JSONDecodeError:
            continue
    if debug:
        head = t[:600].replace("\n", " ")
        tail = t[-200:].replace("\n", " ") if len(t) > 200 else ""
        print(f"    DEBUG unparseable JSON. head={head!r} ... tail={tail!r}")
    return None


def _qwen_message_text(msg) -> str:
    """Extract assistant text from OpenAI-compatible chat message (vLLM / Qwen)."""
    raw = (getattr(msg, "content", None) or "").strip()
    if raw:
        return raw
    for attr in ("reasoning_content",):
        v = getattr(msg, attr, None)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def _vllm_server_root() -> str:
    """http://host:8000/v1 -> http://host:8000 (for /tokenize on vLLM)."""
    u = QWEN_BASE_URL.strip().rstrip("/")
    if u.lower().endswith("/v1"):
        return u[:-3].rstrip("/")
    return u


def _messages_to_counting_prompt(messages: list) -> str:
    """Flatten chat messages for /tokenize (approximates pre-template size; we add overhead below)."""
    parts = []
    for m in messages:
        role = str(m.get("role", ""))
        c = str(m.get("content") or "")
        parts.append(f"## {role}\n{c}")
    return "\n\n".join(parts)


def _vllm_tokenize_count_text(prompt: str) -> int | None:
    """
    vLLM exposes POST /tokenize (server root, not always under /v1).
    OpenAI-style /v1/messages/count_tokens often expects Anthropic payloads → 400; avoid it.
    """
    root = _vllm_server_root()
    bodies = (
        {"prompt": prompt},
        {"prompt": prompt, "model": DISCOVERY_MODEL},
        {"text": prompt},
    )
    for path in ("/tokenize", "/v1/tokenize"):
        url = f"{root}{path}"
        for body in bodies:
            try:
                payload = json.dumps(body).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                cnt = data.get("count")
                if isinstance(cnt, int) and cnt > 0:
                    return int(cnt * 1.10) + 128
                toks = data.get("tokens")
                if toks is None:
                    toks = data.get("token_ids")
                if isinstance(toks, list) and toks:
                    # Apply_chat_template adds special tokens vs raw concat
                    return int(len(toks) * 1.10) + 128
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError, TypeError, KeyError, OSError):
                continue
    return None


def _vllm_count_prompt_tokens(messages: list) -> int | None:
    flat = _messages_to_counting_prompt(messages)
    return _vllm_tokenize_count_text(flat)


def _qwen_effective_max_tokens(messages: list) -> int:
    """
    vLLM rejects when (prompt tokens) + max_tokens > max_model_len; it also validates using
    rendered text length. Prefer POST /tokenize; else a moderate chars→tokens heuristic.
    """
    global _qwen_noted_low_cap, _vllm_tokenize_warned
    n = _vllm_count_prompt_tokens(messages)
    if n is not None:
        est_prompt_tokens = max(1, n)
    else:
        if not _vllm_tokenize_warned:
            _vllm_tokenize_warned = True
            print(
                "    NOTE: vLLM /tokenize unavailable; using QWEN_CHARS_PER_TOKEN heuristic "
                f"({QWEN_CHARS_PER_TOKEN} chars/tok + {QWEN_TEMPLATE_TOKEN_OVERHEAD} template overhead). "
                "If JSON still truncates, tune those env vars or check vLLM /tokenize path."
            )
        char_len = sum(len(str(m.get("content") or "")) for m in messages)
        est_prompt_tokens = int(char_len / QWEN_CHARS_PER_TOKEN) + QWEN_TEMPLATE_TOKEN_OVERHEAD
        est_prompt_tokens = max(256, est_prompt_tokens)
    room = QWEN_MAX_MODEL_LEN - est_prompt_tokens - QWEN_CONTEXT_BUFFER
    if room < 64:
        raise ValueError(
            "Qwen/vLLM: this request does not fit the context window — "
            f"estimated ~{est_prompt_tokens} prompt tokens with "
            f"QWEN_MAX_MODEL_LEN={QWEN_MAX_MODEL_LEN} (must match vLLM --max-model-len). "
            "Raise --max-model-len and QWEN_MAX_MODEL_LEN, lower DISCOVERY_BATCH_SIZE, "
            "or rely on compact discovery prompt (auto when QWEN_MAX_MODEL_LEN<=8192)."
        )
    cap = min(QWEN_MAX_TOKENS, room)
    if cap < 1536 and not _qwen_noted_low_cap:
        _qwen_noted_low_cap = True
        print(
            f"    NOTE: Qwen output capped to ≤{cap} tokens this run (~{est_prompt_tokens} prompt tok / "
            f"{QWEN_MAX_MODEL_LEN} context). For fewer truncations use --max-model-len 32768 or "
            "DISCOVERY_BATCH_SIZE=2."
        )
    if cap < 1024 and LLM_PARSE_DEBUG:
        print(
            f"    DEBUG Qwen completion cap={cap} (est. prompt ~{est_prompt_tokens} tok, "
            f"context={QWEN_MAX_MODEL_LEN})."
        )
    return cap


def _openai_apply_prompt_caching(create_kw: dict, cache_key: str) -> None:
    """
    Attach prompt_cache_key / prompt_cache_retention for OpenAI Prompt Caching.
    See https://developers.openai.com/api/docs/guides/prompt-caching
    (prefix match from start of messages; optional key improves routing for shared long prefixes).
    """
    if cache_key:
        create_kw["prompt_cache_key"] = cache_key
    if OPENAI_PROMPT_CACHE_RETENTION in ("in_memory", "24h"):
        create_kw["prompt_cache_retention"] = OPENAI_PROMPT_CACHE_RETENTION


def _openai_discovery_chat_body(user_prompt: str) -> dict:
    """
    Chat Completions body for discovery. Static system message first, variable user last
    so repeated discovery calls share a cacheable prefix (see OpenAI prompt caching guide).
    """
    system_text = get_discovery_system_prompt()
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]
    body: dict = {
        "model": DISCOVERY_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }
    if OPENAI_JSON_RESPONSE:
        body["response_format"] = {"type": "json_object"}
    _openai_apply_prompt_caching(body, OPENAI_PROMPT_CACHE_KEY)
    return body


def _maybe_log_openai_chat_usage(resp, context: str = "") -> None:
    extra = f" ({context})" if context else ""
    if OPENAI_LOG_USAGE or LLM_PARSE_DEBUG:
        try:
            ch = resp.choices[0] if getattr(resp, "choices", None) else None
            fin = getattr(ch, "finish_reason", None) if ch else None
            if fin == "length":
                print(
                    f"    WARNING: OpenAI response truncated at max_tokens{extra} — "
                    "gpt-4o-mini caps near 16k completion tokens."
                )
        except (IndexError, TypeError, AttributeError):
            pass

    if not OPENAI_LOG_USAGE and not LLM_PARSE_DEBUG:
        return
    u = getattr(resp, "usage", None)
    if not u:
        return
    pt = int(getattr(u, "prompt_tokens", None) or 0)
    ct = int(getattr(u, "completion_tokens", None) or 0)
    details = getattr(u, "prompt_tokens_details", None)
    cached = 0
    if details is not None:
        cached = int(getattr(details, "cached_tokens", None) or 0)
    fr = ""
    try:
        ch2 = resp.choices[0] if getattr(resp, "choices", None) else None
        fin2 = getattr(ch2, "finish_reason", None) if ch2 else None
        if fin2:
            fr = f", finish_reason={fin2!r}"
    except (IndexError, TypeError, AttributeError):
        pass
    print(
        f"    OpenAI usage{extra}: prompt={pt} (cached={cached}, "
        f"non_cached≈{max(0, pt - cached)}) completion={ct}{fr}"
    )


def _openai_parse_batch_output_jsonl(raw: str) -> tuple[dict[str, str], dict[str, int]]:
    """Parse Batch output file: custom_id -> assistant text; accumulate token usage."""
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
                f"{str(resp.get('body'))[:200]}"
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
        if not choices or not isinstance(choices[0], dict):
            continue
        msg = choices[0].get("message") or {}
        if not isinstance(msg, dict):
            msg = {}
        content = msg.get("content") or ""
        if cid:
            out[cid] = content
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
    """Batch API writes failed/expired per-request lines to error_file_id (separate from output)."""
    failed_ids, error_counts = _openai_parse_batch_error_file(error_file_id)
    if not failed_ids:
        return
    err_breakdown = ", ".join(f"{code}={cnt}" for code, cnt in sorted(error_counts.items()))
    tqdm.write(
        f"  Batch error_file: {len(failed_ids)} failed request(s) [{err_breakdown}]."
    )


def _estimate_openai_discovery_cost_usd(acc: dict[str, int], *, batch_api: bool) -> float:
    """
    Rough cost in USD for gpt-4o-mini list pricing:
      input $0.15/M, cached input $0.075/M, output $0.60/M
    Batch API: 50% discount on those rates (see OpenAI Batch guide).
    """
    pin, pcached, pout = 0.15, 0.075, 0.60
    if batch_api:
        pin, pcached, pout = pin * 0.5, pcached * 0.5, pout * 0.5
    non_cached = max(0, acc["prompt"] - acc["cached"])
    incost = (non_cached / 1e6) * pin + (acc["cached"] / 1e6) * pcached
    outcost = (acc["completion"] / 1e6) * pout
    return incost + outcost


def _run_openai_discovery_batch_chunk(
    jobs: list[dict], chunk_idx: int
) -> tuple[dict[str, str], dict[str, int], str, set[str]]:
    """
    One Batch API job (≤ OPENAI_BATCH_MAX_REQUESTS lines).
    Returns (mapping, usage_acc, batch_id, failed_custom_ids).
    Partial results are always collected when an output_file exists.
    """
    input_path = DISCOVERY_DIR / f"openai_batch_discovery_input_{chunk_idx}.jsonl"
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
        metadata={"phase": "phase1_discovery", "chunk": str(chunk_idx)},
    )
    tqdm.write(
        f"  batch_id={batch_job.id}  polling every {OPENAI_BATCH_POLL_SEC}s "
        f"(≤24h completion window per OpenAI Batch API)."
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
    rc = batch_job.request_counts

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
                f" Lower OPENAI_BATCH_MAX_ENQUEUED_TOKENS (currently {OPENAI_BATCH_MAX_ENQUEUED_TOKENS:,}) "
                "to split into smaller chunks."
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


OPENAI_BATCH_MAX_RETRIES = _safe_int_env("OPENAI_BATCH_MAX_RETRIES", 2, vmin=0, vmax=5)


def _run_openai_discovery_batch(jobs: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """
    jobs: [{"custom_id": str, "body": dict}, ...]
    Splits by request count, file size, AND enqueued-token limit so each chunk
    stays under the org's enqueued token quota for gpt-4o-mini.
    Chunks run sequentially (each waits for completion before the next submits).
    Failed items are automatically retried in new batch rounds (up to
    OPENAI_BATCH_MAX_RETRIES rounds) before the caller falls back to sync.
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
            m, a, _bid, failed = _run_openai_discovery_batch_chunk(chunk, chunk_counter)
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
    est = _estimate_openai_discovery_cost_usd(acc_total, batch_api=True)
    tqdm.write(
        f"  Approx. discovery cost (Batch API 50% tier, gpt-4o-mini list rates): ${est:.2f} USD"
    )
    return combined, acc_total


def call_llm(
    prompt: str,
    system_prompt: str = None,
    retries: int = 3,
    *,
    openai_prompt_cache_key: str | None = None,
    openai_max_tokens: int | None = None,
) -> str:
    """
    Generic LLM caller that works with Gemini, OpenAI, and Qwen (local).
    Returns the text response.

    openai_max_tokens: OpenAI only; default 4096. Use OPENAI_CONSOLIDATION_MAX_TOKENS for consolidation.
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

                max_out = openai_max_tokens if openai_max_tokens is not None else 4096
                create_kw = dict(
                    model=DISCOVERY_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_out,
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
                _maybe_log_openai_chat_usage(response)
                return response.choices[0].message.content

            elif LLM_PROVIDER == "qwen":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                max_out = _qwen_effective_max_tokens(messages)
                response = qwen_client.chat.completions.create(
                    model=DISCOVERY_MODEL,
                    messages=messages,
                    max_tokens=max_out,
                    temperature=0.2,
                    top_p=0.9,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                choice = response.choices[0]
                if getattr(choice, "finish_reason", None) == "length":
                    print(
                        "    WARNING: Qwen completion hit max_tokens (JSON may be truncated). "
                        "Lower DISCOVERY_BATCH_SIZE or raise QWEN_MAX_TOKENS / vLLM --max-model-len."
                    )
                return _qwen_message_text(choice.message)
        
        except ValueError as e:
            if "Qwen/vLLM" in str(e):
                raise
            print(f"    API error (attempt {attempt+1}): {str(e)[:120]}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {str(e)[:120]}")
            if "429" in str(e) or "rate limit" in str(e).lower():
                time.sleep(10 * (attempt + 1))
            else:
                time.sleep(2 ** attempt)
    
    raise Exception(f"All {retries} LLM call attempts failed")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Load captions + sampling (stratified or full)
# ──────────────────────────────────────────────────────────────────────────────

# Sampling tiers: label_name class size → number of captions to sample
SAMPLING_TIERS = [
    (5000,  500),   # large classes (>5000 records): 500 captions
    (1000,  300),   # medium classes (1000-5000): 300 captions
    (200,    200),   # small classes (200-1000): 200 captions
    (0,    None),   # tiny classes (<200): take ALL
]


def compute_sample_size(class_count: int) -> int:
    """Determine how many captions to sample from a class based on its size."""
    for threshold, n_sample in SAMPLING_TIERS:
        if class_count > threshold:
            return n_sample if n_sample is not None else class_count
    return class_count


def load_captions_df(csv_path: str, caption_col: str) -> pd.DataFrame:
    """Load CSV, validate caption column, drop empty caption rows."""
    print("Loading full CSV...")
    df = pd.read_csv(csv_path)
    if caption_col not in df.columns:
        raise ValueError(
            f"Caption column {caption_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    df[caption_col] = df[caption_col].fillna("").astype(str)
    df = df[df[caption_col].str.strip().ne("")].copy()
    print(f"  Caption column: {caption_col}")
    print(f"  {len(df):,} records with non-empty captions")
    print(f"  {df['label_name'].nunique()} unique label_names")
    if "disease_label" in df.columns:
        print(f"  {df['disease_label'].nunique()} unique disease_labels\n")
    else:
        print()
    return df


def unique_captions_preserve_order(captions: list[str]) -> tuple[list[str], int]:
    """
    Within one label_name, keep the first occurrence of each caption text.
    Matching is by str.strip(); empty-after-strip strings are dropped.
    Returns (unique_captions, n_skipped) where n_skipped counts duplicates + dropped empty.
    """
    seen: set[str] = set()
    out: list[str] = []
    skipped = 0
    for c in captions:
        key = (c if isinstance(c, str) else str(c)).strip()
        if not key:
            skipped += 1
            continue
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        out.append(key)
    return out, skipped


def apply_stratified_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-label_name stratified sample; within label, stratify by disease_label
    when subsampling (sub-type diversity).
    """
    label_counts = df["label_name"].value_counts()
    sampled_parts = []

    for label_name, total_count in label_counts.items():
        n_sample = compute_sample_size(total_count)
        n_sample = min(n_sample, total_count)

        subset = df[df["label_name"] == label_name]
        n_sublabels = subset["disease_label"].nunique() if "disease_label" in subset.columns else 1

        if n_sublabels > 1 and n_sample < total_count:
            per_sub = max(1, n_sample // n_sublabels)
            stratified = (
                subset.groupby("disease_label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_sub), random_state=42))
            )
            if len(stratified) > n_sample:
                stratified = stratified.sample(n_sample, random_state=42)
            sampled_parts.append(stratified)
        else:
            sampled_parts.append(subset.sample(n_sample, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    print(f"  Sampled {len(sampled):,} captions across {sampled['label_name'].nunique()} label_names")

    tier_counts = {"large(>5K)": 0, "medium(1K-5K)": 0, "small(200-1K)": 0, "tiny(<200)": 0}
    for _, cnt in label_counts.items():
        if cnt > 5000:
            tier_counts["large(>5K)"] += 1
        elif cnt > 1000:
            tier_counts["medium(1K-5K)"] += 1
        elif cnt > 200:
            tier_counts["small(200-1K)"] += 1
        else:
            tier_counts["tiny(<200)"] += 1
    print(f"  Tier breakdown: {tier_counts}\n")
    return sampled


def apply_sampling_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return dataframe for discovery: either stratified subsample or all rows."""
    if mode == "stratified":
        return apply_stratified_sampling(df)
    if mode == "full":
        n = len(df)
        n_labels = df["label_name"].nunique()
        batches = 0
        total_unique = 0
        for ln in df["label_name"].unique():
            caps = df.loc[df["label_name"] == ln, CAPTION_COLUMN].tolist()
            if DISCOVERY_DEDUPE_CAPTIONS:
                uniq, _ = unique_captions_preserve_order([str(x) for x in caps])
                ucnt = len(uniq)
            else:
                ucnt = len(caps)
            total_unique += ucnt
            batches += (ucnt + BATCH_SIZE - 1) // BATCH_SIZE
        dedupe_note = (
            f" (~{total_unique:,} unique captions within labels; duplicates skipped)"
            if DISCOVERY_DEDUPE_CAPTIONS
            else " (all rows, including duplicate caption text)"
        )
        print(
            f"  FULL mode: {n:,} rows, {n_labels} label_names, ~{batches:,} LLM batches{dedupe_note}."
        )
        if n > 50_000:
            print(
                "  WARNING: Full-caption discovery is very expensive vs stratified. "
                "Consider DISCOVERY_SAMPLING_MODE=stratified for iteration.\n"
            )
        else:
            print()
        return df.reset_index(drop=True)
    raise ValueError(
        f"Unknown DISCOVERY_SAMPLING_MODE: {mode!r}. Use 'stratified' or 'full'."
    )


def load_and_sample(csv_path: str, caption_col: str, mode: str) -> pd.DataFrame:
    """Load CSV and apply the chosen sampling strategy."""
    base = load_captions_df(csv_path, caption_col)
    return apply_sampling_mode(base, mode)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: LLM-based feature discovery per label_name
# ──────────────────────────────────────────────────────────────────────────────

DISCOVERY_SYSTEM_PROMPT = """You are a clinical NLP expert specialising in dermatology.
Your task: given a batch of skin disease case captions, enumerate ALL specific
binary features that appear, grouped by subcategory.

CRITICAL RULES:
- Extract ONLY information that is ACTUALLY STATED in these captions.
  Do NOT infer or assume features that are not mentioned.
- Separate compound features (e.g. 'red scaly lesion' → morphology_color: [red], morphology_texture: [scaly])
- Each value must be a specific, distinguishable binary attribute (present / absent)
- Use snake_case for all values (e.g. oral_mucosal, back_of_hand, joint_pain)

You MUST scan for features in ALL of these subcategories (use these exact keys), if present:

1. demographics_age: child, adolescent, adult, middle_aged, elderly, infant, neonate, specific ages, age groups, etc.
2. demographics_sex: male, female, gender, etc.
3. demographics_skin_type: fst1, fst2, fst3, fst4, fst5, fst6, fair, medium, olive, dark, deeply_pigmented, etc.
4. demographics_ethnicity: as mentioned (e.g. caucasian, african_american, asian, hispanic, etc.)

5. morphology_texture:
   - Elevated: raised, papule, plaque, nodule, bump, wart, verruca, wheal, etc.
   - Flat: flat, macular, macule, patch, etc.
   - Surface: rough, scaly, flaky, desquamated, keratotic, hyperkeratotic, etc.
   - Fluid-filled: vesicle, blister, bulla, pustule, fluid_filled, weeping, oozing, etc.
   - Ulceration: ulcer, erosion, crusted, excoriated, etc.
   - Smooth: smooth, shiny, glossy, etc.
   - Other: cyst, abscess, scar, atrophy, benign, malignant, etc.

6. morphology_color:
   - Red spectrum: red, erythematous, pink, violaceous, purple, etc.
   - Pigment: brown, hyperpigmented, hypopigmented, white, depigmented, pale, dark, etc.
   - Other: yellow, black, blue, grey, salmon_colored, skin_colored, etc.

7. morphology_shape:
   - Shape: round, oval, circular, irregular, annular, linear, serpiginous, etc.
   - Border: well_defined, well_demarcated, ill_defined, irregular_border, etc.
   - Size: small, medium, large, etc.

8. morphology_distribution:
   - Pattern: unilateral, bilateral, symmetric, asymmetric, etc.
   - Arrangement: grouped, clustered, herpetiform, dermatomal, linear, streak_like, etc.
   - Extent: localized, widespread, generalized, diffuse, isolated, etc.

9. body_location:
   - Head: face, cheek, forehead, nose, perioral, periorbital, eyelid, lip, scalp, hairline, ear, etc.
   - Neck: neck, etc.
   - Torso: chest, abdomen, back, flank, trunk, etc.
   - Upper limb: arm, forearm, elbow, wrist, hand, palm, finger, back_of_hand, etc.
   - Lower limb: leg, thigh, shin, calf, knee, ankle, foot, sole, heel, toe, etc.
   - Special: genitalia, groin, scrotum, vulva, perineum, perianal, buttocks, etc.
   - Other: mouth, oral_mucosal, tongue, nail, axilla, intertriginous, etc.

10. symptoms_dermatological:
    - Sensations: itching, burning, stinging, pain, tenderness, soreness, etc.
    - Changes: increasing_size, spreading, darkening, lightening, bleeding, etc.
    - Other: bothersome_appearance, cosmetic_concern, numbness, tingling, etc.

11. symptoms_systemic:
    - General: fever, chills, fatigue, malaise, weight_loss, etc.
    - Specific: joint_pain, mouth_sores, shortness_of_breath, etc.
    - Lymphatic: lymphadenopathy, swollen_lymph_nodes, etc.

12. duration:
    - Duration: acute, chronic, subacute, days, weeks, months, years, lifelong, congenital, since_childhood, etc.
    - Onset: sudden_onset, gradual_onset, etc.
    - Pattern: recurrent, relapsing, persistent, intermittent, first_episode, etc.

13. triggers:
    - Environmental: sun_exposure, uv_light, heat, cold, sweating, humidity, seasonal, etc. 
    - Contact: allergens, irritants, chemicals, cosmetics, metals, latex
    - Medications: drug_reaction, antibiotic_reaction, etc.
    - Biological: infection, bacteria, virus, fungus, insect_bite, trauma, friction, etc.
    - Lifestyle: stress, hormonal_changes, pregnancy, diet, etc.
    - Occupational: occupational_exposure, etc.

14. treatments:
    - Topical: topical_steroids, corticosteroids, emollients, moisturizers, etc.
    - Systemic: oral_steroids, antibiotics, antifungals, antihistamines, immunosuppressants, biologics, retinoids, etc.
    - Physical: phototherapy, laser_therapy, surgery, excision, cryotherapy, etc.
    - Supportive: home_remedies, otc, wound_care, etc.

15. clinical_signs:
    - Specific: nikolsky_sign, auspitz_sign, koebner_phenomenon, darier_sign, etc.
    - Patterns: dermoscopic_pattern, wickham_striae, target_lesions, pathergy, etc.     
    - Diagnostic: biopsy_proven, histologically_confirmed, clinically_diagnosed, etc.

16. history:
    - Personal: family_history, genetic_predisposition, atopy, allergies, asthma, etc.
    - Disease: recurrence, previous_episodes, new_onset, etc.
    - Immune: immunocompromised, immunosuppressed, hiv, diabetes, autoimmune, etc.
    - Other: comorbidities, sun_damage, smoking, travel_history, etc.

17. lesion_count:
    - Number: single, few, multiple, numerous, etc.
    - Extent: localized, scattered, generalized, etc.

18. secondary_changes:
    - Chronic: lichenification, thickening, atrophy, etc.
    - Trauma: excoriation, scratch_marks, crusting, fissuring, maceration, etc.
    - Post-inflammatory: post_inflammatory_hyperpigmentation, post_inflammatory_hypopigmentation, scarring, keloid, etc.

19. severity:
    - Scale: mild, moderate, severe, very_severe, life_threatening, etc.
    - Impact: asymptomatic, symptomatic, disabling, quality_of_life_impact, etc.

20. image_metadata:
    - Type: clinical_image, dermoscopy, close_up, macro, microscopic, etc.
    - Quality: clear, blurry, well_lit, poor_lighting, etc.

21. other:
    - Any specific clinical findings not covered above (e.g. contagious, autoimmune_nature, malignancy_risk, etc.)
    - Include any other relevant clinical features found in captions that are not covered by the above subcategories.

Return ONLY valid JSON — no markdown fences, no prose.
Output is a single object where each key is a subcategory name and each value is the list of snake_case feature values found in these captions:
{"demographics_age":["adult","elderly"],"morphology_color":["red","hyperpigmented"],"body_location":["face","arm"],"symptoms_dermatological":["itching","pain"]}
Omit subcategories with no findings. List every distinct value found.
"""

DISCOVERY_COMPACT_SYSTEM_PROMPT = """You are a clinical NLP expert for dermatology. From the captions, extract ONLY features explicitly stated (no guessing).

Subcategory keys: demographics_age, demographics_sex, demographics_skin_type, demographics_ethnicity, morphology_texture, morphology_color, morphology_shape, morphology_distribution, body_location, symptoms_dermatological, symptoms_systemic, duration, triggers, treatments, clinical_signs, history, lesion_count, secondary_changes, severity, image_metadata, other.

Rules: snake_case values; split compound findings; omit empty subcategories.

Return ONLY valid JSON (no markdown fences) — object mapping subcategory to list of values:
{"body_location":["face","arm"],"morphology_color":["red"],"symptoms_dermatological":["itching"]}
"""


def get_discovery_system_prompt() -> str:
    """Full checklist for Gemini/OpenAI; compact default for local Qwen with short context."""
    if LLM_PROVIDER != "qwen":
        return DISCOVERY_SYSTEM_PROMPT
    flag = os.getenv("QWEN_COMPACT_DISCOVERY_PROMPT", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return DISCOVERY_COMPACT_SYSTEM_PROMPT
    if flag in ("0", "false", "no"):
        return DISCOVERY_SYSTEM_PROMPT
    if QWEN_MAX_MODEL_LEN <= 8192:
        return DISCOVERY_COMPACT_SYSTEM_PROMPT
    return DISCOVERY_SYSTEM_PROMPT


def build_discovery_user_prompt(captions: list[str], label_name: str) -> str:
    """User message (variable part) for discovery — keep separate from static system prompt for caching."""
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(captions))
    return (
        f"Extract all binary features from these {len(captions)} dermatology captions "
        f"for the disease category '{label_name}':\n\n{numbered}"
    )


def features_from_discovery_response_text(text: str) -> dict[str, list[str]]:
    """Parse compact discovery JSON {subcategory: [values]} (empty dict if unparseable)."""
    parsed = parse_llm_json(text, debug=LLM_PARSE_DEBUG)
    if not isinstance(parsed, dict):
        return {}
    if "feature_categories" in parsed and isinstance(parsed["feature_categories"], list):
        parsed.pop("feature_categories")
    out: dict[str, list[str]] = {}
    for key, vals in parsed.items():
        k = str(key).strip().lower().replace(" ", "_")
        if not k:
            continue
        if isinstance(vals, list):
            clean = [
                str(v).strip().lower().replace(" ", "_")
                for v in vals
                if isinstance(v, (str, int, float)) and str(v).strip()
            ]
            if clean:
                out[k] = clean
        elif isinstance(vals, str) and vals.strip():
            out[k] = [vals.strip().lower().replace(" ", "_")]
    return out


def discover_features_batch(
    captions: list[str], label_name: str, retries: int = 3
) -> dict[str, list[str]]:
    """Send a batch of captions to LLM and extract features as {subcategory: [values]}."""
    user_prompt = build_discovery_user_prompt(captions, label_name)

    for attempt in range(retries):
        try:
            text = call_llm(user_prompt, get_discovery_system_prompt())
            result = features_from_discovery_response_text(text)
            if not result:
                print(f"    JSON parse error (attempt {attempt+1}), retrying...")
                time.sleep(2 ** attempt)
                continue
            return result
        except json.JSONDecodeError:
            print(f"    JSON parse error (attempt {attempt+1}), retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    print(f"    WARNING: All retries failed for batch, skipping.")
    return {}


def _short_label_desc(label_name, max_len: int = 42) -> str:
    """Truncate label_name for tqdm descriptions (one line, no newlines)."""
    s = str(label_name).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _merge_batch_features(
    accumulator: dict[str, list[str]],
    batch_result: dict[str, list[str]],
) -> None:
    """Merge one batch's {subcategory: [values]} into a label-level accumulator (list, not set, for JSON)."""
    for subcat, vals in batch_result.items():
        if subcat not in accumulator:
            accumulator[subcat] = []
        existing = set(accumulator[subcat])
        for v in vals:
            if v not in existing:
                accumulator[subcat].append(v)
                existing.add(v)


def _merge_label_features_into_global(
    all_features: dict[str, set[str]],
    label_features: dict[str, list[str]],
) -> None:
    """Merge one label's {subcategory: [values]} into global accumulator {subcategory: set}."""
    for subcat, vals in label_features.items():
        if subcat not in all_features:
            all_features[subcat] = set()
        all_features[subcat].update(vals)


def _count_total_values(features: dict[str, list[str] | set[str]]) -> int:
    """Count total individual feature values across all subcategories."""
    return sum(len(v) for v in features.values())


def _prepare_label_captions_block(sample_df: pd.DataFrame, label_name) -> tuple[list[str], int, int]:
    """Return (deduped caption list, raw_row_count, dup_skipped)."""
    label_captions = (
        sample_df[sample_df["label_name"] == label_name][CAPTION_COLUMN].tolist()
    )
    raw_caption_n = len(label_captions)
    dup_skipped = 0
    if DISCOVERY_DEDUPE_CAPTIONS:
        label_captions, dup_skipped = unique_captions_preserve_order(
            [str(x) for x in label_captions]
        )
    else:
        label_captions = [str(x) for x in label_captions]
    return label_captions, raw_caption_n, dup_skipped


def _log_label_caption_counts(label_name, raw_caption_n: int, uniq_n: int, dup_skipped: int) -> None:
    if DISCOVERY_DEDUPE_CAPTIONS:
        dup_part = (
            f"; {dup_skipped:,} duplicate/empty row(s) not sent to LLM" if dup_skipped else ""
        )
        tqdm.write(
            f"  [{label_name}] rows in sample: {raw_caption_n:,} → "
            f"{uniq_n:,} unique caption(s) for extraction{dup_part}"
        )
    else:
        tqdm.write(
            f"  [{label_name}] rows in sample: {raw_caption_n:,} for extraction "
            f"(DISCOVERY_DEDUPE_CAPTIONS off)"
        )


def _discover_all_features_openai_batch(sample_df: pd.DataFrame) -> dict[str, set[str]]:
    """
    OpenAI-only: enqueue all discovery chat completions on the Batch API, then assemble results.
    Per-label disk cache is respected; only missing labels are batched.
    """
    all_features: dict[str, set[str]] = {}
    label_names = sample_df["label_name"].unique()
    jobs: list[dict] = []
    job_meta: dict[str, dict] = {}
    job_seq = 0

    for label_name in tqdm(label_names, desc="Discovery (per label_name)", unit="label"):
        label_captions, raw_caption_n, dup_skipped = _prepare_label_captions_block(
            sample_df, label_name
        )
        uniq_n = len(label_captions)
        _log_label_caption_counts(label_name, raw_caption_n, uniq_n, dup_skipped)

        safe_name = re.sub(r"[^\w\-]", "_", str(label_name))[:60]
        cache_path = DISCOVERY_DIR / f"discovery_{safe_name}_{DISCOVERY_SAMPLING_MODE}.json"

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                label_features: dict[str, list[str]] = json.load(f)
            tqdm.write(
                f"  [{label_name}] loaded {_count_total_values(label_features)} cached values "
                f"in {len(label_features)} subcategories (skipped LLM)"
            )
            _merge_label_features_into_global(all_features, label_features)
            continue

        for start in range(0, uniq_n, BATCH_SIZE):
            batch = label_captions[start : start + BATCH_SIZE]
            cid = f"job_{job_seq}"
            job_seq += 1
            user_prompt = build_discovery_user_prompt(batch, label_name)
            jobs.append({"custom_id": cid, "body": _openai_discovery_chat_body(user_prompt)})
            job_meta[cid] = {
                "label_name": label_name,
                "batch_start": start,
                "captions": batch,
                "cache_path": cache_path,
                "uniq_n": uniq_n,
                "raw_caption_n": raw_caption_n,
            }

    if not jobs:
        tqdm.write("  No OpenAI batch jobs (all labels had disk cache).")
        return all_features

    mapping, _acc = _run_openai_discovery_batch(jobs)

    by_label: dict = defaultdict(list)
    for cid, meta in job_meta.items():
        by_label[meta["label_name"]].append((meta["batch_start"], cid, meta))

    for label_name in label_names:
        if label_name not in by_label:
            continue
        entries = sorted(by_label[label_name], key=lambda x: x[0])
        meta0 = entries[0][2]
        cache_path = meta0["cache_path"]
        uniq_n = meta0["uniq_n"]
        raw_caption_n = meta0["raw_caption_n"]
        label_features: dict[str, list[str]] = {}
        for _start, cid, meta in entries:
            text = mapping.get(cid) or ""
            feats = features_from_discovery_response_text(text)
            if not feats:
                tqdm.write(
                    f"    Batch parse/sync fallback: {meta['label_name']!r} "
                    f"batch @ {meta['batch_start']}"
                )
                feats = discover_features_batch(meta["captions"], meta["label_name"])
            _merge_batch_features(label_features, feats)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(label_features, f, indent=2)
        src_note = (
            f"{uniq_n:,} unique captions"
            if DISCOVERY_DEDUPE_CAPTIONS and raw_caption_n != uniq_n
            else f"{uniq_n:,} captions"
        )
        n_vals = _count_total_values(label_features)
        tqdm.write(
            f"  [{label_name}] discovered {n_vals} values in {len(label_features)} subcategories "
            f"from {src_note} (batch)"
        )
        _merge_label_features_into_global(all_features, label_features)

    return all_features


def _discover_all_features_sequential(sample_df: pd.DataFrame) -> dict[str, set[str]]:
    """Gemini / Qwen / OpenAI sync: one chat completion per caption batch."""
    all_features: dict[str, set[str]] = {}
    label_names = sample_df["label_name"].unique()

    for label_name in tqdm(
        label_names,
        desc="Discovery (per label_name)",
        unit="label",
    ):
        label_captions, raw_caption_n, dup_skipped = _prepare_label_captions_block(
            sample_df, label_name
        )
        uniq_n = len(label_captions)
        _log_label_caption_counts(label_name, raw_caption_n, uniq_n, dup_skipped)

        safe_name = re.sub(r"[^\w\-]", "_", str(label_name))[:60]
        cache_path = DISCOVERY_DIR / f"discovery_{safe_name}_{DISCOVERY_SAMPLING_MODE}.json"

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                label_features: dict[str, list[str]] = json.load(f)
            tqdm.write(
                f"  [{label_name}] loaded {_count_total_values(label_features)} cached values "
                f"in {len(label_features)} subcategories (skipped LLM)"
            )
        else:
            label_features: dict[str, list[str]] = {}
            batch_starts = list(range(0, uniq_n, BATCH_SIZE))
            inner_desc = f"Batches [{_short_label_desc(label_name)}]"
            for start in tqdm(
                batch_starts,
                desc=inner_desc,
                leave=False,
                unit="batch",
                total=len(batch_starts),
            ):
                batch = label_captions[start : start + BATCH_SIZE]
                feats = discover_features_batch(batch, label_name)
                _merge_batch_features(label_features, feats)
                if LLM_PROVIDER == "qwen":
                    time.sleep(0.1)
                elif LLM_PROVIDER == "openai":
                    time.sleep(0.5)
                else:
                    time.sleep(1)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(label_features, f, indent=2)
            src_note = (
                f"{uniq_n:,} unique captions"
                if DISCOVERY_DEDUPE_CAPTIONS and raw_caption_n != uniq_n
                else f"{uniq_n:,} captions"
            )
            n_vals = _count_total_values(label_features)
            tqdm.write(
                f"  [{label_name}] discovered {n_vals} values in {len(label_features)} "
                f"subcategories from {src_note}"
            )

        _merge_label_features_into_global(all_features, label_features)

    return all_features


def discover_all_features(sample_df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Run per-label-name discovery across all sampled captions.
    Returns {subcategory: set_of_values} accumulated across all labels.
    """
    if LLM_PROVIDER == "openai" and OPENAI_USE_BATCH:
        return _discover_all_features_openai_batch(sample_df)
    return _discover_all_features_sequential(sample_df)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Consolidation via LLM
# ──────────────────────────────────────────────────────────────────────────────

CONSOLIDATION_SYSTEM_PROMPT = """You deduplicate and standardize feature values for a dermatology classifier schema.

You receive a JSON object mapping subcategory keys to lists of raw feature values collected from ~73k dermatology image captions and the SCIN questionnaire. The raw lists are very noisy (plurals, lexical variants, rare micro-anatomy, specific ages, compound phrases). Your job: produce a COMPACT, CLEAN canonical vocabulary per subcategory that a downstream one-hot classifier can use.

GUIDING PRINCIPLE
Aggressively consolidate toward a small canonical vocabulary per subcategory. Merge everything that is the same clinical concept (synonyms, lexical variants, plural/singular, sub-region of a named region, compound descriptive phrases). Keep separate ONLY what is clinically distinct (e.g. papule ≠ plaque, nail ≠ finger, pruritus → itching but itching ≠ burning).

TARGET BUDGETS (approximate final count per subcategory — the TOTAL across all subcategories MUST be 250-300):
- body_location:            < 40
- clinical_signs:           ≤ 15
- demographics_age:         EXACTLY the 10 bins listed below (rule 3)
- demographics_ethnicity:   ≤ 10
- demographics_sex:         EXACTLY ["male","female","other"]
- demographics_skin_type:   EXACTLY ["fst1","fst2","fst3","fst4","fst5","fst6"] (prefer Fitzpatrick) OR ≤ 6 descriptive tones — not both
- duration:                 EXACTLY ["hours","days","weeks","months","years","chronic"]
- history:                  ≤ 15
- image_metadata:           ≤ 6
- lesion_count:             EXACTLY ["single","few_2_to_5","multiple_6_to_20","many_20_plus","numerous"]
- morphology_color:         ≤ 12
- morphology_distribution:  ≤ 12
- morphology_shape:         ≤ 12
- morphology_texture:       ≤ 20
- other:                    ≤ 20
- secondary_changes:        ≤ 12
- severity:                 EXACTLY ["mild","moderate","severe","very_severe"]
- symptoms_dermatological:  ≤ 20
- symptoms_systemic:        ≤ 20
- treatments:               ≤ 20
- triggers:                 ≤ 20

If a subcategory's raw list is tiny (e.g. 3 values), keep what's there. If it's huge (e.g. body_location has 1000+), consolidate aggressively down to the budget.

RULES

1. MERGE aggressively within each subcategory. Collapse:
   - Plural → singular: "papules"→"papule", "ankles"→"ankle", "fingernails"→"fingernail".
   - Lexical/spelling variants: "hyperpigmentation"+"hyper_pigmented"→"hyperpigmented"; "erythema"+"erythematous"→"red"; "pruritus"+"pruritic"+"itch"+"itchy"→"itching".
   - Compound descriptive phrases → their head term: "anterior aspect of thigh"→"thigh", "both ankles"→"ankle", "antecubital fossa"→"elbow", "zygomatic cheek"→"cheek", "abdominal skin"→"abdomen", "web spaces between fingers and toes"→"web_space".
   - Sub-regions of a named region → the canonical region: "alar base"/"ala of nose"/"alar region"→"nose"; "nasal dorsum"/"bridge of nose"→"nose"; "anterior chest wall"→"chest"; "lateral thigh"→"thigh".
   - Redundant qualifiers: drop "area", "region", "skin", "surface", "aspect", "side", "both", "bilateral", "left", "right" when the base term is already canonical.
   - SCIN values merged with Derm-1M equivalents when they mean the same thing.

2. KEEP separate only clinically distinct concepts:
   - body_location: nail / fingernail / toenail / nail_bed ARE distinct from finger/toe/hand/foot. Lips, ear, nose, scalp, cheek, forehead, chin, eyelid are distinct from "face". Palm/sole distinct from hand/foot. Mucosa, genital, buttock, axilla, groin are their own sites.
   - morphology: papule ≠ plaque ≠ nodule ≠ macule ≠ patch ≠ vesicle ≠ bulla ≠ pustule ≠ cyst ≠ tumor.
   - secondary_changes: crust ≠ scale ≠ erosion ≠ ulcer ≠ fissure ≠ atrophy ≠ lichenification.

3. ENUM SUBCATEGORIES — use EXACTLY these values (nothing else allowed):
   - demographics_age → ["0-5","5-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-plus"]. Map every specific age ("1-year-old", "2.5-year-old", "23-year-old", "42_yo", "infant_3_months", "newborn", "toddler", "child", "adult", "elderly") onto the appropriate bin. DO NOT output any raw age like "23-year-old" — only bin labels.
   - duration → ["hours","days","weeks","months","years","chronic"]. Map "3_days"→"days", "2_weeks"→"weeks", "longstanding"/"persistent_for_10_years"→"chronic".
   - lesion_count → ["single","few_2_to_5","multiple_6_to_20","many_20_plus","numerous"].
   - severity → ["mild","moderate","severe","very_severe"].
   - demographics_sex → ["male","female","other"].
   - demographics_skin_type → prefer Fitzpatrick ["fst1","fst2","fst3","fst4","fst5","fst6"] if ANY fst values appear in the raw input; otherwise up to 6 descriptive tones (e.g. "fair","medium","olive","brown","dark","deeply_pigmented").

4. REMOVE noise entries: "other", "unknown", "various", "not_specified", "n_a", "none", "none_mentioned", "no_history", "no_treatment", "not_mentioned", and any value that is a disease name itself (belongs to the label, not the feature schema).

5. SUGGESTED canonical body_location vocabulary (use these; add site-specific ones from raw input only if clinically distinct):
   scalp, hairline, forehead, temple, face, eyebrow, eyelid, periorbital, nose, nostril, cheek, ear, earlobe, lip, oral_mucosa, tongue, chin, jaw, neck, nape, shoulder, axilla, chest, breast, back, abdomen, flank, groin, genital, perineum, buttock, arm, elbow, forearm, wrist, hand, palm, dorsum_hand, knuckle, finger, fingernail, web_space, thigh, knee, shin, calf, ankle, foot, sole, heel, dorsum_foot, toe, toenail, nail, trunk, extremity, generalized, flexural, acral, mucosa

6. STANDARDIZE all values to lowercase snake_case ASCII. Hyphens only for numeric range bins ("0-5", "80-plus").

7. DO NOT move values between subcategories; DO NOT add new subcategory keys.

OUTPUT — ONLY valid JSON, no markdown fences, no prose, same top-level shape as input:
{"demographics_age":["0-5","5-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-plus"],"demographics_sex":["male","female","other"],"severity":["mild","moderate","severe","very_severe"],"body_location":["scalp","forehead","face","cheek","nose","lip","ear","chin","neck","chest","back","abdomen","groin","axilla","arm","elbow","forearm","wrist","hand","palm","finger","fingernail","nail","web_space","thigh","knee","shin","ankle","foot","sole","toe","toenail","trunk","extremity","generalized"], ...}
"""


# Hard-enumerated subcategories: LLM output for these MUST be drawn from the canonical
# list below. Anything else is dropped by _enforce_enum_subcategories.
_ENUM_CANONICAL: dict[str, list[str]] = {
    "demographics_age": [
        "0-5", "5-10", "10-20", "20-30", "30-40",
        "40-50", "50-60", "60-70", "70-80", "80-plus",
    ],
    "duration": ["hours", "days", "weeks", "months", "years", "chronic"],
    "lesion_count": [
        "single", "few_2_to_5", "multiple_6_to_20", "many_20_plus", "numerous",
    ],
    "severity": ["mild", "moderate", "severe", "very_severe"],
    "demographics_sex": ["male", "female", "other"],
    # demographics_skin_type is partially-enumerated: EITHER Fitzpatrick OR descriptive tones.
    # Handled specially in _enforce_enum_subcategories.
}

_FITZPATRICK_TONES: list[str] = ["fst1", "fst2", "fst3", "fst4", "fst5", "fst6"]
_DESCRIPTIVE_TONES: list[str] = [
    "fair", "light", "medium", "olive", "brown", "dark", "deeply_pigmented",
]

# Minimal noise tokens that should never survive — applied across every subcategory.
_NOISE_VALUES: set[str] = {
    "", "unknown", "other", "various", "miscellaneous",
    "not_specified", "not_mentioned", "none", "none_mentioned", "no_history",
    "no_treatment", "n_a", "na", "null", "nan",
}


def _normalize_value(v: str) -> str:
    """Lower, strip, collapse whitespace → underscores. Hyphens preserved for numeric bins like 0-5."""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _enforce_enum_subcategories(
    consolidated: dict[str, list[str]],
) -> tuple[dict[str, list[str]], int]:
    """
    Deterministic post-LLM cleanup. For each hard-enumerated subcategory, drop any
    value not in the canonical list. Also strip global noise values from every
    subcategory. Does NOT re-add values — only removes.

    Returns (cleaned_schema, n_values_dropped).
    """
    out: dict[str, list[str]] = {}
    n_dropped = 0

    for subcat, vals in consolidated.items():
        kept: list[str] = []
        dropped_here: list[str] = []

        if subcat in _ENUM_CANONICAL:
            allowed = set(_ENUM_CANONICAL[subcat])
            for v in vals:
                nv = _normalize_value(v)
                if nv in allowed:
                    kept.append(nv)
                else:
                    dropped_here.append(v)
        elif subcat == "demographics_skin_type":
            fitz = set(_FITZPATRICK_TONES)
            desc = set(_DESCRIPTIVE_TONES)
            has_fitz = any(_normalize_value(v) in fitz for v in vals)
            allowed = fitz if has_fitz else desc
            for v in vals:
                nv = _normalize_value(v)
                if nv in allowed:
                    kept.append(nv)
                else:
                    dropped_here.append(v)
        else:
            for v in vals:
                nv = _normalize_value(v)
                if nv and nv not in _NOISE_VALUES:
                    kept.append(nv)
                else:
                    dropped_here.append(v)

        unique_kept = sorted(set(kept))
        if unique_kept:
            out[subcat] = unique_kept
        n_dropped += len(dropped_here)
        if dropped_here:
            preview = ", ".join(str(d) for d in dropped_here[:6])
            ellipsis = f" ... +{len(dropped_here) - 6} more" if len(dropped_here) > 6 else ""
            print(f"    Enum cleanup [{subcat}]: dropped {len(dropped_here)} non-canonical value(s): {preview}{ellipsis}")

    return out, n_dropped


def consolidate_schema(
    all_features: dict[str, set[str]],
    n_captions_sampled: int = 0,
    n_disease_classes: int = 0,
) -> dict[str, list[str]]:
    """
    Deduplicate feature values within each subcategory via LLM, then run a deterministic
    enum-enforcement pass that drops non-canonical values from hard-enumerated subcategories
    (demographics_age/sex/skin_type, duration, lesion_count, severity) and strips global
    noise tokens (unknown/other/n_a/...) from every subcategory.
    Input:  {subcategory: set_of_values}  (raw discovery output)
    Output: {subcategory: sorted_list_of_canonical_values}
    """
    raw: dict[str, list[str]] = {k: sorted(v) for k, v in all_features.items()}
    total_vals = sum(len(v) for v in raw.values())
    print(
        f"  Sending {total_vals} raw values in {len(raw)} subcategories for consolidation..."
    )

    body = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    user = (
        f"Consolidate this dermatology feature schema ({total_vals} total values "
        f"across {len(raw)} subcategories), extracted from {n_captions_sampled:,} captions "
        f"across {n_disease_classes} disease classes.\n\n"
        f"JSON:\n{body}"
    )

    for attempt in range(3):
        try:
            text = call_llm(
                user,
                CONSOLIDATION_SYSTEM_PROMPT,
                openai_prompt_cache_key=OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY,
                openai_max_tokens=OPENAI_CONSOLIDATION_MAX_TOKENS,
            )
            parsed = parse_llm_json(text, debug=LLM_PARSE_DEBUG)
            if not isinstance(parsed, dict):
                print(f"  Consolidation parse error (attempt {attempt+1}), retrying...")
                time.sleep(3)
                continue
            result: dict[str, list[str]] = {}
            for key, vals in parsed.items():
                k = str(key).strip().lower().replace(" ", "_")
                if not k or not isinstance(vals, list):
                    continue
                clean = sorted(set(
                    str(v).strip().lower().replace(" ", "_")
                    for v in vals
                    if isinstance(v, (str, int, float)) and str(v).strip()
                ))
                if clean:
                    result[k] = clean
            new_total = sum(len(v) for v in result.values())
            print(
                f"  Consolidated (LLM): {total_vals} → {new_total} values "
                f"in {len(result)} subcategories"
            )
            result, n_dropped = _enforce_enum_subcategories(result)
            final_total = sum(len(v) for v in result.values())
            if n_dropped:
                print(
                    f"  Enum cleanup: dropped {n_dropped} non-canonical value(s) "
                    f"→ final: {final_total} values across {len(result)} subcategories"
                )
            else:
                print(f"  Enum cleanup: OK — no non-canonical values found "
                      f"({final_total} final values)")
            return result
        except json.JSONDecodeError:
            print(f"  Consolidation JSON parse error (attempt {attempt+1}), retrying...")
            time.sleep(3)
        except Exception as e:
            print(f"  Consolidation API error (attempt {attempt+1}): {e}")
            time.sleep(3)

    print("  WARNING: All consolidation attempts failed; using raw deduplicated values.")
    return {k: sorted(v) for k, v in all_features.items()}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: SCIN column alignment (AFTER final feature list is established)
# ──────────────────────────────────────────────────────────────────────────────

# SCIN canonical name → (subcategory, feature_value)
SCIN_TO_SUBCATEGORY: dict[str, tuple[str, str]] = {
    "age_group":                       ("demographics_age", "age_group"),
    "sex":                             ("demographics_sex", "sex"),
    "fitzpatrick_skin_type":           ("demographics_skin_type", "fitzpatrick_skin_type"),
    "race_american_indian_alaska_native": ("demographics_ethnicity", "american_indian_alaska_native"),
    "race_asian":                      ("demographics_ethnicity", "asian"),
    "race_black_african_american":     ("demographics_ethnicity", "black_african_american"),
    "race_hispanic_latino":            ("demographics_ethnicity", "hispanic_latino"),
    "race_middle_eastern_north_african": ("demographics_ethnicity", "middle_eastern_north_african"),
    "race_native_hawaiian_pacific_islander": ("demographics_ethnicity", "native_hawaiian_pacific_islander"),
    "race_white":                      ("demographics_ethnicity", "white"),
    "race_other":                      ("demographics_ethnicity", "other_race"),
    "race_prefer_not_to_answer":       ("demographics_ethnicity", "prefer_not_to_answer"),
    "race_two_or_more":                ("demographics_ethnicity", "two_or_more_races"),
    "texture_raised":                  ("morphology_texture", "raised"),
    "texture_flat":                    ("morphology_texture", "flat"),
    "texture_rough_flaky":             ("morphology_texture", "rough_flaky"),
    "texture_fluid_filled":            ("morphology_texture", "fluid_filled"),
    "location_head_neck":              ("body_location", "head_neck"),
    "location_arm":                    ("body_location", "arm"),
    "location_palm":                   ("body_location", "palm"),
    "location_back_of_hand":           ("body_location", "back_of_hand"),
    "location_torso_front":            ("body_location", "torso_front"),
    "location_torso_back":             ("body_location", "torso_back"),
    "location_genitalia_groin":        ("body_location", "genitalia_groin"),
    "location_buttocks":               ("body_location", "buttocks"),
    "location_leg":                    ("body_location", "leg"),
    "location_foot_top_side":          ("body_location", "foot_top_side"),
    "location_foot_sole":              ("body_location", "foot_sole"),
    "location_other":                  ("body_location", "other_location"),
    "symptom_bothersome_appearance":   ("symptoms_dermatological", "bothersome_appearance"),
    "symptom_bleeding":                ("symptoms_dermatological", "bleeding"),
    "symptom_increasing_size":         ("symptoms_dermatological", "increasing_size"),
    "symptom_darkening":               ("symptoms_dermatological", "darkening"),
    "symptom_itching":                 ("symptoms_dermatological", "itching"),
    "symptom_burning":                 ("symptoms_dermatological", "burning"),
    "symptom_pain":                    ("symptoms_dermatological", "pain"),
    "symptom_no_relevant_experience":  ("symptoms_dermatological", "no_relevant_experience"),
    "symptom_fever":                   ("symptoms_systemic", "fever"),
    "symptom_chills":                  ("symptoms_systemic", "chills"),
    "symptom_fatigue":                 ("symptoms_systemic", "fatigue"),
    "symptom_joint_pain":              ("symptoms_systemic", "joint_pain"),
    "symptom_mouth_sores":             ("symptoms_systemic", "mouth_sores"),
    "symptom_shortness_of_breath":     ("symptoms_systemic", "shortness_of_breath"),
    "symptom_no_relevant_symptoms":    ("symptoms_systemic", "no_relevant_symptoms"),
    "duration_1_day":                  ("duration", "days"),
    "duration_less_than_1_week":       ("duration", "days"),
    "duration_1-4_weeks":              ("duration", "weeks"),
    "duration_1-3_months":             ("duration", "months"),
    "duration_3-12_months":            ("duration", "months"),
    "duration_over_1_year":            ("duration", "1_year"),
    "duration_over_5_years":           ("duration", "years"),
    "duration_since_childhood":        ("duration", "since_childhood"),
    "related_category_acne":           ("other", "acne_related"),
    "related_category_growth_or_mole": ("other", "growth_or_mole_related"),
    "related_category_hair_loss":      ("other", "hair_loss_related"),
    "related_category_hair_issue":     ("other", "hair_issue_related"),
    "related_category_nail_issue":     ("other", "nail_issue_related"),
    "related_category_pigment_birthmark":  ("other", "pigment_birthmark_related"),
    "related_category_rash":           ("other", "rash_related"),
    "related_category_normal":         ("other", "normal"),
}


def inject_scin_into_raw_features(
    all_features: dict[str, set[str]],
) -> int:
    """
    Inject SCIN feature values into raw discovery features BEFORE consolidation,
    so the LLM can merge synonyms between Derm-1M and SCIN during consolidation.
    Returns count of SCIN values injected.
    """
    injected = 0
    for scin_col, canonical in SCIN_SCHEMA_FEATURES.items():
        mapping = SCIN_TO_SUBCATEGORY.get(canonical)
        if not mapping:
            continue
        subcat, value = mapping
        if subcat not in all_features:
            all_features[subcat] = set()
        if value not in all_features[subcat]:
            all_features[subcat].add(value)
            injected += 1
    return injected


def verify_scin_post_consolidation(
    consolidated: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]], int]:
    """
    After consolidation, verify SCIN features are still present (the LLM may have
    merged some into Derm-1M equivalents). Re-add any that were lost, and build
    the scin_map tracking which features have SCIN equivalents.
    Returns (updated_schema, scin_map {subcategory: [scin_comparable_values]}, count_re_added).
    """
    scin_map: dict[str, list[str]] = defaultdict(list)
    re_added = 0

    for scin_col, canonical in SCIN_SCHEMA_FEATURES.items():
        mapping = SCIN_TO_SUBCATEGORY.get(canonical)
        if not mapping:
            continue
        subcat, value = mapping
        if subcat not in consolidated:
            consolidated[subcat] = []
        if value not in consolidated[subcat]:
            consolidated[subcat].append(value)
            re_added += 1
        scin_map[subcat].append(value)

    n_scin = sum(len(v) for v in scin_map.values())
    print(f"  Verified {n_scin} SCIN features across {len(scin_map)} subcategories")
    if re_added:
        print(f"  Re-added {re_added} SCIN feature values lost during consolidation")
    else:
        print(f"  All SCIN features survived consolidation (no re-adds needed)")
    return consolidated, dict(scin_map), n_scin


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "demographics_age": "Patient age groups",
    "demographics_sex": "Patient sex or gender",
    "demographics_skin_type": "Fitzpatrick skin type or skin tone descriptions",
    "demographics_ethnicity": "Patient ethnicity or race as mentioned",
    "morphology_texture": "Surface texture and elevation of lesions",
    "morphology_color": "Color characteristics of lesions",
    "morphology_shape": "Shape, border, and size of lesions",
    "morphology_distribution": "Spatial distribution and arrangement of lesions",
    "morphology_other": "Other morphological features not covered above",
    "body_location": "Specific body locations affected by the condition",
    "symptoms_dermatological": "Skin-related symptoms and sensations",
    "symptoms_systemic": "Systemic symptoms beyond the skin",
    "duration": "Duration, onset, and temporal pattern of the condition",
    "triggers": "Triggering or exacerbating factors",
    "treatments": "Treatments mentioned, attempted, or suggested",
    "clinical_signs": "Specific clinical signs and diagnostic findings",
    "history": "Patient history, comorbidities, and risk factors",
    "lesion_count": "Number and extent of lesions",
    "secondary_changes": "Secondary changes including post-inflammatory effects",
    "severity": "Severity scale and functional impact",
    "image_metadata": "Image type, quality, and photographic context",
    "other": "Other clinically relevant features not covered above",
}


if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 1: Feature Schema Discovery (Bottom-Up)")
    print("=" * 60 + "\n")

    # ── Step 1: Sample ────────────────────────────────────────────────────────
    print("STEP 1: Loading and sampling captions...\n")
    print(
        f"  DISCOVERY_SAMPLING_MODE={DISCOVERY_SAMPLING_MODE!r}, "
        f"CAPTION_COLUMN={CAPTION_COLUMN!r}, "
        f"DISCOVERY_DEDUPE_CAPTIONS={'on' if DISCOVERY_DEDUPE_CAPTIONS else 'off'}\n"
    )
    sample_df = load_and_sample(CSV_PATH, CAPTION_COLUMN, DISCOVERY_SAMPLING_MODE)

    sample_path = DISCOVERY_DIR / "sampled_captions.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"  Saved sample to {sample_path}\n")

    if ESTIMATE_ONLY:
        n_labels = int(sample_df["label_name"].nunique())
        n_captions = len(sample_df)

        # Simulate per-label deduplication and batching to get accurate API call count
        discovery_api_calls = 0
        total_unique_per_label = 0
        cached_labels = 0
        for label_name in sample_df["label_name"].unique():
            safe_name = re.sub(r"[^\w\-]", "_", str(label_name))[:60]
            cache_path = DISCOVERY_DIR / f"discovery_{safe_name}_{DISCOVERY_SAMPLING_MODE}.json"
            if cache_path.exists():
                cached_labels += 1
                continue
            label_caps = sample_df[sample_df["label_name"] == label_name][CAPTION_COLUMN].tolist()
            if DISCOVERY_DEDUPE_CAPTIONS:
                unique_caps = set(str(x).strip() for x in label_caps if str(x).strip())
                n_unique = len(unique_caps)
            else:
                n_unique = len(label_caps)
            total_unique_per_label += n_unique
            discovery_api_calls += (n_unique + BATCH_SIZE - 1) // BATCH_SIZE

        sys_prompt_words = len(DISCOVERY_SYSTEM_PROMPT.split())
        sys_prompt_tokens = int(sys_prompt_words * 1.3)
        avg_caption_tokens = 120
        avg_input_per_call = sys_prompt_tokens + (BATCH_SIZE * avg_caption_tokens) + 50
        avg_output_per_call = 1500
        disc_input_tokens = discovery_api_calls * avg_input_per_call
        disc_output_tokens = discovery_api_calls * avg_output_per_call

        # Consolidation: 1 call with the full raw schema
        consol_input_tokens = 2000
        consol_output_tokens = OPENAI_CONSOLIDATION_MAX_TOKENS

        total_input = disc_input_tokens + consol_input_tokens
        total_output = disc_output_tokens + consol_output_tokens

        model_name = DISCOVERY_MODEL if LLM_PROVIDER == "openai" else "gpt-4o-mini"
        est = _estimate_cost(model_name, total_input, total_output, cached_fraction=0.8)

        print("  Cost Estimate (ESTIMATE_ONLY=1):")
        print(f"    Model: {model_name}")
        print()
        print(f"    ── Discovery Step ──")
        print(f"    Labels to process:     {n_labels:,} ({cached_labels} already cached on disk)")
        print(f"    Labels needing LLM:    {n_labels - cached_labels:,}")
        print(f"    Captions sampled:      {n_captions:,}")
        print(f"    Unique captions (deduped per label): {total_unique_per_label:,}")
        print(f"    Batch size:            {BATCH_SIZE}")
        print(f"    Est. API calls:        {discovery_api_calls:,}")
        print(f"    Est. input tokens:     ~{disc_input_tokens:,} "
              f"(~{sys_prompt_tokens:,} system prompt cached per call)")
        print(f"    Est. output tokens:    ~{disc_output_tokens:,}")
        print()
        print(f"    ── Consolidation Step ──")
        print(f"    Est. API calls:        1")
        print(f"    Est. input tokens:     ~{consol_input_tokens:,}")
        print(f"    Est. output tokens:    ~{consol_output_tokens:,}")
        print()
        print(f"    ── Total ──")
        print(f"    Total API calls:       {discovery_api_calls + 1:,}")
        print(f"    Total input tokens:    ~{total_input:,}")
        print(f"      Cached (~80%):       ~{est['cached_input_tokens']:,} "
              f"@ ${_MODEL_COSTS.get(model_name, _MODEL_COSTS['gpt-4o-mini'])['cached_input']}/1M")
        print(f"      Uncached (~20%):     ~{est['uncached_input_tokens']:,} "
              f"@ ${_MODEL_COSTS.get(model_name, _MODEL_COSTS['gpt-4o-mini'])['input']}/1M")
        print(f"    Total output tokens:   ~{total_output:,} "
              f"@ ${_MODEL_COSTS.get(model_name, _MODEL_COSTS['gpt-4o-mini'])['output']}/1M")
        print()
        print(f"    ── Estimated Cost ──")
        print(f"    Sync API cost:         ${est['total_cost_usd']:.4f}")
        print(f"    Batch API cost (~50%): ${est['batch_api_cost_usd']:.4f}")
        if OPENAI_USE_BATCH:
            print(f"    → Using Batch API (OPENAI_USE_BATCH=1)")
        else:
            print(f"    → Using sync API (set OPENAI_USE_BATCH=1 for ~50% discount)")
        print("\n  Exiting without running. Set ESTIMATE_ONLY=0 to proceed.")
        import sys
        sys.exit(0)

    # ── Step 2: Per-label discovery ───────────────────────────────────────────
    raw_path = DISCOVERY_DIR / "raw_features_all.json"
    if raw_path.exists():
        print(f"STEP 2: Loading raw discovery features from {raw_path} (skipping LLM)...\n")
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_loaded = json.load(f)
        all_features: dict[str, set[str]] = {
            k: set(v) for k, v in raw_loaded.items() if isinstance(v, list)
        }
        total_raw = sum(len(v) for v in all_features.values())
        print(
            f"  Loaded {total_raw} raw values across {len(all_features)} subcategories "
            f"(delete {raw_path.name} to rerun discovery)\n"
        )
    else:
        print("STEP 2: Running per-label-name LLM feature discovery...\n")
        all_features = discover_all_features(sample_df)
        total_raw = sum(len(v) for v in all_features.values())
        print(
            f"\n  Total raw values discovered: {total_raw} "
            f"across {len(all_features)} subcategories\n"
        )

        raw_serializable = {k: sorted(v) for k, v in all_features.items()}
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_serializable, f, indent=2)
        print(f"  Saved raw features to {raw_path}\n")

    # ── Step 3: Inject SCIN features + Consolidation ──────────────────────────
    print("STEP 3: Injecting SCIN features and consolidating via LLM...\n")
    n_scin_injected = inject_scin_into_raw_features(all_features)
    print(f"  Injected {n_scin_injected} SCIN feature values into raw features before consolidation")
    consolidated = consolidate_schema(
        all_features,
        n_captions_sampled=len(sample_df),
        n_disease_classes=int(sample_df["label_name"].nunique()),
    )

    # ── Step 4: SCIN post-consolidation verification ──────────────────────────
    print("\nSTEP 4: Verifying SCIN features post-consolidation...\n")
    consolidated, scin_map, n_scin_comparable = verify_scin_post_consolidation(consolidated)

    # ── Build final schema ────────────────────────────────────────────────────
    n_categories = len(consolidated)
    n_total_features = sum(len(v) for v in consolidated.values())

    feature_categories = []
    for subcat in sorted(consolidated.keys()):
        feature_categories.append({
            "category": subcat,
            "description": CATEGORY_DESCRIPTIONS.get(subcat, ""),
            "features": sorted(consolidated[subcat]),
            "scin_comparable_features": sorted(scin_map.get(subcat, [])),
        })

    schema = {
        "feature_categories": feature_categories,
        "metadata": {
            "source_csv": CSV_PATH,
            "caption_column": CAPTION_COLUMN,
            "discovery_sampling_mode": DISCOVERY_SAMPLING_MODE,
            "n_captions_sampled": len(sample_df),
            "n_rows_used": len(sample_df),
            "n_label_names": int(sample_df["label_name"].nunique()),
            "n_categories": n_categories,
            "n_total_features": n_total_features,
            "n_scin_comparable": n_scin_comparable,
            "model_used": DISCOVERY_MODEL,
            "llm_provider": LLM_PROVIDER,
            "openai_use_batch": bool(LLM_PROVIDER == "openai" and OPENAI_USE_BATCH),
            "openai_prompt_cache_key": OPENAI_PROMPT_CACHE_KEY if LLM_PROVIDER == "openai" else "",
        },
    }

    print(f"\n  Total subcategories:          {n_categories}")
    print(f"  Total binary features:        {n_total_features}")
    print(f"  SCIN-comparable features:     {n_scin_comparable}")
    print(f"  Feature breakdown by subcategory:")
    for cat_entry in feature_categories:
        c = cat_entry["category"]
        n = len(cat_entry["features"])
        scin_n = len(cat_entry.get("scin_comparable_features", []))
        scin_part = f" ({scin_n} SCIN)" if scin_n else ""
        print(f"    - {c}: {n}{scin_part}")

    with open(SCHEMA_OUT, "w", encoding="utf-8") as fp:
        json.dump(schema, fp, indent=2)

    print(f"\n  Schema saved to {SCHEMA_OUT}")
    print(f"\n  Next: run phase2_bulk_extraction.py")
