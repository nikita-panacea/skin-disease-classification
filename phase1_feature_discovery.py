"""
PHASE 1: Feature Schema Discovery (Bottom-Up)
===============================================
Discover ALL possible clinical features directly FROM the Derm-1M captions,
then consolidate into a canonical feature schema aligned with SCIN.

Strategy:
  1. Load captions from cleaned_caption_Derm1M.csv (column configurable via CAPTION_COLUMN).
     Sampling: stratified (default) or full (all non-empty captions per label).
  2. Per-label-name LLM discovery via Gemini, OpenAI GPT-4o-mini, or Qwen 3.5 9B (local)
  3. Global consolidation to deduplicate synonyms
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
  OPENAI_CONSOLIDATION_CHUNK_SIZE — raw features per batch when auto-chunking (default 72)
  OPENAI_CONSOLIDATION_SINGLE_CALL_MAX — if feature count ≤ this after a reduce round, one final full-schema polish call (default 72)
  OPENAI_CONSOLIDATION_SAVE_CHUNK_JSON — 1/true: write each chunk LLM JSON to discovery_outputs/consolidation_batches/
  OPENAI_PROMPT_CACHE_RETENTION — optional: in_memory | 24h (if model supports extended cache)
  OPENAI_BATCH_POLL_SEC    — seconds between batch status polls (default 20)
  OPENAI_BATCH_MAX_REQUESTS — max lines per batch JSONL (default 50000, API cap 50k; larger jobs split into multiple batches)
  OPENAI_BATCH_MAX_FILE_BYTES — max UTF-8 bytes per batch JSONL (default ~195 MiB; API cap 200 MB per file)
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
OPENAI_JSON_RESPONSE = os.getenv("OPENAI_JSON_RESPONSE", "").lower() in ("1", "true", "yes")
OPENAI_USE_BATCH = os.getenv("OPENAI_USE_BATCH", "").strip().lower() in ("1", "true", "yes")
OPENAI_PROMPT_CACHE_KEY = os.getenv("OPENAI_PROMPT_CACHE_KEY", "phase1_discovery_v1").strip()
OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY = os.getenv(
    "OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY", "phase1_consolidation_v1"
).strip()
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
# Consolidation returns a large JSON schema; gpt-4o-mini max completion ~16k — chunk when list is large.
OPENAI_CONSOLIDATION_MAX_TOKENS = _safe_int_env(
    "OPENAI_CONSOLIDATION_MAX_TOKENS", 16_384, vmin=4_096, vmax=65_536
)
OPENAI_CONSOLIDATION_CHUNK_SIZE = _safe_int_env(
    "OPENAI_CONSOLIDATION_CHUNK_SIZE", 72, vmin=24, vmax=150
)
OPENAI_CONSOLIDATION_SINGLE_CALL_MAX = _safe_int_env(
    "OPENAI_CONSOLIDATION_SINGLE_CALL_MAX", 72, vmin=24, vmax=200
)
OPENAI_CONSOLIDATION_SAVE_CHUNK_JSON = os.getenv(
    "OPENAI_CONSOLIDATION_SAVE_CHUNK_JSON", ""
).strip().lower() in ("1", "true", "yes")
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
            f"≤{openai_batch_max_file_bytes() / (1024 * 1024):.0f} MiB UTF-8/file"
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
    print(
        f"  Consolidation chunking: CHUNK_SIZE={OPENAI_CONSOLIDATION_CHUNK_SIZE}, "
        f"SINGLE_CALL_MAX={OPENAI_CONSOLIDATION_SINGLE_CALL_MAX} "
        f"(auto multi-batch when raw feature count > SINGLE_CALL_MAX)"
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
        "max_tokens": 4096,
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
                    "gpt-4o-mini caps near 16k completion; phase1 uses multi-batch consolidation when "
                    "the feature list is long (OPENAI_CONSOLIDATION_CHUNK_SIZE). "
                    "Or try a model with higher max output / lower chunk size."
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


def _openai_log_batch_error_file_summary(error_file_id: str | None) -> None:
    """Batch API writes failed/expired per-request lines to error_file_id (separate from output)."""
    raw = _openai_download_batch_file_text(error_file_id)
    if not raw.strip():
        return
    n = sum(1 for line in raw.splitlines() if line.strip())
    tqdm.write(
        f"  Batch error_file: {n} line(s). Those custom_ids have no successful output line; "
        "pipeline uses sync fallback where needed."
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
) -> tuple[dict[str, str], dict[str, int], str]:
    """
    One Batch API job (≤ OPENAI_BATCH_MAX_REQUESTS lines). Returns (mapping, usage_acc, batch_id).
    Matches https://developers.openai.com/api/docs/guides/batch (jsonl → upload → create → poll → files.content).
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

    # Allow partial results when window expired (completed requests still in output_file).
    if batch_job.status == "expired" and batch_job.output_file_id:
        tqdm.write(
            "  NOTE: Batch status=expired — reading partial output_file (per Batch API expiration rules)."
        )
    elif batch_job.status != "completed":
        raise RuntimeError(
            f"OpenAI batch ended with status={batch_job.status!r}. "
            f"Inspect batch in dashboard; error_file_id may list per-request failures."
        )
    if not batch_job.output_file_id:
        raise RuntimeError(
            f"No output_file_id for batch status={batch_job.status!r} — nothing to parse."
        )

    raw_text = _openai_download_batch_file_text(batch_job.output_file_id)
    err_id = getattr(batch_job, "error_file_id", None)
    _openai_log_batch_error_file_summary(err_id)

    mapping, acc = _openai_parse_batch_output_jsonl(raw_text)
    return mapping, acc, batch_job.id


def _run_openai_discovery_batch(jobs: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """
    jobs: [{"custom_id": str, "body": dict}, ...]
    Splits at OPENAI_BATCH_MAX_REQUESTS (API max 50k lines per file).
    """
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
            f"(≤{max_r:,} lines each, ≤{file_cap / (1024 * 1024):.0f} MiB UTF-8 per file per Batch API)."
        )

    for ci, chunk in enumerate(chunks):
        m, a, _bid = _run_openai_discovery_batch_chunk(chunk, ci)
        combined.update(m)
        for k in acc_total:
            acc_total[k] += a[k]

    tqdm.write(
        f"  Batch token totals (all chunks): prompt={acc_total['prompt']:,}, "
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
Your task: given a batch of skin disease case captions, extract ALL possible
clinical and contextual feature *categories* that appear.

CRITICAL RULES:
- Extract ONLY information that is ACTUALLY STATED in these captions.
  Do NOT infer or assume features that are not mentioned.
- Separate compound features (e.g. 'red scaly lesion' → color=red, texture=scaly)
- Each feature should be MEDICALLY meaningful and distinguishable
- Look for EXACT phrases in captions and record them as extraction_examples
- If a feature appears in multiple captions, collect ALL unique example phrases

You MUST look for and extract features in ALL of these categories included, but not limited to, if present:

1. DEMOGRAPHICS: 
   - Age: specific ages (e.g., "15 years old"), age ranges (e.g., "40 to 49", "18-29"), age groups (child, adolescent, adult, elderly)
   - Sex/Gender: male, female, man, woman, boy, girl
   - Fitzpatrick Skin Type/Skin tone: FST1, FST2, FST3, FST4, FST5, FST6 or descriptions (very light, fair/light, medium, olive, brown, dark, deeply pigmented)
   - Ethnicity/Race: as mentioned in captions

2. MORPHOLOGY - TEXTURE: 
   - Elevated: raised, bumpy, papule, plaque, nodule, bump, wart, verruca, hives, wheal
   - Flat: flat, macular, macule, patch
   - Surface: rough, scaly, flaky, scale, desquamated, keratotic, hyperkeratotic
   - Fluid-filled: vesicle, blister, bulla, pustule, fluid-filled, weeping, oozing
   - Ulceration: ulcer, ulcerated, erosion, eroded, crusted, crust, excoriated
   - Smooth: smooth, shiny, glossy
   - Other: cyst, abscess, scar, atrophy, cancerous, non-cancerous, benign, malignant, etc.

3. MORPHOLOGY - COLOR: 
   - Red spectrum: red, erythematous, erythema, pink, violaceous, purple
   - Pigment changes: brown, hyperpigmented, hypopigmented, white, depigmented, pale, dark
   - Other colors: yellow, black, blue, grey/gray, slate, salmon-colored, skin-colored


4. MORPHOLOGY - SHAPE/BORDER/SIZE: 
   - Shape: round, oval, circular, irregular, annular, ring-like, linear, serpiginous, snake-like
   - Border: well-defined, well-demarcated, ill-defined, irregular border
   - Size: small (<1cm), medium (1-5cm), large (>5cm), specific measurements
   - Other: other shape, other information, other factors, other details, etc.

5. MORPHOLOGY - DISTRIBUTION: 
   - Pattern: unilateral, bilateral, symmetric, asymmetric
   - Arrangement: grouped, clustered, herpetiform, dermatomal, linear, streak-like
   - Extent: localized, widespread, generalized, diffuse, all over, isolated
   - Other: other distribution, other information, other factors, other details, etc.

6. BODY LOCATION (be specific):
   - Head: face, cheek, forehead, nose, perioral, periorbital, eyelid, lip, scalp, hairline
   - Neck: neck, cervical
   - Torso: chest, abdomen, back, flank, torso, trunk, sternum
   - Upper limb: arm, forearm, elbow, wrist, hand, palm, dorsum, finger, knuckle, back of hand
   - Lower limb: leg, thigh, shin, calf, knee, ankle, foot, sole, heel, toe, dorsum of foot
   - Special areas: genitalia, groin, scrotum, vulva, perineum, perianal, buttocks, gluteal
   - Other: mouth, oral, tongue, mucosa, nail, axilla, armpit, intertriginous
   - Other: other body locations, other information, other factors, other details, etc.

7. SYMPTOMS - DERMATOLOGICAL:
   - Sensations: itching, pruritus, burning, stinging, pain, tenderness, soreness, hurting
   - Changes: increasing size, growing, spreading, expanding, darkening, lightening, bleeding
   - Other: bothersome appearance, cosmetic concern, disfigurement, numbness, tingling
   - Other: other dermatological symptoms, other information, other factors, other details, etc.

8. SYMPTOMS - SYSTEMIC:
   - General: fever, chills, fatigue, tiredness, malaise, lethargy, weight loss
   - Specific: joint pain, arthralgia, arthritis, mouth sores, oral ulcers, shortness of breath, dyspnea
   - Lymphatic: lymphadenopathy, swollen lymph nodes
   - Other: other systemic symptoms, other information, other factors, other details, etc.

9. DURATION/ONSET:
   - Duration: acute, chronic, subacute, hours, days, weeks, months, years, lifelong, congenital, since childhood, since birth
   - Onset: sudden onset, abrupt onset, gradual onset, slow onset, overnight, within hours, within days
   - Pattern: recurrent, relapsing, remitting, persistent, intermittent, first episode


10. TRIGGERS/EXACERBATING FACTORS:
    - Environmental: sun exposure, UV light, heat, cold, sweating, humidity, seasonal changes
    - Contact: allergens, irritants, chemicals, cosmetics, metals (nickel), latex, plants
    - Medications: drugs, medications, antibiotics, new medications
    - Biological: infection, bacteria, virus, fungus, insect bites, stings, trauma, friction, pressure
    - Lifestyle: stress, anxiety, hormonal changes, pregnancy, menstruation, menopause, diet, food
    - Occupational: occupational exposure, work-related
    - Other: other triggers, other excacerbating factors, causes, etc.

11. TREATMENTS (mentioned or suggested):
    - Topical: topical steroids, corticosteroids, creams, ointments, lotions, gels, emollients, moisturizers
    - Systemic medications: oral steroids, antibiotics, antifungals, antihistamines, immunosuppressants, biologics, retinoids, methotrexate
    - Physical: phototherapy, UV therapy, laser therapy, surgery, excision, cryotherapy, freezing
    - Supportive: home remedies, over-the-counter, OTC, self-care, wound care, dressings
    - Other: chemotherapy, radiation therapy (for skin cancer), other treatments, other supportive measures, etc.

12. CLINICAL SIGNS:
    - Specific signs: Nikolsky sign, Auspitz sign, Koebner phenomenon, Darier sign
    - Patterns: dermoscopic patterns, Wickham striae, target lesions, iris lesions, pathergy
    - Diagnostic: biopsy-proven, histologically confirmed, clinically diagnosed

13. HISTORY/CONTEXT:
    - Personal: family history, genetic predisposition, atopy, allergies, asthma
    - Disease course: recurrence, previous episodes, chronic condition, new onset
    - Immune status: immunocompromised, immunosuppressed, HIV, diabetes, autoimmune disease
    - Comorbidities: associated diseases, concurrent conditions
    - Risk factors: sun damage, smoking, occupational hazards, travel history
    - Diagnosis source: self-diagnosed, patient-reported, clinician-diagnosed, dermatologist-confirmed, biopsy-proven
    - Other: other history, other context, other factors, other information, etc.

14. LESION COUNT/EXTENT:
    - Number: single, solitary, one lesion, few, several, multiple, numerous, countless
    - Distribution: localized to one area, scattered, generalized, universal
    - Other: other count, other extent, other measurements, other information, etc.

15. SECONDARY CHANGES:
    - Chronic changes: lichenification, thickening, atrophy
    - Trauma: excoriation, scratch marks, crusting, erosion, fissuring, maceration
    - Post-inflammatory: post-inflammatory hyperpigmentation, post-inflammatory hypopigmentation, scarring, keloid
    - Other: other changes, other modifications, other information, other factors, etc.

16. SEVERITY:
    - Scale: mild, moderate, severe, very severe, life-threatening
    - Impact: asymptomatic, symptomatic, disabling, affecting daily activities, quality of life impact


17. IMAGE/CONTEXT METADATA:
    - Image type: clinical image, dermoscopy, close-up, at angle, macro, microscopic
    - Image quality: clear, blurry, well-lit, poor lighting
    - View: anterior, posterior, lateral, close-up view
    - Other: other metadata, other information, other factors, other details, etc.

18. OTHER UNIQUE FEATURES:
    - Any other specific clinical findings, descriptors, or contextual information not covered above
    - Look for medical terminology, anatomical terms, and disease-specific descriptors
    - Other diseases, skin-diseases, or disease acronyms mentioned in captions
    - Other: other unique features, other information, other factors, other details, etc.

Return ONLY valid JSON — no markdown fences, no prose — in this structure:
{
  "feature_categories": [
    {
      "name": "snake_case_feature_name",
      "category": "one of: demographics | morphology | body_location | symptoms | duration | triggers | treatments | clinical_signs | history | severity | image_metadata | other",
      "description": "brief clinical description of what this feature captures",
      "example_values": ["value1", "value2", "value3"],
      "is_binary": true or false,
      "extraction_examples": ["phrase from caption that indicated this feature"]
    }
  ]
}
"""

# Shorter instructions for Qwen + vLLM --max-model-len 8192 (leaves room for captions + JSON).
DISCOVERY_COMPACT_SYSTEM_PROMPT = """You are a clinical NLP expert for dermatology. From the captions, extract ONLY features explicitly stated (no guessing).

Scan: demographics (age, sex, FST/skin tone, ethnicity); morphology (texture, color, shape, border, size, distribution); body_location; dermatologic + systemic symptoms; duration/onset; triggers; treatments; clinical signs; history; lesion count/extent; secondary changes; severity; image metadata; other distinct terms.

Rules: snake_case names; split compound findings; extraction_examples must quote caption phrases; merge duplicates.

Return ONLY valid JSON (no markdown fences):
{
  "feature_categories": [
    {
      "name": "snake_case_feature_name",
      "category": "demographics | morphology | body_location | symptoms | duration | triggers | treatments | clinical_signs | history | severity | image_metadata | other",
      "description": "brief clinical description",
      "example_values": ["value1", "value2"],
      "is_binary": true or false,
      "extraction_examples": ["phrase from caption"]
    }
  ]
}
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
        f"Extract all feature categories from these {len(captions)} dermatology captions "
        f"for the disease category '{label_name}':\n\n{numbered}"
    )


def features_from_discovery_response_text(text: str) -> list[dict]:
    """Parse discovery JSON into feature dicts (empty list if unparseable)."""
    parsed = parse_llm_json(text, debug=LLM_PARSE_DEBUG)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        return list(parsed.get("feature_categories") or [])
    if isinstance(parsed, list):
        return parsed
    return []


def discover_features_batch(
    captions: list[str], label_name: str, retries: int = 3
) -> list[dict]:
    """Send a batch of captions to LLM and extract feature categories."""
    user_prompt = build_discovery_user_prompt(captions, label_name)

    for attempt in range(retries):
        try:
            text = call_llm(user_prompt, get_discovery_system_prompt())
            parsed = parse_llm_json(text, debug=LLM_PARSE_DEBUG)
            if parsed is None:
                print(f"    JSON parse error (attempt {attempt+1}), retrying...")
                time.sleep(2 ** attempt)
                continue
            if isinstance(parsed, dict):
                return parsed.get("feature_categories", [])
            elif isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            print(f"    JSON parse error (attempt {attempt+1}), retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    print(f"    WARNING: All retries failed for batch, skipping.")
    return []


def _short_label_desc(label_name, max_len: int = 42) -> str:
    """Truncate label_name for tqdm descriptions (one line, no newlines)."""
    s = str(label_name).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _coerce_discovery_feature_items(items: list | None) -> list[dict]:
    """
    Some models (e.g. Qwen) return feature_categories as a mix of objects and bare strings.
    Merge expects dicts with a "name" key; coerce strings into minimal dicts, skip junk.
    """
    if not items:
        return []
    out: list[dict] = []
    for feat in items:
        if isinstance(feat, dict):
            out.append(feat)
            continue
        if isinstance(feat, str):
            s = feat.strip()
            if not s:
                continue
            out.append(
                {
                    "name": s,
                    "category": "other",
                    "description": "",
                    "example_values": [],
                    "is_binary": False,
                    "extraction_examples": [],
                }
            )
    return out


def _stringify_listish(x) -> list[str]:
    """Coerce example_values / extraction_examples to a list of strings (avoids set() on nested types)."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x] if x.strip() else []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def _normalize_schema_feature_categories(items: list | None) -> list[dict]:
    """Canonical schema list for phase 1 output and phase 2 input: dicts with non-empty names only."""
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


def _merge_raw_label_features_into_global(
    all_features: dict[str, dict], label_features: list, label_name
) -> None:
    """Merge one label's discovery list into the global feature dict (by snake_case name)."""
    for feat in _coerce_discovery_feature_items(label_features):
        name = str(feat.get("name") or "").lower().strip().replace(" ", "_")
        if not name:
            continue
        if name not in all_features:
            all_features[name] = feat
            all_features[name]["found_in_labels"] = [label_name]
        else:
            existing_ex = set(_stringify_listish(all_features[name].get("example_values")))
            new_ex = set(_stringify_listish(feat.get("example_values")))
            all_features[name]["example_values"] = list(existing_ex | new_ex)[:10]
            existing_extr = set(_stringify_listish(all_features[name].get("extraction_examples")))
            new_extr = set(_stringify_listish(feat.get("extraction_examples")))
            all_features[name]["extraction_examples"] = list(existing_extr | new_extr)[:5]
            if label_name not in all_features[name].get("found_in_labels", []):
                all_features[name]["found_in_labels"].append(label_name)


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


def _discover_all_features_openai_batch(sample_df: pd.DataFrame) -> dict[str, dict]:
    """
    OpenAI-only: enqueue all discovery chat completions on the Batch API, then assemble results.
    Per-label disk cache is respected; only missing labels are batched.
    """
    all_features: dict[str, dict] = {}
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
                label_features = json.load(f)
            tqdm.write(
                f"  [{label_name}] loaded {len(label_features)} cached features (skipped LLM)"
            )
            _merge_raw_label_features_into_global(all_features, label_features, label_name)
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
        label_features: list = []
        for _start, cid, meta in entries:
            text = mapping.get(cid) or ""
            feats = features_from_discovery_response_text(text)
            if not feats:
                tqdm.write(
                    f"    Batch parse/sync fallback: {meta['label_name']!r} "
                    f"batch @ {meta['batch_start']}"
                )
                feats = discover_features_batch(meta["captions"], meta["label_name"])
            label_features.extend(feats)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(label_features, f, indent=2)
        src_note = (
            f"{uniq_n:,} unique captions"
            if DISCOVERY_DEDUPE_CAPTIONS and raw_caption_n != uniq_n
            else f"{uniq_n:,} captions"
        )
        tqdm.write(
            f"  [{label_name}] discovered {len(label_features)} feature(s) from {src_note} (batch)"
        )
        _merge_raw_label_features_into_global(all_features, label_features, label_name)

    return all_features


def _discover_all_features_sequential(sample_df: pd.DataFrame) -> dict[str, dict]:
    """Gemini / Qwen / OpenAI sync: one chat completion per caption batch."""
    all_features: dict[str, dict] = {}
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
                label_features = json.load(f)
            tqdm.write(
                f"  [{label_name}] loaded {len(label_features)} cached features (skipped LLM)"
            )
        else:
            label_features = []
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
                label_features.extend(feats)
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
            tqdm.write(
                f"  [{label_name}] discovered {len(label_features)} feature(s) from {src_note}"
            )

        _merge_raw_label_features_into_global(all_features, label_features, label_name)

    return all_features


def discover_all_features(sample_df: pd.DataFrame) -> dict[str, dict]:
    """
    Run per-label-name discovery across all sampled captions.
    Returns a dict of {feature_name: feature_dict}.
    """
    if LLM_PROVIDER == "openai" and OPENAI_USE_BATCH:
        return _discover_all_features_openai_batch(sample_df)
    return _discover_all_features_sequential(sample_df)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Consolidation via LLM
# ──────────────────────────────────────────────────────────────────────────────

CONSOLIDATION_SYSTEM_PROMPT = """You are a senior clinical NLP engineer building a canonical
feature schema for a skin disease classification system.

Your task:
1. DEDUPLICATE synonyms (e.g. 'lesion_colour' and 'color_of_lesion' → 'lesion_color';
   'itch' and 'pruritus' → 'symptom_itching')
2. MERGE overlapping features into canonical names
3. STANDARDISE names to snake_case
4. DO NOT REMOVE any unique features that are useful for differential diagnosis
5. ENSURE the following categories are well-represented:
   - demographics (age, sex, ethnicity, fitzpatrick skin type, skin tone), morphology (texture, color, shape/border, distribution),
   - body_location, symptoms (dermatological + systemic),
   - duration/onset, triggers, treatments, clinical_signs, history
   - lesion count/extent, secondary changes, severity, other
6. For each feature, decide: is_binary (true = present/absent encoding)
   or categorical (false = needs value strings like "topical|systemic|surgical")

Return ONLY valid JSON — no markdown fences, no prose — with this structure:
{
  "feature_categories": [
    {
      "name": "snake_case_canonical_name",
      "category": "demographics | morphology | body_location | symptoms | duration | triggers | treatments | clinical_signs | history | severity | other",
      "description": "brief clinical description",
      "example_values": ["val1", "val2"],
      "is_binary": true/false,
      "regex_extractable": true/false
    }
  ]
}
"""

# Shorter system prompt for batched reduce steps (smaller completion than one-shot 700+ features).
CONSOLIDATION_CHUNK_SYSTEM_PROMPT = """You consolidate dermatology feature definitions for a classifier schema.

You receive ONE BATCH of raw features that is part of a larger list (other batches exist).
Merge obvious duplicates and synonyms WITHIN this batch only. Keep clinically distinct concepts separate.
Use snake_case names. description ≤120 characters. At most 3 strings in example_values per feature.
Set is_binary and regex_extractable sensibly (same meaning as in the full schema).

Return ONLY valid JSON — no markdown — exactly:
{"feature_categories":[{"name":"...","category":"...","description":"...","example_values":[],"is_binary":true,"regex_extractable":false}, ...]}
"""


def _merge_consolidation_features_by_name(features: list) -> list[dict]:
    """Collapse features that share the same normalised snake_case name."""
    out: dict[str, dict] = {}
    for raw in features:
        if not isinstance(raw, dict):
            continue
        nm = str(raw.get("name") or "").strip()
        if not nm:
            continue
        key = nm.lower().replace(" ", "_")
        if key not in out:
            d = dict(raw)
            d["name"] = nm
            if "n_labels_found" not in d:
                d["n_labels_found"] = int(raw.get("n_labels_found", 0) or 0)
            out[key] = d
        else:
            ex = out[key]
            oev = _stringify_listish(ex.get("example_values"))
            nev = _stringify_listish(raw.get("example_values"))
            ex["example_values"] = list(dict.fromkeys(oev + nev))[:8]
            d1 = str(ex.get("description") or "")
            d2 = str(raw.get("description") or "")
            if len(d2) > len(d1):
                ex["description"] = d2
            ex["n_labels_found"] = max(
                int(ex.get("n_labels_found", 0) or 0),
                int(raw.get("n_labels_found", 0) or 0),
            )
    return list(out.values())


def _parsed_to_feature_list(parsed) -> list | None:
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "feature_categories" in parsed:
        return parsed.get("feature_categories")
    return None


def _consolidate_schema_single_pass(
    compact: list[dict],
    n_captions_sampled: int,
    n_disease_classes: int,
) -> dict | None:
    """One full-schema LLM call; returns None on failure."""
    raw_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        f"You are given a raw list of {len(compact)} clinical feature categories that were\n"
        f"extracted by an LLM from {n_captions_sampled} dermatology case captions across\n"
        f"{n_disease_classes} disease classes.\n\n"
        f"Here is the raw feature list:\n{raw_json}"
    )
    for attempt in range(3):
        try:
            text = call_llm(
                prompt,
                CONSOLIDATION_SYSTEM_PROMPT,
                openai_prompt_cache_key=OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY,
                openai_max_tokens=OPENAI_CONSOLIDATION_MAX_TOKENS,
            )
            parsed = parse_llm_json(text, debug=LLM_PARSE_DEBUG)
            items = _parsed_to_feature_list(parsed)
            if items is None:
                print(
                    f"  Consolidation JSON parse error (attempt {attempt+1}), retrying... "
                    "(large lists use chunked consolidation automatically if this keeps failing)"
                )
                time.sleep(3)
                continue
            if isinstance(parsed, list):
                parsed = {"feature_categories": items}
            else:
                parsed = dict(parsed)
                parsed["feature_categories"] = items
            parsed["feature_categories"] = _normalize_schema_feature_categories(
                parsed.get("feature_categories")
            )
            print(f"  Consolidated to {len(parsed['feature_categories'])} features")
            return parsed
        except json.JSONDecodeError:
            print(f"  Consolidation JSON parse error (attempt {attempt+1}), retrying...")
            time.sleep(3)
        except Exception as e:
            print(f"  Consolidation API error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


def _llm_consolidation_chunk(
    batch: list[dict],
    *,
    round_idx: int,
    batch_idx: int,
    n_batches: int,
) -> list[dict] | None:
    body = json.dumps(batch, ensure_ascii=False, separators=(",", ":"))
    user = (
        f"Consolidation batch {batch_idx} of {n_batches} in reduce round {round_idx}.\n"
        f"Merge duplicates only within this batch ({len(batch)} features).\n\n"
        f"JSON:\n{body}"
    )
    for attempt in range(3):
        try:
            text = call_llm(
                user,
                CONSOLIDATION_CHUNK_SYSTEM_PROMPT,
                openai_prompt_cache_key=OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY + "_chunk",
                openai_max_tokens=OPENAI_CONSOLIDATION_MAX_TOKENS,
            )
            parsed = parse_llm_json(text, debug=LLM_PARSE_DEBUG)
            items = _parsed_to_feature_list(parsed)
            if not items:
                print(
                    f"    Chunk r{round_idx} b{batch_idx}: parse failed "
                    f"(attempt {attempt+1}), retrying..."
                )
                time.sleep(2)
                continue
            norm = _normalize_schema_feature_categories(items)
            if OPENAI_CONSOLIDATION_SAVE_CHUNK_JSON:
                out_dir = DISCOVERY_DIR / "consolidation_batches"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"round{round_idx}_batch{batch_idx}_of{n_batches}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"feature_categories": norm}, f, indent=2, ensure_ascii=False)
                print(f"    Wrote chunk response → {out_path}")
            return norm
        except Exception as e:
            print(f"    Chunk r{round_idx} b{batch_idx} error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def _consolidate_schema_final_polish(
    current: list[dict],
    n_captions_sampled: int,
    n_disease_classes: int,
) -> dict | None:
    raw_json = json.dumps(current, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        f"After batched merging, there are {len(current)} candidate feature categories.\n"
        f"Perform a GLOBAL polish: merge synonyms across the whole list, standardise snake_case, "
        f"fix categories. Keep clinically distinct features.\n"
        f"Corpus: {n_captions_sampled} captions sampled, {n_disease_classes} disease classes.\n\n"
        f"JSON:\n{raw_json}"
    )
    for attempt in range(3):
        try:
            text = call_llm(
                prompt,
                CONSOLIDATION_SYSTEM_PROMPT,
                openai_prompt_cache_key=OPENAI_CONSOLIDATION_PROMPT_CACHE_KEY,
                openai_max_tokens=OPENAI_CONSOLIDATION_MAX_TOKENS,
            )
            parsed = parse_llm_json(text, debug=LLM_PARSE_DEBUG)
            items = _parsed_to_feature_list(parsed)
            if items is None:
                print(
                    f"  Final polish JSON parse error (attempt {attempt+1}), retrying... "
                    "(if truncated, lower OPENAI_CONSOLIDATION_SINGLE_CALL_MAX or CHUNK_SIZE)"
                )
                time.sleep(3)
                continue
            out = {
                "feature_categories": _normalize_schema_feature_categories(items),
            }
            print(f"  Final polish: {len(out['feature_categories'])} features")
            return out
        except Exception as e:
            print(f"  Final polish error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


def _consolidate_schema_chunked_reduce(
    compact: list[dict],
    n_captions_sampled: int,
    n_disease_classes: int,
) -> dict:
    chunk_sz = OPENAI_CONSOLIDATION_CHUNK_SIZE
    single_max = OPENAI_CONSOLIDATION_SINGLE_CALL_MAX
    current = list(compact)
    round_idx = 0
    max_rounds = 50

    print(
        f"  Chunked consolidation: CHUNK_SIZE={chunk_sz}, "
        f"SINGLE_CALL_MAX={single_max} (gpt-4o-mini output ~16k tokens)."
    )
    if OPENAI_CONSOLIDATION_SAVE_CHUNK_JSON:
        batch_dir = DISCOVERY_DIR / "consolidation_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Per-batch JSON → {batch_dir}/ (OPENAI_CONSOLIDATION_SAVE_CHUNK_JSON)")

    while len(current) > single_max and round_idx < max_rounds:
        round_idx += 1
        prev_len = len(current)
        n_batches = (len(current) + chunk_sz - 1) // chunk_sz
        print(
            f"  Reduce round {round_idx}: {len(current)} features in {n_batches} LLM batch(es)"
        )
        next_parts: list[dict] = []
        for bi in range(n_batches):
            sl = current[bi * chunk_sz : (bi + 1) * chunk_sz]
            got = _llm_consolidation_chunk(
                sl, round_idx=round_idx, batch_idx=bi + 1, n_batches=n_batches
            )
            if got is None:
                print("  WARNING: Chunk LLM failed; using name-merged list without further LLM steps.")
                return {
                    "feature_categories": _normalize_schema_feature_categories(
                        _merge_consolidation_features_by_name(current)
                    ),
                }
            next_parts.extend(got)
        current = _merge_consolidation_features_by_name(next_parts)
        print(f"  → {len(current)} features after name-merge")
        if len(current) >= prev_len and round_idx >= 4:
            tqdm.write(
                "  NOTE: Slow shrinkage — consider lowering OPENAI_CONSOLIDATION_CHUNK_SIZE."
            )

    if round_idx >= max_rounds and len(current) > single_max:
        print(
            f"  WARNING: Hit max reduce rounds ({max_rounds}); final polish may still truncate."
        )

    final = _consolidate_schema_final_polish(
        current, n_captions_sampled, n_disease_classes
    )
    if final is not None:
        return final

    print(
        "  WARNING: Final polish failed or truncated; returning merge-only feature list "
        "(run again or tighten OPENAI_CONSOLIDATION_SINGLE_CALL_MAX)."
    )
    return {
        "feature_categories": _normalize_schema_feature_categories(current),
    }


def consolidate_schema(
    all_features: dict[str, dict],
    n_captions_sampled: int = 0,
    n_disease_classes: int = 0,
) -> dict:
    """Deduplicate discovered features via LLM (chunked when the list is large)."""
    compact: list[dict] = []
    for name, feat in all_features.items():
        compact.append(
            {
                "name": name,
                "category": feat.get("category", "other"),
                "description": feat.get("description", ""),
                "example_values": feat.get("example_values", [])[:5],
                "is_binary": feat.get("is_binary", True),
                "n_labels_found": len(feat.get("found_in_labels", [])),
            }
        )

    print(f"  Sending {len(compact)} raw features for consolidation...")

    if len(compact) <= OPENAI_CONSOLIDATION_SINGLE_CALL_MAX:
        done = _consolidate_schema_single_pass(
            compact, n_captions_sampled, n_disease_classes
        )
        if done is not None:
            return done
        print("  Single-pass consolidation failed; using chunked reduce on the full list...")

    return _consolidate_schema_chunked_reduce(
        compact, n_captions_sampled, n_disease_classes
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: SCIN column alignment (AFTER final feature list is established)
# ──────────────────────────────────────────────────────────────────────────────
# SCIN column → canonical names: shared module scin_feature_map.SCIN_SCHEMA_FEATURES

def add_scin_alignment(schema: dict) -> dict:
    """
    Tag each schema feature with its SCIN column counterpart (if any).
    This is done AFTER the final feature list is established from Derm-1M.
    """
    fc = schema.get("feature_categories")
    schema["feature_categories"] = _normalize_schema_feature_categories(
        fc if isinstance(fc, list) else []
    )
    if isinstance(fc, list) and len(fc) > 0 and len(schema["feature_categories"]) == 0:
        print(
            "  WARNING: feature_categories had no valid dict/string entries after normalization."
        )

    existing_names = {f["name"] for f in schema["feature_categories"]}

    # Track which features are SCIN-comparable
    scin_comparable_count = 0

    for scin_col, canonical in SCIN_SCHEMA_FEATURES.items():
        # Find feature in schema or add it as SCIN-sourced
        matched = next(
            (f for f in schema["feature_categories"] if f["name"] == canonical), None
        )
        if matched:
            matched["scin_column"] = scin_col
            matched["scin_comparable"] = True
            scin_comparable_count += 1
        else:
            # Add missing SCIN feature to schema so it gets extracted
            schema["feature_categories"].append({
                "name": canonical,
                "category": _infer_category(canonical),
                "description": f"Mapped from SCIN column: {scin_col}",
                "example_values": [],
                "is_binary": True,
                "regex_extractable": _is_regex_extractable(canonical),
                "scin_column": scin_col,
                "scin_comparable": True,
                "derm1m_sourced": False,
            })
            scin_comparable_count += 1
    
    print(f"  Aligned {scin_comparable_count} features with SCIN columns")
    return schema


def _infer_category(name: str) -> str:
    if name.startswith("symptom_"):   return "symptoms"
    if name.startswith("location_"):  return "body_location"
    if name.startswith("texture_"):   return "morphology"
    if name.startswith("color_"):     return "morphology"
    if name.startswith("trigger_"):   return "triggers"
    if name.startswith("treatment_"): return "treatments"
    if name.startswith("race_"):        return "demographics"
    if name in ("age_group", "sex", "fitzpatrick_skin_type"): return "demographics"
    if name == "duration":            return "duration"
    if name == "related_category":    return "condition_metadata"
    return "other"


def _is_regex_extractable(name: str) -> bool:
    """Heuristic: features that can be reliably extracted via keyword/regex."""
    regex_prefixes = (
        "symptom_", "location_", "texture_", "color_",
        "distribution_", "race_",
    )
    regex_names = {
        "age_group", "sex", "fitzpatrick_skin_type", "duration",
        "onset_sudden", "lesion_count", "diagnosis_confidence",
        "related_category",
    }
    return name.startswith(regex_prefixes) or name in regex_names


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
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

    # Save sample for reference
    sample_path = DISCOVERY_DIR / "sampled_captions.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"  Saved sample to {sample_path}\n")

    # ── Step 2: Per-label discovery ───────────────────────────────────────────
    print("STEP 2: Running per-label-name LLM feature discovery...\n")
    all_features = discover_all_features(sample_df)
    print(f"\n  Total raw features discovered: {len(all_features)}\n")

    # Save raw features
    raw_path = DISCOVERY_DIR / "raw_features_all.json"
    # Convert for JSON serialization (found_in_labels might cause issues)
    raw_serializable = {}
    for k, v in all_features.items():
        entry = dict(v)
        entry["found_in_labels"] = list(entry.get("found_in_labels", []))
        raw_serializable[k] = entry
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_serializable, f, indent=2)
    print(f"  Saved raw features to {raw_path}\n")

    # ── Step 3: Consolidation ─────────────────────────────────────────────────
    print("STEP 3: Consolidating features via LLM...\n")
    schema = consolidate_schema(
        all_features,
        n_captions_sampled=len(sample_df),
        n_disease_classes=int(sample_df["label_name"].nunique()),
    )

    # ── Step 4: SCIN alignment (AFTER final feature list is established) ─────
    print("\nSTEP 4: Aligning with SCIN columns (after final feature list established)...\n")
    schema = add_scin_alignment(schema)

    n_total = len(schema["feature_categories"])
    n_comparable = sum(
        1 for f in schema["feature_categories"] if f.get("scin_comparable")
    )
    n_regex = sum(
        1 for f in schema["feature_categories"] if f.get("regex_extractable")
    )
    n_llm_only = n_total - n_regex

    print(f"  Total canonical features:     {n_total}")
    print(f"  SCIN-comparable features:     {n_comparable}")
    print(f"  Regex-extractable features:   {n_regex}")
    print(f"  LLM-only features:            {n_llm_only}\n")

    # ── Save schema ───────────────────────────────────────────────────────────
    # Add metadata
    schema["metadata"] = {
        "source_csv": CSV_PATH,
        "caption_column": CAPTION_COLUMN,
        "discovery_sampling_mode": DISCOVERY_SAMPLING_MODE,
        "n_captions_sampled": len(sample_df),
        "n_rows_used": len(sample_df),
        "n_label_names": int(sample_df["label_name"].nunique()),
        "n_total_features": n_total,
        "n_scin_comparable": n_comparable,
        "n_regex_extractable": n_regex,
        "n_llm_only": n_llm_only,
        "model_used": DISCOVERY_MODEL,
        "llm_provider": LLM_PROVIDER,
        "openai_use_batch": bool(LLM_PROVIDER == "openai" and OPENAI_USE_BATCH),
        "openai_prompt_cache_key": OPENAI_PROMPT_CACHE_KEY if LLM_PROVIDER == "openai" else "",
    }

    with open(SCHEMA_OUT, "w", encoding="utf-8") as fp:
        json.dump(schema, fp, indent=2)

    print(f"  Schema saved to {SCHEMA_OUT}")
    print(f"  Next: run phase2_bulk_extraction.py")
