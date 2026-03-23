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
LLM_PARSE_DEBUG = os.getenv("LLM_PARSE_DEBUG", "").lower() in ("1", "true", "yes")
# Must match the server's --max-model-len; vLLM returns 400 if prompt_tokens + max_tokens exceeds this.
QWEN_MAX_MODEL_LEN = int(os.getenv("QWEN_MAX_MODEL_LEN", "8192"))
# Full discovery JSON often needs >8k completion tokens when context allows; cap still clamped per-request.
_qwen_mt_env = os.getenv("QWEN_MAX_TOKENS", "").strip()
if _qwen_mt_env:
    QWEN_MAX_TOKENS = int(_qwen_mt_env)
else:
    QWEN_MAX_TOKENS = min(16384, max(8192, QWEN_MAX_MODEL_LEN // 2))
QWEN_CONTEXT_BUFFER = int(os.getenv("QWEN_CONTEXT_BUFFER", "256"))
# Heuristic when /tokenize is unavailable (chars per token ~2.5–3 for English + medical terms)
QWEN_CHARS_PER_TOKEN = float(os.getenv("QWEN_CHARS_PER_TOKEN", "2.75"))
QWEN_TEMPLATE_TOKEN_OVERHEAD = int(os.getenv("QWEN_TEMPLATE_TOKEN_OVERHEAD", "800"))
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
    genai.configure(api_key=GEMINI_API_KEY)
    DISCOVERY_MODEL = "gemini-2.5-flash-lite"
    model = genai.GenerativeModel(DISCOVERY_MODEL)
elif LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        raise ValueError("Set OPENAI_API_KEY in your .env file for OpenAI provider")
    import openai
    openai.api_key = OPENAI_API_KEY
    DISCOVERY_MODEL = "gpt-4o-mini"
    model = None
elif LLM_PROVIDER == "qwen":
    from openai import OpenAI as QwenClient
    qwen_client = QwenClient(base_url=QWEN_BASE_URL, api_key="EMPTY")
    DISCOVERY_MODEL = QWEN_MODEL_NAME
    model = None
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'gemini', 'openai', or 'qwen'")

# Captions per LLM call — smaller batches for local Qwen + short max_model_len reduce truncated JSON
_bs_env = os.getenv("DISCOVERY_BATCH_SIZE", "").strip()
if _bs_env:
    BATCH_SIZE = max(1, int(_bs_env))
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


def call_llm(prompt: str, system_prompt: str = None, retries: int = 3) -> str:
    """
    Generic LLM caller that works with Gemini, OpenAI, and Qwen (local).
    Returns the text response.
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
                    model=DISCOVERY_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                if OPENAI_JSON_RESPONSE:
                    create_kw["response_format"] = {"type": "json_object"}
                response = openai.chat.completions.create(**create_kw)
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


def discover_features_batch(
    captions: list[str], label_name: str, retries: int = 3
) -> list[dict]:
    """Send a batch of captions to LLM and extract feature categories."""
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(captions))
    prompt = (
        f"Extract all feature categories from these {len(captions)} dermatology captions "
        f"for the disease category '{label_name}':\n\n{numbered}"
    )

    for attempt in range(retries):
        try:
            text = call_llm(prompt, get_discovery_system_prompt())
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


def discover_all_features(sample_df: pd.DataFrame) -> dict[str, dict]:
    """
    Run per-label-name discovery across all sampled captions.
    Returns a dict of {feature_name: feature_dict}.
    """
    all_features: dict[str, dict] = {}
    label_names = sample_df["label_name"].unique()

    for label_name in tqdm(
        label_names,
        desc="Discovery (per label_name)",
        unit="label",
    ):
        label_captions = (
            sample_df[sample_df["label_name"] == label_name][CAPTION_COLUMN]
            .tolist()
        )
        raw_caption_n = len(label_captions)
        dup_skipped = 0
        if DISCOVERY_DEDUPE_CAPTIONS:
            label_captions, dup_skipped = unique_captions_preserve_order(
                [str(x) for x in label_captions]
            )
        else:
            label_captions = [str(x) for x in label_captions]

        uniq_n = len(label_captions)
        if DISCOVERY_DEDUPE_CAPTIONS:
            dup_part = (
                f"; {dup_skipped:,} duplicate/empty row(s) not sent to LLM"
                if dup_skipped
                else ""
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

        # Check for cached per-label discovery (separate cache per sampling mode)
        safe_name = re.sub(r'[^\w\-]', '_', str(label_name))[:60]
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
                # Rate limiting — local qwen needs no delay, remote APIs do
                if LLM_PROVIDER == "qwen":
                    time.sleep(0.1)
                elif LLM_PROVIDER == "openai":
                    time.sleep(0.5)
                else:
                    time.sleep(1)

            # Cache per-label results
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

        # Merge into global feature dict (deduplicate by name)
        for feat in label_features:
            name = feat.get("name", "").lower().strip().replace(" ", "_")
            if not name:
                continue
            if name not in all_features:
                all_features[name] = feat
                all_features[name]["found_in_labels"] = [label_name]
            else:
                # Merge example values and track which labels have this feature
                existing_ex = set(all_features[name].get("example_values", []))
                new_ex = set(feat.get("example_values", []))
                all_features[name]["example_values"] = list(existing_ex | new_ex)[:10]
                existing_extr = set(all_features[name].get("extraction_examples", []))
                new_extr = set(feat.get("extraction_examples", []))
                all_features[name]["extraction_examples"] = list(existing_extr | new_extr)[:5]
                if label_name not in all_features[name].get("found_in_labels", []):
                    all_features[name]["found_in_labels"].append(label_name)

    return all_features


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

def consolidate_schema(
    all_features: dict[str, dict],
    n_captions_sampled: int = 0,
    n_disease_classes: int = 0,
) -> dict:
    """Send all discovered features to LLM for deduplication and consolidation."""
    # Prepare a compact version for the prompt (drop found_in_labels to save tokens)
    compact = []
    for name, feat in all_features.items():
        compact.append({
            "name": name,
            "category": feat.get("category", "other"),
            "description": feat.get("description", ""),
            "example_values": feat.get("example_values", [])[:5],
            "is_binary": feat.get("is_binary", True),
            "n_labels_found": len(feat.get("found_in_labels", [])),
        })

    raw_json = json.dumps(compact, indent=1)
    prompt = (
        f"You are given a raw list of {len(compact)} clinical feature categories that were\n"
        f"extracted by an LLM from {n_captions_sampled} dermatology case captions across\n"
        f"{n_disease_classes} disease classes.\n\n"
        f"Here is the raw feature list:\n{raw_json}"
    )

    print(f"  Sending {len(compact)} raw features for consolidation...")

    for attempt in range(3):
        try:
            text = call_llm(prompt, CONSOLIDATION_SYSTEM_PROMPT)
            parsed = parse_llm_json(text, debug=LLM_PARSE_DEBUG)
            if parsed is None:
                print(f"  Consolidation JSON parse error (attempt {attempt+1}), retrying...")
                time.sleep(3)
                continue
            if isinstance(parsed, list):
                parsed = {"feature_categories": parsed}
            if isinstance(parsed, dict) and "feature_categories" in parsed:
                print(f"  Consolidated to {len(parsed['feature_categories'])} features")
                return parsed
        except json.JSONDecodeError:
            print(f"  Consolidation JSON parse error (attempt {attempt+1}), retrying...")
            time.sleep(3)
        except Exception as e:
            print(f"  Consolidation API error (attempt {attempt+1}): {e}")
            time.sleep(3)

    # Fallback: use raw features as-is
    print("  WARNING: Consolidation failed, using raw discovered features.")
    return {"feature_categories": compact}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: SCIN column alignment (AFTER final feature list is established)
# ──────────────────────────────────────────────────────────────────────────────
# SCIN column → canonical names: shared module scin_feature_map.SCIN_SCHEMA_FEATURES

def add_scin_alignment(schema: dict) -> dict:
    """
    Tag each schema feature with its SCIN column counterpart (if any).
    This is done AFTER the final feature list is established from Derm-1M.
    """
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
    }

    with open(SCHEMA_OUT, "w", encoding="utf-8") as fp:
        json.dump(schema, fp, indent=2)

    print(f"  Schema saved to {SCHEMA_OUT}")
    print(f"  Next: run phase2_bulk_extraction.py")
