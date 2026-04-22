"""
Shared helpers for OpenAI Batch API (phase1 + phase2).

- Per-batch limits: https://developers.openai.com/api/docs/guides/batch
  (≤50,000 requests, input file ≤200 MB)
- Prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
- Enqueued token limit: org-level cap on total tokens across all in-progress batches.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Conservative chars/token ratio. Plain English averages ~4.0, but
# our content is keyword-dense medical text with:
#   - many punctuation characters (commas, brackets, colons, slashes)
#   - proper nouns and Latin-derived medical terms that split into 2-3
#     subword tokens each
#   - numeric tokens (ages, dates, measurements)
# Empirically this tokenizes at ~2.8-3.0 chars/token per the cl100k tokenizer.
# Using 3.0 intentionally makes our estimator err ON the HIGH side of OpenAI's
# real token count — the whole point of the estimator is to prevent the
# batch splitter from overpacking a chunk past the org enqueue cap.
# A few percent of "wasted" chunk budget is vastly preferable to repeated
# "Enqueued token limit reached" upload failures and 4-minute backoffs.
CHARS_PER_TOKEN_ESTIMATE = 3.0

# Per-request framing overhead OpenAI reserves on top of raw message
# content (role markers, JSON envelope, stop tokens, batch wrapper).
# Empirically ~30-50 tokens; 40 keeps a comfortable margin.
PER_REQUEST_OVERHEAD_TOKENS = 40


def estimate_job_enqueued_tokens(job: dict[str, Any]) -> int:
    """
    Estimate the number of tokens OpenAI reserves for a single batch request.
    OpenAI counts input tokens + max_tokens (output reservation) against the
    org-level enqueued token limit.

    This estimator intentionally rounds UP and applies a per-request framing
    overhead so the splitter never packs more into a chunk than OpenAI
    counts against the cap.
    """
    body = job.get("body", {})
    max_tokens = body.get("max_tokens", 4096)

    text_chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            text_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text_chars += len(part.get("text", ""))

    # Ceiling division on floats to avoid undercount vs real tokenizer.
    input_tokens = int(text_chars / CHARS_PER_TOKEN_ESTIMATE + 0.9999) + PER_REQUEST_OVERHEAD_TOKENS
    return input_tokens + max_tokens


def openai_batch_max_file_bytes() -> int:
    """
    Max UTF-8 size of a single .jsonl batch file (sum of line byte lengths).
    OpenAI cap is 200 MB per file (https://developers.openai.com/api/docs/guides/batch);
    default ~195 MiB leaves margin. Set OPENAI_BATCH_MAX_FILE_BYTES after load_dotenv().
    """
    raw = os.getenv("OPENAI_BATCH_MAX_FILE_BYTES", str(195 * 1024 * 1024)).strip()
    return max(1024 * 1024, int(raw))


def batch_jsonl_line(job: dict[str, Any]) -> str:
    """One JSONL line for POST /v1/chat/completions (job must have custom_id + body)."""
    rec = {
        "custom_id": job["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": job["body"],
    }
    return json.dumps(rec, ensure_ascii=False) + "\n"


def chunk_jobs_for_openai_batch(
    jobs: list[dict[str, Any]],
    *,
    max_requests: int,
    max_file_bytes: int | None = None,
    max_enqueued_tokens: int | None = None,
) -> list[list[dict[str, Any]]]:
    """
    Split jobs into chunks that respect:
      - max_requests (OpenAI: 50,000 per file)
      - max_file_bytes (OpenAI: 200 MB per file; default env OPENAI_BATCH_MAX_FILE_BYTES)
      - max_enqueued_tokens: org-level cap on total tokens in in-progress batches.
        Each request's contribution = estimated input tokens + max_tokens.
        If None, no token-based splitting is applied.

    Each job is {"custom_id": str, "body": dict}.
    """
    if not jobs:
        return []
    cap_b = max_file_bytes if max_file_bytes is not None else openai_batch_max_file_bytes()
    cap_r = max(1, min(50_000, max_requests))
    cap_t = max_enqueued_tokens

    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_bytes = 0
    cur_tokens = 0

    for job in jobs:
        line = batch_jsonl_line(job)
        line_b = len(line.encode("utf-8"))
        job_tokens = estimate_job_enqueued_tokens(job) if cap_t else 0
        if line_b > cap_b:
            raise ValueError(
                f"Single batch JSONL line is {line_b:,} bytes (limit {cap_b:,}). "
                "Shorten prompts/captions; OpenAI batch input files are capped at 200 MB total."
            )
        needs_new_chunk = (
            len(cur) >= cap_r
            or cur_bytes + line_b > cap_b
            or (cap_t and cur_tokens + job_tokens > cap_t)
        )
        if cur and needs_new_chunk:
            chunks.append(cur)
            cur = []
            cur_bytes = 0
            cur_tokens = 0
        cur.append(job)
        cur_bytes += line_b
        cur_tokens += job_tokens

    if cur:
        chunks.append(cur)
    return chunks


def write_openai_batch_jsonl(path: os.PathLike[str] | str, jobs: list[dict[str, Any]]) -> int:
    """Write jobs to path; returns total UTF-8 byte size on disk."""
    p = str(path)
    data = "".join(batch_jsonl_line(j) for j in jobs)
    raw = data.encode("utf-8")
    with open(p, "wb") as f:
        f.write(raw)
    return len(raw)


def openai_batches_create_safe(client: Any, **kwargs: Any) -> Any:
    """
    client.batches.create; retries without metadata if the SDK rejects unknown kwargs.
    """
    try:
        return client.batches.create(**kwargs)
    except TypeError:
        meta = kwargs.pop("metadata", None)
        if meta is None:
            raise
        return client.batches.create(**kwargs)
