# OpenAI Token Exhaustion Analysis — Phase 2 Feature Extraction

## The Problem

Running `phase2_bulk_extraction.py` with the default configuration exhausts OpenAI API tokens far faster than expected. The root cause is a mismatch between the configured `max_tokens` output limit and the actual volume of JSON output the model must generate per API call.

---

## Dataset & Schema Scale

| Parameter                | Value                        |
| ------------------------ | ---------------------------- |
| CSV file                 | `cleaned_caption_Derm1M.csv` |
| Total data rows          | 182,062                      |
| Unique labels            | 95                           |
| Features to extract      | 182                          |
| Default `LLM_BATCH_SIZE` | 25 captions per API call     |
| Configured `max_tokens`  | 4,096                        |

---

## Actual Caption Length Distribution

Verified from `cleaned_caption_Derm1M.csv` (column: `truncated_caption`):

| Metric                   | Value       |
| ------------------------ | ----------- |
| Total captions with data | 181,268     |
| Average length           | 272.5 chars |
| Median length            | 235 chars   |
| Min                      | 1 char      |
| Max                      | 1,911 chars |

| Range (chars) | Count  | Percentage |
| ------------- | ------ | ---------- |
| 0–50          | 11,762 | 6.5%       |
| 50–100        | 23,427 | 12.9%      |
| 100–150       | 19,355 | 10.7%      |
| 150–200       | 12,506 | 6.9%       |
| 200–250       | 35,547 | 19.6%      |
| 250–300       | 24,396 | 13.5%      |
| 300–400       | 26,093 | 14.4%      |
| 400–500       | 8,892  | 4.9%       |
| 500–1,000     | 15,427 | 8.5%       |
| 1,000+        | 3,863  | 2.1%       |

Only ~37% of captions are under 200 characters. The bulk (47.5%) sits in the 200–400 range.

---

## The Math Behind Token Consumption

### Step 1: Output tokens required per caption

Each caption must produce a JSON object with all 182 features. A single feature key-value pair looks like:

```json
{ "symptom_itching": 1 }
```

Token estimation per feature key-value pair:

| Component                     | Characters | Tokens (approx) |
| ----------------------------- | ---------- | --------------- |
| Key name (avg 25 chars)       | 25         | 5               |
| Colon + space + value + comma | 5          | 2               |
| **Per feature**               | **30**     | **7**           |

Per caption output:

```
182 features × 7 tokens = 1,274 tokens
JSON brackets/overhead        =    40 tokens
                              ─────────────────
Total per caption             ≈ 1,314 tokens
```

### Step 2: Output tokens required per batch (25 captions)

```
25 captions × 1,314 tokens/caption = 32,850 tokens required
Configured max_tokens              =  4,096 tokens available
                                     ─────────────────────
Shortfall                          = 28,754 tokens (87.5% missing)
```

The model is asked to produce ~32,850 output tokens but is hard-capped at 4,096. It can only generate ~12.5% of the required JSON before being cut off.

### Step 3: Input tokens per API call

The system prompt contains the full feature schema and extraction rules:

```
182 feature names (JSON serialized) ≈ 1,500 tokens
Extraction rules & guidelines        ≈   800 tokens
Category summary                     ≈   200 tokens
                                     ─────────────
System prompt                        ≈ 2,500 tokens
```

The user prompt contains 25 numbered captions:

```
25 captions × avg 272.5 chars = 6,812 chars ≈ 2,044 tokens
Numbering/formatting overhead               ≈   200 tokens
                                            ─────────────
User prompt                                 ≈ 2,244 tokens
```

> Note: The original estimate used "avg 200 chars" but the actual average
> measured from the CSV is 272.5 chars (median 235). This increases the
> input token cost by ~32% vs the original estimate.

Total input per call:

```
System prompt  = 2,500 tokens
User prompt    = 2,244 tokens
               ─────────────
Total input    ≈ 4,744 tokens
```

### Step 4: Minimum API calls without retries

```
182,062 captions ÷ 25 captions/batch = 7,283 API calls (minimum)
```

### Step 5: Actual API calls with retries

Because 87.5% of responses are truncated mid-JSON, `json.loads()` fails and triggers retries (up to `MAX_RETRIES = 5`). Conservative estimate:

| Scenario                  | Rate | Calls | Subtotal    |
| ------------------------- | ---- | ----- | ----------- |
| First attempt             | 100% | 7,283 | 7,283       |
| Retry 1 (truncated)       | 70%  | 5,098 | 5,098       |
| Retry 2 (still truncated) | 50%  | 3,642 | 3,642       |
| Retry 3                   | 30%  | 2,185 | 2,185       |
| Retry 4                   | 20%  | 1,457 | 1,457       |
| Retry 5 (final fallback)  | 10%  | 728   | 728         |
| **Total**                 |      |       | **~20,393** |

Conservative estimate: **~20,000 actual API calls**.

### Step 6: Tokens consumed (broken configuration)

```
Input tokens:   20,000 calls × 4,744 tokens  =  94,880,000 tokens
Output tokens:  20,000 calls × 4,096 tokens  =  81,920,000 tokens
                                               ────────────────────
Total tokens consumed                       ≈ 176,800,000 tokens
```

Cost at gpt-4o-mini rates ($0.15/M input, $0.60/M output):

```
Input cost:   94.9M × $0.15/M  = $14.23
Output cost:  81.9M × $0.60/M  = $49.15
                                ─────────
Raw API cost                   ≈ $63.38
```

However, only ~12.5% of each output is valid JSON. The remaining 87.5% is wasted truncated text that gets discarded on retry. Effective cost per useful token is ~8x higher than normal.

If sync fallback kicks in (no Batch API 50% discount) for failed batches, costs double for those calls.

---

## Root Cause Summary

```
                     REQUIRED          AVAILABLE        RATIO
Output tokens:       ~32,850           4,096            0.125x (12.5%)
                     per batch         max_tokens

Result: ~87.5% of every response is truncated → parse failure → retry loop → token waste
```

The `max_tokens=4096` setting was reasonable when the schema had ~30-50 features, but at 182 features it produces outputs 8x larger than the cap allows.

---

## The Fix

### Change 1: Increase `max_tokens`

**File:** `phase2_bulk_extraction.py`, lines 189 and 405

```python
# FROM:
"max_tokens": 4096,

# TO:
"max_tokens": 16384,
```

This must be changed in two places:

1. `_openai_extraction_chat_body()` (line 189) — batch API body
2. `call_llm()` (line 405) — sync fallback caller

### Change 2: Reduce default batch size

**File:** `phase2_bulk_extraction.py`, line 156

```python
# FROM:
LLM_BATCH_SIZE = _safe_int_env("LLM_BATCH_SIZE", 25, vmin=1)

# TO:
LLM_BATCH_SIZE = _safe_int_env("LLM_BATCH_SIZE", 4, vmin=1, vmax=8)
```

Safe batch size at 182 features with `max_tokens=16384`:

```
16,384 × 0.80 (safety margin) = 13,107 usable tokens
13,107 ÷ 1,314 tokens/caption  = 9.9 captions max
With margin                     = 8 captions max (safe)
Recommended                     = 4 captions (conservative)
```

### Change 3: Add automatic batch size validation

**File:** `phase2_bulk_extraction.py`, add after line 908 (after loading schema):

```python
def _estimate_output_tokens_per_caption(n_features: int) -> int:
    """Estimate output tokens for one caption's feature JSON."""
    return n_features * 7 + 40

def _safe_batch_size_for_output(
    n_features: int,
    max_output_tokens: int = 16384,
    safety_margin: float = 0.8,
) -> int:
    """Calculate max captions per batch that fits within max_tokens."""
    per_caption = _estimate_output_tokens_per_caption(n_features)
    usable = int(max_output_tokens * safety_margin)
    return max(1, usable // per_caption)

# In run_extraction(), after line 908:
safe_batch = _safe_batch_size_for_output(len(all_feature_names))
if LLM_BATCH_SIZE > safe_batch:
    print(f"  WARNING: LLM_BATCH_SIZE={LLM_BATCH_SIZE} exceeds safe limit of {safe_batch} "
          f"for {len(all_feature_names)} features. Reducing to {safe_batch}.")
    LLM_BATCH_SIZE = safe_batch
```

---

## Expected Token Usage After Fix

With `max_tokens=16384` and `LLM_BATCH_SIZE=4`:

### Per API call

```
Input tokens:   ~3,030 tokens (system prompt + 4 captions)
Output tokens:  ~5,456 tokens (4 captions × 1,314 + overhead)
Total:          ~8,486 tokens
```

### Total API calls

```
182,062 captions ÷ 4 captions/batch = 45,516 API calls
+ ~2% retry rate                      = ~46,426 API calls
```

### Total tokens (fixed configuration)

```
Input tokens:   46,426 × 3,030  =  140,670,780 tokens
Output tokens:  46,426 × 5,456  =  253,300,256 tokens
                                 ────────────────────
Total                           ≈ 393,971,036 tokens
```

### Cost comparison

| Configuration                   | Input Tokens | Output Tokens | Input Cost | Output Cost | Total Cost   | Waste Rate |
| ------------------------------- | ------------ | ------------- | ---------- | ----------- | ------------ | ---------- |
| **Broken** (batch=25, max=4096) | 95M          | 82M           | $14.23     | $49.15      | **~$63.38**  | ~87.5%     |
| **Fixed** (batch=4, max=16384)  | 141M         | 253M          | $21.10     | $151.98     | **~$173.08** | ~0%        |

The fixed version costs more in raw dollars because it produces 8x more **valid** output, but every token is productive — zero waste. The broken version appears cheaper on paper but delivers almost nothing useful.

With Batch API 50% discount (if `OPENAI_USE_BATCH=1`):

```
Fixed cost with Batch API: $173.08 × 0.50 = ~$86.54
```

With prompt caching (reuses system prefix across calls), input cost drops further:

```
Cached input: ~60% of 141M = 85M cached at $0.075/M = $6.33
Uncached input: 40% of 141M = 56M at $0.15/M = $8.44
Total input cost with caching: ~$14.77 (vs $21.10 uncached)
```

### Best case cost (Batch API + prompt caching)

```
Input:  $14.77
Output: $151.98 × 0.50 = $75.99
                         ─────────
Total:                   ≈ $90.76
```

---

## Quick Reference: Token Estimation Formula

For any dataset with `N` captions and `F` features:

```
tokens_per_caption  = F × 7 + 40
safe_batch_size     = floor(max_tokens × 0.8 / tokens_per_caption)
min_api_calls       = ceil(N / safe_batch_size)
tokens_per_call     = system_prompt_tokens + user_prompt_tokens + output_tokens
                    ≈ 2,500 + (batch_size × 273) + (batch_size × tokens_per_caption)
total_tokens        = min_api_calls × tokens_per_call × retry_multiplier
```

Where `retry_multiplier` is:

- ~1.02 with the fix (2% failure rate)
- ~2.8 with the broken config (70% truncation rate)
