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

# ── Load API keys from .env ───────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── LLM Provider Selection ─────────────────────────────────────────────────
# Options: "gemini", "openai", or "qwen"
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "openai")  # Default to openai for cost efficiency

# Qwen local server config
QWEN_BASE_URL   = os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3.5-9B")

# Model configuration
if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Set GEMINI_API_KEY in your .env file for Gemini provider")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-2.0-flash-thinking-exp"
    model = genai.GenerativeModel(MODEL_NAME)
elif LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        raise ValueError("Set OPENAI_API_KEY in your .env file for OpenAI provider")
    import openai
    openai.api_key = OPENAI_API_KEY
    MODEL_NAME = "gpt-4o-mini"
    model = None
elif LLM_PROVIDER == "qwen":
    from openai import OpenAI as QwenClient
    qwen_client = QwenClient(base_url=QWEN_BASE_URL, api_key="EMPTY")
    MODEL_NAME = QWEN_MODEL_NAME
    model = None
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'gemini', 'openai', or 'qwen'")

print(f"Using LLM Provider: {LLM_PROVIDER} with model: {MODEL_NAME}")
if LLM_PROVIDER == "qwen":
    print(f"  Qwen base URL: {QWEN_BASE_URL}")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH         = "cleaned_caption_Derm1M.csv"
SCHEMA_PATH      = "feature_schema.json"
OUTPUT_CSV       = "derm1m_features.csv"
STATS_FILE       = "extraction_stats.json"
CHECKPOINT_DIR   = Path("checkpoints")
LLM_BATCH_SIZE   = 25              # captions per API call
MAX_CAPTION_LEN  = 500             # truncate before sending to LLM (chars)
MAX_RETRIES      = 5               # max retries per batch
RATE_LIMIT_SLEEP = 0.1 if LLM_PROVIDER == "qwen" else 0.5

CHECKPOINT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LLM API Wrapper Functions
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, system_prompt: str = None, retries: int = MAX_RETRIES) -> str:
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
                
                response = openai.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
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
    """Load the feature schema produced by Phase 1."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema.get("feature_categories", [])
    print(f"  Loaded schema: {len(features)} features")
    
    # Show feature breakdown by category
    categories = defaultdict(int)
    for feat in features:
        cat = feat.get("category", "other")
        categories[cat] += 1
    
    print(f"  Feature breakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count}")
    
    return schema


def get_all_feature_names(schema: dict) -> list[str]:
    """Get list of all feature names from the schema."""
    return [f["name"] for f in schema.get("feature_categories", [])]


def get_feature_categories(schema: dict) -> dict[str, str]:
    """Get mapping of feature name to category."""
    return {f["name"]: f.get("category", "other") 
            for f in schema.get("feature_categories", [])}


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
        
        # Validate values are 0, 1, or 2
        for feat in expected_features:
            val = result.get(feat, 2)
            if val not in [0, 1, 2]:
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
    """
    truncated = [c[:MAX_CAPTION_LEN] for c in captions]
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(truncated))
    
    full_prompt = system_prompt + "\n\n" + numbered
    
    stats = {
        "attempts": 0,
        "success": False,
        "validation_fixes": 0,
    }
    
    for attempt in range(retries):
        try:
            stats["attempts"] = attempt + 1
            text = call_llm(full_prompt, retries=1)  # Handle retries ourselves
            text = text.strip()
            
            # Strip markdown fences if present
            text = re.sub(r"```json\s*|```\s*", "", text).strip()
            
            # Parse JSON
            parsed = json.loads(text)
            
            if not isinstance(parsed, list):
                print(f"    Warning: LLM returned {type(parsed).__name__}, expected list")
                if isinstance(parsed, dict) and "feature_categories" in parsed:
                    # Sometimes LLM returns wrong structure
                    parsed = [{} for _ in range(len(captions))]
                else:
                    raise ValueError(f"Expected list, got {type(parsed).__name__}")
            
            # Validate and fix results
            valid_results, num_fixed = validate_batch_results(
                parsed, all_feature_names, len(captions)
            )
            stats["validation_fixes"] = num_fixed
            
            # Check count mismatch
            if len(valid_results) != len(captions):
                print(f"    Warning: Expected {len(captions)} results, got {len(valid_results)}")
                # Pad or truncate
                if len(valid_results) < len(captions):
                    fallback = {k: 2 for k in all_feature_names}
                    valid_results.extend([fallback] * (len(captions) - len(valid_results)))
                valid_results = valid_results[:len(captions)]
            
            stats["success"] = True
            return valid_results, stats
            
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
    
    # All retries failed - return unknown for all
    print(f"    ERROR: All {retries} attempts failed. Returning unknown values.")
    fallback = [{k: 2 for k in all_feature_names} for _ in captions]
    return fallback, stats


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
    df["truncated_caption"] = df["truncated_caption"].fillna("")
    n = len(df)
    print(f"  {n:,} records loaded\n")

    # ── LLM extraction for ALL features ──────────────────────────────────────────
    print(f"Running LLM extraction for ALL {len(all_feature_names)} features...")
    print(f"  Provider: {LLM_PROVIDER} | Model: {MODEL_NAME}")
    print(f"  This will process all {n:,} captions in batches of {LLM_BATCH_SIZE}\n")

    system_prompt = build_llm_system_prompt(all_feature_names, feature_categories)

    # Pre-allocate results array aligned to df index
    all_results_list = [None] * n
    
    # Statistics tracking
    global_stats = {
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
    }

    # Process per label_name for checkpointing
    label_names = df["label_name"].unique()
    
    for label_idx, label_name in enumerate(tqdm(label_names, desc="Processing labels"), 1):
        safe_name = re.sub(r'[^\w\-]', '_', str(label_name))[:60]
        ckpt_path = CHECKPOINT_DIR / f"llm_{safe_name}.json"

        # Get indices for this label
        label_mask = df["label_name"] == label_name
        label_indices = df.index[label_mask].tolist()
        label_captions = df.loc[label_indices, "truncated_caption"].tolist()
        
        n_label = len(label_captions)
        n_batches_label = (n_label + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE

        # Check for existing checkpoint
        if ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    label_results = json.load(f)
                if len(label_results) == len(label_indices):
                    print(f"  [{label_idx}/{len(label_names)}] {label_name}: "
                          f"Loaded {len(label_results):,} results from checkpoint")
                    for idx, result in zip(label_indices, label_results):
                        all_results_list[idx] = result
                    global_stats["labels_skipped"].append(label_name)
                    continue
                else:
                    print(f"  [{label_idx}/{len(label_names)}] {label_name}: "
                          f"Checkpoint incomplete ({len(label_results)}/{len(label_indices)}), re-extracting...")
            except Exception as e:
                print(f"  [{label_idx}/{len(label_names)}] {label_name}: "
                      f"Error loading checkpoint: {e}, re-extracting...")

        # Extract features for this label
        print(f"  [{label_idx}/{len(label_names)}] {label_name}: "
              f"Processing {n_label:,} captions in {n_batches_label} batches...")
        
        label_results = []
        label_stats = {
            "batches": 0,
            "retries": 0,
            "validation_fixes": 0,
        }
        
        batch_pbar = tqdm(range(0, len(label_captions), LLM_BATCH_SIZE), 
                         desc=f"  Batches", leave=False, total=n_batches_label)
        
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
            
            # Update progress bar description
            batch_pbar.set_postfix({
                "retries": label_stats["retries"],
                "fixes": label_stats["validation_fixes"]
            })
            
            # Rate limiting
            time.sleep(RATE_LIMIT_SLEEP)
        
        # Save checkpoint for this label
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(label_results, f)
        
        # Assign results to the correct indices
        for idx, result in zip(label_indices, label_results):
            all_results_list[idx] = result
        
        global_stats["labels_processed"].append({
            "label": label_name,
            "captions": n_label,
            "batches": label_stats["batches"],
            "retries": label_stats["retries"],
            "fixes": label_stats["validation_fixes"],
        })

    # Fill any remaining None entries with unknown
    fallback = {k: 2 for k in all_feature_names}
    none_count = sum(1 for r in all_results_list if r is None)
    if none_count > 0:
        print(f"\n  Warning: {none_count:,} records have no results, filling with unknown (2)")
        for i in range(n):
            if all_results_list[i] is None:
                all_results_list[i] = fallback

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
