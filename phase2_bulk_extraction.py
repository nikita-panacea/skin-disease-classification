"""
PHASE 2: Bulk Feature Extraction (Hybrid NLP + LLM)
=====================================================
Strategy:
  - FAST LANE: Rule-based regex/NLP for structured features (demographics,
    body location, colours, textures, duration) — handles regex-extractable
    features from the schema discovered in Phase 1.
  - SLOW LANE: Gemini 2.0 Flash Thinking batch calls for semantically complex
    features (triggers, treatments, clinical signs, morphology details, etc.)
    Organized per label_name with checkpointing.
  - Encoding: 0 = explicitly absent, 1 = present, 2 = unknown/not mentioned.

Prerequisites:
  pip install pandas numpy tqdm google-generativeai python-dotenv

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
import google.generativeai as genai

# ── Load API key from .env ───────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    raise ValueError("Set GEMINI_API_KEY in your .env file")
genai.configure(api_key=GEMINI_API_KEY)

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH         = "cleaned_caption_Derm1M.csv"
SCHEMA_PATH      = "feature_schema.json"
OUTPUT_CSV       = "derm1m_features.csv"
CHECKPOINT_DIR   = Path("checkpoints")
LLM_BATCH_SIZE   = 30              # captions per API call for LLM lane
MODEL_NAME       = "gemini-2.0-flash-thinking-exp"
MAX_CAPTION_LEN  = 500             # truncate before sending to LLM (chars)

CHECKPOINT_DIR.mkdir(exist_ok=True)
model = genai.GenerativeModel(MODEL_NAME)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_schema(schema_path: str) -> dict:
    """Load the feature schema produced by Phase 1."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema.get("feature_categories", [])
    print(f"  Loaded schema: {len(features)} features")
    return schema


def split_features_by_lane(schema: dict) -> tuple[list[dict], list[dict]]:
    """
    Partition features into rule-based (fast lane) and LLM (slow lane)
    based on the 'regex_extractable' flag set during Phase 1.
    """
    regex_feats = []
    llm_feats = []
    for feat in schema.get("feature_categories", []):
        if feat.get("regex_extractable", False):
            regex_feats.append(feat)
        else:
            llm_feats.append(feat)
    return regex_feats, llm_feats


# ══════════════════════════════════════════════════════════════════════════════
# FAST LANE: RULE-BASED EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════
# These cover well-defined, keyword-matchable features. They remain hardcoded
# because regex patterns are reliable for these categories and much faster
# than LLM calls. The schema guides which ones to include in the output.

# ── Demographics ──
AGE_RE = re.compile(
    r"\b(\d{1,3})[- ]?(?:year|yr)s?[- ]?old\b|\bage[d]?\s*:?\s*(\d{1,3})\b", re.I
)
SEX_RE = re.compile(
    r"\b(male|female|man|woman|boy|girl|he\b|she\b|his\b|her\b)\b", re.I
)

def extract_age(text: str) -> tuple:
    """Returns (age_numeric, age_group_bucket)."""
    m = AGE_RE.search(text)
    if not m:
        return None, "unknown"
    age = int(m.group(1) or m.group(2))
    if age < 13:   return age, "child"
    if age < 18:   return age, "adolescent"
    if age < 30:   return age, "18_29"
    if age < 45:   return age, "30_44"
    if age < 60:   return age, "45_59"
    if age < 75:   return age, "60_74"
    return age, "75_plus"

def extract_sex(text: str) -> int:
    """1=male, 0=female, 2=unknown."""
    m = SEX_RE.search(text)
    if not m:
        return 2
    w = m.group(1).lower()
    if w in ("male", "man", "boy", "he", "his"):
        return 1
    if w in ("female", "woman", "girl", "she", "her"):
        return 0
    return 2

# ── Fitzpatrick ──
def extract_fitzpatrick(text: str) -> int:
    m = re.search(r"fitzpatrick\s*(type|skin)?\s*(\d)", text, re.I)
    if m:
        return int(m.group(2))
    lower = text.lower()
    if re.search(r"fair|type [i1]\b|very light",     lower): return 1
    if re.search(r"light|type [ii2]\b",              lower): return 2
    if re.search(r"medium|olive|type [iii3]\b",      lower): return 3
    if re.search(r"brown|type [iv4]\b",              lower): return 4
    if re.search(r"dark brown|type [v5]\b",          lower): return 5
    if re.search(r"deeply pigmented|type [vi6]\b",   lower): return 6
    return 0  # 0 = unknown

# ── Body location ──
LOCATION_PATTERNS: dict[str, list[str]] = {
    "location_face":        ["face", "cheek", "forehead", "nose", "chin",
                             "perioral", "periorbital", "eyelid", "lip"],
    "location_scalp":       ["scalp", "hair", "hairline"],
    "location_head_neck":   ["head", "neck", "cervical"],
    "location_trunk":       ["trunk", "torso", "chest", "abdomen", "back", "flank"],
    "location_torso_front": ["chest", "abdomen", "torso front", "anterior trunk"],
    "location_torso_back":  ["back", "posterior trunk", "dorsal"],
    "location_arm":         ["arm", "forearm", "elbow", "upper arm", "antecubital"],
    "location_hand":        ["hand", "finger", "knuckle", "wrist", "dorsum of hand"],
    "location_palm":        ["palm", "palmar"],
    "location_back_of_hand":["back of hand", "dorsum of hand"],
    "location_leg":         ["leg", "thigh", "shin", "calf", "popliteal", "knee"],
    "location_foot":        ["foot", "feet", "toe", "ankle"],
    "location_foot_top_side":["dorsum of foot", "top of foot"],
    "location_foot_sole":   ["sole", "plantar", "heel"],
    "location_genitalia_groin": ["genital", "groin", "inguinal", "scrotal", "vulva",
                                 "perineal", "perianal"],
    "location_buttocks":    ["buttock", "gluteal"],
    "location_mouth":       ["mouth", "oral", "mucosa", "tongue", "buccal", "gum"],
    "location_nail":        ["nail", "ungual", "periungual", "subungual"],
    "location_axilla":      ["axilla", "axillary", "armpit"],
    "location_widespread":  ["widespread", "generalised", "generalized",
                             "whole body", "diffuse", "all over"],
}

def extract_locations(text: str) -> dict[str, int]:
    lower = text.lower()
    return {
        feat: (1 if any(kw in lower for kw in kws) else 2)
        for feat, kws in LOCATION_PATTERNS.items()
    }

# ── Texture / morphology ──
TEXTURE_PATTERNS: dict[str, list[str]] = {
    "texture_raised":       ["raised", "elevated", "papule", "nodule", "plaque",
                             "bump", "boil", "wart", "verruca", "papular"],
    "texture_flat":         ["flat", "macular", "macule", "patch"],
    "texture_rough_flaky":  ["rough", "scaly", "scale", "flak", "desquam",
                             "keratotic", "hyperkeratotic"],
    "texture_fluid_filled": ["vesicle", "blister", "bulla", "pustule",
                             "fluid", "weeping", "oozing"],
    "texture_ulcerated":    ["ulcer", "ulcerat", "erosion", "erode",
                             "crusted", "crust", "excoriat"],
    "texture_smooth":       ["smooth", "shiny", "glossy"],
}

COLOR_PATTERNS: dict[str, list[str]] = {
    "color_red":        ["red", "erythema", "erythematous", "pink",
                         "violaceous", "purpura", "petechiae"],
    "color_brown":      ["brown", "hyperpigment", "pigment", "tan", "bronze"],
    "color_white":      ["white", "hypopigment", "depigment", "pale", "leucoder"],
    "color_yellow":     ["yellow", "xanth"],
    "color_black":      ["black", "dark", "melanotic", "melanin"],
    "color_blue_grey":  ["blue", "grey", "gray", "slate"],
}

DISTRIBUTION_PATTERNS: dict[str, list[str]] = {
    "distribution_unilateral":  ["unilateral", "one side", "left ", "right "],
    "distribution_bilateral":   ["bilateral", "both sides", "symmetric"],
    "distribution_dermatomal":  ["dermatome", "dermatomal"],
    "distribution_grouped":     ["grouped", "cluster", "herpetiform"],
    "distribution_linear":      ["linear", "streak", "along"],
    "distribution_annular":     ["annular", "ring", "ring-like", "circular"],
}

def extract_morphology(text: str) -> dict[str, int]:
    lower = text.lower()
    result = {}
    all_patterns = {
        **TEXTURE_PATTERNS, **COLOR_PATTERNS, **DISTRIBUTION_PATTERNS
    }
    for feat, kws in all_patterns.items():
        result[feat] = 1 if any(kw in lower for kw in kws) else 2
    return result

# ── Symptoms ──
SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "symptom_itching":          ["itch", "pruritic", "pruritus"],
    "symptom_burning":          ["burn", "burning", "stinging"],
    "symptom_pain":             ["pain", "painful", "tender", "sore", "hurt"],
    "symptom_bleeding":         ["bleed", "hemorrhage", "haemorrhage"],
    "symptom_increasing_size":  ["growing", "enlarg", "increas", "spreading", "expand"],
    "symptom_darkening":        ["darken", "pigment increas", "getting darker"],
    "symptom_bothersome_appearance": ["bothersome", "unsightly", "cosmetic concern",
                                      "disfigur"],
    "symptom_fever":            ["fever", "febrile", "pyrexia"],
    "symptom_chills":           ["chill", "rigor"],
    "symptom_fatigue":          ["fatigue", "tired", "malaise", "lethargy"],
    "symptom_joint_pain":       ["joint pain", "arthralgia", "arthrit"],
    "symptom_mouth_sores":      ["mouth sore", "oral ulcer", "aphth", "mucosal lesion"],
    "symptom_shortness_of_breath": ["shortness of breath", "dyspnea", "dyspnoea",
                                    "breathing difficult"],
    "symptom_nail_change":      ["nail change", "onychol", "nail dystrophy",
                                 "nail pitting"],
    "symptom_hair_loss":        ["hair loss", "alopecia", "bald"],
}

def extract_symptoms(text: str) -> dict[str, int]:
    lower = text.lower()
    return {
        feat: (1 if any(kw in lower for kw in kws) else 2)
        for feat, kws in SYMPTOM_PATTERNS.items()
    }

# ── Duration ──
DURATION_MAP = [
    (r"\b(\d+)\s*hour",                        "hours"),
    (r"\b(\d+)\s*day",                          "days"),
    (r"\b(\d+)\s*week",                         "weeks"),
    (r"\b(\d+)\s*month",                        "months"),
    (r"\b(\d+)\s*year",                         "years"),
    (r"since childhood|lifelong|congenital|birth", "lifelong"),
    (r"chronic|long.stand",                     "chronic"),
    (r"acute|sudden|abrupt",                    "acute"),
]

def extract_duration(text: str) -> str:
    lower = text.lower()
    for pat, bucket in DURATION_MAP:
        if re.search(pat, lower):
            return bucket
    return "unknown"

# ── Onset ──
def extract_onset(text: str) -> int:
    """1=sudden, 0=gradual, 2=unknown."""
    lower = text.lower()
    if re.search(r"sudden|abrupt|acute onset|overnight|appeared\s+(yesterday|today)",
                 lower):
        return 1
    if re.search(r"gradual|slowly|progressive|over (time|weeks|months|years)", lower):
        return 0
    return 2

# ── Diagnosis confidence ──
def extract_diagnosis_confidence(text: str) -> str:
    lower = text.lower()
    if re.search(r"confirmed|biopsy.proven|histolog|patholog|laboratory", lower):
        return "confirmed"
    if re.search(r"clinically diagnosed|consistent with|impression", lower):
        return "clinical"
    if re.search(r"self.assumed|self.diagnos|possible|likely|suspected|probable",
                 lower):
        return "suspected"
    if re.search(r"no definitive|uncertain|unclear|differential", lower):
        return "uncertain"
    return "unknown"

# ── Lesion count / extent ──
def extract_lesion_count(text: str) -> str:
    lower = text.lower()
    if re.search(r"\bsingle\b|solitary|one lesion|isolated", lower):
        return "single"
    if re.search(r"\bfew\b|several|multiple|numerous|many", lower):
        return "multiple"
    if re.search(r"diffuse|widespread|generali[sz]ed", lower):
        return "widespread"
    return "unknown"


def extract_row_rules(caption: str) -> dict:
    """Fast-lane: all rule-based features from one caption."""
    age_num, age_bucket = extract_age(caption)
    row = {
        "age_numeric":           age_num,
        "age_group":             age_bucket,
        "sex":                   extract_sex(caption),
        "fitzpatrick_skin_type": extract_fitzpatrick(caption),
        "duration_bucket":       extract_duration(caption),
        "onset_sudden":          extract_onset(caption),
        "diagnosis_confidence":  extract_diagnosis_confidence(caption),
        "lesion_count":          extract_lesion_count(caption),
    }
    row.update(extract_locations(caption))
    row.update(extract_morphology(caption))
    row.update(extract_symptoms(caption))
    return row


# ══════════════════════════════════════════════════════════════════════════════
# SLOW LANE: LLM EXTRACTOR (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_feature_list(llm_feats: list[dict]) -> list[str]:
    """Get the list of LLM feature names from the schema."""
    return [f["name"] for f in llm_feats]


def build_llm_system_prompt(llm_feature_names: list[str]) -> str:
    """Build the extraction prompt based on discovered schema features."""
    feature_list_str = json.dumps(llm_feature_names, indent=2)
    return f"""You are a clinical dermatology NLP specialist.
Extract the following features from each skin disease caption.
Return ONLY a valid JSON array — one object per caption — with these exact keys:
{feature_list_str}

ENCODING RULES:
- For binary features (present/absent):
  1 = feature is present or mentioned in the caption
  0 = feature is explicitly stated as absent (e.g. "non-itchy", "no fever")
  2 = no information about this feature in the caption
- For categorical features (multiple possible values):
  Use the most appropriate value string, or "unknown" if not mentioned.

IMPORTANT:
- Extract ONLY what is ACTUALLY STATED. Do not infer or guess.
- If a caption mentions a trigger (e.g. "caused by sun exposure"), set
  trigger_identified=1 and trigger_type to the appropriate value.
- If a caption mentions treatment (e.g. "treated with topical steroids"),
  set treatment_mentioned=1 and treatment_type to the appropriate value.
- If a caption mentions clinical signs (e.g. "Nikolsky sign positive"),
  set the corresponding sign feature to 1.

No preamble, no markdown fences, pure JSON array only.
"""


def extract_llm_features(
    captions: list[str],
    llm_feature_names: list[str],
    system_prompt: str,
    retries: int = 3,
) -> list[dict]:
    """Batch LLM call. Returns list of dicts aligned to input captions."""
    truncated = [c[:MAX_CAPTION_LEN] for c in captions]
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(truncated))

    full_prompt = system_prompt + "\n\n" + numbered

    for attempt in range(retries):
        try:
            resp = model.generate_content(full_prompt)
            text = resp.text.strip()
            text = re.sub(r"```json\s*|```\s*", "", text).strip()
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == len(captions):
                return parsed
            elif isinstance(parsed, list) and len(parsed) > 0:
                # LLM returned wrong count — pad or truncate
                if len(parsed) < len(captions):
                    fallback = {k: 2 for k in llm_feature_names}
                    parsed.extend([fallback] * (len(captions) - len(parsed)))
                return parsed[:len(captions)]
        except json.JSONDecodeError:
            print(f"    JSON parse error (attempt {attempt+1}), retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            err_str = str(e)
            print(f"    LLM attempt {attempt+1} failed: {err_str[:120]}")
            # If rate limited, wait longer
            if "429" in err_str or "quota" in err_str.lower():
                time.sleep(10 * (attempt + 1))
            else:
                time.sleep(2 ** attempt)

    # Fallback: all unknown
    return [{k: 2 for k in llm_feature_names} for _ in captions]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(csv_path: str, schema_path: str, use_llm: bool = True):
    print("=" * 60)
    print("  PHASE 2: Bulk Feature Extraction (Hybrid)")
    print("=" * 60 + "\n")

    # ── Load schema ───────────────────────────────────────────────────────────
    print("Loading feature schema...")
    schema = load_schema(schema_path)
    regex_feats, llm_feats = split_features_by_lane(schema)
    llm_feature_names = build_llm_feature_list(llm_feats)
    print(f"  Regex-extractable: {len(regex_feats)} features")
    print(f"  LLM-extractable:   {len(llm_feats)} features\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df["truncated_caption"] = df["truncated_caption"].fillna("")
    n = len(df)
    print(f"  {n:,} records loaded\n")

    # ── Fast lane: rule-based ─────────────────────────────────────────────────
    print("Running rule-based extraction (fast lane)...")
    rule_results = (
        df["truncated_caption"]
        .apply(extract_row_rules)
        .apply(pd.Series)
    )
    print(f"  Done. {len(rule_results.columns)} rule-based features extracted.\n")

    # ── Slow lane: LLM batching per label_name ────────────────────────────────
    if use_llm and llm_feature_names:
        print(f"Running LLM extraction (slow lane, batches of {LLM_BATCH_SIZE})...")
        print(f"  Features to extract via LLM: {len(llm_feature_names)}\n")

        system_prompt = build_llm_system_prompt(llm_feature_names)

        # Pre-allocate LLM results array aligned to df index
        llm_results_list = [None] * n

        # Process per label_name for checkpointing and label-aware context
        label_names = df["label_name"].unique()
        total_batches = 0

        for label_name in tqdm(label_names, desc="Processing labels"):
            safe_name = re.sub(r'[^\w\-]', '_', str(label_name))[:60]
            ckpt_path = CHECKPOINT_DIR / f"llm_{safe_name}.json"

            # Get indices for this label
            label_mask = df["label_name"] == label_name
            label_indices = df.index[label_mask].tolist()
            label_captions = df.loc[label_indices, "truncated_caption"].tolist()

            if ckpt_path.exists():
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    label_llm = json.load(f)
                if len(label_llm) == len(label_indices):
                    for idx, result in zip(label_indices, label_llm):
                        llm_results_list[idx] = result
                    continue
                # else: checkpoint is stale/incomplete, re-extract

            label_llm = []
            for b_start in range(0, len(label_captions), LLM_BATCH_SIZE):
                batch = label_captions[b_start:b_start + LLM_BATCH_SIZE]
                batch_results = extract_llm_features(
                    batch, llm_feature_names, system_prompt
                )
                label_llm.extend(batch_results)
                total_batches += 1
                # Rate limiting
                time.sleep(0.5)

            # Save checkpoint
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(label_llm, f)

            # Assign results to the correct indices
            for idx, result in zip(label_indices, label_llm):
                llm_results_list[idx] = result

        # Fill any remaining None entries with unknown
        fallback = {k: 2 for k in llm_feature_names}
        for i in range(n):
            if llm_results_list[i] is None:
                llm_results_list[i] = fallback

        llm_df = pd.DataFrame(llm_results_list)
        print(f"\n  Done. {len(llm_df.columns)} LLM features extracted "
              f"({total_batches} API calls).\n")
    else:
        print("Skipping LLM extraction (use_llm=False or no LLM features).\n")
        llm_df = pd.DataFrame(
            [{k: 2 for k in llm_feature_names} for _ in range(n)]
        )

    # ── Combine ───────────────────────────────────────────────────────────────
    print("Combining results...")
    meta_cols = df[["image", "label_name", "disease_label"]].reset_index(drop=True)
    final_df = pd.concat(
        [meta_cols,
         rule_results.reset_index(drop=True),
         llm_df.reset_index(drop=True)],
        axis=1,
    )

    # Fill NaN → 2 (unknown) for binary/ternary feature columns
    NON_BINARY_COLS = {
        "image", "label_name", "disease_label",
        "age_numeric", "age_group", "duration_bucket",
        "diagnosis_confidence", "lesion_count",
    }
    # Also exclude any categorical LLM features (not binary)
    categorical_llm = {
        f["name"] for f in llm_feats if not f.get("is_binary", True)
    }
    non_binary = NON_BINARY_COLS | categorical_llm

    binary_cols = [
        c for c in final_df.columns if c not in non_binary
    ]
    for col in binary_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
            final_df[col] = final_df[col].fillna(2).astype(int)

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Feature matrix saved: {OUTPUT_CSV}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Total features: {final_df.shape[1] - 3}")  # minus meta cols
    return final_df


if __name__ == "__main__":
    # Set use_llm=False for a fast rule-only run first (useful for debugging)
    result_df = run_extraction(CSV_PATH, SCHEMA_PATH, use_llm=True)

    print("\nFeature value counts (sample):")
    # Show first 10 feature columns (skip meta)
    feat_cols = [c for c in result_df.columns
                 if c not in ("image", "label_name", "disease_label")]
    for col in feat_cols[:15]:
        vc = result_df[col].value_counts().to_dict()
        print(f"  {col}: {vc}")
