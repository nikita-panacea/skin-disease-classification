"""
PHASE 1: Feature Schema Discovery (Bottom-Up)
===============================================
Discover ALL possible clinical features directly FROM the Derm-1M captions,
then consolidate into a canonical feature schema aligned with SCIN.

Strategy:
  1. Per-label-name stratified sampling (auto-calculated from label distribution)
  2. Per-label-name LLM discovery via Gemini 2.0 Flash Thinking
  3. Global consolidation to deduplicate synonyms
  4. SCIN column alignment

Run BEFORE phase2_bulk_extraction.py

Prerequisites:
  pip install pandas google-generativeai python-dotenv tqdm
"""

import pandas as pd
import json
import re
import os
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai

# ── Load API key from .env ───────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    raise ValueError("Set GEMINI_API_KEY in your .env file")
genai.configure(api_key=GEMINI_API_KEY)

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH        = "cleaned_caption_Derm1M.csv"
SCHEMA_OUT      = "feature_schema.json"
DISCOVERY_DIR   = Path("discovery_outputs")
DISCOVERY_DIR.mkdir(exist_ok=True)

BATCH_SIZE      = 20                    # captions per LLM call
DISCOVERY_MODEL = "gemini-2.5-flash-lite"

model = genai.GenerativeModel(DISCOVERY_MODEL)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Per-label stratified sampling
# ──────────────────────────────────────────────────────────────────────────────

# Sampling tiers: label_name class size → number of captions to sample
SAMPLING_TIERS = [
    (5000,  200),   # large classes (>5000 records): 200 captions
    (1000,  150),   # medium classes (1000-5000): 150 captions
    (200,    100),   # small classes (200-1000): 100 captions
    (0,    None),   # tiny classes (<200): take ALL
]

def compute_sample_size(class_count: int) -> int:
    """Determine how many captions to sample from a class based on its size."""
    for threshold, n_sample in SAMPLING_TIERS:
        if class_count > threshold:
            return n_sample if n_sample is not None else class_count
    return class_count  # fallback: take all

def load_and_sample(csv_path: str) -> pd.DataFrame:
    """
    Load the Derm-1M CSV and create a per-label-name stratified sample.
    Within each label_name, further stratify by disease_label to capture
    sub-type diversity (e.g., 'dermatitis' → allergic/atopic/contact/etc.).
    """
    print("Loading full CSV...")
    df = pd.read_csv(csv_path)
    df["truncated_caption"] = df["truncated_caption"].fillna("")
    # Drop rows with empty captions
    df = df[df["truncated_caption"].str.strip().ne("")].copy()
    print(f"  {len(df):,} records with non-empty captions")
    print(f"  {df['label_name'].nunique()} unique label_names")
    print(f"  {df['disease_label'].nunique()} unique disease_labels\n")

    label_counts = df["label_name"].value_counts()
    sampled_parts = []

    for label_name, total_count in label_counts.items():
        n_sample = compute_sample_size(total_count)
        n_sample = min(n_sample, total_count)

        subset = df[df["label_name"] == label_name]
        n_sublabels = subset["disease_label"].nunique()

        if n_sublabels > 1 and n_sample < total_count:
            # Stratified within disease_label sub-types
            per_sub = max(1, n_sample // n_sublabels)
            stratified = (
                subset.groupby("disease_label", group_keys=False)
                      .apply(lambda g: g.sample(min(len(g), per_sub), random_state=42))
            )
            # If we got more than needed, downsample; if fewer, that's fine
            if len(stratified) > n_sample:
                stratified = stratified.sample(n_sample, random_state=42)
            sampled_parts.append(stratified)
        else:
            sampled_parts.append(subset.sample(n_sample, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    print(f"  Sampled {len(sampled):,} captions across {sampled['label_name'].nunique()} label_names")

    # Print tier breakdown
    tier_counts = {"large(>5K)": 0, "medium(1K-5K)": 0, "small(200-1K)": 0, "tiny(<200)": 0}
    for label_name, cnt in label_counts.items():
        if cnt > 5000:   tier_counts["large(>5K)"] += 1
        elif cnt > 1000: tier_counts["medium(1K-5K)"] += 1
        elif cnt > 200:  tier_counts["small(200-1K)"] += 1
        else:            tier_counts["tiny(<200)"] += 1
    print(f"  Tier breakdown: {tier_counts}\n")

    return sampled

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: LLM-based feature discovery per label_name
# ──────────────────────────────────────────────────────────────────────────────

DISCOVERY_PROMPT = """You are a clinical NLP expert specialising in dermatology.

Given the following batch of skin disease case captions (all belonging to the
disease category "{label_name}"), extract ALL possible clinical and contextual
feature *categories* that are ACTUALLY MENTIONED or DESCRIBED in these captions.

IMPORTANT RULES:
- Extract ONLY information that is ACTUALLY STATED in these captions.
  Do NOT infer or assume features that are not mentioned.
- Separate compound features (e.g. 'red scaly lesion' → color=red, texture=scaly)
- Each feature should be MEDICALLY meaningful and distinguishable

You MUST look for and extract features in ALL of these categories if present:

1. DEMOGRAPHICS: age, sex/gender, ethnicity, fitzpatrick skin type
2. MORPHOLOGY - TEXTURE: raised/bumpy, flat, rough/scaly/flaky, fluid-filled,
   ulcerated, smooth, crusted, papule, plaque, vesicle, bulla, pustule, nodule,
   macule, patch, wheal, comedo, cyst, abscess, scar, atrophy
3. MORPHOLOGY - COLOR: red/erythematous, brown/hyperpigmented, white/depigmented,
   yellow, black/dark, blue/grey, pink, violaceous/purple, salmon-colored, skin-colored
4. MORPHOLOGY - SHAPE/BORDER/SIZE: round, oval, irregular, annular, linear, serpiginous,
   well-defined, ill-defined, small, medium, large, size measurements
5. MORPHOLOGY - DISTRIBUTION: unilateral, bilateral, dermatomal, grouped/clustered,
   linear, annular, widespread/generalized, localized, symmetric, asymmetric
6. BODY LOCATION: face, scalp, neck, trunk/torso, arm, hand, palm, leg, foot, sole,
   genitalia, groin, buttocks, mouth/oral, nails, axilla, intertriginous areas
7. SYMPTOMS - DERMATOLOGICAL: itching, burning, pain, bleeding, increasing size,
   darkening, tenderness, numbness, tingling
8. SYMPTOMS - SYSTEMIC: fever, chills, fatigue, joint pain, mouth sores,
   shortness of breath, weight loss, malaise, lymphadenopathy
9. DURATION/ONSET: acute, chronic, sudden onset, gradual onset, recurrent,
   congenital/lifelong, specific time periods (hours/days/weeks/months/years)
10. TRIGGERS: sun exposure, drugs/medications, food, allergens/contact, stress,
    trauma, infection, heat/cold, chemicals, insect bites, pregnancy, menstruation
11. TREATMENTS: topical steroids, antibiotics (topical/oral), antifungals,
    antihistamines, immunosuppressants, phototherapy, surgery, cryotherapy,
    laser therapy, biologics, retinoids, emollients/moisturizers, home remedies,
    chemotherapy, radiation
12. CLINICAL SIGNS: Nikolsky sign, Auspitz sign, Koebner phenomenon,
    dermoscopic patterns, Wickham striae, target lesions, pathergy
13. HISTORY/CONTEXT: family history, recurrence, immunocompromised status,
    associated diseases/comorbidities, contagious risk, diagnosis confidence,
    biopsy/histology mentioned, self-diagnosed vs clinician-diagnosed
14. LESION COUNT/EXTENT: single, multiple, few, numerous, widespread
15. SECONDARY CHANGES: lichenification, excoriation, post-inflammatory changes,
    scarring, fissuring, maceration

Return ONLY valid JSON — no markdown fences, no prose — in this structure:
{{
  "feature_categories": [
    {{
      "name": "snake_case_feature_name",
      "category": "one of: demographics | morphology | body_location | symptoms | duration | triggers | treatments | clinical_signs | history | severity | other",
      "description": "brief clinical description of what this feature captures",
      "example_values": ["value1", "value2", "value3"],
      "is_binary": true or false,
      "extraction_examples": ["phrase from caption that indicated this feature"]
    }}
  ]
}}
"""

def discover_features_batch(
    captions: list[str], label_name: str, retries: int = 3
) -> list[dict]:
    """Send a batch of captions to Gemini and extract feature categories."""
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(captions))
    prompt = (
        DISCOVERY_PROMPT.format(label_name=label_name)
        + f"\n\nHere are {len(captions)} captions for '{label_name}':\n\n{numbered}"
    )

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Strip any accidental markdown fences
            text = re.sub(r"```json\s*|```\s*", "", text).strip()
            parsed = json.loads(text)
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


def discover_all_features(sample_df: pd.DataFrame) -> dict[str, dict]:
    """
    Run per-label-name discovery across all sampled captions.
    Returns a dict of {feature_name: feature_dict}.
    """
    all_features: dict[str, dict] = {}
    label_names = sample_df["label_name"].unique()

    for label_name in tqdm(label_names, desc="Discovering features per label"):
        label_captions = (
            sample_df[sample_df["label_name"] == label_name]["truncated_caption"]
            .tolist()
        )

        # Check for cached per-label discovery
        safe_name = re.sub(r'[^\w\-]', '_', str(label_name))[:60]
        cache_path = DISCOVERY_DIR / f"discovery_{safe_name}.json"

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                label_features = json.load(f)
            print(f"  [{label_name}] Loaded {len(label_features)} cached features")
        else:
            label_features = []
            for start in range(0, len(label_captions), BATCH_SIZE):
                batch = label_captions[start:start + BATCH_SIZE]
                feats = discover_features_batch(batch, label_name)
                label_features.extend(feats)
                # Rate limiting — be polite to the API
                time.sleep(1)

            # Cache per-label results
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(label_features, f, indent=2)
            print(f"  [{label_name}] Discovered {len(label_features)} features "
                  f"from {len(label_captions)} captions")

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

CONSOLIDATION_PROMPT = """You are a senior clinical NLP engineer building a canonical
feature schema for a skin disease classification system.

You are given a raw list of {n_features} clinical feature categories that were
extracted by an LLM from {n_captions_sampled} dermatology case captions across
{n_disease_classes} disease classes.

Your task:
1. DEDUPLICATE synonyms (e.g. 'lesion_colour' and 'color_of_lesion' → 'lesion_color';
   'itch' and 'pruritus' → 'symptom_itching')
2. MERGE overlapping features into canonical names
3. STANDARDISE names to snake_case
4. DO NOT REMOVE any uniques features that are useful for differential diagnosis
5. ENSURE the following categories are well-represented:
   - demographics, morphology (texture, color, shape/border, distribution),
   - body_location, symptoms (dermatological + systemic),
   - duration/onset, triggers, treatments, clinical_signs, history
6. For each feature, decide: is_binary (true = present/absent encoding)
   or categorical (false = needs value strings like "topical|systemic|surgical")

Return ONLY valid JSON — no markdown fences, no prose — with this structure:
{{
  "feature_categories": [
    {{
      "name": "snake_case_canonical_name",
      "category": "demographics | morphology | body_location | symptoms | duration | triggers | treatments | clinical_signs | history | severity | other",
      "description": "brief clinical description",
      "example_values": ["val1", "val2"],
      "is_binary": true/false,
      "regex_extractable": true/false
    }}
  ]
}}

Here is the raw feature list:
{raw_features}
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
    prompt = CONSOLIDATION_PROMPT.format(
        n_features=len(compact),
        n_captions_sampled=n_captions_sampled,
        n_disease_classes=n_disease_classes,
        raw_features=raw_json,
    )

    print(f"  Sending {len(compact)} raw features for consolidation...")

    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            text = re.sub(r"```json\s*|```\s*", "", text).strip()
            parsed = json.loads(text)
            if isinstance(parsed, list):
                parsed = {"feature_categories": parsed}
            if "feature_categories" in parsed:
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
# STEP 4: SCIN column alignment
# ──────────────────────────────────────────────────────────────────────────────

SCIN_FEATURES = {
    # SCIN column name → canonical feature name in our schema
    "textures_raised_or_bumpy":                     "texture_raised",
    "textures_flat":                                "texture_flat",
    "textures_rough_or_flaky":                      "texture_rough_flaky",
    "textures_fluid_filled":                        "texture_fluid_filled",
    "condition_symptoms_itching":                   "symptom_itching",
    "condition_symptoms_burning":                   "symptom_burning",
    "condition_symptoms_pain":                      "symptom_pain",
    "condition_symptoms_bleeding":                  "symptom_bleeding",
    "condition_symptoms_increasing_size":            "symptom_increasing_size",
    "condition_symptoms_darkening":                 "symptom_darkening",
    "condition_symptoms_bothersome_appearance":      "symptom_bothersome_appearance",
    "other_symptoms_fever":                         "symptom_fever",
    "other_symptoms_chills":                        "symptom_chills",
    "other_symptoms_fatigue":                       "symptom_fatigue",
    "other_symptoms_joint_pain":                    "symptom_joint_pain",
    "other_symptoms_mouth_sores":                   "symptom_mouth_sores",
    "other_symptoms_shortness_of_breath":            "symptom_shortness_of_breath",
    "body_parts_head_or_neck":                      "location_head_neck",
    "body_parts_arm":                               "location_arm",
    "body_parts_palm":                              "location_palm",
    "body_parts_back_of_hand":                      "location_back_of_hand",
    "body_parts_torso_front":                       "location_torso_front",
    "body_parts_torso_back":                        "location_torso_back",
    "body_parts_genitalia_or_groin":                "location_genitalia_groin",
    "body_parts_buttocks":                          "location_buttocks",
    "body_parts_leg":                               "location_leg",
    "body_parts_foot_top_or_side":                  "location_foot_top_side",
    "body_parts_foot_sole":                         "location_foot_sole",
    "age_group":                                    "age_group",
    "sex_at_birth":                                 "sex",
    "fitzpatrick_skin_type":                        "fitzpatrick_skin_type",
    "condition_duration":                           "duration",
}

def add_scin_alignment(schema: dict) -> dict:
    """Tag each schema feature with its SCIN column counterpart (if any)."""
    existing_names = {f["name"] for f in schema["feature_categories"]}

    for scin_col, canonical in SCIN_FEATURES.items():
        # Find feature in schema or add it as SCIN-sourced
        matched = next(
            (f for f in schema["feature_categories"] if f["name"] == canonical), None
        )
        if matched:
            matched["scin_column"] = scin_col
            matched["scin_comparable"] = True
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

    return schema


def _infer_category(name: str) -> str:
    if name.startswith("symptom_"):   return "symptoms"
    if name.startswith("location_"):  return "body_location"
    if name.startswith("texture_"):   return "morphology"
    if name.startswith("color_"):     return "morphology"
    if name.startswith("trigger_"):   return "triggers"
    if name.startswith("treatment_"): return "treatments"
    if name in ("age_group", "sex", "fitzpatrick_skin_type"): return "demographics"
    if name == "duration":            return "duration"
    return "other"


def _is_regex_extractable(name: str) -> bool:
    """Heuristic: features that can be reliably extracted via keyword/regex."""
    regex_prefixes = (
        "symptom_", "location_", "texture_", "color_",
        "distribution_",
    )
    regex_names = {
        "age_group", "sex", "fitzpatrick_skin_type", "duration",
        "onset_sudden", "lesion_count", "diagnosis_confidence",
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
    sample_df = load_and_sample(CSV_PATH)

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

    # ── Step 4: SCIN alignment ────────────────────────────────────────────────
    print("\nSTEP 4: Aligning with SCIN columns...\n")
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
        "n_captions_sampled": len(sample_df),
        "n_label_names": int(sample_df["label_name"].nunique()),
        "n_total_features": n_total,
        "n_scin_comparable": n_comparable,
        "n_regex_extractable": n_regex,
        "n_llm_only": n_llm_only,
        "model_used": DISCOVERY_MODEL,
    }

    with open(SCHEMA_OUT, "w", encoding="utf-8") as fp:
        json.dump(schema, fp, indent=2)

    print(f"  Schema saved to {SCHEMA_OUT}")
    print(f"  Next: run phase2_bulk_extraction.py")
