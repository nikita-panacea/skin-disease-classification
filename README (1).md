# Derm-1M Feature Extraction Pipeline

## Overview

Three-phase pipeline to extract all features from the `caption` column of the Derm-1M dataset (413K records, 300MB CSV), encode them as 0/1/2, and compare against SCIN.

---

## Feature Encoding
| Value | Meaning |
|-------|---------|
| **1** | Feature is present / mentioned |
| **0** | Feature is explicitly absent |
| **2** | No information available (unknown / not mentioned) |

---

## Strategy: Why Hybrid (NLP Rules + LLM)?

| Approach | Speed | Accuracy | Cost | Best for |
|----------|-------|----------|------|----------|
| Pure regex/NLP | ✅ Very fast | ⚠️ Misses semantics | Free | Demographics, location, colour |
| Pure LLM | ❌ 50h+ for 413K rows | ✅ Best | Expensive | Complex reasoning |
| **Hybrid** ✅ | ✅ Fast | ✅ Good | Low | **Full pipeline** |

The hybrid approach runs **rule-based extractors** for ~25 structured features at near-instant speed, then runs **LLM batch calls** (50 captions/call) for ~14 semantically complex features. LLM processing of 413K rows at 50/call = ~8,260 API calls (≈4–6 hours, resumable via checkpoints).

---

## Files

```
phase1_feature_discovery.py   # Discover feature schema via LLM sampling
phase2_bulk_extraction.py     # Extract features from all 413K captions
phase3_analysis.py            # Feature importance, SCIN comparison, questionnaire
```

---

## Phase 1: Feature Schema Discovery

**Goal:** Before extracting features at scale, discover the complete feature taxonomy.

**Method:**
1. **Stratified sample** 1,000 captions across all 50 disease classes
2. Send in batches of 20 to Claude Opus for feature category discovery
3. **Consolidate** discovered features (remove synonyms/duplicates)
4. **Align** with SCIN columns to flag which features are comparable

**Output:** `feature_schema.json` — canonical feature list with SCIN mappings

**Run:**
```bash
pip install anthropic pandas
python phase1_feature_discovery.py
```

---

## Phase 2: Bulk Feature Extraction

### Fast Lane (Rule-Based) — ~5 minutes for 413K rows
Handles structured, explicit information using regex patterns:

| Feature Group | Features | Method |
|--------------|----------|--------|
| Demographics | age_numeric, age_group, sex, fitzpatrick | Regex |
| Body location | 12 anatomical regions | Keyword matching |
| Morphology | texture (7), color (6), distribution (6) | Keyword matching |
| Symptoms | 13 clinical symptoms | Keyword matching |
| Duration | hours/days/weeks/months/years/lifelong | Regex |
| Onset | sudden vs gradual | Regex |
| Diagnosis confidence | confirmed/clinical/suspected/uncertain | Regex |
| Lesion count | single/multiple/widespread | Regex |

### Slow Lane (LLM) — ~4–6 hours for 413K rows
Handles semantically complex features requiring reasoning:

| Feature | Description |
|---------|-------------|
| systemic_involvement | Fever + rash + joint → systemic disease |
| trigger_identified | Sun, drug, stress, contact allergen |
| trigger_type | Type of identified trigger |
| contagious_risk | Infectious risk level |
| chronic_vs_acute | Disease temporality |
| treatment_mentioned | Topical / systemic / surgical |
| recurrence | First episode vs recurrent |
| family_history | Genetic component |
| immunocompromised | Host immune status |
| associated_disease | Comorbidities |
| lesion_border | Well-defined vs ill-defined |
| lesion_shape | Round / oval / irregular / annular |
| lesion_size | Small (<1cm) / medium / large (>5cm) |
| secondary_change | Lichenification / excoriation / PIH |

**Checkpointing:** Saves every 5,000 rows to `checkpoints/` so crashes don't lose progress.

**Run:**
```bash
pip install pandas spacy tqdm anthropic
python -m spacy download en_core_web_sm
# Fast-only test (no API cost):
# Set use_llm=False in __main__ first
python phase2_bulk_extraction.py
```

**Output:** `derm1m_features.csv` — 413K rows × ~55 features

---

## Phase 3: Analysis

### 3.1 Feature Coverage Analysis
For each feature: % present / absent / unknown in Derm-1M

**Key insight:** Features with high "unknown%" indicate the dataset
is missing that information → potential overfitting risk.

### 3.2 Derm-1M vs SCIN Comparison
Maps overlapping features and computes:
- Derm-1M: % of records where feature is mentioned
- SCIN: % of records where SCIN column is filled

**Interpretation:**
- **Large gap (Derm-1M >> SCIN):** Derm-1M contains richer features
  → Classifier likely relies on features not available in SCIN
  → This explains the performance drop (52% vs 84%)
- **Small gap:** Both datasets comparably rich
  → Model may be overfitting

### 3.3 Global Feature Importance
Uses **Mutual Information** between each feature and the 50-class disease label.
Higher MI = feature is more discriminative across all diseases.

### 3.4 Class-wise Feature Importance
For each of 50 disease classes:
- Features with high **odds ratio** (present in class >> present in other classes)
- Used to identify the key distinguishing features per condition

### 3.5 Questionnaire Generation
For a given disease confusion cluster (e.g. eczema/psoriasis/contact dermatitis),
ranks questions by their discriminatory power.

**Run:**
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
python phase3_analysis.py
```

**Outputs:**
- `analysis_outputs/feature_coverage.csv` + `.png`
- `analysis_outputs/derm_vs_scin_comparison.csv` + `.png`
- `analysis_outputs/feature_importance_global.csv`
- `analysis_outputs/classwise_importance/<disease>.csv` (50 files)
- `analysis_outputs/questionnaire_<cluster>.json`

---

## Expected Feature Schema (after Phase 1)

### Demographics (4)
age_group, age_numeric, sex, fitzpatrick_skin_type

### Body Location (12)
location_face, location_scalp, location_neck, location_trunk, location_arm,
location_hand, location_leg, location_foot, location_genitalia, location_buttocks,
location_mouth, location_widespread

### Morphology — Texture (7)
texture_raised, texture_flat, texture_rough_flaky, texture_fluid_filled,
texture_ulcerated, texture_smooth, texture_scarring

### Morphology — Colour (6)
color_red, color_brown, color_white, color_yellow, color_black, color_blue_grey

### Morphology — Distribution (6)
distribution_unilateral, distribution_bilateral, distribution_dermatomal,
distribution_grouped, distribution_linear, distribution_annular

### Morphology — LLM (3)
lesion_border, lesion_shape, lesion_size

### Symptoms — Dermatological (7)
symptom_itching, symptom_burning, symptom_pain, symptom_bleeding,
symptom_increasing_size, symptom_darkening, symptom_nail_change, symptom_hair_loss

### Symptoms — Systemic (6)
symptom_fever, symptom_chills, symptom_fatigue, symptom_joint_pain,
symptom_mouth_sores, symptom_shortness_of_breath

### History / Context (7)
onset_sudden, duration_bucket, diagnosis_confidence, lesion_count,
recurrence, family_history, immunocompromised

### Triggers / Associations (3)
trigger_identified, trigger_type, associated_disease

### Clinical Complexity (4)
systemic_involvement, contagious_risk, chronic_vs_acute, treatment_mentioned

**Total: ~65 features**
**SCIN-comparable: ~21 features**

---

## Interpreting the Derm-1M vs SCIN Gap

| Gap Size | Interpretation | Action |
|----------|---------------|--------|
| > 30% avg gap | Derm-1M is significantly richer | Classifier relies on Derm-1M-specific features → need richer SCIN input or retrain on less-informative data |
| 10–30% gap | Moderate informativeness difference | Add clinical questionnaire to supplement SCIN at inference |
| < 10% gap | Datasets comparably rich | Model is overfitting → retrain with stronger regularisation |
