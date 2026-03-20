"""
Shared SCIN ↔ canonical feature mappings for phase1, phase3, and phase3b.

- SCIN_TO_CANONICAL: SCIN case columns that map to extracted feature columns
  (textures, body parts, symptoms). Used for Derm-1M vs SCIN comparison and
  SCIN feature matrix construction in phase3/3b.

- SCIN_SCHEMA_FEATURES: Full alignment used in phase1 schema (includes
  demographics, race, duration, related_category) so the LLM schema matches
  SCIN questionnaire fields.
"""

# Subset for phase3 / phase3b — material findings overlap with caption extraction
SCIN_TO_CANONICAL = {
    "textures_raised_or_bumpy": "texture_raised",
    "textures_flat": "texture_flat",
    "textures_rough_or_flaky": "texture_rough_flaky",
    "textures_fluid_filled": "texture_fluid_filled",
    "condition_symptoms_itching": "symptom_itching",
    "condition_symptoms_burning": "symptom_burning",
    "condition_symptoms_pain": "symptom_pain",
    "condition_symptoms_bleeding": "symptom_bleeding",
    "condition_symptoms_increasing_size": "symptom_increasing_size",
    "condition_symptoms_darkening": "symptom_darkening",
    "condition_symptoms_bothersome_appearance": "symptom_bothersome_appearance",
    "other_symptoms_fever": "symptom_fever",
    "other_symptoms_chills": "symptom_chills",
    "other_symptoms_fatigue": "symptom_fatigue",
    "other_symptoms_joint_pain": "symptom_joint_pain",
    "other_symptoms_mouth_sores": "symptom_mouth_sores",
    "other_symptoms_shortness_of_breath": "symptom_shortness_of_breath",
    "body_parts_head_or_neck": "location_head_neck",
    "body_parts_arm": "location_arm",
    "body_parts_palm": "location_palm",
    "body_parts_back_of_hand": "location_back_of_hand",
    "body_parts_torso_front": "location_torso_front",
    "body_parts_torso_back": "location_torso_back",
    "body_parts_genitalia_or_groin": "location_genitalia_groin",
    "body_parts_buttocks": "location_buttocks",
    "body_parts_leg": "location_leg",
    "body_parts_foot_top_or_side": "location_foot_top_side",
    "body_parts_foot_sole": "location_foot_sole",
}

# Phase 1 schema alignment — superset including demographics & metadata
SCIN_SCHEMA_FEATURES = {
    "age_group": "age_group",
    "sex_at_birth": "sex",
    "fitzpatrick_skin_type": "fitzpatrick_skin_type",
    "race_ethnicity_american_indian_or_alaska_native": "race_american_indian_alaska_native",
    "race_ethnicity_asian": "race_asian",
    "race_ethnicity_black_or_african_american": "race_black_african_american",
    "race_ethnicity_hispanic_latino_or_spanish_origin": "race_hispanic_latino",
    "race_ethnicity_middle_eastern_or_north_african": "race_middle_eastern_north_african",
    "race_ethnicity_native_hawaiian_or_pacific_islander": "race_native_hawaiian_pacific_islander",
    "race_ethnicity_white": "race_white",
    "race_ethnicity_other_race": "race_other",
    "race_ethnicity_prefer_not_to_answer": "race_prefer_not_to_answer",
    "race_ethnicity_two_or_more_after_mitigation": "race_two_or_more",
    "textures_raised_or_bumpy": "texture_raised",
    "textures_flat": "texture_flat",
    "textures_rough_or_flaky": "texture_rough_flaky",
    "textures_fluid_filled": "texture_fluid_filled",
    "body_parts_head_or_neck": "location_head_neck",
    "body_parts_arm": "location_arm",
    "body_parts_palm": "location_palm",
    "body_parts_back_of_hand": "location_back_of_hand",
    "body_parts_torso_front": "location_torso_front",
    "body_parts_torso_back": "location_torso_back",
    "body_parts_genitalia_or_groin": "location_genitalia_groin",
    "body_parts_buttocks": "location_buttocks",
    "body_parts_leg": "location_leg",
    "body_parts_foot_top_or_side": "location_foot_top_side",
    "body_parts_foot_sole": "location_foot_sole",
    "body_parts_other": "location_other",
    "condition_symptoms_bothersome_appearance": "symptom_bothersome_appearance",
    "condition_symptoms_bleeding": "symptom_bleeding",
    "condition_symptoms_increasing_size": "symptom_increasing_size",
    "condition_symptoms_darkening": "symptom_darkening",
    "condition_symptoms_itching": "symptom_itching",
    "condition_symptoms_burning": "symptom_burning",
    "condition_symptoms_pain": "symptom_pain",
    "condition_symptoms_no_relevant_experience": "symptom_no_relevant_experience",
    "other_symptoms_fever": "symptom_fever",
    "other_symptoms_chills": "symptom_chills",
    "other_symptoms_fatigue": "symptom_fatigue",
    "other_symptoms_joint_pain": "symptom_joint_pain",
    "other_symptoms_mouth_sores": "symptom_mouth_sores",
    "other_symptoms_shortness_of_breath": "symptom_shortness_of_breath",
    "other_symptoms_no_relevant_symptoms": "symptom_no_relevant_symptoms",
    "condition_duration": "duration",
    "related_category": "related_category",
}

# Canonical names that appear in SCIN_TO_CANONICAL (for “has SCIN column?” flags)
SCIN_COMPARABLE_CANONICALS = frozenset(SCIN_TO_CANONICAL.values())
