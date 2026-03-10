"""
PHASE 3: Feature Analysis
==========================
1. Overall feature coverage (how often each feature is present vs unknown in Derm-1M)
2. Derm-1M vs SCIN feature availability comparison
3. Disease class-wise feature importance (chi-square + mutual information)
4. Generates questionnaire priority ranking per disease cluster

Run after phase2_bulk_extraction.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
DERM_FEATURES_CSV = "derm1m_features.csv"
SCIN_CSV          = "SCIN-dataset/dataset_scin_cases.csv"
SCHEMA_PATH       = "feature_schema.json"
OUTPUT_DIR        = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# SCIN column → our canonical feature name (matches phase2 output columns)
# Updated to match the new schema from phase1/phase2
SCIN_TO_CANONICAL = {
    # Textures
    "textures_raised_or_bumpy":                     "texture_raised",
    "textures_flat":                                "texture_flat",
    "textures_rough_or_flaky":                      "texture_rough_flaky",
    "textures_fluid_filled":                        "texture_fluid_filled",
    # Condition symptoms
    "condition_symptoms_itching":                   "symptom_itching",
    "condition_symptoms_burning":                   "symptom_burning",
    "condition_symptoms_pain":                      "symptom_pain",
    "condition_symptoms_bleeding":                  "symptom_bleeding",
    "condition_symptoms_increasing_size":            "symptom_increasing_size",
    "condition_symptoms_darkening":                 "symptom_darkening",
    "condition_symptoms_bothersome_appearance":      "symptom_bothersome_appearance",
    # Other symptoms
    "other_symptoms_fever":                         "symptom_fever",
    "other_symptoms_chills":                        "symptom_chills",
    "other_symptoms_fatigue":                       "symptom_fatigue",
    "other_symptoms_joint_pain":                    "symptom_joint_pain",
    "other_symptoms_mouth_sores":                   "symptom_mouth_sores",
    "other_symptoms_shortness_of_breath":            "symptom_shortness_of_breath",
    # Body parts → canonical location names from phase2
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
}

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1: Feature Coverage in Derm-1M
# ──────────────────────────────────────────────────────────────────────────────
def analyse_feature_coverage(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """For each feature: % present, % absent, % unknown."""
    rows = []
    for col in feature_cols:
        vc = df[col].value_counts()
        n = len(df)
        rows.append({
            "feature":       col,
            "n_present":     vc.get(1, 0),
            "n_absent":      vc.get(0, 0),
            "n_unknown":     vc.get(2, 0),
            "pct_present":   round(100 * vc.get(1, 0) / n, 2),
            "pct_absent":    round(100 * vc.get(0, 0) / n, 2),
            "pct_unknown":   round(100 * vc.get(2, 0) / n, 2),
            "informativeness": round(100 * (vc.get(1, 0) + vc.get(0, 0)) / n, 2),
        })
    return pd.DataFrame(rows).sort_values("informativeness", ascending=False)

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: Derm-1M vs SCIN Feature Availability Comparison
# ──────────────────────────────────────────────────────────────────────────────
def compare_derm_vs_scin(
    derm_df: pd.DataFrame,
    scin_df: pd.DataFrame,
    feature_cols: list[str]
) -> pd.DataFrame:
    """
    For each overlapping feature:
    - Derm-1M: % of rows where feature is actually mentioned (non-unknown)
    - SCIN: % of rows where column is filled (non-null, non-empty)
    """
    rows = []
    for scin_col, canonical in SCIN_TO_CANONICAL.items():
        if canonical not in feature_cols or scin_col not in scin_df.columns:
            continue

        # Derm-1M informativeness (1 or 0, not 2)
        derm_info = (derm_df[canonical] != 2).mean() * 100

        # SCIN availability (YES present or non-null)
        s_col = scin_df[scin_col]
        if s_col.dtype == object:
            scin_avail = (s_col.notna() & (s_col.str.upper() == "YES")).mean() * 100
        else:
            scin_avail = s_col.notna().mean() * 100

        rows.append({
            "canonical_feature": canonical,
            "scin_column":       scin_col,
            "derm1m_pct_informed": round(derm_info, 2),
            "scin_pct_available":  round(scin_avail, 2),
            "gap":                 round(derm_info - scin_avail, 2),
        })

    cmp_df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    return cmp_df

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: Disease Class-wise Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
def compute_feature_importance(
    df: pd.DataFrame, feature_cols: list[str], label_col: str = "label_name"
) -> pd.DataFrame:
    """
    Mutual information between each feature and disease label.
    Only uses rows where feature is 0 or 1 (not unknown=2).
    Uses label_name (the cleaned/merged labels used by the model).
    """
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))

    mi_scores = []
    for col in feature_cols:
        mask = df[col] != 2      # exclude unknowns
        x_sub = df.loc[mask, col].values.reshape(-1, 1)
        y_sub = y[mask.values]

        if len(x_sub) < 100:     # too few informed rows
            mi = 0.0
        else:
            mi = mutual_info_classif(x_sub, y_sub, discrete_features=True)[0]

        mi_scores.append({
            "feature": col,
            "mutual_information": round(mi, 5),
            "n_informed": int(mask.sum()),
        })

    return pd.DataFrame(mi_scores).sort_values("mutual_information", ascending=False)

def compute_classwise_importance(
    df: pd.DataFrame, feature_cols: list[str],
    label_col: str = "label_name", top_features: int = 20
) -> dict[str, pd.DataFrame]:
    """
    For each disease class (label_name), compute feature prevalence
    vs rest (odds ratio).
    """
    results = {}
    labels = df[label_col].unique()

    for label in labels:
        in_class = df[df[label_col] == label]
        out_class = df[df[label_col] != label]
        rows = []
        for col in feature_cols:
            in_pct = (in_class[col] == 1).sum() / max(len(in_class), 1)
            out_pct = (out_class[col] == 1).sum() / max(len(out_class), 1)
            # Odds ratio with Laplace smoothing
            or_val = (
                ((in_pct + 1e-6) / (1 - in_pct + 1e-6))
                / ((out_pct + 1e-6) / (1 - out_pct + 1e-6))
            )
            rows.append({
                "feature": col,
                "in_class_pct": round(in_pct * 100, 2),
                "out_class_pct": round(out_pct * 100, 2),
                "odds_ratio": round(or_val, 3),
            })
        results[label] = (
            pd.DataFrame(rows)
            .sort_values("odds_ratio", ascending=False)
            .head(top_features)
        )
    return results

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4: Questionnaire Priority
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_TO_QUESTION: dict[str, str] = {
    "symptom_itching":          "Does the rash itch?",
    "symptom_burning":          "Does the affected area feel like it is burning or stinging?",
    "symptom_pain":             "Is the area painful or tender to touch?",
    "symptom_bleeding":         "Has the lesion bled?",
    "symptom_fever":            "Have you had a fever recently?",
    "symptom_joint_pain":       "Have you experienced any joint pain or swelling?",
    "symptom_fatigue":          "Have you felt unusually tired or fatigued?",
    "symptom_mouth_sores":      "Do you have any sores or ulcers in your mouth?",
    "symptom_increasing_size":  "Has the lesion been growing or spreading?",
    "symptom_darkening":        "Has the area been getting darker over time?",
    "symptom_shortness_of_breath": "Have you experienced any shortness of breath?",
    "symptom_chills":           "Have you experienced chills or rigors?",
    "symptom_bothersome_appearance": "Is the appearance of the lesion bothersome to you?",
    "texture_raised":           "Is the lesion raised or bumpy above the skin surface?",
    "texture_flat":             "Is the lesion flat (not raised)?",
    "texture_rough_flaky":      "Is the area rough, scaly, or flaky?",
    "texture_fluid_filled":     "Are there any blisters or fluid-filled bumps?",
    "texture_ulcerated":        "Is there any open wound, ulcer, or crusting?",
    "onset_sudden":             "Did the rash appear suddenly (within hours or overnight)?",
    "trigger_identified":       "Did anything trigger or worsen the rash (e.g. sun, medication, food)?",
    "family_history":           "Does anyone in your family have a similar skin condition?",
    "recurrence":               "Have you had this before?",
    "immunocompromised":        "Do you have any condition that affects your immune system?",
    "associated_disease":       "Do you have any other diagnosed medical conditions?",
    "location_widespread":      "Is the rash affecting multiple areas of the body?",
    "location_face":            "Is the rash on your face?",
    "location_trunk":           "Is the rash on your chest, abdomen, or back?",
    "location_genitalia_groin": "Is the rash in the genital or groin area?",
    "treatment_mentioned":      "Have you tried any treatment for this condition?",
    "trigger_type":             "What triggered or worsened the rash?",
}

def generate_questionnaire_for_cluster(
    disease_cluster: list[str],
    classwise_importance: dict[str, pd.DataFrame],
    top_n: int = 10
) -> list[dict]:
    """
    Given a cluster of diseases that the model confuses,
    rank questions by their discriminatory power across the cluster.
    """
    # Aggregate feature importance across cluster members
    feature_scores: dict[str, list[float]] = {}
    for disease in disease_cluster:
        if disease not in classwise_importance:
            continue
        for _, row in classwise_importance[disease].iterrows():
            feat = row["feature"]
            if feat not in feature_scores:
                feature_scores[feat] = []
            feature_scores[feat].append(row["odds_ratio"])

    # Score = mean odds ratio across cluster members
    ranked = sorted(
        [(feat, np.mean(scores)) for feat, scores in feature_scores.items()],
        key=lambda x: -x[1]
    )[:top_n]

    questionnaire = []
    for feat, score in ranked:
        q = FEATURE_TO_QUESTION.get(
            feat,
            f"Does the patient show signs of {feat.replace('_', ' ')}?"
        )
        questionnaire.append({
            "feature": feat,
            "discriminatory_score": round(score, 3),
            "question": q,
        })

    return questionnaire

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
def plot_coverage(coverage_df: pd.DataFrame, top_n: int = 40):
    top = coverage_df.head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(top))
    ax.bar(x, top["pct_present"], label="Present (1)", color="#2196F3")
    ax.bar(x, top["pct_absent"], bottom=top["pct_present"],
           label="Absent (0)", color="#90CAF9")
    ax.bar(x, top["pct_unknown"],
           bottom=top["pct_present"] + top["pct_absent"],
           label="Unknown (2)", color="#E0E0E0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(top["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of records")
    ax.set_title(f"Feature Coverage in Derm-1M (top {top_n} most informative)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_coverage.png", dpi=150)
    plt.close()

def plot_derm_vs_scin_comparison(cmp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(cmp_df))
    ax.barh(list(y_pos), cmp_df["derm1m_pct_informed"],
            label="Derm-1M informed%", color="#1976D2")
    ax.barh(list(y_pos), cmp_df["scin_pct_available"],
            label="SCIN available%", color="#EF5350", alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cmp_df["canonical_feature"], fontsize=9)
    ax.set_xlabel("% of records with feature information")
    ax.set_title("Feature Informativeness: Derm-1M vs SCIN")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "derm_vs_scin_comparison.png", dpi=150)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Phase 3: Feature Analysis ===\n")

    # Load data
    derm_df = pd.read_csv(DERM_FEATURES_CSV)
    scin_df = pd.read_csv(SCIN_CSV)

    # Meta columns to exclude from feature analysis
    META_COLS = {"image", "label_name", "disease_label", "age_numeric",
                 "age_group", "duration_bucket", "diagnosis_confidence",
                 "lesion_count"}
    feature_cols = [
        c for c in derm_df.columns
        if c not in META_COLS
        and derm_df[c].isin([0, 1, 2]).mean() > 0.8  # binary/ternary cols
    ]

    # 1. Coverage
    print("Computing feature coverage...")
    coverage = analyse_feature_coverage(derm_df, feature_cols)
    coverage.to_csv(OUTPUT_DIR / "feature_coverage.csv", index=False)
    plot_coverage(coverage)
    print(f"  Saved feature_coverage.csv and .png\n")

    # 2. Derm-1M vs SCIN comparison
    print("Comparing Derm-1M vs SCIN feature availability...")
    cmp_df = compare_derm_vs_scin(derm_df, scin_df, feature_cols)
    cmp_df.to_csv(OUTPUT_DIR / "derm_vs_scin_comparison.csv", index=False)
    plot_derm_vs_scin_comparison(cmp_df)
    derm_mean = cmp_df["derm1m_pct_informed"].mean()
    scin_mean = cmp_df["scin_pct_available"].mean()
    print(f"  Derm-1M avg feature informativeness: {derm_mean:.1f}%")
    print(f"  SCIN avg feature availability:       {scin_mean:.1f}%")
    print(f"  Gap (Derm-1M advantage):             {derm_mean - scin_mean:.1f}%\n")

    # 3. Global feature importance (uses label_name = cleaned model labels)
    print("Computing global mutual information feature importance...")
    mi_df = compute_feature_importance(derm_df, feature_cols, label_col="label_name")
    mi_df.to_csv(OUTPUT_DIR / "feature_importance_global.csv", index=False)
    print(f"  Top 10 features:\n{mi_df.head(10).to_string(index=False)}\n")

    # 4. Class-wise importance (per label_name)
    print("Computing class-wise feature importance...")
    cw_importance = compute_classwise_importance(
        derm_df, feature_cols, label_col="label_name"
    )
    # Save one CSV per disease
    cw_dir = OUTPUT_DIR / "classwise_importance"
    cw_dir.mkdir(exist_ok=True)
    for disease, df_imp in cw_importance.items():
        safe_name = disease.replace("/", "_").replace(" ", "_")[:60]
        df_imp.to_csv(cw_dir / f"{safe_name}.csv", index=False)
    print(f"  Saved {len(cw_importance)} class-wise importance files\n")

    # 5. Example questionnaire for a confusion cluster
    example_cluster = [
        "eczema", "dermatitis", "psoriasis",
    ]
    questionnaire = generate_questionnaire_for_cluster(
        example_cluster, cw_importance
    )
    q_path = OUTPUT_DIR / "questionnaire_eczema_cluster.json"
    with open(q_path, "w") as f:
        json.dump(
            {"cluster": example_cluster, "questions": questionnaire},
            f, indent=2,
        )
    print(f"Example questionnaire for cluster {example_cluster}:")
    for item in questionnaire:
        print(f"  [{item['discriminatory_score']:.3f}] {item['question']}")

    # 6. Features only in Derm-1M (not available in SCIN)
    scin_canonical_set = set(SCIN_TO_CANONICAL.values())
    derm_only_features = [
        c for c in feature_cols if c not in scin_canonical_set
    ]
    derm_only_coverage = coverage[coverage["feature"].isin(derm_only_features)]
    derm_only_coverage.to_csv(
        OUTPUT_DIR / "derm_only_features.csv", index=False
    )
    print(f"\n  Features in Derm-1M but NOT in SCIN: {len(derm_only_features)}")
    print(f"  Saved derm_only_features.csv")
    if len(derm_only_coverage) > 0:
        top_derm_only = derm_only_coverage.head(10)
        print(f"  Top 10 Derm-1M-only features by informativeness:")
        for _, row in top_derm_only.iterrows():
            print(f"    {row['feature']:40s} "
                  f"present={row['pct_present']:.1f}% "
                  f"informed={row['informativeness']:.1f}%")

    print(f"\n  All outputs saved to {OUTPUT_DIR}/")
