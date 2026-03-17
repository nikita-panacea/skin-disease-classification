"""
PHASE 3: Feature Analysis with Detailed EDA
===========================================
1. Overall feature coverage (how often each feature is present vs unknown in Derm-1M)
2. Derm-1M vs SCIN feature availability comparison
3. Disease class-wise feature importance (chi-square + mutual information)
4. Generates questionnaire priority ranking per disease cluster
5. DETAILED EDA: Feature distribution, disease-wise distribution, top features,
   disease-wise feature occurrence, correlation analysis

Run after phase2_bulk_extraction.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ── Config ────────────────────────────────────────────────────────────────────
DERM_FEATURES_CSV = "derm1m_features.csv"
SCIN_CSV          = "SCIN-dataset/dataset_scin_cases.csv"
SCHEMA_PATH       = "feature_schema.json"
OUTPUT_DIR        = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
EDA_DIR           = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(exist_ok=True)

# SCIN column → our canonical feature name (matches phase2 output columns)
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
# DETAILED EDA FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_distribution(coverage_df: pd.DataFrame, top_n: int = 50):
    """Plot distribution of feature informativeness across all features."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top N most informative features
    top = coverage_df.head(top_n)
    ax1 = axes[0, 0]
    x = range(len(top))
    ax1.bar(x, top["pct_present"], label="Present (1)", color="#2196F3")
    ax1.bar(x, top["pct_absent"], bottom=top["pct_present"],
           label="Absent (0)", color="#90CAF9")
    ax1.bar(x, top["pct_unknown"],
           bottom=top["pct_present"] + top["pct_absent"],
           label="Unknown (2)", color="#E0E0E0")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(top["feature"], rotation=45, ha="right", fontsize=6)
    ax1.set_ylabel("% of records")
    ax1.set_title(f"Top {top_n} Most Informative Features")
    ax1.legend()
    
    # 2. Informativeness histogram
    ax2 = axes[0, 1]
    ax2.hist(coverage_df["informativeness"], bins=30, color="#1976D2", edgecolor="white")
    ax2.set_xlabel("Informativeness (%)")
    ax2.set_ylabel("Number of features")
    ax2.set_title("Distribution of Feature Informativeness")
    ax2.axvline(coverage_df["informativeness"].mean(), color="red", linestyle="--",
               label=f"Mean: {coverage_df['informativeness'].mean():.1f}%")
    ax2.legend()
    
    # 3. Present vs Unknown scatter
    ax3 = axes[1, 0]
    ax3.scatter(coverage_df["pct_present"], coverage_df["pct_unknown"],
               alpha=0.6, c=coverage_df["informativeness"], cmap="viridis")
    ax3.set_xlabel("% Present")
    ax3.set_ylabel("% Unknown")
    ax3.set_title("Present vs Unknown by Feature")
    plt.colorbar(ax3.collections[0], ax=ax3, label="Informativeness")
    
    # 4. Feature category breakdown (if category info available)
    ax4 = axes[1, 1]
    # Extract category from feature name
    categories = []
    for feat in coverage_df["feature"]:
        if feat.startswith("symptom_"):
            categories.append("symptoms")
        elif feat.startswith("location_"):
            categories.append("location")
        elif feat.startswith("texture_") or feat.startswith("color_"):
            categories.append("morphology")
        elif feat.startswith("trigger_"):
            categories.append("triggers")
        elif feat.startswith("treatment_"):
            categories.append("treatments")
        else:
            categories.append("other")
    
    coverage_df_copy = coverage_df.copy()
    coverage_df_copy["category"] = categories
    cat_stats = coverage_df_copy.groupby("category")["informativeness"].mean().sort_values(ascending=False)
    cat_stats.plot(kind="bar", ax=ax4, color="#1565C0")
    ax4.set_xlabel("Feature Category")
    ax4.set_ylabel("Mean Informativeness (%)")
    ax4.set_title("Informativeness by Feature Category")
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(EDA_DIR / "feature_distribution_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature distribution overview to {EDA_DIR / 'feature_distribution_overview.png'}")


def plot_disease_wise_feature_heatmap(df: pd.DataFrame, feature_cols: list[str], 
                                       top_diseases: int = 20, top_features: int = 30):
    """Create heatmap of feature prevalence by disease."""
    # Get top diseases by count
    disease_counts = df["label_name"].value_counts().head(top_diseases)
    top_disease_names = disease_counts.index.tolist()
    
    # Get top features by overall informativeness
    feature_prev = {}
    for feat in feature_cols:
        feature_prev[feat] = (df[feat] == 1).mean()
    top_feature_names = sorted(feature_prev, key=feature_prev.get, reverse=True)[:top_features]
    
    # Create prevalence matrix
    prevalence_matrix = []
    for disease in top_disease_names:
        disease_df = df[df["label_name"] == disease]
        row = []
        for feat in top_feature_names:
            prev = (disease_df[feat] == 1).mean() * 100
            row.append(prev)
        prevalence_matrix.append(row)
    
    prevalence_df = pd.DataFrame(prevalence_matrix, 
                                  index=top_disease_names,
                                  columns=[f.replace("_", " ") for f in top_feature_names])
    
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(prevalence_df, annot=True, fmt=".0f", cmap="YlOrRd", 
                cbar_kws={"label": "Prevalence (%)"}, ax=ax, linewidths=0.5)
    ax.set_title(f"Feature Prevalence by Disease (Top {top_diseases} Diseases, Top {top_features} Features)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Diseases")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "disease_feature_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved disease-feature heatmap to {EDA_DIR / 'disease_feature_heatmap.png'}")


def plot_top_features_by_disease(df: pd.DataFrame, feature_cols: list[str], 
                                  disease: str, top_n: int = 15):
    """Plot top features for a specific disease."""
    disease_df = df[df["label_name"] == disease]
    if len(disease_df) == 0:
        return
    
    feature_prev = []
    for feat in feature_cols:
        prev = (disease_df[feat] == 1).mean() * 100
        feature_prev.append((feat, prev))
    
    feature_prev.sort(key=lambda x: x[1], reverse=True)
    top_features = feature_prev[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    feats = [f[0].replace("_", " ") for f in top_features]
    prevs = [f[1] for f in top_features]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feats)))
    ax.barh(range(len(feats)), prevs, color=colors)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_xlabel("Prevalence (%)")
    ax.set_title(f"Top {top_n} Features for {disease} (n={len(disease_df)})")
    ax.invert_yaxis()
    
    plt.tight_layout()
    safe_name = disease.replace("/", "_").replace(" ", "_")[:50]
    plt.savefig(EDA_DIR / f"top_features_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_eda_summary_tables(df: pd.DataFrame, feature_cols: list[str], coverage_df: pd.DataFrame):
    """Generate summary tables for EDA."""
    
    # 1. Overall feature statistics
    stats = {
        "total_records": len(df),
        "total_features": len(feature_cols),
        "total_diseases": df["label_name"].nunique(),
        "mean_informativeness": coverage_df["informativeness"].mean(),
        "median_informativeness": coverage_df["informativeness"].median(),
        "features_with_>50%_informativeness": (coverage_df["informativeness"] > 50).sum(),
        "features_with_<10%_informativeness": (coverage_df["informativeness"] < 10).sum(),
    }
    
    with open(EDA_DIR / "eda_summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved EDA summary stats to {EDA_DIR / 'eda_summary_stats.json'}")
    
    # 2. Top 20 most informative features
    top_20 = coverage_df.head(20)[["feature", "pct_present", "pct_unknown", "informativeness"]]
    top_20.to_csv(EDA_DIR / "top_20_informative_features.csv", index=False)
    print(f"  Saved top 20 features to {EDA_DIR / 'top_20_informative_features.csv'}")
    
    # 3. Bottom 20 least informative features
    bottom_20 = coverage_df.tail(20)[["feature", "pct_present", "pct_unknown", "informativeness"]]
    bottom_20.to_csv(EDA_DIR / "bottom_20_informative_features.csv", index=False)
    print(f"  Saved bottom 20 features to {EDA_DIR / 'bottom_20_informative_features.csv'}")
    
    # 4. Disease-wise record counts
    disease_counts = df["label_name"].value_counts().reset_index()
    disease_counts.columns = ["disease", "count"]
    disease_counts["percentage"] = (disease_counts["count"] / len(df) * 100).round(2)
    disease_counts.to_csv(EDA_DIR / "disease_distribution.csv", index=False)
    print(f"  Saved disease distribution to {EDA_DIR / 'disease_distribution.csv'}")
    
    # 5. Feature prevalence by disease (top 10 diseases, all features)
    top_10_diseases = df["label_name"].value_counts().head(10).index.tolist()
    prevalence_by_disease = {}
    for disease in top_10_diseases:
        disease_df = df[df["label_name"] == disease]
        prev_row = {}
        for feat in feature_cols[:50]:  # Limit to top 50 features for brevity
            prev_row[feat] = round((disease_df[feat] == 1).mean() * 100, 2)
        prevalence_by_disease[disease] = prev_row
    
    prev_df = pd.DataFrame(prevalence_by_disease).T
    prev_df.to_csv(EDA_DIR / "feature_prevalence_top10_diseases.csv")
    print(f"  Saved feature prevalence by disease to {EDA_DIR / 'feature_prevalence_top10_diseases.csv'}")


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
    print("=" * 60)
    print("  PHASE 3: Feature Analysis with Detailed EDA")
    print("=" * 60 + "\n")

    # Load data
    print("Loading data...")
    derm_df = pd.read_csv(DERM_FEATURES_CSV)
    scin_df = pd.read_csv(SCIN_CSV)
    print(f"  Derm-1M: {len(derm_df):,} records")
    print(f"  SCIN: {len(scin_df):,} records\n")

    # Meta columns to exclude from feature analysis
    META_COLS = {"image", "label_name", "disease_label", "age_numeric",
                 "age_group", "duration_bucket", "diagnosis_confidence",
                 "lesion_count"}
    feature_cols = [
        c for c in derm_df.columns
        if c not in META_COLS
        and derm_df[c].isin([0, 1, 2]).mean() > 0.8  # binary/ternary cols
    ]
    print(f"  Total features to analyze: {len(feature_cols)}\n")

    # 1. Coverage
    print("1. Computing feature coverage...")
    coverage = analyse_feature_coverage(derm_df, feature_cols)
    coverage.to_csv(OUTPUT_DIR / "feature_coverage.csv", index=False)
    plot_coverage(coverage)
    print(f"  Saved feature_coverage.csv and .png\n")

    # 2. Derm-1M vs SCIN comparison
    print("2. Comparing Derm-1M vs SCIN feature availability...")
    cmp_df = compare_derm_vs_scin(derm_df, scin_df, feature_cols)
    cmp_df.to_csv(OUTPUT_DIR / "derm_vs_scin_comparison.csv", index=False)
    plot_derm_vs_scin_comparison(cmp_df)
    derm_mean = cmp_df["derm1m_pct_informed"].mean()
    scin_mean = cmp_df["scin_pct_available"].mean()
    print(f"  Derm-1M avg feature informativeness: {derm_mean:.1f}%")
    print(f"  SCIN avg feature availability:       {scin_mean:.1f}%")
    print(f"  Gap (Derm-1M advantage):             {derm_mean - scin_mean:.1f}%\n")

    # 3. Global feature importance
    print("3. Computing global mutual information feature importance...")
    mi_df = compute_feature_importance(derm_df, feature_cols, label_col="label_name")
    mi_df.to_csv(OUTPUT_DIR / "feature_importance_global.csv", index=False)
    print(f"  Top 10 features by mutual information:")
    print(mi_df.head(10).to_string(index=False))
    print()

    # 4. Class-wise importance
    print("4. Computing class-wise feature importance...")
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
    print("5. Generating example questionnaire for confusion cluster...")
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
    print()

    # 6. Features only in Derm-1M (not available in SCIN)
    print("6. Analyzing Derm-1M-only features...")
    scin_canonical_set = set(SCIN_TO_CANONICAL.values())
    derm_only_features = [
        c for c in feature_cols if c not in scin_canonical_set
    ]
    derm_only_coverage = coverage[coverage["feature"].isin(derm_only_features)]
    derm_only_coverage.to_csv(
        OUTPUT_DIR / "derm_only_features.csv", index=False
    )
    print(f"  Features in Derm-1M but NOT in SCIN: {len(derm_only_features)}")
    print(f"  Saved derm_only_features.csv")
    if len(derm_only_coverage) > 0:
        top_derm_only = derm_only_coverage.head(10)
        print(f"  Top 10 Derm-1M-only features by informativeness:")
        for _, row in top_derm_only.iterrows():
            print(f"    {row['feature']:40s} "
                  f"present={row['pct_present']:.1f}% "
                  f"informed={row['informativeness']:.1f}%")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # DETAILED EDA
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  DETAILED EDA")
    print("=" * 60 + "\n")

    # 7. Feature distribution analysis
    print("7. Generating feature distribution analysis...")
    plot_feature_distribution(coverage, top_n=50)
    print()

    # 8. Disease-wise feature heatmap
    print("8. Generating disease-wise feature heatmap...")
    plot_disease_wise_feature_heatmap(derm_df, feature_cols, 
                                       top_diseases=20, top_features=30)
    print()

    # 9. Top features by disease
    print("9. Generating top features by disease plots...")
    top_5_diseases = derm_df["label_name"].value_counts().head(5).index.tolist()
    for disease in top_5_diseases:
        plot_top_features_by_disease(derm_df, feature_cols, disease, top_n=15)
    print(f"  Generated top features plots for top 5 diseases\n")

    # 10. Generate summary tables
    print("10. Generating EDA summary tables...")
    generate_eda_summary_tables(derm_df, feature_cols, coverage)
    print()

    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to:")
    print(f"  - Main outputs: {OUTPUT_DIR}/")
    print(f"  - EDA outputs: {EDA_DIR}/")
    print(f"\nKey files:")
    print(f"  - feature_coverage.csv - Feature coverage statistics")
    print(f"  - feature_importance_global.csv - Global feature importance")
    print(f"  - derm_vs_scin_comparison.csv - Derm-1M vs SCIN comparison")
    print(f"  - eda/feature_distribution_overview.png - Feature distribution visualizations")
    print(f"  - eda/disease_feature_heatmap.png - Disease-feature prevalence heatmap")
    print(f"  - eda/top_features_*.png - Top features by disease")
    print(f"  - eda/*.csv - Various summary tables")
