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

Env (optional):
  LABEL_COL — target column for MI / chi-square / classwise OR (default: label_name; use disease_label for fine-grained)
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

from scin_feature_map import SCIN_TO_CANONICAL, SCIN_COMPARABLE_CANONICALS

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

# Target column for MI / chi-square / classwise OR (label_name = merged class; disease_label = fine-grained)
LABEL_COL = os.getenv("LABEL_COL", "label_name")


def load_schema_category_map(schema_path: str = SCHEMA_PATH) -> dict[str, str]:
    """
    Load feature schema and return {feature_full_name: category} mapping.
    Feature names are constructed as {category}_{value}.
    Falls back to prefix-based heuristic if schema file is missing.
    """
    if not Path(schema_path).exists():
        return {}
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    mapping: dict[str, str] = {}
    for entry in schema.get("feature_categories", []):
        if not isinstance(entry, dict):
            continue
        cat = str(entry.get("category", "other"))
        for val in entry.get("features", []):
            v = str(val).strip()
            if v:
                mapping[f"{cat}_{v}"] = cat
    return mapping


def get_feature_category(feature_name: str, schema_map: dict[str, str]) -> str:
    """Get the category for a feature, using schema map with prefix fallback."""
    if schema_map and feature_name in schema_map:
        return schema_map[feature_name]
    parts = feature_name.split("_", 1)
    return parts[0] if parts else "other"

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

    columns = [
        "canonical_feature", "scin_column",
        "derm1m_pct_informed", "scin_pct_available", "gap",
    ]
    if not rows:
        print(
            "  WARNING: no overlap between SCIN_TO_CANONICAL values and feature_cols; "
            "update scin_feature_map.SCIN_TO_CANONICAL to match current feature_schema.json. "
            "Returning empty comparison frame."
        )
        return pd.DataFrame(columns=columns)

    cmp_df = pd.DataFrame(rows, columns=columns).sort_values("gap", ascending=False)
    return cmp_df

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: Disease Class-wise Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
def _cramers_v_from_table(observed: np.ndarray) -> tuple[float, float, float]:
    """
    Return (chi2, p_value, cramers_v). Cramér's V in [0,1] for effect size.

    Drops rows/columns with zero marginal before calling chi2_contingency
    (scipy refuses to compute expected frequencies when any marginal is 0),
    and degrades gracefully on degenerate tables.
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.size == 0:
        return 0.0, 1.0, 0.0

    row_sums = obs.sum(axis=1)
    col_sums = obs.sum(axis=0)
    obs = obs[row_sums > 0][:, col_sums > 0]
    if obs.size == 0 or min(obs.shape) < 2:
        return 0.0, 1.0, 0.0

    n = float(obs.sum())
    if n <= 0:
        return 0.0, 1.0, 0.0

    try:
        chi2, p, dof, expected = chi2_contingency(obs)
    except ValueError:
        return 0.0, 1.0, 0.0

    r, k = obs.shape
    denom = n * (min(r, k) - 1)
    v = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
    v = min(v, 1.0)
    p_out = float(p) if np.isfinite(p) else 1.0
    return float(chi2), p_out, v


def compute_feature_importance(
    df: pd.DataFrame, feature_cols: list[str], label_col: str = "label_name"
) -> pd.DataFrame:
    """
    Per feature vs disease label (informed rows only, values 0/1):
    - Mutual information (discrete)
    - Chi-square independence test on feature × label crosstab
    - Cramér's V (effect size)
    """
    if label_col not in df.columns:
        raise ValueError(f"LABEL_COL {label_col!r} not in dataframe columns")

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    n_classes = len(le.classes_)

    rows_out = []
    for col in feature_cols:
        mask = df[col] != 2
        n_inf = int(mask.sum())
        if n_inf < 100:
            rows_out.append({
                "feature": col,
                "mutual_information": 0.0,
                "chi2_statistic": 0.0,
                "chi2_pvalue": 1.0,
                "cramers_v": 0.0,
                "n_informed": n_inf,
            })
            continue

        x_sub = df.loc[mask, col].astype(int).values
        y_sub = y[mask.to_numpy()]

        mi = mutual_info_classif(
            x_sub.reshape(-1, 1), y_sub, discrete_features=True
        )[0]

        ct = pd.crosstab(
            pd.Series(x_sub, name=col),
            pd.Series(y_sub, name="label"),
        )
        # Ensure 2 rows (0/1) for binary feature
        for xi in (0, 1):
            if xi not in ct.index:
                ct.loc[xi] = 0
        ct = ct.sort_index()
        # Ensure all label columns 0..n_classes-1 present
        for c in range(n_classes):
            if c not in ct.columns:
                ct[c] = 0
        ct = ct.reindex(sorted(ct.columns), axis=1)
        obs = ct.to_numpy(dtype=float)
        chi2, p, v = _cramers_v_from_table(obs)

        rows_out.append({
            "feature": col,
            "mutual_information": round(float(mi), 5),
            "chi2_statistic": round(chi2, 4),
            "chi2_pvalue": float(p) if np.isfinite(p) else 1.0,
            "cramers_v": round(v, 5),
            "n_informed": n_inf,
        })

    return pd.DataFrame(rows_out).sort_values("mutual_information", ascending=False)

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
            log2_or = float(np.log2(max(or_val, 1e-9)))
            rows.append({
                "feature": col,
                "in_class_pct": round(in_pct * 100, 2),
                "out_class_pct": round(out_pct * 100, 2),
                "odds_ratio": round(or_val, 3),
                "log2_odds_ratio": round(log2_or, 4),
            })
        results[label] = (
            pd.DataFrame(rows)
            .sort_values("odds_ratio", ascending=False)
            .head(top_features)
        )
    return results


def export_classwise_importance_long(
    classwise: dict[str, pd.DataFrame], path: Path
) -> None:
    """Single long CSV: disease, feature, odds_ratio, log2_odds_ratio, ..."""
    parts = []
    for disease, sub in classwise.items():
        t = sub.copy()
        t.insert(0, "disease", disease)
        parts.append(t)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(path, index=False)


def build_feature_importance_scin_context(
    mi_df: pd.DataFrame, cmp_df: pd.DataFrame | None
) -> pd.DataFrame:
    """
    Join global importance with SCIN overlap flags and Derm vs SCIN gap (where mapped).
    High MI + not in SCIN compare set + large gap → questionnaire / data collection priority.
    """
    out = mi_df.copy()
    out["in_scin_compare_set"] = out["feature"].isin(SCIN_COMPARABLE_CANONICALS)
    if cmp_df is not None and len(cmp_df) > 0:
        m = cmp_df[["canonical_feature", "derm1m_pct_informed", "scin_pct_available", "gap"]].copy()
        out = out.merge(
            m,
            left_on="feature",
            right_on="canonical_feature",
            how="left",
        )
        out = out.drop(columns=["canonical_feature"], errors="ignore")
    else:
        out["derm1m_pct_informed"] = np.nan
        out["scin_pct_available"] = np.nan
        out["gap"] = np.nan
    return out


def write_explainability_report(path: Path, label_col: str, context_df: pd.DataFrame) -> None:
    """Short markdown summary for interpretation."""
    top = context_df.nlargest(15, "mutual_information")
    high_mi_no_scin = context_df[
        (context_df["mutual_information"] > 0.01)
        & (~context_df["in_scin_compare_set"])
    ].sort_values("chi2_pvalue").head(10)
    lines = [
        "# Feature importance vs SCIN (Phase 3 summary)",
        "",
        f"- **Label column:** `{label_col}`",
        "- **in_scin_compare_set:** feature has a direct SCIN questionnaire column "
        "(textures, symptoms, body parts in `scin_feature_map.SCIN_TO_CANONICAL`).",
        "",
        "## Top 15 features by mutual information",
        "",
        "```",
        top[["feature", "mutual_information", "cramers_v", "in_scin_compare_set"]].to_string(index=False),
        "```",
        "",
        "## High MI features not in SCIN compare set (up to 10 by chi-square p-value)",
        "",
    ]
    if len(high_mi_no_scin) > 0:
        lines.append("```")
        lines.append(
            high_mi_no_scin[
                ["feature", "mutual_information", "cramers_v", "chi2_pvalue"]
            ].to_string(index=False)
        )
        lines.append("```")
    else:
        lines.append("_None or insufficient data._")
    path.write_text("\n".join(lines), encoding="utf-8")

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

def plot_feature_distribution(
    coverage_df: pd.DataFrame, top_n: int = 50, schema_map: dict[str, str] | None = None
):
    """Plot distribution of feature informativeness across all features."""
    _smap = schema_map or load_schema_category_map()
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
    
    # 4. Feature category breakdown using schema
    ax4 = axes[1, 1]
    categories = [get_feature_category(f, _smap) for f in coverage_df["feature"]]
    
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


def plot_disease_wise_feature_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    top_diseases: int = 20,
    top_features: int = 30,
    label_col: str = "label_name",
):
    """Create heatmap of feature prevalence by disease."""
    disease_counts = df[label_col].value_counts().head(top_diseases)
    top_disease_names = disease_counts.index.tolist()
    
    # Get top features by overall informativeness
    feature_prev = {}
    for feat in feature_cols:
        feature_prev[feat] = (df[feat] == 1).mean()
    top_feature_names = sorted(feature_prev, key=feature_prev.get, reverse=True)[:top_features]
    
    # Create prevalence matrix
    prevalence_matrix = []
    for disease in top_disease_names:
        disease_df = df[df[label_col] == disease]
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


def plot_top_features_by_disease(
    df: pd.DataFrame,
    feature_cols: list[str],
    disease: str,
    top_n: int = 15,
    label_col: str = "label_name",
):
    """Plot top features for a specific disease."""
    disease_df = df[df[label_col] == disease]
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


def generate_eda_summary_tables(
    df: pd.DataFrame,
    feature_cols: list[str],
    coverage_df: pd.DataFrame,
    label_col: str = "label_name",
):
    """Generate summary tables for EDA."""

    # 1. Overall feature statistics (cast numpy scalars → native Python for JSON)
    stats = {
        "total_records": int(len(df)),
        "total_features": int(len(feature_cols)),
        "total_diseases": int(df[label_col].nunique()),
        "label_col_used": str(label_col),
        "mean_informativeness": float(coverage_df["informativeness"].mean()),
        "median_informativeness": float(coverage_df["informativeness"].median()),
        "features_with_>50%_informativeness": int((coverage_df["informativeness"] > 50).sum()),
        "features_with_<10%_informativeness": int((coverage_df["informativeness"] < 10).sum()),
    }

    with open(EDA_DIR / "eda_summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
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
    disease_counts = df[label_col].value_counts().reset_index()
    disease_counts.columns = ["disease", "count"]
    disease_counts["percentage"] = (disease_counts["count"] / len(df) * 100).round(2)
    disease_counts.to_csv(EDA_DIR / "disease_distribution.csv", index=False)
    print(f"  Saved disease distribution to {EDA_DIR / 'disease_distribution.csv'}")
    
    # 5. Feature prevalence by disease (top 10 diseases, all features)
    top_10_diseases = df[label_col].value_counts().head(10).index.tolist()
    prevalence_by_disease = {}
    for disease in top_10_diseases:
        disease_df = df[df[label_col] == disease]
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
# ADDITIONAL ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def detect_feature_redundancy(
    df: pd.DataFrame,
    feature_cols: list[str],
    phi_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Identify highly correlated feature pairs (phi > threshold) that may be redundant.
    Returns DataFrame of redundant pairs sorted by correlation strength.
    """
    from scipy.stats import pearsonr

    pairs = []
    for i, f1 in enumerate(feature_cols):
        for j, f2 in enumerate(feature_cols):
            if j <= i:
                continue
            mask = (df[f1] != 2) & (df[f2] != 2)
            if mask.sum() < 50:
                continue
            x = (df.loc[mask, f1] == 1).astype(int).values
            y = (df.loc[mask, f2] == 1).astype(int).values
            if x.std() == 0 or y.std() == 0:
                continue
            r, _ = pearsonr(x, y)
            if abs(r) >= phi_threshold:
                pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "phi_coefficient": round(float(r), 4),
                    "n_informed": int(mask.sum()),
                })
    return pd.DataFrame(pairs).sort_values("phi_coefficient", ascending=False, key=abs)


def generate_scin_coverage_report(
    feature_cols: list[str],
    coverage_df: pd.DataFrame,
    schema_map: dict[str, str],
) -> pd.DataFrame:
    """
    For each feature in the schema, report: category, SCIN equivalent exists,
    SCIN column name, informativeness in Derm-1M.
    """
    scin_canonical_set = set(SCIN_TO_CANONICAL.values())
    scin_reverse = {v: k for k, v in SCIN_TO_CANONICAL.items()}

    rows = []
    for feat in feature_cols:
        cat = get_feature_category(feat, schema_map)
        has_scin = feat in scin_canonical_set
        scin_col = scin_reverse.get(feat, "")
        cov_row = coverage_df[coverage_df["feature"] == feat]
        info = float(cov_row["informativeness"].iloc[0]) if len(cov_row) > 0 else 0.0
        pct_present = float(cov_row["pct_present"].iloc[0]) if len(cov_row) > 0 else 0.0

        rows.append({
            "feature": feat,
            "category": cat,
            "has_scin_equivalent": has_scin,
            "scin_column": scin_col,
            "derm1m_informativeness": round(info, 2),
            "derm1m_pct_present": round(pct_present, 2),
        })

    return pd.DataFrame(rows).sort_values(
        ["has_scin_equivalent", "derm1m_informativeness"],
        ascending=[True, False],
    )


def generate_confusion_cluster_questionnaires(
    classwise_importance: dict[str, pd.DataFrame],
    confusion_pairs_csv: str | None = None,
) -> dict[str, list[dict]]:
    """
    Generate questionnaires for disease clusters derived from confusion pairs.
    If confusion_pairs_csv is available, use it; otherwise use default pairs.
    """
    default_clusters = {
        "eczema_dermatitis_psoriasis": ["eczema", "dermatitis", "psoriasis"],
        "skin_cancer": ["melanoma", "basal cell carcinoma", "squamous cell carcinoma"],
        "viral_skin": ["herpes simplex virus", "herpes zoster (shingles)", "warts"],
        "acne_folliculitis": ["acne", "folliculitis (inflamed hair follicles)"],
    }

    if confusion_pairs_csv and Path(confusion_pairs_csv).is_file():
        try:
            pairs_df = pd.read_csv(confusion_pairs_csv)
            if "true_label" in pairs_df.columns and "confused_with" in pairs_df.columns:
                cluster_members: dict[str, set[str]] = {}
                for _, row in pairs_df.iterrows():
                    a, b = str(row["true_label"]).strip(), str(row["confused_with"]).strip()
                    key = a
                    if key not in cluster_members:
                        cluster_members[key] = {a}
                    cluster_members[key].add(b)
                for key, members in cluster_members.items():
                    safe_key = key.replace(" ", "_").replace("/", "_")[:40]
                    default_clusters[safe_key] = sorted(members)
        except Exception:
            pass

    all_questionnaires: dict[str, list[dict]] = {}
    for cluster_name, cluster_diseases in default_clusters.items():
        q = generate_questionnaire_for_cluster(cluster_diseases, classwise_importance, top_n=10)
        if q:
            all_questionnaires[cluster_name] = q
    return all_questionnaires


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
    print(f"  SCIN: {len(scin_df):,} records")
    print(f"  LABEL_COL for MI / classwise OR: {LABEL_COL!r}\n")
    if LABEL_COL not in derm_df.columns:
        raise ValueError(
            f"LABEL_COL={LABEL_COL!r} not in derm1m_features.csv. "
            f"Use label_name or disease_label, or add column."
        )

    # Load schema-based category mapping
    schema_map = load_schema_category_map(SCHEMA_PATH)
    if schema_map:
        print(f"  Loaded schema: {len(schema_map)} feature→category mappings")
    else:
        print(f"  WARNING: Schema not found at {SCHEMA_PATH}, using prefix-based category inference")

    # Meta columns to exclude from feature analysis
    META_COLS = {"image", "image_path", "label", "label_name", "disease_label",
                 "age_numeric", "age_group", "duration_bucket",
                 "diagnosis_confidence", "lesion_count",
                 "truncated_caption", "caption", "text_caption",
                 "extracted_features"}
    feature_cols = [
        c for c in derm_df.columns
        if c not in META_COLS
        and derm_df[c].isin([0, 1, 2]).mean() > 0.8
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

    # 3. Global feature importance (MI + chi-square + Cramér's V)
    print("3. Computing global feature importance (MI + chi-square + Cramér's V)...")
    mi_df = compute_feature_importance(derm_df, feature_cols, label_col=LABEL_COL)
    mi_df.to_csv(OUTPUT_DIR / "feature_importance_global.csv", index=False)
    print("  Top 10 features by mutual information:")
    print(mi_df.head(10).to_string(index=False))
    print()

    # 3b. Explainability: importance × SCIN mapping × gap
    context_df = build_feature_importance_scin_context(mi_df, cmp_df)
    context_df.to_csv(
        OUTPUT_DIR / "feature_importance_with_scin_context.csv", index=False
    )
    write_explainability_report(
        OUTPUT_DIR / "explainability_report.md", LABEL_COL, context_df
    )
    print(f"  Saved feature_importance_with_scin_context.csv and explainability_report.md\n")

    # 4. Class-wise importance
    print("4. Computing class-wise feature importance...")
    cw_importance = compute_classwise_importance(
        derm_df, feature_cols, label_col=LABEL_COL
    )
    # Save one CSV per disease
    cw_dir = OUTPUT_DIR / "classwise_importance"
    cw_dir.mkdir(exist_ok=True)
    for disease, df_imp in cw_importance.items():
        safe_name = disease.replace("/", "_").replace(" ", "_")[:60]
        df_imp.to_csv(cw_dir / f"{safe_name}.csv", index=False)
    export_classwise_importance_long(
        cw_importance, OUTPUT_DIR / "classwise_importance_all.csv"
    )
    print(f"  Saved {len(cw_importance)} class-wise CSVs + classwise_importance_all.csv\n")

    # 5. Questionnaires for all confusion clusters
    print("5. Generating questionnaires for confusion clusters...")
    confusion_csv = os.getenv("CONFUSION_PAIRS_CSV", "").strip() or None
    all_questionnaires = generate_confusion_cluster_questionnaires(
        cw_importance, confusion_pairs_csv=confusion_csv
    )
    q_all_path = OUTPUT_DIR / "questionnaires_all_clusters.json"
    with open(q_all_path, "w") as f:
        json.dump(all_questionnaires, f, indent=2)
    print(f"  Generated questionnaires for {len(all_questionnaires)} clusters")
    for cluster_name, q_items in all_questionnaires.items():
        print(f"  [{cluster_name}]: {len(q_items)} questions")
        for item in q_items[:3]:
            print(f"    [{item['discriminatory_score']:.3f}] {item['question']}")
    print()

    # 5b. Feature redundancy detection
    print("5b. Detecting redundant feature pairs (phi >= 0.8)...")
    redundancy_df = detect_feature_redundancy(derm_df, feature_cols, phi_threshold=0.8)
    redundancy_df.to_csv(OUTPUT_DIR / "feature_redundancy.csv", index=False)
    print(f"  Found {len(redundancy_df)} highly correlated feature pairs")
    if len(redundancy_df) > 0:
        for _, row in redundancy_df.head(5).iterrows():
            print(f"    {row['feature_1']} <-> {row['feature_2']}  phi={row['phi_coefficient']:.3f}")
    print()

    # 5c. SCIN feature coverage report
    print("5c. Generating SCIN feature coverage report...")
    scin_coverage_report = generate_scin_coverage_report(
        feature_cols, coverage, schema_map
    )
    scin_coverage_report.to_csv(OUTPUT_DIR / "scin_feature_coverage_report.csv", index=False)
    n_with_scin = scin_coverage_report["has_scin_equivalent"].sum()
    n_without = len(scin_coverage_report) - n_with_scin
    print(f"  {n_with_scin} features have SCIN equivalents, {n_without} do not")
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
    plot_feature_distribution(coverage, top_n=50, schema_map=schema_map)
    print()

    # 8. Disease-wise feature heatmap
    print("8. Generating disease-wise feature heatmap...")
    plot_disease_wise_feature_heatmap(
        derm_df, feature_cols, top_diseases=20, top_features=30, label_col=LABEL_COL
    )
    print()

    # 9. Top features by disease
    print("9. Generating top features by disease plots...")
    top_5_diseases = derm_df[LABEL_COL].value_counts().head(5).index.tolist()
    for disease in top_5_diseases:
        plot_top_features_by_disease(
            derm_df, feature_cols, disease, top_n=15, label_col=LABEL_COL
        )
    print(f"  Generated top features plots for top 5 diseases\n")

    # 10. Generate summary tables
    print("10. Generating EDA summary tables...")
    generate_eda_summary_tables(derm_df, feature_cols, coverage, label_col=LABEL_COL)
    print()

    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to:")
    print(f"  - Main outputs: {OUTPUT_DIR}/")
    print(f"  - EDA outputs: {EDA_DIR}/")
    print(f"\nKey files:")
    print(f"  - feature_coverage.csv - Feature coverage statistics")
    print(f"  - feature_importance_global.csv - MI + chi-square + Cramér's V")
    print(f"  - feature_importance_with_scin_context.csv - importance vs SCIN gap")
    print(f"  - explainability_report.md - short interpretation summary")
    print(f"  - classwise_importance_all.csv - long-format per-class OR")
    print(f"  - derm_vs_scin_comparison.csv - Derm-1M vs SCIN comparison")
    print(f"  - eda/feature_distribution_overview.png - Feature distribution visualizations")
    print(f"  - eda/disease_feature_heatmap.png - Disease-feature prevalence heatmap")
    print(f"  - eda/top_features_*.png - Top features by disease")
    print(f"  - eda/*.csv - Various summary tables")
