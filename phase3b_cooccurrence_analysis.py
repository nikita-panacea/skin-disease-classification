"""
PHASE 3b: Per-Disease Feature Co-occurrence Analysis
=====================================================
Core insight: The classifier learned feature *combinations*, not individual features.
A disease label's "diagnostic signature" is the set of features that consistently
co-occur. Comparing these signatures between Derm-1M and SCIN reveals exactly
which co-occurrences the model relies on that SCIN cannot provide.

Analyses:
  1. Per-disease feature co-occurrence matrix (Phi coefficient)
  2. Per-disease "diagnostic signature" — the minimal feature set that
     distinguishes a disease (frequent itemsets via Apriori-style mining)
  3. Signature completeness in SCIN: for each disease's signature,
     how many features are actually available in SCIN cases?
  4. Signature-level gap: which feature *pairs/triplets* are present in
     Derm-1M but missing/broken in SCIN → root cause of performance drop
  5. Confusion-aware gap: for pairs of diseases the model confuses,
     which distinguishing co-occurrence is absent in SCIN?

Run after phase2_bulk_extraction.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DERM_FEATURES_CSV  = "derm1m_features.csv"
SCIN_CSV           = "SCIN-dataset/dataset_scin_cases.csv"
SCHEMA_PATH        = "feature_schema.json"
OUTPUT_DIR         = Path("analysis_outputs/cooccurrence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SUPPORT        = 0.15   # feature must be present in >= 15% of class to count
MIN_PAIR_SUPPORT   = 0.10   # co-occurring pair must appear in >= 10% of class
TOP_FEATURES_PLOT  = 20     # features shown in heatmaps

# SCIN column -> canonical feature (must match phase2 output column names)
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
    # Body parts -> canonical location names from phase2
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
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Phi coefficient for two binary arrays.
    Measures strength of co-occurrence beyond chance.
    Excludes rows where either feature is unknown (value=2).
    Range: -1 to +1. We care about strong positives (both features present together).
    """
    mask = (x != 2) & (y != 2)
    if mask.sum() < 10:
        return 0.0
    x_b = (x[mask] == 1).astype(int)
    y_b = (y[mask] == 1).astype(int)
    if x_b.std() == 0 or y_b.std() == 0:
        return 0.0
    r, _ = pearsonr(x_b, y_b)
    return round(float(r), 4)

def binary_presence(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Convert 0/1/2 matrix to binary 0/1 matrix (2 -> NaN).
    Used for co-occurrence computation where unknowns are excluded.
    """
    binary = df[feature_cols].copy().astype(float)
    binary[binary == 2] = np.nan
    return binary

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Per-disease co-occurrence matrix
# ──────────────────────────────────────────────────────────────────────────────
def compute_disease_cooccurrence(
    df: pd.DataFrame, feature_cols: list[str], disease: str,
    label_col: str = "label_name"
) -> pd.DataFrame:
    """
    For one disease class (label_name), compute pairwise Phi coefficient
    for all feature pairs. Returns a square DataFrame (feature x feature).
    """
    subset = df[df[label_col] == disease][feature_cols]
    n_rows, n_cols = subset.shape
    phi_matrix = np.zeros((n_cols, n_cols))

    for i, f1 in enumerate(feature_cols):
        for j, f2 in enumerate(feature_cols):
            if i == j:
                phi_matrix[i, j] = 1.0
            elif i < j:
                phi = phi_coefficient(subset[f1].values, subset[f2].values)
                phi_matrix[i, j] = phi
                phi_matrix[j, i] = phi  # symmetric

    return pd.DataFrame(phi_matrix, index=feature_cols, columns=feature_cols)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Diagnostic signature — frequent co-occurring feature sets per disease
# ──────────────────────────────────────────────────────────────────────────────
def extract_diagnostic_signature(
    df: pd.DataFrame,
    feature_cols: list[str],
    disease: str,
    label_col: str = "label_name",
    min_support: float = MIN_SUPPORT,
    min_pair_support: float = MIN_PAIR_SUPPORT,
    top_k_singles: int = 15,
) -> dict:
    """
    For one disease (label_name):
    1. Find frequent single features (prevalence >= min_support within class)
    2. Find frequent feature pairs (both present >= min_pair_support within class)
    3. Find frequent triplets from the top single features
    Returns a dict with 'singles', 'pairs', 'triplets', 'n_cases'
    """
    subset = df[df[label_col] == disease]
    n = len(subset)
    if n < 20:
        return {
            "disease": disease, "n_cases": n,
            "singles": [], "pairs": [], "triplets": [],
        }

    # Single features
    singles = []
    for f in feature_cols:
        col = subset[f]
        n_informed = (col != 2).sum()
        n_present = (col == 1).sum()
        support = n_present / n if n > 0 else 0
        if support >= min_support:
            singles.append({
                "feature":          f,
                "support":          round(support, 3),
                "n_present":        int(n_present),
                "n_informed":       int(n_informed),
                "pct_of_informed":  round(n_present / n_informed, 3) if n_informed > 0 else 0,
            })
    singles.sort(key=lambda x: -x["support"])

    # Pairs (only from top singles to keep tractable)
    top_feats = [s["feature"] for s in singles[:top_k_singles]]
    pairs = []
    for f1, f2 in combinations(top_feats, 2):
        mask = (subset[f1] != 2) & (subset[f2] != 2)
        n_both = ((subset.loc[mask, f1] == 1) & (subset.loc[mask, f2] == 1)).sum()
        support = n_both / n if n > 0 else 0
        phi = phi_coefficient(subset[f1].values, subset[f2].values)
        if support >= min_pair_support:
            pairs.append({
                "feature_1": f1, "feature_2": f2,
                "joint_support": round(support, 3),
                "n_both": int(n_both),
                "phi_coefficient": phi,
            })
    pairs.sort(key=lambda x: -x["joint_support"])

    # Triplets (from top 10 singles)
    top10 = [s["feature"] for s in singles[:10]]
    triplets = []
    for f1, f2, f3 in combinations(top10, 3):
        mask = (subset[f1] != 2) & (subset[f2] != 2) & (subset[f3] != 2)
        n_all3 = (
            (subset.loc[mask, f1] == 1)
            & (subset.loc[mask, f2] == 1)
            & (subset.loc[mask, f3] == 1)
        ).sum()
        support = n_all3 / n if n > 0 else 0
        if support >= min_pair_support:
            triplets.append({
                "features": [f1, f2, f3],
                "joint_support": round(support, 3),
                "n_all_present": int(n_all3),
            })
    triplets.sort(key=lambda x: -x["joint_support"])

    return {
        "disease":   disease,
        "n_cases":   n,
        "singles":   singles,
        "pairs":     pairs[:30],
        "triplets":  triplets[:20],
    }

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: SCIN signature completeness
# ──────────────────────────────────────────────────────────────────────────────
def build_scin_feature_matrix(scin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SCIN raw columns to canonical feature names.
    Binary: 1 = YES, 0 = NO/absent, 2 = unknown.
    """
    result = pd.DataFrame(index=scin_df.index)

    for scin_col, canonical in SCIN_TO_CANONICAL.items():
        if scin_col not in scin_df.columns:
            continue
        col = scin_df[scin_col]
        if col.dtype == object:
            vals = col.str.upper().str.strip()
            encoded = vals.map({"YES": 1}).fillna(
                vals.map({"NO": 0}).fillna(2)
            ).astype(int)
        else:
            encoded = col.notna().astype(int) * 1  # 1 if filled, else 0
        # If canonical already added (e.g. torso_front + torso_back -> location_trunk), OR
        if canonical in result.columns:
            result[canonical] = result[canonical].combine(
                encoded,
                lambda a, b: 1 if (a == 1 or b == 1)
                else (2 if (a == 2 and b == 2) else 0),
            )
        else:
            result[canonical] = encoded

    return result

def signature_completeness_in_scin(
    signature: dict,
    scin_features: pd.DataFrame,
    canonical_available: set[str],
    scin_label_col: str = None,
    disease_label: str = None,
) -> dict:
    """
    For a disease's Derm-1M diagnostic signature:
    - Which single/pair features are AVAILABLE in SCIN (column exists)?
    - Which are INFORMATIVE in SCIN (column filled >= 10% of SCIN cases)?
    - What is the overall "signature coverage" score?

    If scin_label_col and disease_label provided, restricts to matching SCIN cases.
    """
    disease = signature["disease"]

    # Optionally filter SCIN to same disease
    if scin_label_col and disease_label and scin_label_col in scin_features.columns:
        scin_sub = scin_features[scin_features[scin_label_col] == disease_label]
    else:
        scin_sub = scin_features

    n_scin = max(len(scin_sub), 1)

    # -- Single feature coverage --
    singles_coverage = []
    for s in signature["singles"]:
        feat = s["feature"]
        available = feat in canonical_available
        if available and feat in scin_sub.columns:
            pct_informed = (scin_sub[feat] != 2).mean()
            pct_present = (scin_sub[feat] == 1).mean()
        else:
            pct_informed, pct_present = 0.0, 0.0
        singles_coverage.append({
            "feature":            feat,
            "derm1m_support":     s["support"],
            "scin_col_available": available,
            "scin_pct_informed":  round(pct_informed, 3),
            "scin_pct_present":   round(pct_present, 3),
            "gap":                round(s["support"] - pct_present, 3),
        })

    # -- Pair coverage --
    pairs_coverage = []
    for p in signature["pairs"]:
        f1, f2 = p["feature_1"], p["feature_2"]
        both_avail = (f1 in canonical_available) and (f2 in canonical_available)
        if both_avail and f1 in scin_sub.columns and f2 in scin_sub.columns:
            mask = (scin_sub[f1] != 2) & (scin_sub[f2] != 2)
            n_both = (
                (scin_sub.loc[mask, f1] == 1) & (scin_sub.loc[mask, f2] == 1)
            ).sum()
            scin_joint = n_both / n_scin
            phi_scin = phi_coefficient(scin_sub[f1].values, scin_sub[f2].values)
        else:
            scin_joint, phi_scin = 0.0, 0.0
        pairs_coverage.append({
            "feature_1":            f1,
            "feature_2":            f2,
            "derm1m_joint_support": p["joint_support"],
            "derm1m_phi":           p["phi_coefficient"],
            "both_cols_in_scin":    both_avail,
            "scin_joint_support":   round(scin_joint, 3),
            "scin_phi":             round(phi_scin, 3),
            "joint_support_gap":    round(p["joint_support"] - scin_joint, 3),
            "phi_gap":              round(p["phi_coefficient"] - phi_scin, 3),
        })

    # -- Summary score --
    n_singles = len(singles_coverage)
    n_covered = sum(1 for s in singles_coverage if s["scin_pct_informed"] >= 0.05)
    coverage_score = n_covered / n_singles if n_singles > 0 else 0

    n_pairs = len(pairs_coverage)
    n_intact = sum(
        1 for p in pairs_coverage
        if p["derm1m_phi"] > 0 and p["scin_phi"] >= 0.5 * p["derm1m_phi"]
    )
    pair_integrity = n_intact / n_pairs if n_pairs > 0 else 0

    return {
        "disease":          disease,
        "n_derm1m_cases":   signature["n_cases"],
        "n_scin_cases":     n_scin,
        "singles_coverage": singles_coverage,
        "pairs_coverage":   pairs_coverage,
        "summary": {
            "n_signature_features":      n_singles,
            "n_features_in_scin":        n_covered,
            "single_feature_coverage":   round(coverage_score, 3),
            "n_signature_pairs":         n_pairs,
            "n_pairs_intact_in_scin":    n_intact,
            "pair_integrity_score":      round(pair_integrity, 3),
            "signature_reproducibility": round((coverage_score + pair_integrity) / 2, 3),
        },
    }

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Confusion-aware gap
# ──────────────────────────────────────────────────────────────────────────────
def confusion_aware_gap(
    signatures: dict[str, dict],
    confusion_pairs: list[tuple[str, str]],  # (true_label, confused_with)
) -> list[dict]:
    """
    For each confused pair (A, B):
    Find feature pairs that strongly co-occur in A but NOT in B (discriminators).
    Then check if those discriminating co-occurrences are available in SCIN.
    This shows EXACTLY what signal the model loses when evaluated on SCIN.
    """
    results = []

    for label_a, label_b in confusion_pairs:
        if label_a not in signatures or label_b not in signatures:
            continue

        sig_a = signatures[label_a]
        sig_b = signatures[label_b]

        # Features strong in A but not B (discriminating singles)
        a_singles = {s["feature"]: s["support"] for s in sig_a["singles"]}
        b_singles = {s["feature"]: s["support"] for s in sig_b["singles"]}

        discriminating = []
        for feat, a_sup in a_singles.items():
            b_sup = b_singles.get(feat, 0)
            lift = a_sup - b_sup
            if lift >= 0.10:  # at least 10pp more common in A
                discriminating.append({
                    "feature": feat,
                    "support_in_A": a_sup,
                    "support_in_B": b_sup,
                    "discriminating_lift": round(lift, 3),
                })
        discriminating.sort(key=lambda x: -x["discriminating_lift"])

        # Discriminating pairs in A not in B
        a_pairs = {(p["feature_1"], p["feature_2"]): p for p in sig_a["pairs"]}
        b_pair_keys = {(p["feature_1"], p["feature_2"]) for p in sig_b["pairs"]}

        disc_pairs = []
        for (f1, f2), p in a_pairs.items():
            if (f1, f2) not in b_pair_keys:
                disc_pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "joint_support_in_A": p["joint_support"],
                    "phi_in_A": p["phi_coefficient"],
                    "present_in_B_signature": False,
                })

        results.append({
            "true_label":            label_a,
            "confused_with":         label_b,
            "discriminating_singles": discriminating[:10],
            "discriminating_pairs":   disc_pairs[:10],
            "n_discriminators":       len(discriminating),
        })

    return results

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
def plot_cooccurrence_heatmap(
    phi_df: pd.DataFrame, disease: str, top_n: int = TOP_FEATURES_PLOT
):
    """Plot Phi-coefficient heatmap for a disease, showing top N most co-occurring features."""
    max_phi = phi_df.abs().replace(1.0, 0).max(axis=1)
    top_feats = max_phi.nlargest(top_n).index.tolist()
    sub = phi_df.loc[top_feats, top_feats]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.eye(len(top_feats), dtype=bool)
    sns.heatmap(
        sub, mask=mask, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, ax=ax, square=True,
    )
    ax.set_title(f"Feature Co-occurrence (Phi) - {disease}", fontsize=12, pad=12)
    plt.tight_layout()
    safe = disease.replace("/", "_").replace(" ", "_")[:50]
    path = OUTPUT_DIR / f"cooccurrence_{safe}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path

def plot_signature_gap(completeness: dict, top_n: int = 15):
    """Bar chart: Derm-1M feature support vs SCIN feature support per disease signature."""
    singles = completeness["singles_coverage"][:top_n]
    if not singles:
        return

    feats = [s["feature"].replace("_", " ") for s in singles]
    derm = [s["derm1m_support"] for s in singles]
    scin = [s["scin_pct_present"] for s in singles]

    x = np.arange(len(feats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w / 2, derm, w, label="Derm-1M support", color="#1565C0")
    ax.bar(x + w / 2, scin, w, label="SCIN present%", color="#EF5350", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Proportion of class records")
    disease = completeness["disease"]
    ax.set_title(
        f"Signature Feature Gap - {disease}\n"
        f"Coverage: {completeness['summary']['single_feature_coverage']:.0%}  |  "
        f"Pair Integrity: {completeness['summary']['pair_integrity_score']:.0%}  |  "
        f"Reproducibility: {completeness['summary']['signature_reproducibility']:.0%}"
    )
    ax.legend()
    plt.tight_layout()
    safe = disease.replace("/", "_").replace(" ", "_")[:50]
    path = OUTPUT_DIR / f"signature_gap_{safe}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path

def plot_reproducibility_summary(all_completeness: list[dict]):
    """Summary chart: signature reproducibility score per disease class."""
    rows = [
        (c["disease"], c["summary"]["signature_reproducibility"],
         c["summary"]["single_feature_coverage"],
         c["summary"]["pair_integrity_score"])
        for c in all_completeness
    ]
    rows.sort(key=lambda x: x[1])

    diseases = [r[0] for r in rows]
    reprod = [r[1] for r in rows]
    single = [r[2] for r in rows]
    pair_int = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(14, max(6, len(diseases) * 0.3)))
    y = np.arange(len(diseases))
    ax.barh(y, reprod, color="#1565C0", label="Signature Reproducibility")
    ax.barh(y - 0.25, single, height=0.2, color="#42A5F5", alpha=0.8,
            label="Single-feature Coverage")
    ax.barh(y + 0.25, pair_int, height=0.2, color="#EF9A9A", alpha=0.8,
            label="Pair Integrity")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(diseases, fontsize=8)
    ax.set_xlabel("Score (0-1)")
    ax.set_title(
        "SCIN Signature Reproducibility per Disease Class\n"
        "(Low score = model loses diagnostic signal when evaluated on SCIN)"
    )
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = OUTPUT_DIR / "reproducibility_summary.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Phase 3b: Co-occurrence Analysis ===\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading feature matrices...")
    derm_df = pd.read_csv(DERM_FEATURES_CSV)
    scin_raw = pd.read_csv(SCIN_CSV)

    # Meta columns to exclude from feature analysis (matches phase2 output)
    META_COLS = {
        "image", "label_name", "disease_label", "age_numeric",
        "age_group", "duration_bucket", "diagnosis_confidence", "lesion_count",
    }
    feature_cols = [
        c for c in derm_df.columns
        if c not in META_COLS and derm_df[c].isin([0, 1, 2]).mean() > 0.8
    ]

    # Use label_name (cleaned/merged labels used by the model)
    diseases = derm_df["label_name"].value_counts()
    print(f"  {len(derm_df):,} Derm-1M records | "
          f"{len(feature_cols)} features | "
          f"{len(diseases)} disease classes (label_name)\n")

    # ── Build SCIN canonical feature matrix ───────────────────────────────────
    print("Building SCIN canonical feature matrix...")
    scin_features = build_scin_feature_matrix(scin_raw)
    canonical_in_scin = set(scin_features.columns)
    print(f"  {len(canonical_in_scin)} canonical features available in SCIN\n")

    # ── Step 1+2: Extract signatures for all diseases (by label_name) ─────────
    print("Extracting diagnostic signatures per disease (label_name)...")
    all_signatures: dict[str, dict] = {}
    for disease in diseases.index:
        sig = extract_diagnostic_signature(
            derm_df, feature_cols, disease, label_col="label_name"
        )
        all_signatures[disease] = sig

    sig_path = OUTPUT_DIR / "all_signatures.json"
    with open(sig_path, "w") as f:
        json.dump(all_signatures, f, indent=2)
    print(f"  Saved {len(all_signatures)} signatures -> {sig_path}\n")

    # ── Step 3: SCIN signature completeness ───────────────────────────────────
    print("Computing SCIN signature completeness per disease...")
    all_completeness = []
    for disease, sig in all_signatures.items():
        comp = signature_completeness_in_scin(sig, scin_features, canonical_in_scin)
        all_completeness.append(comp)
        plot_signature_gap(comp)

    # Save summary table
    summary_rows = [
        c["summary"] | {
            "disease": c["disease"],
            "n_derm1m": c["n_derm1m_cases"],
            "n_scin": c["n_scin_cases"],
        }
        for c in all_completeness
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("signature_reproducibility")
    summary_df.to_csv(OUTPUT_DIR / "signature_completeness_summary.csv", index=False)

    print(f"  Reproducibility scores (low = SCIN cannot reproduce Derm-1M signature):")
    print(
        summary_df[
            ["disease", "signature_reproducibility",
             "single_feature_coverage", "pair_integrity_score"]
        ].to_string(index=False)
    )

    plot_reproducibility_summary(all_completeness)

    # ── Step 4: Co-occurrence heatmaps for key diseases ───────────────────────
    worst_diseases = summary_df.head(10)["disease"].tolist()
    print(f"\nPlotting co-occurrence heatmaps for "
          f"{len(worst_diseases)} lowest-reproducibility diseases...")
    for disease in worst_diseases:
        phi_df = compute_disease_cooccurrence(
            derm_df, feature_cols, disease, label_col="label_name"
        )
        plot_cooccurrence_heatmap(phi_df, disease)

    # ── Step 5: Confusion-aware gap ───────────────────────────────────────────
    # Confusion pairs using label_name (the cleaned/merged model labels)
    # Replace/extend with actual pairs from your confusion matrices
    confusion_pairs = [
        ("eczema",                      "dermatitis"),
        ("eczema",                      "psoriasis"),
        ("psoriasis",                   "dermatitis"),
        ("melanoma",                    "nevus (mole)"),
        ("basal cell carcinoma",        "squamous cell carcinoma"),
        ("tinea (ringworm)",            "eczema"),
        ("urticaria (hives)",           "dermatitis"),
        ("dermatitis",                  "eczema"),
        ("acne",                        "folliculitis (inflamed hair follicles)"),
        ("herpes simplex virus",        "herpes zoster (shingles)"),
    ]
    print(f"\nComputing confusion-aware gaps for {len(confusion_pairs)} pairs...")
    conf_gaps = confusion_aware_gap(all_signatures, confusion_pairs)
    with open(OUTPUT_DIR / "confusion_aware_gaps.json", "w") as f:
        json.dump(conf_gaps, f, indent=2)

    for gap in conf_gaps:
        print(f"\n  [{gap['true_label']}] confused with [{gap['confused_with']}]:")
        print(f"    {gap['n_discriminators']} discriminating features, top 3:")
        for d in gap["discriminating_singles"][:3]:
            print(
                f"      {d['feature']:35s}  "
                f"A={d['support_in_A']:.2f}  B={d['support_in_B']:.2f}  "
                f"lift={d['discriminating_lift']:.2f}"
            )
        if gap["discriminating_pairs"]:
            print(f"    Top discriminating pair unique to [{gap['true_label']}]:")
            for p in gap["discriminating_pairs"][:2]:
                print(
                    f"      ({p['feature_1']} + {p['feature_2']})  "
                    f"support={p['joint_support_in_A']:.2f}  "
                    f"phi={p['phi_in_A']:.2f}"
                )

    print(f"\n  All co-occurrence outputs -> {OUTPUT_DIR}/")
    print("   Key files:")
    print("   - all_signatures.json               -- full per-disease feature signatures")
    print("   - signature_completeness_summary.csv -- reproducibility scores per disease")
    print("   - reproducibility_summary.png        -- visual summary across all classes")
    print("   - confusion_aware_gaps.json          -- exact missing signal per confused pair")
    print("   - cooccurrence_<disease>.png         -- heatmaps for worst-reproducibility diseases")
    print("   - signature_gap_<disease>.png        -- Derm-1M vs SCIN bar charts per disease")
