"""
PHASE 3b: Per-Disease Feature Co-occurrence Analysis with Enhanced EDA
=======================================================================
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
  6. Enhanced EDA: Co-occurrence networks, feature correlation matrices,
     signature visualizations

Run after phase2_bulk_extraction.py

Env (optional):
  CONFUSION_PAIRS_CSV   — CSV with columns true_label, confused_with (appends to defaults)
  COOCCURRENCE_PHI_TOP_K — max features for phi heatmaps (default 60; 0 = all features)
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

from scin_feature_map import SCIN_TO_CANONICAL

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ── Config ────────────────────────────────────────────────────────────────────
DERM_FEATURES_CSV  = "derm1m_features.csv"
SCIN_CSV           = "SCIN-dataset/dataset_scin_cases.csv"
SCHEMA_PATH        = "feature_schema.json"
OUTPUT_DIR         = Path("analysis_outputs/cooccurrence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR            = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(exist_ok=True)

MIN_SUPPORT        = 0.15   # default; overridden per-class when adaptive
MIN_PAIR_SUPPORT   = 0.10
TOP_FEATURES_PLOT  = 20     # features shown in heatmaps
# Max features for pairwise phi matrix (full grid is O(k^2)); 0 = use all
COOCCURRENCE_PHI_TOP_K = int(os.getenv("COOCCURRENCE_PHI_TOP_K", "60"))
CONFUSION_PAIRS_CSV = os.getenv("CONFUSION_PAIRS_CSV", "").strip()


def adaptive_supports(n_class: int) -> tuple[float, float]:
    """Scale min support with class size so small classes are not empty."""
    if n_class <= 0:
        return MIN_SUPPORT, MIN_PAIR_SUPPORT
    ms = max(0.05, min(0.15, 30.0 / max(n_class, 1)))
    mps = max(0.05, min(0.10, 20.0 / max(n_class, 1)))
    return ms, mps

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
    label_col: str = "label_name",
    top_k_features: int | None = None,
) -> pd.DataFrame:
    """
    For one disease class (label_name), compute pairwise Phi coefficient
    for all feature pairs. Returns a square DataFrame (feature x feature).
    If top_k_features is set, only the top-K features by P(feature=1) are used.
    """
    sub_df = df[df[label_col] == disease]
    use_cols = list(feature_cols)
    if top_k_features and len(use_cols) > top_k_features:
        prev = {f: (sub_df[f] == 1).mean() for f in use_cols}
        use_cols = sorted(prev, key=prev.get, reverse=True)[:top_k_features]

    subset = sub_df[use_cols]
    n_rows, n_cols = subset.shape
    phi_matrix = np.zeros((n_cols, n_cols))

    for i, f1 in enumerate(use_cols):
        for j, f2 in enumerate(use_cols):
            if i == j:
                phi_matrix[i, j] = 1.0
            elif i < j:
                phi = phi_coefficient(subset[f1].values, subset[f2].values)
                phi_matrix[i, j] = phi
                phi_matrix[j, i] = phi  # symmetric

    return pd.DataFrame(phi_matrix, index=use_cols, columns=use_cols)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Diagnostic signature — frequent co-occurring feature sets per disease
# ──────────────────────────────────────────────────────────────────────────────
def extract_diagnostic_signature(
    df: pd.DataFrame,
    feature_cols: list[str],
    disease: str,
    label_col: str = "label_name",
    min_support: float | None = None,
    min_pair_support: float | None = None,
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
    if min_support is None or min_pair_support is None:
        ms, mps = adaptive_supports(n)
        min_support = min_support if min_support is not None else ms
        min_pair_support = min_pair_support if min_pair_support is not None else mps
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
# ENHANCED EDA FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_signature_network(signature: dict, disease: str, min_phi: float = 0.3):
    """
    Plot a network visualization of feature co-occurrences for a disease.
    Shows features as nodes and strong co-occurrences as edges.
    """
    import matplotlib.patches as mpatches
    
    pairs = [p for p in signature["pairs"] if p["phi_coefficient"] >= min_phi]
    if len(pairs) < 3:
        return
    
    # Create adjacency matrix for network layout
    features = list(set([p["feature_1"] for p in pairs] + [p["feature_2"] for p in pairs]))
    if len(features) < 3:
        return
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Simple circular layout
    n = len(features)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = {feat: (np.cos(angle), np.sin(angle)) for feat, angle in zip(features, angles)}
    
    # Draw edges
    for p in pairs:
        f1, f2 = p["feature_1"], p["feature_2"]
        if f1 in pos and f2 in pos:
            x1, y1 = pos[f1]
            x2, y2 = pos[f2]
            alpha = min(1.0, p["phi_coefficient"])
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=alpha, linewidth=2*alpha)
    
    # Draw nodes
    for feat, (x, y) in pos.items():
        # Find support for this feature
        support = next((s["support"] for s in signature["singles"] if s["feature"] == feat), 0.2)
        size = 300 + 2000 * support
        ax.scatter(x, y, s=size, c='#1976D2', alpha=0.8, edgecolors='white', linewidth=2)
        ax.text(x, y, feat.replace("_", " "), ha='center', va='center', 
               fontsize=8, fontweight='bold', wrap=True)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Feature Co-occurrence Network for {disease}\n(φ >= {min_phi}, node size = support)")
    
    plt.tight_layout()
    safe = disease.replace("/", "_").replace(" ", "_")[:50]
    plt.savefig(EDA_DIR / f"signature_network_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved signature network for {disease}")


def plot_feature_correlation_matrix(df: pd.DataFrame, feature_cols: list[str], 
                                     disease: str, top_n: int = 25):
    """Plot correlation matrix of top features for a disease."""
    subset = df[df["label_name"] == disease][feature_cols]
    if len(subset) < 20:
        return
    
    # Get top features by prevalence
    prev = {f: (subset[f] == 1).mean() for f in feature_cols}
    top_features = sorted(prev, key=prev.get, reverse=True)[:top_n]
    
    # Compute phi correlation matrix
    corr_matrix = np.zeros((len(top_features), len(top_features)))
    for i, f1 in enumerate(top_features):
        for j, f2 in enumerate(top_features):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                phi = phi_coefficient(subset[f1].values, subset[f2].values)
                corr_matrix[i, j] = phi
                corr_matrix[j, i] = phi
    
    corr_df = pd.DataFrame(corr_matrix, index=top_features, columns=top_features)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(corr_df, mask=mask, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                annot=True, fmt=".2f", annot_kws={"size": 7}, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Feature Correlation Matrix (Phi) for {disease}\nTop {top_n} features by prevalence")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    safe = disease.replace("/", "_").replace(" ", "_")[:50]
    plt.savefig(EDA_DIR / f"correlation_matrix_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_signature_summary_table(all_signatures: dict[str, dict], 
                                      all_completeness: list[dict]):
    """Generate comprehensive summary table of all signatures."""
    rows = []
    for comp in all_completeness:
        disease = comp["disease"]
        sig = all_signatures.get(disease, {})
        summary = comp["summary"]
        
        rows.append({
            "disease": disease,
            "n_cases": comp["n_derm1m_cases"],
            "n_signature_features": summary["n_signature_features"],
            "n_pairs": summary["n_signature_pairs"],
            "top_feature": sig.get("singles", [{}])[0].get("feature", "N/A") if sig.get("singles") else "N/A",
            "top_feature_support": sig.get("singles", [{}])[0].get("support", 0) if sig.get("singles") else 0,
            "scin_reproducibility": summary["signature_reproducibility"],
            "single_coverage": summary["single_feature_coverage"],
            "pair_integrity": summary["pair_integrity_score"],
        })
    
    summary_df = pd.DataFrame(rows).sort_values("scin_reproducibility")
    summary_df.to_csv(EDA_DIR / "signature_summary_all_diseases.csv", index=False)
    print(f"  Saved signature summary to {EDA_DIR / 'signature_summary_all_diseases.csv'}")
    return summary_df


def compute_hierarchical_disease_clusters(
    all_signatures: dict[str, dict],
    top_k: int = 20,
) -> tuple[pd.DataFrame, list[list[str]]]:
    """
    Build a disease similarity matrix based on Jaccard similarity of top-K features.
    Returns (similarity_df, discovered_clusters).
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    diseases = sorted(all_signatures.keys())
    n = len(diseases)

    disease_top_feats: dict[str, set[str]] = {}
    for d in diseases:
        sig = all_signatures[d]
        singles = sorted(sig.get("singles", []), key=lambda s: -s["support"])[:top_k]
        disease_top_feats[d] = {s["feature"] for s in singles}

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                a, b = disease_top_feats[diseases[i]], disease_top_feats[diseases[j]]
                union = len(a | b)
                jaccard = len(a & b) / union if union > 0 else 0.0
                sim_matrix[i, j] = jaccard
                sim_matrix[j, i] = jaccard

    sim_df = pd.DataFrame(sim_matrix, index=diseases, columns=diseases)

    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=0.7, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for d, cl in zip(diseases, labels):
        clusters.setdefault(int(cl), []).append(d)

    multi_clusters = [members for members in clusters.values() if len(members) >= 2]

    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.25)))
    sns.heatmap(sim_df, cmap="YlOrRd", vmin=0, vmax=1, annot=False,
                linewidths=0.3, ax=ax, square=True)
    ax.set_title(f"Disease Similarity (Jaccard of Top-{top_k} Features)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "disease_similarity_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()

    sim_df.to_csv(EDA_DIR / "disease_similarity_matrix.csv")
    return sim_df, multi_clusters


def compute_cross_dataset_signature_comparison(
    derm_signatures: dict[str, dict],
    scin_df: pd.DataFrame,
    scin_features: pd.DataFrame,
    feature_cols: list[str],
    scin_label_col: str = "skin_condition_label",
    top_k: int = 15,
) -> pd.DataFrame:
    """
    For diseases present in both Derm-1M and SCIN, compare top-K feature signatures
    using Jaccard similarity.
    """
    canonical_in_scin = set(scin_features.columns)

    if scin_label_col not in scin_df.columns:
        for candidate in ("condition", "label", "skin_condition"):
            if candidate in scin_df.columns:
                scin_label_col = candidate
                break

    scin_diseases = set()
    if scin_label_col in scin_df.columns:
        scin_diseases = set(scin_df[scin_label_col].dropna().unique())

    rows = []
    for disease, sig in derm_signatures.items():
        derm_singles = sorted(sig.get("singles", []), key=lambda s: -s["support"])[:top_k]
        derm_top = {s["feature"] for s in derm_singles}

        scin_top: set[str] = set()
        n_scin_cases = 0
        if disease in scin_diseases and scin_label_col in scin_df.columns:
            scin_sub = scin_df[scin_df[scin_label_col] == disease]
            n_scin_cases = len(scin_sub)
            if n_scin_cases >= 10:
                feat_prev = {}
                for f in feature_cols:
                    if f in scin_features.columns:
                        scin_vals = scin_features.loc[scin_sub.index, f]
                        feat_prev[f] = (scin_vals == 1).mean()
                scin_top = set(
                    sorted(feat_prev, key=feat_prev.get, reverse=True)[:top_k]
                )

        overlap = derm_top & scin_top
        union = derm_top | scin_top
        jaccard = len(overlap) / len(union) if union else 0.0
        derm_only = derm_top - scin_top
        scin_only = scin_top - derm_top

        rows.append({
            "disease": disease,
            "n_derm1m_cases": sig.get("n_cases", 0),
            "n_scin_cases": n_scin_cases,
            "derm_top_k": len(derm_top),
            "scin_top_k": len(scin_top),
            "overlap": len(overlap),
            "jaccard_similarity": round(jaccard, 3),
            "derm_only_features": ", ".join(sorted(derm_only)[:5]),
            "scin_only_features": ", ".join(sorted(scin_only)[:5]),
        })

    return pd.DataFrame(rows).sort_values("jaccard_similarity")


def weight_confusion_gaps_by_importance(
    conf_gaps: list[dict],
    mi_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Weight discriminating features in confusion gaps by global MI score
    to prioritize diagnostically valuable gaps.
    """
    if mi_df is None:
        return conf_gaps
    mi_map = dict(zip(mi_df["feature"], mi_df["mutual_information"]))

    for gap in conf_gaps:
        for d in gap.get("discriminating_singles", []):
            feat = d["feature"]
            d["global_mi"] = round(mi_map.get(feat, 0.0), 5)
            d["weighted_lift"] = round(
                d["discriminating_lift"] * (1.0 + mi_map.get(feat, 0.0) * 10), 4
            )
        gap["discriminating_singles"] = sorted(
            gap.get("discriminating_singles", []),
            key=lambda x: -x.get("weighted_lift", x.get("discriminating_lift", 0)),
        )
    return conf_gaps


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


def load_confusion_pairs(default: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Load extra (true_label, confused_with) pairs from CONFUSION_PAIRS_CSV if set."""
    if not CONFUSION_PAIRS_CSV:
        return list(default)
    path = Path(CONFUSION_PAIRS_CSV)
    if not path.is_file():
        print(f"  WARNING: CONFUSION_PAIRS_CSV not found: {path}, using defaults only")
        return list(default)
    extra = pd.read_csv(path)
    for col in ("true_label", "confused_with"):
        if col not in extra.columns:
            raise ValueError(
                f"{path} must have columns true_label, confused_with; got {list(extra.columns)}"
            )
    pairs = list(default)
    for _, row in extra.iterrows():
        a, b = str(row["true_label"]).strip(), str(row["confused_with"]).strip()
        if a and b:
            pairs.append((a, b))
    return pairs

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 3b: Co-occurrence Analysis with Enhanced EDA")
    print("=" * 60 + "\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading feature matrices...")
    derm_df = pd.read_csv(DERM_FEATURES_CSV)
    scin_raw = pd.read_csv(SCIN_CSV)

    # Meta columns to exclude from feature analysis (matches phase2 output)
    META_COLS = {
        "image", "image_path", "label", "label_name", "disease_label",
        "age_numeric", "age_group", "duration_bucket",
        "diagnosis_confidence", "lesion_count",
        "truncated_caption", "caption", "text_caption",
        "extracted_features",
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

    phi_top_k = COOCCURRENCE_PHI_TOP_K if COOCCURRENCE_PHI_TOP_K > 0 else None

    # ── Step 4: Co-occurrence heatmaps for key diseases ───────────────────────
    worst_diseases = summary_df.head(10)["disease"].tolist()
    print(f"\nPlotting co-occurrence heatmaps for "
          f"{len(worst_diseases)} lowest-reproducibility diseases...")
    for disease in worst_diseases:
        phi_df = compute_disease_cooccurrence(
            derm_df, feature_cols, disease, label_col="label_name",
            top_k_features=phi_top_k,
        )
        plot_cooccurrence_heatmap(phi_df, disease)

    # ── Step 5: Confusion-aware gap ───────────────────────────────────────────
    default_confusion_pairs = [
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
    confusion_pairs = load_confusion_pairs(default_confusion_pairs)
    if CONFUSION_PAIRS_CSV:
        print(f"  Loaded confusion pairs from {CONFUSION_PAIRS_CSV} (merged with defaults)")
    print(f"\nComputing confusion-aware gaps for {len(confusion_pairs)} pairs...")
    conf_gaps = confusion_aware_gap(all_signatures, confusion_pairs)

    # Weight confusion gaps by global MI if available
    mi_path = Path("analysis_outputs/feature_importance_global.csv")
    mi_df = None
    if mi_path.exists():
        try:
            mi_df = pd.read_csv(mi_path)
            conf_gaps = weight_confusion_gaps_by_importance(conf_gaps, mi_df)
            print(f"  Weighted confusion gaps by global MI scores from {mi_path}")
        except Exception:
            pass

    with open(OUTPUT_DIR / "confusion_aware_gaps.json", "w") as f:
        json.dump(conf_gaps, f, indent=2)

    for gap in conf_gaps:
        print(f"\n  [{gap['true_label']}] confused with [{gap['confused_with']}]:")
        print(f"    {gap['n_discriminators']} discriminating features, top 3:")
        for d in gap["discriminating_singles"][:3]:
            mi_str = f"  MI={d.get('global_mi', 0):.4f}" if "global_mi" in d else ""
            print(
                f"      {d['feature']:35s}  "
                f"A={d['support_in_A']:.2f}  B={d['support_in_B']:.2f}  "
                f"lift={d['discriminating_lift']:.2f}{mi_str}"
            )
        if gap["discriminating_pairs"]:
            print(f"    Top discriminating pair unique to [{gap['true_label']}]:")
            for p in gap["discriminating_pairs"][:2]:
                print(
                    f"      ({p['feature_1']} + {p['feature_2']})  "
                    f"support={p['joint_support_in_A']:.2f}  "
                    f"phi={p['phi_in_A']:.2f}"
                )

    # ═══════════════════════════════════════════════════════════════════════
    # ENHANCED EDA
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ENHANCED EDA")
    print("=" * 60 + "\n")

    # 6. Generate signature summary table
    print("6. Generating signature summary table...")
    generate_signature_summary_table(all_signatures, all_completeness)
    print()

    # 7. Plot signature networks for top 5 diseases by case count (not dict order)
    print("7. Plotting signature networks for top diseases...")
    top_5_diseases = diseases.head(5).index.tolist()
    for disease in top_5_diseases:
        sig = all_signatures[disease]
        if sig.get("pairs"):
            plot_signature_network(sig, disease, min_phi=0.3)
    print(f"  Generated signature networks for top 5 diseases\n")

    # 8. Plot feature correlation matrices for top diseases
    print("8. Plotting feature correlation matrices for top diseases...")
    for disease in top_5_diseases:
        plot_feature_correlation_matrix(derm_df, feature_cols, disease, top_n=25)
    print(f"  Generated correlation matrices for top 5 diseases\n")

    # 9. Hierarchical disease clustering
    print("9. Computing hierarchical disease clusters...")
    sim_df, disease_clusters = compute_hierarchical_disease_clusters(
        all_signatures, top_k=20
    )
    print(f"  Found {len(disease_clusters)} multi-disease clusters:")
    for ci, cluster in enumerate(disease_clusters):
        print(f"    Cluster {ci + 1}: {', '.join(cluster)}")
    cluster_path = OUTPUT_DIR / "disease_clusters.json"
    with open(cluster_path, "w") as f:
        json.dump(
            {"clusters": disease_clusters, "n_clusters": len(disease_clusters)},
            f, indent=2,
        )
    print()

    # 10. Cross-dataset signature comparison
    print("10. Computing cross-dataset signature comparison (Derm-1M vs SCIN)...")
    cross_comparison = compute_cross_dataset_signature_comparison(
        all_signatures, scin_raw, scin_features, feature_cols, top_k=15
    )
    cross_comparison.to_csv(OUTPUT_DIR / "cross_dataset_signature_comparison.csv", index=False)
    n_with_scin = (cross_comparison["n_scin_cases"] > 0).sum()
    if n_with_scin > 0:
        avg_jaccard = cross_comparison[cross_comparison["n_scin_cases"] > 0]["jaccard_similarity"].mean()
        print(f"  {n_with_scin} diseases have cases in both datasets")
        print(f"  Average signature Jaccard similarity: {avg_jaccard:.3f}")
        worst = cross_comparison[cross_comparison["n_scin_cases"] > 0].head(5)
        print(f"  Lowest overlap diseases:")
        for _, row in worst.iterrows():
            print(f"    {row['disease']:35s} Jaccard={row['jaccard_similarity']:.3f} "
                  f"(Derm={row['n_derm1m_cases']}, SCIN={row['n_scin_cases']})")
    print()

    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print(f"EDA outputs saved to {EDA_DIR}/")
    print("\nKey files:")
    print("   - all_signatures.json               -- full per-disease feature signatures")
    print("   - signature_completeness_summary.csv -- reproducibility scores per disease")
    print("   - reproducibility_summary.png        -- visual summary across all classes")
    print("   - confusion_aware_gaps.json          -- exact missing signal per confused pair")
    print("   - cooccurrence_<disease>.png         -- heatmaps for worst-reproducibility diseases")
    print("   - signature_gap_<disease>.png        -- Derm-1M vs SCIN bar charts per disease")
    print("   - disease_clusters.json              -- hierarchical disease clusters")
    print("   - disease_similarity_matrix.png      -- disease similarity heatmap")
    print("   - cross_dataset_signature_comparison.csv -- Derm-1M vs SCIN signature overlap")
    print("   - eda/signature_network_*.png        -- feature co-occurrence networks")
    print("   - eda/correlation_matrix_*.png       -- feature correlation matrices")
    print("   - eda/signature_summary_all_diseases.csv -- comprehensive signature summary")
