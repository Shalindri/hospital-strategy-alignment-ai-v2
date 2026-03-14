"""
Comprehensive Evaluation Module for the ISPS Alignment System.

Compares the ISPS system (sentence-transformer embeddings) against two
baselines — TF-IDF cosine similarity and keyword overlap — using a
58-pair human-annotated ground truth dataset.

Metrics computed per approach:
    - Classification: Precision, Recall, F1, Accuracy, Confusion Matrix
    - Regression:     MAE, RMSE, Pearson r, Spearman rho
    - Ranking:        ROC curve, AUC (binary: aligned vs not-aligned)
    - Statistical:    Paired t-test (system vs each baseline)

Usage::

    python -m tests.evaluation                 # full evaluation
    python -m tests.evaluation --threshold 0.5 # custom classification cutoff
    python -m tests.evaluation --no-plots      # skip chart generation

Outputs (written to ``outputs/evaluation/``)::

    evaluation_results.json   — all metrics in machine-readable form
    comparison_table.csv      — side-by-side metrics table
    roc_curves.png            — ROC curves for all approaches
    method_comparison.png     — grouped bar chart of P/R/F1/Accuracy
    confusion_matrices.png    — side-by-side confusion matrices
    score_correlation.png     — predicted vs ground-truth scatter plots
    mae_comparison.png        — MAE bar chart

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = OUTPUT_DIR / "evaluation"
TESTS_DIR = PROJECT_ROOT / "tests"
GT_FILE = TESTS_DIR / "ground_truth.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBJECTIVE_CODES = ["A", "B", "C", "D", "E"]

# Classification threshold: scores >= this are "aligned", below = "not aligned"
DEFAULT_THRESHOLD = 0.5

# Label mapping for binary classification
# Ground-truth labels:  1.0, 0.7 → aligned (1);  0.4, 0.0 → not aligned (0)
BINARY_THRESHOLD_GT = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_ground_truth() -> list[dict]:
    """Load the human-annotated ground truth pairs."""
    with open(GT_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_alignment_matrix() -> tuple[np.ndarray, list[str], list[int]]:
    """Load the system's alignment similarity matrix.

    Returns:
        (matrix, objective_codes, action_numbers)
        matrix shape: (n_objectives, n_actions)
    """
    with open(DATA_DIR / "alignment_report.json", encoding="utf-8") as f:
        report = json.load(f)
    matrix = np.array(report["alignment_matrix"])
    row_labels = report["matrix_row_labels"]      # ['A', 'B', 'C', 'D', 'E']
    col_labels = report["matrix_col_labels"]       # [1, 2, ..., 25]
    return matrix, row_labels, col_labels


def load_text_data() -> tuple[dict[str, str], dict[int, str]]:
    """Build text representations for objectives and actions.

    Returns:
        (objective_texts, action_texts)
        - objective_texts: {code: concatenated text}
        - action_texts:    {action_number: concatenated text}
    """
    with open(DATA_DIR / "strategic_plan.json", encoding="utf-8") as f:
        sp = json.load(f)
    with open(DATA_DIR / "action_plan.json", encoding="utf-8") as f:
        ap = json.load(f)

    objective_texts: dict[str, str] = {}
    for obj in sp.get("objectives", []):
        parts = [
            obj.get("name", ""),
            obj.get("goal_statement", ""),
        ]
        for g in obj.get("strategic_goals", []):
            if isinstance(g, dict):
                parts.append(g.get("description", ""))
            elif isinstance(g, str):
                parts.append(g)
        for kpi in obj.get("kpis", []):
            if isinstance(kpi, dict):
                parts.append(kpi.get("description", kpi.get("metric", "")))
            elif isinstance(kpi, str):
                parts.append(kpi)
        parts.extend(obj.get("keywords", []))
        objective_texts[obj["code"]] = " ".join(parts)

    action_texts: dict[int, str] = {}
    for act in ap.get("actions", []):
        parts = [
            act.get("title", ""),
            act.get("description", ""),
            act.get("expected_outcome", ""),
        ]
        for kpi in act.get("kpis", []):
            if isinstance(kpi, dict):
                parts.append(kpi.get("description", kpi.get("metric", "")))
            elif isinstance(kpi, str):
                parts.append(kpi)
        parts.extend(act.get("keywords", []))
        action_texts[act["action_number"]] = " ".join(parts)

    return objective_texts, action_texts


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 1: TF-IDF + Cosine Similarity
# ═══════════════════════════════════════════════════════════════════════════

def compute_tfidf_scores(
    objective_texts: dict[str, str],
    action_texts: dict[int, str],
) -> dict[tuple[str, int], float]:
    """Compute TF-IDF cosine similarity for every (objective, action) pair.

    Returns:
        {(obj_code, action_number): cosine_similarity}
    """
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    # Build corpus: objectives first, then actions
    obj_codes = sorted(objective_texts.keys())
    act_nums = sorted(action_texts.keys())

    corpus = [objective_texts[c] for c in obj_codes] + \
             [action_texts[n] for n in act_nums]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    n_obj = len(obj_codes)
    obj_vectors = tfidf_matrix[:n_obj]
    act_vectors = tfidf_matrix[n_obj:]

    sim_matrix = sk_cosine(obj_vectors, act_vectors)

    scores: dict[tuple[str, int], float] = {}
    for i, code in enumerate(obj_codes):
        for j, num in enumerate(act_nums):
            scores[(code, num)] = float(sim_matrix[i, j])

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 2: Keyword Overlap (Jaccard-like)
# ═══════════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"[a-z]{3,}")

# Common English stop words to filter out
_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "from",
    "this", "that", "with", "will", "each", "make", "like", "long",
    "many", "over", "such", "take", "than", "them", "well", "were",
    "into", "year", "your", "some", "could", "about", "other", "which",
    "their", "would", "there", "these", "more", "also", "through",
})


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keyword tokens from text (>= 3 chars, no stops)."""
    words = set(_WORD_RE.findall(text.lower()))
    return words - _STOP_WORDS


def compute_keyword_scores(
    objective_texts: dict[str, str],
    action_texts: dict[int, str],
) -> dict[tuple[str, int], float]:
    """Compute keyword overlap score for every (objective, action) pair.

    Uses a weighted Jaccard similarity:
        score = |intersection| / |union|

    Returns:
        {(obj_code, action_number): overlap_score}
    """
    obj_kw = {code: _extract_keywords(text)
              for code, text in objective_texts.items()}
    act_kw = {num: _extract_keywords(text)
              for num, text in action_texts.items()}

    scores: dict[tuple[str, int], float] = {}
    for code, okw in obj_kw.items():
        for num, akw in act_kw.items():
            intersection = len(okw & akw)
            union = len(okw | akw)
            scores[(code, num)] = intersection / max(union, 1)

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Extract system (embedding) scores for ground-truth pairs
# ═══════════════════════════════════════════════════════════════════════════

def extract_system_scores(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[int],
    gt_pairs: list[dict],
) -> dict[tuple[str, int], float]:
    """Look up the system's embedding similarity for each GT pair.

    Returns:
        {(obj_code, action_number): similarity_score}
    """
    row_idx = {code: i for i, code in enumerate(row_labels)}
    col_idx = {num: j for j, num in enumerate(col_labels)}

    scores: dict[tuple[str, int], float] = {}
    for pair in gt_pairs:
        obj = pair["objective_code"]
        act = pair["action_number"]
        ri = row_idx.get(obj)
        ci = col_idx.get(act)
        if ri is not None and ci is not None:
            scores[(obj, act)] = float(matrix[ri, ci])
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════

def _find_optimal_threshold(gt_binary: np.ndarray, pred_scores: np.ndarray) -> float:
    """Find the threshold that maximises Youden's J (sensitivity + specificity - 1).

    Falls back to median of predicted scores if ROC cannot be computed.
    """
    try:
        fpr, tpr, thresholds = roc_curve(gt_binary, pred_scores)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        # roc_curve thresholds are sorted descending; clamp to valid range
        optimal = float(thresholds[best_idx])
        return max(0.0, min(optimal, float(np.max(pred_scores))))
    except ValueError:
        return float(np.median(pred_scores))


def compute_metrics(
    gt_pairs: list[dict],
    predicted_scores: dict[tuple[str, int], float],
    threshold: float | None = None,
    method_name: str = "System",
) -> dict[str, Any]:
    """Compute full metrics for a single approach.

    Args:
        gt_pairs:         Ground truth records with ``alignment_label``.
        predicted_scores: {(obj, act): score} from the approach.
        threshold:        Binary classification cutoff for predictions.
                          If ``None``, the optimal threshold is chosen
                          automatically via Youden's J statistic on the
                          ROC curve so that baselines with different score
                          ranges are evaluated fairly.
        method_name:      Human-readable label.

    Returns:
        Dict with all metrics, arrays for ROC, etc.
    """
    gt_labels: list[float] = []
    pred_scores_list: list[float] = []
    gt_binary_list: list[int] = []

    for pair in gt_pairs:
        key = (pair["objective_code"], pair["action_number"])
        if key not in predicted_scores:
            continue

        gt_label = pair["alignment_label"]
        pred_score = predicted_scores[key]

        gt_labels.append(gt_label)
        pred_scores_list.append(pred_score)

        # Binary: GT >= 0.5 (i.e. 0.7 and 1.0) → aligned (1)
        #         GT <  0.5 (i.e. 0.0 and 0.4) → not aligned (0)
        gt_binary_list.append(1 if gt_label >= BINARY_THRESHOLD_GT else 0)

    gt_arr = np.array(gt_labels)
    pred_arr = np.array(pred_scores_list)
    gt_bin = np.array(gt_binary_list)

    n = len(gt_arr)
    if n == 0:
        return {"method": method_name, "n_pairs": 0, "error": "No matching pairs"}

    # ── ROC / AUC (compute first to find optimal threshold) ───────────
    try:
        fpr, tpr, roc_thresholds = roc_curve(gt_bin, pred_arr)
        roc_auc = float(auc(fpr, tpr))
    except ValueError:
        fpr, tpr, roc_thresholds = np.array([0, 1]), np.array([0, 1]), np.array([0])
        roc_auc = 0.5

    # Find optimal threshold if not specified
    if threshold is None:
        threshold = _find_optimal_threshold(gt_bin, pred_arr)

    pred_bin = (pred_arr >= threshold).astype(int)

    # ── Classification metrics ────────────────────────────────────────
    prec = float(precision_score(gt_bin, pred_bin, zero_division=0))
    rec = float(recall_score(gt_bin, pred_bin, zero_division=0))
    f1 = float(f1_score(gt_bin, pred_bin, zero_division=0))
    acc = float(accuracy_score(gt_bin, pred_bin))
    cm = confusion_matrix(gt_bin, pred_bin, labels=[0, 1]).tolist()

    # ── Regression metrics ────────────────────────────────────────────
    mae = float(mean_absolute_error(gt_arr, pred_arr))
    rmse = float(np.sqrt(np.mean((gt_arr - pred_arr) ** 2)))

    # Correlation
    if np.std(pred_arr) > 1e-10 and np.std(gt_arr) > 1e-10:
        pearson_r, pearson_p = sp_stats.pearsonr(gt_arr, pred_arr)
        spearman_rho, spearman_p = sp_stats.spearmanr(gt_arr, pred_arr)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = 0.0

    return {
        "method":        method_name,
        "n_pairs":       n,
        "threshold":     round(threshold, 4),
        # Classification
        "precision":     round(prec, 4),
        "recall":        round(rec, 4),
        "f1":            round(f1, 4),
        "accuracy":      round(acc, 4),
        "confusion_matrix": cm,
        # ROC
        "roc_fpr":       [round(float(x), 4) for x in fpr],
        "roc_tpr":       [round(float(x), 4) for x in tpr],
        "roc_auc":       round(roc_auc, 4),
        # Regression
        "mae":           round(mae, 4),
        "rmse":          round(rmse, 4),
        "pearson_r":     round(float(pearson_r), 4),
        "pearson_p":     round(float(pearson_p), 6),
        "spearman_rho":  round(float(spearman_rho), 4),
        "spearman_p":    round(float(spearman_p), 6),
        # Raw arrays (for plotting, not serialised to JSON)
        "_gt_labels":    gt_arr,
        "_pred_scores":  pred_arr,
        "_gt_binary":    gt_bin,
        "_pred_binary":  pred_bin,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Statistical significance testing
# ═══════════════════════════════════════════════════════════════════════════

def paired_comparison(
    gt_pairs: list[dict],
    scores_a: dict[tuple[str, int], float],
    scores_b: dict[tuple[str, int], float],
    name_a: str = "System",
    name_b: str = "Baseline",
) -> dict[str, Any]:
    """Paired t-test comparing absolute errors of two approaches.

    H0: Mean absolute error of approach A = mean absolute error of approach B.

    Returns:
        Dict with t-statistic, p-value, and interpretation.
    """
    errors_a: list[float] = []
    errors_b: list[float] = []

    for pair in gt_pairs:
        key = (pair["objective_code"], pair["action_number"])
        if key in scores_a and key in scores_b:
            gt = pair["alignment_label"]
            errors_a.append(abs(gt - scores_a[key]))
            errors_b.append(abs(gt - scores_b[key]))

    if len(errors_a) < 2:
        return {
            "comparison": f"{name_a} vs {name_b}",
            "error": "Insufficient paired samples",
        }

    arr_a = np.array(errors_a)
    arr_b = np.array(errors_b)

    t_stat, p_value = sp_stats.ttest_rel(arr_a, arr_b)

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))

    if p_value < 0.01:
        significance = "highly significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    elif p_value < 0.10:
        significance = "marginally significant (p < 0.10)"
    else:
        significance = "not significant (p >= 0.10)"

    if mean_a < mean_b:
        winner = name_a
    elif mean_b < mean_a:
        winner = name_b
    else:
        winner = "tie"

    return {
        "comparison":       f"{name_a} vs {name_b}",
        "n_pairs":          len(errors_a),
        "mean_abs_error_a": round(mean_a, 4),
        "mean_abs_error_b": round(mean_b, 4),
        "t_statistic":      round(float(t_stat), 4),
        "p_value":          round(float(p_value), 6),
        "significance":     significance,
        "lower_error":      winner,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(
    all_metrics: list[dict],
    stat_tests: list[dict],
    output_dir: Path,
) -> list[str]:
    """Generate all evaluation charts as PNG files.

    Returns:
        List of generated file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots.")
        return []

    generated: list[str] = []
    colours = ["#1565C0", "#2E7D32", "#FF8F00"]

    # ── 1. ROC Curves ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, m in enumerate(all_metrics):
        ax.plot(
            m["roc_fpr"], m["roc_tpr"],
            color=colours[i % len(colours)],
            linewidth=2,
            label=f"{m['method']} (AUC={m['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Alignment Detection", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    path = output_dir / "roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    # ── 2. Method Comparison Bar Chart ────────────────────────────────
    methods = [m["method"] for m in all_metrics]
    metrics_names = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(methods))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colours = ["#1565C0", "#2E7D32", "#FF8F00", "#7B1FA2"]
    for j, metric in enumerate(metrics_names):
        vals = [m[metric] for m in all_metrics]
        bars = ax.bar(x + j * width, vals, width, label=metric.capitalize(),
                       color=bar_colours[j], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Classification Metrics Comparison", fontsize=13)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    path = output_dir / "method_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    # ── 3. Confusion Matrices ─────────────────────────────────────────
    n_methods = len(all_metrics)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]
    for ax, m, col in zip(axes, all_metrics, colours):
        cm = np.array(m["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Aligned", "Aligned"], fontsize=8)
        ax.set_yticklabels(["Not Aligned", "Aligned"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title(m["method"], fontsize=10, color=col)
        for (ii, jj), val in np.ndenumerate(cm):
            ax.text(jj, ii, str(val), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if val > cm.max() / 2 else "black")
    fig.suptitle("Confusion Matrices", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    # ── 4. Score Correlation Scatter ──────────────────────────────────
    fig, axes = plt.subplots(1, n_methods, figsize=(4.5 * n_methods, 4.5))
    if n_methods == 1:
        axes = [axes]
    for ax, m, col in zip(axes, all_metrics, colours):
        gt = m["_gt_labels"]
        pred = m["_pred_scores"]
        ax.scatter(gt, pred, alpha=0.6, color=col, edgecolor="white", s=50)
        # Perfect prediction line
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
        # Regression line
        if len(gt) > 2 and np.std(gt) > 0 and np.std(pred) > 0:
            z = np.polyfit(gt, pred, 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(0, 1, 50)
            ax.plot(x_range, p_line(x_range), color=col, linewidth=1.5,
                    linestyle="-", alpha=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Ground Truth Label", fontsize=9)
        ax.set_ylabel("Predicted Score", fontsize=9)
        ax.set_title(
            f"{m['method']}\nr={m['pearson_r']:.3f}, MAE={m['mae']:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.3)
    fig.suptitle("Predicted vs Ground Truth Scores", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "score_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    # ── 5. MAE / RMSE Comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    width = 0.3
    mae_vals = [m["mae"] for m in all_metrics]
    rmse_vals = [m["rmse"] for m in all_metrics]
    bars1 = ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#1565C0")
    bars2 = ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#FF8F00")
    for bar, v in zip(list(bars1) + list(bars2),
                      mae_vals + rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Error", fontsize=11)
    ax.set_title("Mean Absolute Error & RMSE", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    path = output_dir / "mae_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    logger.info("Generated %d evaluation plots in %s", len(generated), output_dir)
    return generated


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def build_comparison_table(all_metrics: list[dict]) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame."""
    rows = []
    for m in all_metrics:
        rows.append({
            "Method":       m["method"],
            "Threshold":    m["threshold"],
            "Precision":    m["precision"],
            "Recall":       m["recall"],
            "F1":           m["f1"],
            "Accuracy":     m["accuracy"],
            "AUC":          m["roc_auc"],
            "MAE":          m["mae"],
            "RMSE":         m["rmse"],
            "Pearson r":    m["pearson_r"],
            "Spearman rho": m["spearman_rho"],
            "N Pairs":      m["n_pairs"],
        })
    return pd.DataFrame(rows)


def serialise_results(
    all_metrics: list[dict],
    stat_tests: list[dict],
    threshold: float,
) -> dict:
    """Build a JSON-serialisable results dict (strip numpy arrays)."""
    clean_metrics = []
    for m in all_metrics:
        clean = {k: v for k, v in m.items() if not k.startswith("_")}
        clean_metrics.append(clean)

    return {
        "evaluation_config": {
            "classification_threshold": threshold,
            "binary_gt_threshold": BINARY_THRESHOLD_GT,
            "ground_truth_file": str(GT_FILE),
            "n_ground_truth_pairs": len(load_ground_truth()),
        },
        "method_results": clean_metrics,
        "statistical_tests": stat_tests,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    threshold: float = DEFAULT_THRESHOLD,
    generate_plots_flag: bool = True,
) -> dict:
    """Execute the full evaluation pipeline.

    Steps:
        1. Load ground truth and system predictions
        2. Compute TF-IDF and keyword baselines
        3. Calculate metrics for each approach
        4. Run statistical significance tests
        5. Generate charts and export results

    Returns:
        The full results dict (also saved to disk).
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading ground truth and system predictions...")
    gt_pairs = load_ground_truth()
    matrix, row_labels, col_labels = load_alignment_matrix()
    obj_texts, act_texts = load_text_data()

    print(f"  Ground truth: {len(gt_pairs)} labelled pairs")
    gt_dist = Counter(p["alignment_label"] for p in gt_pairs)
    for label in sorted(gt_dist.keys(), reverse=True):
        print(f"    {label:.1f} ({SCORE_TO_TEXT.get(label, '?'):<18}): {gt_dist[label]}")

    # ── System scores (sentence-transformers) ─────────────────────────
    print("\nExtracting system (sentence-transformer) scores...")
    system_scores = extract_system_scores(matrix, row_labels, col_labels, gt_pairs)
    print(f"  {len(system_scores)} pairs matched.")

    # ── Baseline 1: TF-IDF ────────────────────────────────────────────
    print("Computing TF-IDF baseline...")
    tfidf_scores = compute_tfidf_scores(obj_texts, act_texts)
    print(f"  {len(tfidf_scores)} total pairs; "
          f"{sum(1 for p in gt_pairs if (p['objective_code'], p['action_number']) in tfidf_scores)} matched to GT.")

    # ── Baseline 2: Keyword overlap ───────────────────────────────────
    print("Computing keyword overlap baseline...")
    keyword_scores = compute_keyword_scores(obj_texts, act_texts)
    print(f"  {len(keyword_scores)} total pairs; "
          f"{sum(1 for p in gt_pairs if (p['objective_code'], p['action_number']) in keyword_scores)} matched to GT.")

    # ── Compute metrics ───────────────────────────────────────────────
    # Use per-method optimal thresholds (Youden's J) so baselines with
    # different score ranges are evaluated fairly.  The --threshold flag
    # overrides this for the ISPS system only; baselines always auto-tune.
    print(f"\nComputing metrics...")
    all_methods = [
        ("ISPS (Sentence-Transformers)", system_scores,  threshold),
        ("TF-IDF Cosine Similarity",     tfidf_scores,   None),
        ("Keyword Overlap (Jaccard)",     keyword_scores, None),
    ]

    all_metrics: list[dict] = []
    for name, scores, thr in all_methods:
        m = compute_metrics(gt_pairs, scores, threshold=thr, method_name=name)
        all_metrics.append(m)
        print(f"\n  {name} (threshold={m['threshold']:.4f}):")
        print(f"    P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}")
        print(f"    AUC={m['roc_auc']:.3f}  MAE={m['mae']:.3f}  "
              f"RMSE={m['rmse']:.3f}")
        print(f"    Pearson r={m['pearson_r']:.3f} (p={m['pearson_p']:.4f})  "
              f"Spearman rho={m['spearman_rho']:.3f}")

    # ── Statistical significance tests ────────────────────────────────
    print("\nStatistical significance tests (paired t-test on absolute errors):")
    stat_tests = []
    system_name, system_scores_dict, _ = all_methods[0]
    for name, scores, _ in all_methods[1:]:
        test = paired_comparison(
            gt_pairs, system_scores_dict, scores,
            name_a=system_name, name_b=name,
        )
        stat_tests.append(test)
        print(f"\n  {test['comparison']}:")
        print(f"    MAE(system)={test.get('mean_abs_error_a', '?'):.4f}  "
              f"MAE(baseline)={test.get('mean_abs_error_b', '?'):.4f}")
        print(f"    t={test.get('t_statistic', '?'):.4f}  "
              f"p={test.get('p_value', '?'):.6f}")
        print(f"    → {test.get('significance', '?')} "
              f"(lower error: {test.get('lower_error', '?')})")

    # ── Comparison table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    df_compare = build_comparison_table(all_metrics)
    print(df_compare.to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = EVAL_DIR / "comparison_table.csv"
    df_compare.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Save JSON ─────────────────────────────────────────────────────
    results = serialise_results(all_metrics, stat_tests, threshold)
    json_path = EVAL_DIR / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # ── Generate plots ────────────────────────────────────────────────
    if generate_plots_flag:
        print("\nGenerating evaluation plots...")
        plots = generate_plots(all_metrics, stat_tests, EVAL_DIR)
        for p in plots:
            print(f"  Saved: {p}")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Label text lookup (used by ground truth builder)
# ═══════════════════════════════════════════════════════════════════════════

SCORE_TO_TEXT: dict[float, str] = {
    1.0: "Strongly Aligned",
    0.7: "Aligned",
    0.4: "Weakly Aligned",
    0.0: "Not Aligned",
}


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ISPS alignment system against baselines.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Binary classification threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation (faster, no matplotlib needed).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_evaluation(
        threshold=args.threshold,
        generate_plots_flag=not args.no_plots,
    )


if __name__ == "__main__":
    main()
