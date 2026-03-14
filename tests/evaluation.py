"""
evaluation.py
-------------
Evaluate the ISPS alignment scorer against a human-annotated ground truth dataset.

How it fits in the pipeline:
  alignment_scorer → [this module] → outputs/evaluation_results.json

Beginner note:
  To know if our system works well, we compare its predictions against labels
  made by a human expert (the "ground truth"). Key metrics:

    Precision  = Of all pairs our system called "aligned", what fraction truly are?
    Recall     = Of all truly aligned pairs, what fraction did our system find?
    F1         = Balanced combination of Precision and Recall (harmonic mean)
    AUC (ROC)  = How well our system ranks aligned pairs above non-aligned ones
                 (0.5 = random, 1.0 = perfect)
    Pearson r  = Linear correlation between our continuous scores and human labels

  Ground truth file format (58 pairs):
    {objective_code, objective_name, action_number, action_title,
     alignment_label (float 0-1), label_text}
"""

# --- Standard library ---
import json
import os
import sys

# Add project root to path so this can be run directly with: python tests/evaluation.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Third-party ---
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from scipy.stats import pearsonr

# --- Local ---
from src.config import THRESHOLD_FAIR, OUTPUTS_DIR, STRATEGIC_PLAN_FILE, ACTION_PLAN_FILE
from src import alignment_scorer

GROUND_TRUTH_FILE = os.path.join("tests", "ground_truth.json")

# Threshold to binarise ground-truth labels:
# alignment_label >= GT_BINARY_THRESHOLD → label = 1 (aligned)
GT_BINARY_THRESHOLD = 0.5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ground_truth(filepath: str = GROUND_TRUTH_FILE) -> list:
    """
    Load the human-annotated ground truth dataset.

    Each entry has:
      objective_code (str), objective_name (str),
      action_number (int), action_title (str),
      alignment_label (float 0-1), label_text (str)

    Args:
        filepath (str): Path to ground_truth.json.

    Returns:
        list: Ground truth pairs.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Ground truth file not found: {filepath}\n"
            "Expected 58-pair annotated dataset at tests/ground_truth.json."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} ground truth pairs from {filepath}.")
    return data


def load_data_for_evaluation() -> tuple:
    """
    Load strategic plan objectives and action plan items, normalising the
    existing JSON format into the simple {id, title, description} format
    that alignment_scorer expects.

    The existing data files use:
      Objectives: code (A/B/C), name, goal_statement
      Actions:    action_number (1/2/3), title, description

    We convert to:
      Objectives: id="A",  title=name,  description=goal_statement
      Actions:    id="1",  title=title, description=description

    Returns:
        tuple: (objectives list, actions list)
    """
    # Load strategic plan
    if not os.path.exists(STRATEGIC_PLAN_FILE):
        raise FileNotFoundError(
            f"Strategic plan JSON not found: {STRATEGIC_PLAN_FILE}\n"
            "Run pdf_processor.process_pdf() first to generate it."
        )

    with open(STRATEGIC_PLAN_FILE, "r", encoding="utf-8") as f:
        sp_data = json.load(f)

    raw_objectives = sp_data.get("objectives", [])
    objectives = []
    for obj in raw_objectives:
        if "id" in obj:
            # New simplified format (created by our pdf_processor) — use as-is
            objectives.append(obj)
        elif "code" in obj:
            # Old rich format (pre-existing data) — convert
            objectives.append({
                "id":          obj["code"],
                "title":       obj.get("name", obj["code"]),
                "description": obj.get("goal_statement", ""),
            })

    # Load action plan
    if not os.path.exists(ACTION_PLAN_FILE):
        raise FileNotFoundError(
            f"Action plan JSON not found: {ACTION_PLAN_FILE}\n"
            "Run pdf_processor.process_pdf() first to generate it."
        )

    with open(ACTION_PLAN_FILE, "r", encoding="utf-8") as f:
        ap_data = json.load(f)

    raw_actions = ap_data.get("actions", [])
    actions = []
    for act in raw_actions:
        if "id" in act:
            # New simplified format — use as-is
            actions.append(act)
        elif "action_number" in act:
            # Old rich format — convert
            actions.append({
                "id":          str(act["action_number"]),
                "title":       act.get("title", str(act["action_number"])),
                "description": act.get("description", ""),
            })

    print(f"Loaded {len(objectives)} objectives and {len(actions)} actions.")
    return objectives, actions


# =============================================================================
# MATCH GROUND TRUTH TO PREDICTIONS
# =============================================================================

def get_predicted_scores(ground_truth: list, alignment_result: dict) -> tuple:
    """
    For each ground-truth pair, look up the system's predicted alignment score.

    Matching uses: objective_code → objective id, action_number → action id.

    Args:
        ground_truth (list): 58-pair annotated dataset.
        alignment_result (dict): Output of alignment_scorer.run_alignment().

    Returns:
        tuple: (y_true: list[float], y_scores: list[float])
               y_true   = human alignment labels (0-1 continuous)
               y_scores = system cosine similarity scores (0-1 continuous)
    """
    # Build lookup: (objective_id, action_id) -> score
    score_lookup: dict = {}
    for c in alignment_result["classifications"]:
        key = (str(c["objective_id"]), str(c["action_id"]))
        score_lookup[key] = c["score"]

    y_true   = []
    y_scores = []
    matched  = 0
    missed   = 0

    for pair in ground_truth:
        obj_key = str(pair.get("objective_code", pair.get("objective_id", "")))
        act_key = str(pair.get("action_number",  pair.get("action_id",    "")))

        true_label      = float(pair.get("alignment_label", pair.get("label", 0.0)))
        predicted_score = score_lookup.get((obj_key, act_key), None)

        if predicted_score is None:
            predicted_score = 0.0
            missed += 1
        else:
            matched += 1

        y_true.append(true_label)
        y_scores.append(float(predicted_score))

    print(f"   Matched {matched}/{len(ground_truth)} pairs "
          f"({missed} not found — defaulted to score 0.0).")
    return y_true, y_scores


# =============================================================================
# THRESHOLD SWEEP — find the optimal binarisation cutoff
# =============================================================================

def find_optimal_threshold(y_true: list, y_scores: list,
                            lo: float = 0.25, hi: float = 0.70,
                            step: float = 0.02) -> dict:
    """
    Try every threshold between lo and hi (in steps of step) and return
    the one that gives the highest F1 score.

    This tells you the ideal THRESHOLD_FAIR value for your data and model.

    Args:
        y_true (list[float]): Ground-truth continuous labels (0-1).
        y_scores (list[float]): Predicted cosine similarity scores (0-1).
        lo (float): Lowest threshold to try. Default 0.25.
        hi (float): Highest threshold to try. Default 0.70.
        step (float): Step size between candidates. Default 0.02.

    Returns:
        dict: {
            "best_threshold": float,
            "best_f1":        float,
            "best_precision": float,
            "best_recall":    float,
            "sweep_table":    list of {threshold, precision, recall, f1}
        }
    """
    y_true_binary = [1 if t >= GT_BINARY_THRESHOLD else 0 for t in y_true]

    sweep_table = []
    best = {"threshold": lo, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Try every candidate threshold
    threshold = lo
    while threshold <= hi + 1e-9:
        t = round(threshold, 4)
        y_pred = [1 if s >= t else 0 for s in y_scores]

        p  = precision_score(y_true_binary, y_pred, zero_division=0)
        r  = recall_score(y_true_binary, y_pred, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred, zero_division=0)

        sweep_table.append({"threshold": t, "precision": round(p, 4),
                             "recall": round(r, 4), "f1": round(f1, 4)})

        if f1 > best["f1"]:
            best = {"threshold": t, "f1": round(f1, 4),
                    "precision": round(p, 4), "recall": round(r, 4)}

        threshold += step

    # Print sweep table
    print("\n  Threshold sweep (THRESHOLD_FAIR candidates):")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-"*42)
    for row in sweep_table:
        marker = " ◄ BEST" if row["threshold"] == best["threshold"] else ""
        print(f"  {row['threshold']:>10.2f}  {row['precision']:>10.4f}  "
              f"{row['recall']:>8.4f}  {row['f1']:>8.4f}{marker}")

    print(f"\n  ✅ Optimal THRESHOLD_FAIR = {best['threshold']} "
          f"(F1={best['f1']:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f})")
    print(f"  → Update THRESHOLD_FAIR in src/config.py to {best['threshold']}")

    return {
        "best_threshold": best["threshold"],
        "best_f1":        best["f1"],
        "best_precision": best["precision"],
        "best_recall":    best["recall"],
        "sweep_table":    sweep_table,
    }


# =============================================================================
# COMPUTE METRICS
# =============================================================================

def compute_metrics(y_true: list, y_scores: list) -> dict:
    """
    Compute Precision, Recall, F1, AUC (ROC), and Pearson r.

    Binarisation:
      Ground truth:  alignment_label >= 0.5  → 1 (aligned)
      Predictions:   cosine score   >= THRESHOLD_FAIR → 1 (aligned)

    Args:
        y_true (list[float]): Ground-truth continuous labels (0-1).
        y_scores (list[float]): System's predicted cosine similarity scores (0-1).

    Returns:
        dict: Keys: precision, recall, f1, auc, pearson_r, pearson_p,
              y_true_binary, y_pred_binary, confusion_matrix.
    """
    y_true_binary = [1 if t >= GT_BINARY_THRESHOLD else 0 for t in y_true]
    y_pred_binary = [1 if s >= THRESHOLD_FAIR      else 0 for s in y_scores]

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall    = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1        = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    try:
        auc = roc_auc_score(y_true_binary, y_scores)
    except ValueError:
        auc = 0.5
        print("   AUC could not be computed (only one class in ground truth).")

    try:
        pearson_r, pearson_p = pearsonr(y_true, y_scores)
    except Exception:
        pearson_r, pearson_p = 0.0, 1.0

    cm = confusion_matrix(y_true_binary, y_pred_binary).tolist()

    return {
        "precision":        round(float(precision), 4),
        "recall":           round(float(recall), 4),
        "f1":               round(float(f1), 4),
        "auc":              round(float(auc), 4),
        "pearson_r":        round(float(pearson_r), 4),
        "pearson_p":        round(float(pearson_p), 4),
        "y_true_binary":    y_true_binary,
        "y_pred_binary":    y_pred_binary,
        "confusion_matrix": cm,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_evaluation(objectives: list = None) -> dict:
    """
    Full evaluation pipeline: load data -> run alignment -> compute metrics -> save.

    Args:
        objectives (list, optional): Pre-loaded objectives. If None, loads from file.

    Returns:
        dict: Evaluation metrics and confusion matrix.
    """
    print("\n" + "="*60)
    print("  RUNNING EVALUATION")
    print("="*60)

    # Step 1 — Load ground truth
    print("📌 Step 1: Loading ground truth...")
    ground_truth = load_ground_truth()

    # Step 2 — Load objectives if not provided
    if objectives is None:
        print("📌 Step 1b: Loading data files...")
        objectives, _ = load_data_for_evaluation()

    # Step 3 — Run alignment scorer
    print("\n📌 Step 2: Running alignment scorer...")
    alignment_result = alignment_scorer.run_alignment(objectives)

    # Step 4 — Match GT to predictions
    print("\n📌 Step 3: Matching ground truth pairs to predictions...")
    y_true, y_scores = get_predicted_scores(ground_truth, alignment_result)

    # Step 5 — Run threshold sweep to find optimal cutoff
    print("\n📌 Step 4: Finding optimal threshold...")
    sweep = find_optimal_threshold(y_true, y_scores)

    # Step 5b — Compute final metrics using the current THRESHOLD_FAIR
    print(f"\n📌 Step 5: Computing metrics at current THRESHOLD_FAIR={THRESHOLD_FAIR}...")
    metrics = compute_metrics(y_true, y_scores)
    metrics["optimal_threshold"] = sweep["best_threshold"]
    metrics["optimal_f1"]        = sweep["best_f1"]
    metrics["sweep_table"]       = sweep["sweep_table"]

    # Step 6 — Print summary table
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"  {'Metric':<22} {'Value':>8}")
    print("-"*35)
    print(f"  {'Precision':<22} {metrics['precision']:>8.4f}")
    print(f"  {'Recall':<22} {metrics['recall']:>8.4f}")
    print(f"  {'F1 Score':<22} {metrics['f1']:>8.4f}")
    print(f"  {'AUC (ROC)':<22} {metrics['auc']:>8.4f}")
    print(f"  {'Pearson r':<22} {metrics['pearson_r']:>8.4f}")
    print(f"  {'Pearson p-value':<22} {metrics['pearson_p']:>8.4f}")
    print("="*60)

    cm = metrics["confusion_matrix"]
    if len(cm) >= 2 and len(cm[0]) >= 2:
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"    True Not Aligned  → Pred Not: {cm[0][0]:>4}  | Pred Yes: {cm[0][1]:>4}")
        print(f"    True Aligned      → Pred Not: {cm[1][0]:>4}  | Pred Yes: {cm[1][1]:>4}")

    # Step 7 — Save results
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, "evaluation_results.json")

    # Build save-friendly dict (strip the binary lists from the saved file)
    save_data = {
        "precision":             metrics["precision"],
        "recall":                metrics["recall"],
        "f1":                    metrics["f1"],
        "auc":                   metrics["auc"],
        "pearson_r":             metrics["pearson_r"],
        "pearson_p":             metrics["pearson_p"],
        "confusion_matrix":      metrics["confusion_matrix"],
        "n_pairs":               len(y_true),
        "gt_binary_threshold":   GT_BINARY_THRESHOLD,
        "pred_binary_threshold": THRESHOLD_FAIR,
        "optimal_threshold":     metrics.get("optimal_threshold", THRESHOLD_FAIR),
        "optimal_f1":            metrics.get("optimal_f1", metrics["f1"]),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n💾 Results saved to {output_path}")
    print("\n✅ Evaluation complete.")
    return metrics


# =============================================================================
# RUN DIRECTLY
# =============================================================================
if __name__ == "__main__":
    run_evaluation()
