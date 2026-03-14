"""
alignment_scorer.py
-------------------
Compute cosine-similarity scores between strategic objectives and action items.

How it fits in the pipeline:
  pdf_processor → vector_store → [this module] → rag_engine / dashboard

Beginner note:
  "Cosine similarity" measures the angle between two vectors.
  Score = 1.0  → vectors point the same direction (identical meaning)
  Score = 0.0  → vectors are orthogonal (no relation)
  Score = -1.0 → vectors point opposite directions (opposite meanings)

  We build an (objectives × actions) matrix so we can see at a glance which
  actions support which objectives, and by how much.
"""

# --- Standard library ---
import json

# --- Third-party ---
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- Local ---
from src.config import (
    EMBEDDING_MODEL,
    ACTION_PLAN_FILE,
    THRESHOLD_EXCELLENT,
    THRESHOLD_GOOD,
    THRESHOLD_FAIR,
    ORPHAN_THRESHOLD,
    classify_score,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_action_plan(filepath: str = ACTION_PLAN_FILE) -> list:
    """
    Load action items from the JSON file produced by pdf_processor.

    Args:
        filepath (str): Path to action_plan.json.

    Returns:
        list: List of action dicts with keys id, title, description.

    Raises:
        FileNotFoundError: If action_plan.json has not been generated yet.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Action plan JSON not found: {filepath}\n"
            "Run pdf_processor.process_pdf() first to generate it."
        )

    actions = data.get("actions", [])
    print(f"✅ Loaded {len(actions)} actions from {filepath}")
    return actions


# =============================================================================
# STEP 1: COMPUTE THE SIMILARITY MATRIX
# =============================================================================

def compute_alignment_matrix(objectives: list, actions: list) -> np.ndarray:
    """
    Encode all objectives and actions as vectors, then compute cosine similarity
    for every (objective, action) pair.

    Args:
        objectives (list): List of objective dicts (id, title, description).
        actions (list): List of action dicts (id, title, description).

    Returns:
        np.ndarray: Shape (n_objectives, n_actions).
                    matrix[i][j] = cosine similarity between objective i and action j.
    """
    print(f"\n📌 Step 1: Computing alignment matrix "
          f"({len(objectives)} objectives × {len(actions)} actions)...")

    # Load the embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build text strings: title + description gives the model richer context
    obj_texts = [
        obj["title"] + " " + obj.get("description", "")
        for obj in objectives
    ]
    act_texts = [
        act["title"] + " " + act.get("description", "")
        for act in actions
    ]

    # Encode both lists into vectors
    # encode() returns a 2D numpy array: shape (n_items, embedding_dim)
    print("   Encoding objectives...")
    obj_vectors = model.encode(obj_texts, show_progress_bar=False)

    print("   Encoding actions...")
    act_vectors = model.encode(act_texts, show_progress_bar=False)

    # Compute cosine similarity between every pair.
    # Result shape: (n_objectives, n_actions)
    matrix = cosine_similarity(obj_vectors, act_vectors)

    # Clip to [0, 1] — cosine similarity can technically go negative
    # but for same-domain text it's always >= 0 in practice
    matrix = np.clip(matrix, 0.0, 1.0)

    print(f"✅ Step 1 complete. Matrix shape: {matrix.shape}")
    return matrix


# =============================================================================
# STEP 2: CLASSIFY EACH SCORE INTO A TIER
# =============================================================================

def classify_matrix(matrix: np.ndarray, objectives: list, actions: list) -> list:
    """
    Convert the raw score matrix into a list of labelled (objective, action) pairs.

    Args:
        matrix (np.ndarray): Shape (n_objectives, n_actions).
        objectives (list): List of objective dicts.
        actions (list): List of action dicts.

    Returns:
        list: One dict per pair:
              {objective_id, action_id, score (float), tier (str)}
    """
    classifications = []

    for i, obj in enumerate(objectives):
        for j, act in enumerate(actions):
            score = float(matrix[i][j])
            tier  = classify_score(score)  # "Excellent", "Good", "Fair", or "Poor"

            classifications.append({
                "objective_id": obj["id"],
                "action_id":    act["id"],
                "score":        round(score, 4),
                "tier":         tier,
            })

    return classifications


# =============================================================================
# STEP 3: IDENTIFY ORPHAN ACTIONS
# =============================================================================

def find_orphan_actions(matrix: np.ndarray, actions: list) -> list:
    """
    Find actions that score below ORPHAN_THRESHOLD against EVERY objective.

    These "orphan" actions are operationally active but have no meaningful
    link to any strategic goal — a key finding for decision-makers.

    Args:
        matrix (np.ndarray): Shape (n_objectives, n_actions).
        actions (list): List of action dicts.

    Returns:
        list: Subset of action dicts that are orphans.
    """
    orphans = []

    for j, act in enumerate(actions):
        # max score this action achieves against any objective
        best_score = float(matrix[:, j].max())

        if best_score < ORPHAN_THRESHOLD:
            orphans.append(act)
            print(f"   ⚠️  Orphan action: [{act['id']}] {act['title']} "
                  f"(best score: {best_score:.3f})")

    return orphans


# =============================================================================
# STEP 4: PRINT SUMMARY TABLE
# =============================================================================

def print_matrix_summary(matrix: np.ndarray, objectives: list, actions: list) -> None:
    """
    Print a readable alignment matrix to the terminal so the student
    can inspect results without opening any files.

    Shows each action's best-matching objective and score.

    Args:
        matrix (np.ndarray): Shape (n_objectives, n_actions).
        objectives (list): Objective dicts.
        actions (list): Action dicts.
    """
    print("\n" + "="*70)
    print("  ALIGNMENT MATRIX SUMMARY")
    print("="*70)
    print(f"  {'Action ID':<10} {'Best Score':>11}  {'Tier':<12}  Best Objective")
    print("-"*70)

    for j, act in enumerate(actions):
        # Find which objective this action aligns with best
        best_obj_idx = int(matrix[:, j].argmax())
        best_score   = float(matrix[best_obj_idx, j])
        tier         = classify_score(best_score)
        best_obj_id  = objectives[best_obj_idx]["id"]

        print(f"  {act['id']:<10} {best_score:>11.3f}  {tier:<12}  → {best_obj_id}")

    print("="*70 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_alignment(objectives: list) -> dict:
    """
    Run the full alignment scoring pipeline.

    1. Load action items from data/action_plan.json
    2. Compute cosine similarity matrix
    3. Classify every (objective, action) pair into a tier
    4. Find orphan actions
    5. Print a summary table

    Args:
        objectives (list): Objective dicts loaded from data/strategic_plan.json.

    Returns:
        dict: {
            "overall_score"   (float): Mean of all matrix scores.
            "matrix"          (list):  2D list — matrix[i][j] = score.
            "classifications" (list):  Per-pair dicts with objective_id, action_id,
                                       score, tier.
            "orphan_actions"  (list):  Action dicts with no strategic alignment.
            "objectives"      (list):  The input objectives (passed through).
            "actions"         (list):  The loaded action items.
        }
    """
    print("\n" + "="*60)
    print("  RUNNING ALIGNMENT SCORER")
    print("="*60)

    # Step 1 — Load actions
    actions = load_action_plan()

    # Step 2 — Build similarity matrix
    matrix = compute_alignment_matrix(objectives, actions)

    # Step 3 — Classify all pairs
    print("📌 Step 2: Classifying alignment tiers...")
    classifications = classify_matrix(matrix, objectives, actions)
    print(f"✅ Step 2 complete. {len(classifications)} pairs classified.")

    # Step 4 — Find orphan actions
    print("📌 Step 3: Identifying orphan actions...")
    orphan_actions = find_orphan_actions(matrix, actions)
    print(f"✅ Step 3 complete. {len(orphan_actions)} orphan action(s) found.")

    # Step 5 — Overall score (mean of all matrix values)
    overall_score = float(np.mean(matrix))

    # Print a readable summary to the terminal
    print_matrix_summary(matrix, objectives, actions)

    print(f"📊 Overall alignment score: {overall_score:.3f}")
    print(f"   Tier breakdown:")

    # Count how many pairs fall into each tier
    tier_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
    for c in classifications:
        tier_counts[c["tier"]] += 1
    for tier, count in tier_counts.items():
        print(f"     {tier}: {count} pairs")

    return {
        "overall_score":   overall_score,
        "matrix":          matrix.tolist(),  # convert to plain Python list for JSON serialisability
        "classifications": classifications,
        "orphan_actions":  orphan_actions,
        "objectives":      objectives,
        "actions":         actions,
    }
