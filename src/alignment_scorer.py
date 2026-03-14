"""
alignment_scorer.py — Compute cosine similarity matrix between objectives and actions.
"""

import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    ACTION_PLAN_FILE,
    THRESHOLD_FAIR,
    ORPHAN_THRESHOLD,
    classify_score,
)


def load_action_plan(filepath: str = ACTION_PLAN_FILE) -> list:
    """Load action items from action_plan.json."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Action plan not found: {filepath}. Run pdf_processor first.")

    actions = data.get("actions", [])
    print(f"✅ Loaded {len(actions)} actions.")
    return actions


def compute_alignment_matrix(objectives: list, actions: list) -> np.ndarray:
    """
    Encode objectives and actions as vectors and compute cosine similarity.

    Returns an ndarray of shape (n_objectives, n_actions).
    """
    print(f"\n📌 Computing alignment matrix ({len(objectives)} × {len(actions)})...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    obj_texts = [o["title"] + " " + o.get("description", "") for o in objectives]
    act_texts = [a["title"] + " " + a.get("description", "") for a in actions]

    print("   Encoding objectives...")
    obj_vectors = model.encode(obj_texts, show_progress_bar=False)

    print("   Encoding actions...")
    act_vectors = model.encode(act_texts, show_progress_bar=False)

    matrix = np.clip(cosine_similarity(obj_vectors, act_vectors), 0.0, 1.0)
    print(f"✅ Matrix shape: {matrix.shape}")
    return matrix


def classify_matrix(matrix: np.ndarray, objectives: list, actions: list) -> list:
    """Return a list of {objective_id, action_id, score, tier} dicts for all pairs."""
    return [
        {
            "objective_id": obj["id"],
            "action_id":    act["id"],
            "score":        round(float(matrix[i][j]), 4),
            "tier":         classify_score(float(matrix[i][j])),
        }
        for i, obj in enumerate(objectives)
        for j, act in enumerate(actions)
    ]


def find_orphan_actions(matrix: np.ndarray, actions: list) -> list:
    """Return actions that score below ORPHAN_THRESHOLD against every objective."""
    orphans = []
    for j, act in enumerate(actions):
        best = float(matrix[:, j].max())
        if best < ORPHAN_THRESHOLD:
            orphans.append(act)
            print(f"   ⚠️  Orphan: [{act['id']}] {act['title']} (best: {best:.3f})")
    return orphans


def print_matrix_summary(matrix: np.ndarray, objectives: list, actions: list) -> None:
    """Print each action's best-matching objective and score to the terminal."""
    print("\n" + "="*70)
    print(f"  {'Action ID':<10} {'Best Score':>11}  {'Tier':<12}  Best Objective")
    print("-"*70)
    for j, act in enumerate(actions):
        best_i = int(matrix[:, j].argmax())
        score  = float(matrix[best_i, j])
        print(f"  {act['id']:<10} {score:>11.3f}  {classify_score(score):<12}  → {objectives[best_i]['id']}")
    print("="*70 + "\n")


def run_alignment(objectives: list) -> dict:
    """
    Full alignment pipeline: load actions → matrix → classify → orphans.

    Returns:
        {overall_score, matrix, classifications, orphan_actions, objectives, actions}
    """
    print("\n" + "="*60)
    print("  RUNNING ALIGNMENT SCORER")
    print("="*60)

    actions         = load_action_plan()
    matrix          = compute_alignment_matrix(objectives, actions)
    classifications = classify_matrix(matrix, objectives, actions)
    orphan_actions  = find_orphan_actions(matrix, actions)

    best_per_action = np.max(matrix, axis=0)
    overall_score   = float(np.mean(best_per_action))

    print_matrix_summary(matrix, objectives, actions)
    print(f"📊 Overall alignment score: {overall_score:.3f}")

    tier_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
    for c in classifications:
        tier_counts[c["tier"]] += 1
    for tier, count in tier_counts.items():
        print(f"     {tier}: {count}")

    return {
        "overall_score":   overall_score,
        "matrix":          matrix.tolist(),
        "classifications": classifications,
        "orphan_actions":  orphan_actions,
        "objectives":      objectives,
        "actions":         actions,
    }
