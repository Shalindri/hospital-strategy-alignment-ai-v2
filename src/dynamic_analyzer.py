"""
Dynamic Analyzer for uploaded documents.

Runs the alignment pipeline on in-memory parsed data (from PDF uploads)
using a temporary ChromaDB instance so the main vector store is not
affected.

Typical usage (from the Streamlit dashboard)::

    from src.dynamic_analyzer import run_dynamic_analysis

    report = run_dynamic_analysis(strategic_data, action_data)
    # report is a dict with overall_score, alignment_matrix, etc.

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.vector_store import (
    VectorStore,
    STRATEGIC_COLLECTION,
    ACTION_COLLECTION,
    _compose_objective_text,
    _compose_action_text,
    _compose_objective_metadata,
    _compose_action_metadata,
)
from src.synchronization_analyzer import (
    SynchronizationAnalyzer,
    ActionAlignment,
    ObjectiveAlignment,
    SynchronizationReport,
    THRESHOLD_EXCELLENT,
    THRESHOLD_GOOD,
    THRESHOLD_FAIR,
    ORPHAN_THRESHOLD,
    _classify_score,
)

logger = logging.getLogger("dynamic_analyzer")


def _build_objective_text_generic(obj: dict) -> str:
    """Build embedding text from an objective dict (flexible schema).

    Handles both the full schema from document_processor.py and the
    simpler schema from LLM-based PDF extraction.
    """
    parts: list[str] = []
    if obj.get("goal_statement"):
        parts.append(obj["goal_statement"])
    if obj.get("name"):
        parts.append(obj["name"])
    for goal in obj.get("strategic_goals", []):
        desc = goal.get("description", "") if isinstance(goal, dict) else str(goal)
        if desc:
            parts.append(desc)
    for kpi in obj.get("kpis", []):
        if isinstance(kpi, dict):
            parts.append(kpi.get("KPI", kpi.get("kpi", "")))
        elif isinstance(kpi, str):
            parts.append(kpi)
    return " ".join(p for p in parts if p)


def _build_action_text_generic(action: dict) -> str:
    """Build embedding text from an action dict (flexible schema)."""
    parts: list[str] = [
        action.get("title", ""),
        action.get("description", ""),
        action.get("expected_outcome", ""),
    ]
    for kpi in action.get("kpis", []):
        if isinstance(kpi, str):
            parts.append(kpi)
        elif isinstance(kpi, dict):
            parts.append(kpi.get("KPI", ""))
    return " ".join(p for p in parts if p)


def run_dynamic_analysis(
    strategic_data: dict[str, Any],
    action_data: dict[str, Any],
) -> dict[str, Any]:
    """Run the full alignment analysis on uploaded document data.

    Creates a temporary ChromaDB, embeds the objectives and actions,
    computes the similarity matrix, and returns the alignment report.

    Args:
        strategic_data: Parsed strategic plan (must have 'objectives' key).
        action_data:    Parsed action plan (must have 'actions' key).

    Returns:
        A dictionary containing the alignment report with keys:
        overall_score, overall_classification, alignment_matrix,
        objective_alignments, action_alignments, etc.
    """
    objectives = strategic_data.get("objectives", [])
    actions = action_data.get("actions", [])

    if not objectives:
        raise ValueError("Strategic plan has no objectives.")
    if not actions:
        raise ValueError("Action plan has no actions.")

    logger.info("Dynamic analysis: %d objectives × %d actions",
                len(objectives), len(actions))

    # Create temporary ChromaDB directory
    temp_dir = tempfile.mkdtemp(prefix="isps_dynamic_")
    try:
        # Initialize vector store in temp directory
        vs = VectorStore(chroma_dir=temp_dir)

        # Embed objectives
        obj_texts: list[str] = []
        obj_ids: list[str] = []
        obj_metas: list[dict] = []

        for obj in objectives:
            text = _build_objective_text_generic(obj)
            obj_texts.append(text)
            code = obj.get("code", chr(65 + len(obj_ids)))  # A, B, C, ...
            obj_ids.append(f"obj_{code}")
            obj_metas.append({
                "code": code,
                "name": obj.get("name", f"Objective {code}"),
                "goal_statement": obj.get("goal_statement", "")[:500],
                "num_goals": len(obj.get("strategic_goals", [])),
                "num_kpis": len(obj.get("kpis", [])),
                "keywords": ", ".join(obj.get("keywords", [])),
            })

        vs.embed_documents(
            texts=obj_texts,
            collection_name=STRATEGIC_COLLECTION,
            ids=obj_ids,
            metadatas=obj_metas,
        )

        # Embed actions
        act_texts: list[str] = []
        act_ids: list[str] = []
        act_metas: list[dict] = []

        for act in actions:
            text = _build_action_text_generic(act)
            act_texts.append(text)
            num = act.get("action_number", len(act_ids) + 1)
            act_ids.append(f"action_{num}")
            act_metas.append({
                "action_number": num,
                "title": act.get("title", f"Action {num}"),
                "strategic_objective_code": act.get("strategic_objective_code", ""),
                "strategic_objective_name": act.get("strategic_objective_name", ""),
                "action_owner": act.get("action_owner", ""),
                "budget_lkr_millions": act.get("budget_lkr_millions", 0.0),
                "timeline": act.get("timeline", ""),
                "quarters": ", ".join(act.get("quarters", [])),
                "keywords": ", ".join(act.get("keywords", [])),
            })

        vs.embed_documents(
            texts=act_texts,
            collection_name=ACTION_COLLECTION,
            ids=act_ids,
            metadatas=act_metas,
        )

        # Compute similarity matrix
        obj_emb_data = vs.get_all_embeddings(STRATEGIC_COLLECTION)
        act_emb_data = vs.get_all_embeddings(ACTION_COLLECTION)

        obj_emb = np.array(obj_emb_data["embeddings"])
        act_emb = np.array(act_emb_data["embeddings"])

        matrix = np.clip(obj_emb @ act_emb.T, -1.0, 1.0)

        # Build codes and numbers
        obj_codes = [oid.replace("obj_", "") for oid in obj_emb_data["ids"]]
        act_numbers = [int(aid.replace("action_", "")) for aid in act_emb_data["ids"]]

        obj_idx = {code: i for i, code in enumerate(obj_codes)}
        act_idx = {num: j for j, num in enumerate(act_numbers)}

        # Build action lookup
        action_lookup = {a.get("action_number", i + 1): a for i, a in enumerate(actions)}

        # Build objective name map
        obj_names = {}
        for obj in objectives:
            code = obj.get("code", "")
            obj_names[code] = obj.get("name", f"Objective {code}")

        # Per-action alignment
        action_alignments: list[dict] = []
        for j, act_num in enumerate(act_numbers):
            act_meta = action_lookup.get(act_num, {})
            sims = {code: round(float(matrix[i, j]), 4) for code, i in obj_idx.items()}
            best_code = max(sims, key=sims.get)
            best_score = sims[best_code]
            declared = act_meta.get("strategic_objective_code", "")

            action_alignments.append({
                "action_id": f"action_{act_num}",
                "action_number": act_num,
                "title": act_meta.get("title", f"Action {act_num}"),
                "declared_objective": declared,
                "action_owner": act_meta.get("action_owner", ""),
                "budget_lkr_millions": act_meta.get("budget_lkr_millions", 0.0),
                "similarities": sims,
                "best_objective": best_code,
                "best_score": round(best_score, 4),
                "classification": _classify_score(best_score),
                "is_orphan": best_score < ORPHAN_THRESHOLD,
                "declared_match": best_code == declared if declared else True,
            })

        # Per-objective alignment
        objective_alignments: list[dict] = []
        for code in obj_codes:
            i = obj_idx[code]
            row = matrix[i, :]
            top5_indices = np.argsort(row)[::-1][:5]
            top_actions = [(act_numbers[j], round(float(row[j]), 4)) for j in top5_indices]

            aligned_count = int((row >= THRESHOLD_FAIR).sum())
            declared_nums = [a["action_number"] for a in action_alignments
                             if a["declared_objective"] == code]

            gap_actions = []
            for act_num in declared_nums:
                j = act_idx.get(act_num)
                if j is not None and row[j] < THRESHOLD_FAIR:
                    gap_actions.append(act_num)

            objective_alignments.append({
                "code": code,
                "name": obj_names.get(code, code),
                "top_actions": top_actions,
                "mean_similarity": round(float(row.mean()), 4),
                "max_similarity": round(float(row.max()), 4),
                "aligned_action_count": aligned_count,
                "declared_action_count": len(declared_nums),
                "coverage_score": round(aligned_count / len(act_numbers), 4) if act_numbers else 0,
                "gap_actions": gap_actions,
            })

        # Aggregate
        best_scores = np.array([a["best_score"] for a in action_alignments])
        overall_score = round(float(best_scores.mean()), 4)
        flat = matrix.flatten()

        distribution = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
        for a in action_alignments:
            distribution[a["classification"]] += 1

        report = {
            "overall_score": overall_score,
            "overall_classification": _classify_score(overall_score),
            "mean_similarity": round(float(flat.mean()), 4),
            "std_similarity": round(float(flat.std()), 4),
            "median_similarity": round(float(np.median(flat)), 4),
            "distribution": distribution,
            "objective_alignments": objective_alignments,
            "action_alignments": action_alignments,
            "well_aligned_actions": sorted(
                a["action_number"] for a in action_alignments
                if a["classification"] in ("Good", "Excellent")
            ),
            "poorly_aligned_actions": sorted(
                a["action_number"] for a in action_alignments
                if a["classification"] == "Poor"
            ),
            "orphan_actions": sorted(
                a["action_number"] for a in action_alignments
                if a["is_orphan"]
            ),
            "mismatched_actions": [
                {
                    "action_number": a["action_number"],
                    "title": a["title"],
                    "declared_objective": a["declared_objective"],
                    "best_objective": a["best_objective"],
                    "best_score": a["best_score"],
                }
                for a in action_alignments
                if not a["declared_match"]
            ],
            "alignment_matrix": np.round(matrix, 4).tolist(),
            "matrix_row_labels": obj_codes,
            "matrix_col_labels": act_numbers,
            # Pass through parsed data for display
            "strategic_data": strategic_data,
            "action_data": action_data,
        }

        logger.info("Dynamic analysis complete: overall=%.4f (%s)",
                     overall_score, report["overall_classification"])
        return report

    finally:
        # Clean up temporary ChromaDB
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleaned up temp directory: %s", temp_dir)
