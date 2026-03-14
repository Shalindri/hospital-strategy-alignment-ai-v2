"""
Data adapter for converting upload_report into the data dict format
expected by all dashboard page functions.

When the user uploads PDFs and runs analysis, the result is stored as
``st.session_state["upload_report"]``. This module converts that into the
same ``data`` dict structure that ``load_analysis_results()`` returns.

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

from typing import Any


def build_data_dict(
    upload_report: dict[str, Any],
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an upload_report into the data dict all pages expect.

    Args:
        upload_report: The alignment report from ``dynamic_analyzer``.
        session_state: Streamlit session state dict (for dynamic_* keys).

    Returns:
        Dict with keys: alignment, strategic, actions, rag, mappings,
        gaps, kg, agent_recs, agent_trace.
    """
    ss = session_state or {}

    alignment = {
        "overall_score": upload_report["overall_score"],
        "overall_classification": upload_report["overall_classification"],
        "mean_similarity": upload_report.get("mean_similarity", 0),
        "std_similarity": upload_report.get("std_similarity", 0),
        "median_similarity": upload_report.get("median_similarity", 0),
        "distribution": upload_report["distribution"],
        "objective_alignments": upload_report["objective_alignments"],
        "action_alignments": upload_report["action_alignments"],
        "alignment_matrix": upload_report["alignment_matrix"],
        "matrix_row_labels": upload_report["matrix_row_labels"],
        "matrix_col_labels": upload_report["matrix_col_labels"],
        "orphan_actions": upload_report.get("orphan_actions", []),
        "poorly_aligned_actions": upload_report.get("poorly_aligned_actions", []),
        "well_aligned_actions": upload_report.get("well_aligned_actions", []),
        "mismatched_actions": upload_report.get("mismatched_actions", []),
    }

    # Extract hospital name from strategic plan metadata if available
    strategic = upload_report.get("strategic_data", {})
    hospital_name = strategic.get("metadata", {}).get("title", "")

    return {
        "alignment": alignment,
        "strategic": strategic,
        "actions": upload_report.get("action_data", {}),
        "rag": ss.get("dynamic_rag", {}),
        "mappings": ss.get("dynamic_mappings", {}),
        "gaps": ss.get("dynamic_gaps", {}),
        "kg": ss.get("dynamic_kg", {}),
        "agent_recs": ss.get("dynamic_agent_recs", {}),
        "agent_trace": ss.get("dynamic_agent_trace", {}),
        "hospital_name": hospital_name,
        "ground_truth_orphans": ss.get("ground_truth_orphans", []),
    }
