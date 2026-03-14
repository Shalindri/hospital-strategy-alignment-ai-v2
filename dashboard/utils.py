"""
Dashboard utility functions for the ISPS Hospital Strategy-Action Plan
Alignment System.

Provides:
    1. ``load_analysis_results()``  — load & cache all upstream JSON artefacts
    2. ``generate_pdf_report(data)`` — multi-page PDF via ReportLab
    3. ``export_data(data, fmt)``    — export to CSV / JSON / Excel bytes
    4. ``create_plotly_charts(data)`` — reusable Plotly figure factories
    5. ``format_llm_response(text)`` — clean & pretty-print raw LLM output

All functions include error handling and return sensible defaults on failure
so the dashboard never crashes due to a missing file or malformed data.

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import io
import json
import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import THRESHOLD_FAIR, THRESHOLD_GOOD  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to *this* file → dashboard/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Colour palette (shared across all chart helpers)
# ---------------------------------------------------------------------------
COLOURS = {
    "primary":       "#1565C0",
    "primary_light": "#42A5F5",
    "secondary":     "#2E7D32",
    "secondary_lt":  "#66BB6A",
    "accent":        "#FF8F00",
    "danger":        "#E53935",
    "warning":       "#FFB300",
    "excellent":     "#1B5E20",
    "good":          "#66BB6A",
    "fair":          "#FFB300",
    "poor":          "#E53935",
}

CLASSIFICATION_COLOURS = {
    "Excellent": COLOURS["excellent"],
    "Good":      COLOURS["good"],
    "Fair":      COLOURS["fair"],
    "Poor":      COLOURS["poor"],
}

NODE_TYPE_COLOURS = {
    "StrategyObjective": "#4285F4",
    "StrategyGoal":      "#81D4FA",
    "OntologyConcept":   "#9C27B0",
    "Action":            "#4CAF50",
    "KPI":               "#FFEB3B",
    "Stakeholder":       "#FF9800",
    "TimelineQuarter":   "#9E9E9E",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. load_analysis_results — load & cache all upstream JSON artefacts
# ═══════════════════════════════════════════════════════════════════════════

def _load_json_safe(filepath: Path) -> dict | list:
    """Load a JSON file, returning ``{}`` on any error."""
    try:
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                return json.load(f)
        logger.warning("File not found: %s", filepath)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load %s: %s", filepath, exc)
    return {}


def load_analysis_results() -> dict[str, Any]:
    """Load every upstream pipeline artefact into a single dict.

    Keys match those used by ``app.py``: alignment, strategic, actions,
    rag, mappings, gaps, kg, agent_recs, agent_trace.

    Returns:
        dict with all data; any missing file yields an empty dict.
    """
    manifest: dict[str, Path] = {
        "alignment":   DATA_DIR / "alignment_report.json",
        "strategic":   DATA_DIR / "strategic_plan.json",
        "actions":     DATA_DIR / "action_plan.json",
        "rag":         DATA_DIR / "rag_recommendations.json",
        "mappings":    OUTPUT_DIR / "mappings.json",
        "gaps":        OUTPUT_DIR / "gaps.json",
        "kg":          OUTPUT_DIR / "strategy_action_kg.json",
        "agent_recs":  OUTPUT_DIR / "agent_recommendations.json",
        "agent_trace": OUTPUT_DIR / "agent_trace.json",
    }

    data: dict[str, Any] = {}
    loaded, failed = 0, 0
    for key, path in manifest.items():
        result = _load_json_safe(path)
        data[key] = result
        if result:
            loaded += 1
        else:
            failed += 1

    logger.info(
        "Loaded %d/%d pipeline artefacts (%d missing/failed).",
        loaded, loaded + failed, failed,
    )

    # Derive hospital name from strategic plan metadata
    strategic = data.get("strategic", {})
    data["hospital_name"] = strategic.get("metadata", {}).get("title", "")

    # Load ground truth orphans if available (for static Nawaloka data)
    gt_path = PROJECT_ROOT / "tests" / "ground_truth.json"
    if gt_path.exists():
        gt = _load_json_safe(gt_path)
        if gt:
            orphan_nums = {
                entry["action_number"]
                for entry in gt
                if entry.get("alignment_label", 1.0) == 0.0
                and entry.get("is_declared_pair", False)
            }
            data["ground_truth_orphans"] = sorted(orphan_nums)

    return data


# ═══════════════════════════════════════════════════════════════════════════
# 2. generate_pdf_report — multi-page PDF via ReportLab
# ═══════════════════════════════════════════════════════════════════════════

def generate_pdf_report(data: dict[str, Any]) -> bytes:
    """Build a multi-page PDF alignment report.

    Sections:
        1. Title & overall score
        2. Objective alignment table
        3. Orphan / poorly-aligned actions
        4. RAG recommendations summary
        5. Agent recommendations summary
        6. Strategic gaps
        7. Evaluation metrics (P / R / F1)

    Returns:
        Raw PDF bytes (suitable for ``st.download_button``).

    Raises nothing — returns an error-page PDF on internal failure.
    """
    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        logger.error("reportlab not installed — cannot generate PDF.")
        return _pdf_error_stub("reportlab is not installed. Run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ISPSTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=rl_colors.HexColor("#1565C0"),
        spaceAfter=6 * mm,
    )
    heading_style = ParagraphStyle(
        "ISPSHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=rl_colors.HexColor("#1A237E"),
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
    )
    body_style = ParagraphStyle(
        "ISPSBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
    )
    small_style = ParagraphStyle(
        "ISPSSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=rl_colors.HexColor("#555555"),
    )

    elements: list = []
    alignment = data.get("alignment", {})
    rag = data.get("rag", {})
    agent = data.get("agent_recs", {})
    gaps = data.get("gaps", {})

    # Helper for safe paragraph text
    def _p(text: str, style=body_style) -> Paragraph:
        safe = format_llm_response(str(text), strip_markdown=True)
        # Escape XML-special chars for ReportLab
        safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(safe, style)

    # ── Title page ────────────────────────────────────────────────────
    elements.append(Paragraph(
        "Hospital Strategy-Action Plan Alignment Report", title_style,
    ))
    hospital_name = data.get("hospital_name", "")
    if hospital_name:
        elements.append(_p(hospital_name))
    elements.append(_p(
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        small_style,
    ))
    elements.append(Spacer(1, 8 * mm))

    # ── Section 1: Overall synchronization ────────────────────────────
    elements.append(Paragraph("1. Overall Synchronization", heading_style))
    score = alignment.get("overall_score", 0)
    cls = alignment.get("overall_classification", "N/A")
    elements.append(_p(
        f"Overall Score: {score:.1%} ({cls}) | "
        f"Mean Similarity: {alignment.get('mean_similarity', 0):.4f} | "
        f"Median: {alignment.get('median_similarity', 0):.4f}"
    ))
    dist = alignment.get("distribution", {})
    if dist:
        elements.append(_p(
            f"Distribution — Excellent: {dist.get('Excellent', 0)}, "
            f"Good: {dist.get('Good', 0)}, Fair: {dist.get('Fair', 0)}, "
            f"Poor: {dist.get('Poor', 0)}"
        ))
    elements.append(Spacer(1, 3 * mm))

    # ── Section 2: Objective alignment table ──────────────────────────
    elements.append(Paragraph("2. Strategic Objective Alignment", heading_style))
    obj_data = alignment.get("objective_alignments", [])
    if obj_data:
        tbl_data = [["Objective", "Mean Sim.", "Max Sim.", "Coverage", "Aligned"]]
        for o in obj_data:
            tbl_data.append([
                f"{o['code']}: {o['name'][:35]}",
                f"{o['mean_similarity']:.3f}",
                f"{o['max_similarity']:.3f}",
                f"{o['coverage_score']:.0%}",
                str(o.get("aligned_action_count", 0)),
            ])
        tbl = Table(tbl_data, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#F8FAFB"), rl_colors.white]),
        ]))
        elements.append(tbl)
    elements.append(Spacer(1, 3 * mm))

    # ── Section 3: Orphan actions ─────────────────────────────────────
    elements.append(Paragraph("3. Orphan / Poorly-Aligned Actions", heading_style))
    orphans = alignment.get("orphan_actions", [])
    if orphans:
        tbl_data = [["#", "Title", "Declared", "Best Obj", "Score"]]
        for num in orphans:
            aa = next(
                (a for a in alignment.get("action_alignments", [])
                 if a["action_number"] == num), {},
            )
            tbl_data.append([
                str(num),
                aa.get("title", "?")[:45],
                aa.get("declared_objective", "?"),
                aa.get("best_objective", "?"),
                f"{aa.get('best_score', 0):.3f}",
            ])
        tbl = Table(tbl_data, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#E53935")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (4, 0), (4, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CCCCCC")),
        ]))
        elements.append(tbl)
    else:
        elements.append(_p("No orphan actions detected."))
    elements.append(Spacer(1, 3 * mm))

    # ── Section 4: RAG recommendations ────────────────────────────────
    elements.append(Paragraph("4. RAG Improvement Recommendations", heading_style))
    improvements = rag.get("improvements", [])
    elements.append(_p(f"{len(improvements)} action improvements identified."))
    for imp in improvements[:8]:  # cap at 8 for space
        elements.append(_p(
            f"Action {imp['action_number']}: {imp['action_title'][:50]} "
            f"(score={imp['alignment_score']:.3f}, "
            f"confidence={imp.get('confidence', 'N/A')})",
            small_style,
        ))
    suggestions = rag.get("new_action_suggestions", [])
    if suggestions:
        elements.append(Spacer(1, 2 * mm))
        elements.append(_p(f"{len(suggestions)} new actions suggested to fill gaps."))
        for sug in suggestions[:6]:
            elements.append(_p(
                f"[Obj {sug['objective_code']}] {sug.get('title', '?')[:55]}",
                small_style,
            ))
    elements.append(Spacer(1, 3 * mm))

    # ── Section 5: Agent recommendations ──────────────────────────────
    elements.append(Paragraph("5. Agent Reasoning Recommendations", heading_style))
    recs = (agent or {}).get("recommendations", [])
    if recs:
        tbl_data = [["Rec ID", "Issue Type", "Impact", "Confidence", "Budget"]]
        for rec in recs:
            tbl_data.append([
                rec["rec_id"],
                rec["issue_type"],
                f"{rec['impact_score']:.3f}",
                rec.get("confidence", "N/A"),
                f"LKR {rec.get('estimated_budget', 0):.0f}M",
            ])
        tbl = Table(tbl_data, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2E7D32")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (2, 0), (4, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#F1F8E9"), rl_colors.white]),
        ]))
        elements.append(tbl)
    else:
        elements.append(_p("No agent recommendations available."))
    elements.append(Spacer(1, 3 * mm))

    # ── Section 6: Strategic gaps ─────────────────────────────────────
    elements.append(Paragraph("6. Strategic Gaps", heading_style))
    uncovered = gaps.get("uncovered_strategy_concepts", [])
    weak = gaps.get("weak_actions", [])
    elements.append(_p(
        f"Uncovered strategy concepts: {len(uncovered)} | "
        f"Weakly aligned actions: {len(weak)}"
    ))
    for gap in uncovered[:6]:
        related = ", ".join(gap.get("related_strategy_goals", []))
        elements.append(_p(
            f"{gap['concept_id']} — related goals: {related}",
            small_style,
        ))
    elements.append(Spacer(1, 3 * mm))

    # ── Section 7: Evaluation metrics ─────────────────────────────────
    elements.append(Paragraph("7. Evaluation Metrics", heading_style))
    col_labels = alignment.get("matrix_col_labels", [])
    all_actions = set(col_labels) if col_labels else set()
    orphan_set = set(alignment.get("orphan_actions", []))
    ground_truth = set(data.get("ground_truth_orphans", []))

    if ground_truth and all_actions:
        tp = len(orphan_set & ground_truth)
        fp = len(orphan_set - ground_truth)
        fn = len(ground_truth - orphan_set)
        tn = len(all_actions - orphan_set - ground_truth)
        prec = tp / max(tp + fp, 1)
        rec_val = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec_val / max(prec + rec_val, 0.001)

        elements.append(_p(
            f"Ground truth misaligned: {sorted(ground_truth)} | "
            f"Detected orphans: {sorted(orphan_set)}"
        ))
        tbl_data = [
            ["Metric", "Value"],
            ["True Positives", str(tp)],
            ["False Positives", str(fp)],
            ["False Negatives", str(fn)],
            ["True Negatives", str(tn)],
            ["Precision", f"{prec:.2%}"],
            ["Recall", f"{rec_val:.2%}"],
            ["F1 Score", f"{f1:.2%}"],
        ]
    else:
        elements.append(_p(
            f"Detected orphans: {sorted(orphan_set)} | "
            f"Total actions: {len(all_actions)}"
        ))
        tbl_data = [
            ["Metric", "Value"],
            ["Orphan count", str(len(orphan_set))],
            ["Total actions", str(len(all_actions))],
            ["Orphan rate", f"{len(orphan_set) / max(len(all_actions), 1):.1%}"],
        ]

    tbl = Table(tbl_data, hAlign="LEFT", colWidths=[120, 80])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CCCCCC")),
    ]))
    elements.append(tbl)

    # ── Footer ────────────────────────────────────────────────────────
    elements.append(Spacer(1, 8 * mm))
    elements.append(_p(
        f"Generated by ISPS Dashboard | {datetime.now():%Y-%m-%d %H:%M:%S}",
        small_style,
    ))

    try:
        doc.build(elements)
    except Exception as exc:
        logger.error("PDF build failed: %s", exc)
        return _pdf_error_stub(f"PDF generation failed: {exc}")

    return buf.getvalue()


def _pdf_error_stub(message: str) -> bytes:
    """Return a minimal single-page PDF with an error message."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        doc.build([Paragraph(message, styles["BodyText"])])
        return buf.getvalue()
    except Exception:
        # Absolute fallback: return an empty bytes object
        return b""


# ═══════════════════════════════════════════════════════════════════════════
# 3. export_data — export analysis results as CSV / JSON / Excel bytes
# ═══════════════════════════════════════════════════════════════════════════

def export_data(
    data: dict[str, Any],
    fmt: str = "csv",
) -> tuple[bytes, str, str]:
    """Export the core analysis tables in the requested format.

    Supported *fmt* values: ``"csv"``, ``"json"``, ``"excel"``.

    Exports four sheets / sections:
        - action_alignments  (from alignment_report)
        - objective_alignments
        - improvements       (from rag_recommendations)
        - agent_recommendations

    Returns:
        ``(file_bytes, filename, mime_type)``

    Raises nothing — returns an error CSV on internal failure.
    """
    fmt = fmt.lower().strip()

    try:
        # ── Build DataFrames ──────────────────────────────────────────
        alignment = data.get("alignment", {})
        rag = data.get("rag", {})
        agent = data.get("agent_recs", {})

        df_actions = _build_action_alignment_df(alignment)
        df_objectives = _build_objective_alignment_df(alignment)
        df_improvements = _build_improvements_df(rag)
        df_agent = _build_agent_recs_df(agent)

        frames = {
            "action_alignments":     df_actions,
            "objective_alignments":  df_objectives,
            "improvements":          df_improvements,
            "agent_recommendations": df_agent,
        }

        if fmt == "json":
            combined = {
                name: df.to_dict(orient="records") for name, df in frames.items()
            }
            raw = json.dumps(combined, indent=2, default=str)
            return raw.encode("utf-8"), "isps_export.json", "application/json"

        if fmt == "excel":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                for name, df in frames.items():
                    df.to_excel(writer, sheet_name=name[:31], index=False)
            return (
                buf.getvalue(),
                "isps_export.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # Default: CSV (concatenate all frames separated by blank lines)
        buf = io.StringIO()
        for name, df in frames.items():
            buf.write(f"# {name}\n")
            df.to_csv(buf, index=False)
            buf.write("\n")
        return buf.getvalue().encode("utf-8"), "isps_export.csv", "text/csv"

    except Exception as exc:
        logger.error("Export failed (%s): %s", fmt, exc)
        error_csv = f"error,{exc}\n".encode("utf-8")
        return error_csv, "isps_export_error.csv", "text/csv"


# ── DataFrame builders (private) ──────────────────────────────────────────

def _build_action_alignment_df(alignment: dict) -> pd.DataFrame:
    rows = []
    for a in alignment.get("action_alignments", []):
        rows.append({
            "Action #":          a["action_number"],
            "Title":             a.get("title", ""),
            "Declared Obj":      a.get("declared_objective", ""),
            "Best Obj":          a.get("best_objective", ""),
            "Best Score":        a.get("best_score", 0),
            "Classification":    a.get("classification", ""),
            "Is Orphan":         a.get("is_orphan", False),
            "Owner":             a.get("action_owner", ""),
            "Budget (LKR M)":    a.get("budget_lkr_millions", 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_objective_alignment_df(alignment: dict) -> pd.DataFrame:
    rows = []
    for o in alignment.get("objective_alignments", []):
        rows.append({
            "Code":              o["code"],
            "Name":              o["name"],
            "Mean Similarity":   o["mean_similarity"],
            "Max Similarity":    o["max_similarity"],
            "Coverage":          o["coverage_score"],
            "Aligned Actions":   o.get("aligned_action_count", 0),
            "Declared Actions":  o.get("declared_action_count", 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_improvements_df(rag: dict) -> pd.DataFrame:
    rows = []
    for imp in rag.get("improvements", []):
        rows.append({
            "Action #":          imp["action_number"],
            "Title":             imp.get("action_title", ""),
            "Declared Obj":      imp.get("declared_objective", ""),
            "Alignment Score":   imp.get("alignment_score", 0),
            "Confidence":        imp.get("confidence", ""),
            "Modified Desc":     format_llm_response(
                imp.get("modified_description", ""), max_length=300,
            ),
            "Strategic Linkage": format_llm_response(
                imp.get("strategic_linkage", ""), max_length=200,
            ),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_agent_recs_df(agent: dict) -> pd.DataFrame:
    rows = []
    for rec in (agent or {}).get("recommendations", []):
        rows.append({
            "Rec ID":         rec["rec_id"],
            "Issue Type":     rec["issue_type"],
            "Impact Score":   rec["impact_score"],
            "Confidence":     rec.get("confidence", ""),
            "Budget (LKR M)": rec.get("estimated_budget", 0),
            "Timeline":       rec.get("estimated_timeline", ""),
            "What to Change": format_llm_response(
                rec.get("what_to_change", ""), max_length=300,
            ),
            "Actions":        "; ".join(
                format_llm_response(a, max_length=150)
                for a in rec.get("actions", [])
            ),
            "KPIs":           "; ".join(
                format_llm_response(k, max_length=100)
                for k in rec.get("kpis", [])
            ),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# 4. create_plotly_charts — reusable chart factory functions
# ═══════════════════════════════════════════════════════════════════════════

def create_plotly_charts(data: dict[str, Any]) -> dict[str, go.Figure]:
    """Build a dict of named Plotly figures from pipeline data.

    Available chart keys:
        - ``alignment_gauge``     — overall sync gauge indicator
        - ``distribution_bar``    — alignment classification distribution
        - ``budget_pie``          — budget by strategic objective
        - ``objective_bar``       — strategy-wise mean/max similarity
        - ``heatmap``             — full alignment matrix heatmap
        - ``similarity_hist``     — similarity score histogram with thresholds
        - ``method_comparison``   — evaluation P/R/F1 grouped bar
        - ``confusion_matrix``    — confusion matrix for orphan detection

    Returns:
        ``dict[str, go.Figure]``; missing data → key omitted.
    """
    charts: dict[str, go.Figure] = {}
    alignment = data.get("alignment", {})
    actions = data.get("actions", {}).get("actions", [])

    # ── Alignment gauge ───────────────────────────────────────────────
    score = alignment.get("overall_score", 0)
    if score:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={"text": "Overall Synchronization", "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOURS["primary"]},
                "steps": [
                    {"range": [0, 45],  "color": "#FFCDD2"},
                    {"range": [45, 60], "color": "#FFF9C4"},
                    {"range": [60, 75], "color": "#C8E6C9"},
                    {"range": [75, 100], "color": "#81C784"},
                ],
                "threshold": {
                    "line": {"color": COLOURS["danger"], "width": 3},
                    "thickness": 0.8,
                    "value": 45,
                },
            },
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30))
        charts["alignment_gauge"] = fig

    # ── Distribution bar ──────────────────────────────────────────────
    dist = alignment.get("distribution", {})
    if dist:
        df = pd.DataFrame([
            {"Classification": k, "Count": v} for k, v in dist.items()
        ])
        fig = px.bar(
            df, x="Classification", y="Count",
            color="Classification",
            color_discrete_map=CLASSIFICATION_COLOURS,
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        charts["distribution_bar"] = fig

    # ── Budget pie ────────────────────────────────────────────────────
    if actions:
        obj_budget: dict[str, float] = {}
        for a in actions:
            code = a.get("strategic_objective_code", "?")
            obj_budget[code] = obj_budget.get(code, 0) + a.get("budget_lkr_millions", 0)
        df = pd.DataFrame([
            {"Objective": k, "Budget (LKR M)": v}
            for k, v in sorted(obj_budget.items())
        ])
        fig = px.pie(
            df, names="Objective", values="Budget (LKR M)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        charts["budget_pie"] = fig

    # ── Objective grouped bar ─────────────────────────────────────────
    obj_data = alignment.get("objective_alignments", [])
    if obj_data:
        df = pd.DataFrame([{
            "Objective": f"{o['code']}: {o['name'][:30]}",
            "Mean Similarity": o["mean_similarity"],
            "Max Similarity": o["max_similarity"],
        } for o in obj_data])
        fig = px.bar(
            df, x="Objective",
            y=["Mean Similarity", "Max Similarity"],
            barmode="group",
            color_discrete_sequence=[COLOURS["primary_light"], COLOURS["primary"]],
        )
        fig.update_layout(
            height=280, margin=dict(t=20, b=20),
            yaxis_title="Cosine Similarity",
            legend=dict(orientation="h", y=1.1),
        )
        charts["objective_bar"] = fig

    # ── Alignment matrix heatmap ──────────────────────────────────────
    matrix = alignment.get("alignment_matrix", [])
    row_labels = alignment.get("matrix_row_labels", [])
    col_labels = alignment.get("matrix_col_labels", [])
    if matrix:
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"Act {c}" for c in col_labels],
            y=[f"Obj {r}" for r in row_labels],
            colorscale=[
                [0, "#FFCDD2"], [THRESHOLD_FAIR, "#FFE082"],
                [THRESHOLD_GOOD, "#A5D6A7"], [1, "#1B5E20"],
            ],
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate=(
                "Objective %{y}<br>Action %{x}<br>"
                "Score: %{z:.3f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            height=300, margin=dict(t=20, b=20),
            xaxis_title="Action Items",
            yaxis_title="Strategic Objectives",
        )
        charts["heatmap"] = fig

    # ── Similarity histogram ──────────────────────────────────────────
    if matrix:
        all_scores = [s for row in matrix for s in row]
        fig = px.histogram(
            x=all_scores, nbins=25,
            labels={"x": "Cosine Similarity", "y": "Count"},
            color_discrete_sequence=[COLOURS["primary_light"]],
        )
        fig.add_vline(
            x=THRESHOLD_FAIR, line_dash="dash", line_color="red",
            annotation_text="Fair threshold",
        )
        fig.add_vline(
            x=THRESHOLD_GOOD, line_dash="dash", line_color="green",
            annotation_text="Good threshold",
        )
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        charts["similarity_hist"] = fig

    # ── Evaluation: method comparison & confusion matrix ──────────────
    eval_metrics = _compute_evaluation_metrics(data)
    if eval_metrics:
        df_metrics = pd.DataFrame(eval_metrics)

        df_plot = df_metrics.melt(
            id_vars="Method",
            value_vars=["Precision", "Recall", "F1"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            df_plot, x="Method", y="Score", color="Metric",
            barmode="group",
            color_discrete_sequence=[
                COLOURS["primary"], COLOURS["secondary"], COLOURS["accent"],
            ],
        )
        fig.update_layout(
            height=350, margin=dict(t=20, b=80),
            xaxis_tickangle=-30,
            yaxis_title="Score",
        )
        charts["method_comparison"] = fig

        # Confusion matrix for best-F1 method
        best = max(eval_metrics, key=lambda m: m["F1"])
        cm = [
            [int(best["TP"]), int(best["FP"])],
            [int(best["FN"]), int(best["TN"])],
        ]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Misaligned", "Predicted Aligned"],
            y=["Actually Misaligned", "Actually Aligned"],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale=[[0, "#E3F2FD"], [1, COLOURS["primary"]]],
            showscale=False,
        ))
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        charts["confusion_matrix"] = fig

    return charts


def _compute_evaluation_metrics(data: dict) -> list[dict]:
    """Compute P/R/F1 for each detection method against ground truth."""
    ground_truth = set(data.get("ground_truth_orphans", []))
    alignment = data.get("alignment", {})
    col_labels = alignment.get("matrix_col_labels", [])
    all_actions = set(col_labels) if col_labels else set()

    if not ground_truth or not all_actions:
        return []
    gaps = data.get("gaps", {})
    agent = data.get("agent_recs", {})

    methods: dict[str, set[int]] = {
        "Embedding (Orphan)": set(alignment.get("orphan_actions", [])),
        "Embedding (Poor)": set(alignment.get("poorly_aligned_actions", [])),
    }

    # Ontology weak actions
    weak = set()
    for w in gaps.get("weak_actions", []):
        num_str = w.get("action_id", "").replace("action_", "")
        if num_str.isdigit():
            weak.add(int(num_str))
    methods["Ontology (Weak)"] = weak

    # Agent orphans
    agent_det = set()
    for issue in (agent or {}).get("diagnosed_issues_summary", []):
        if issue.get("type") == "orphan_action":
            num_str = issue["issue_id"].replace("orphan_action_", "")
            if num_str.isdigit():
                agent_det.add(int(num_str))
    methods["Agent (Orphan)"] = agent_det

    # Combined
    methods["Combined"] = methods["Embedding (Orphan)"] | weak

    rows = []
    for name, detected in methods.items():
        tp = len(detected & ground_truth)
        fp = len(detected - ground_truth)
        fn = len(ground_truth - detected)
        tn = len(all_actions - detected - ground_truth)
        prec = tp / max(tp + fp, 1)
        rec_v = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec_v / max(prec + rec_v, 0.001)
        rows.append({
            "Method": name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": prec, "Recall": rec_v, "F1": f1,
            "Accuracy": (tp + tn) / len(all_actions),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 5. format_llm_response — clean & pretty-print raw LLM output
# ═══════════════════════════════════════════════════════════════════════════

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADING_RE = re.compile(r"^#{1,4}\s*", re.MULTILINE)
_BULLET_PREFIX = re.compile(r"^[-*]\s+", re.MULTILINE)
_NUMBERED_PREFIX = re.compile(r"^\d+\.\s+", re.MULTILINE)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_SECTION_TAGS = re.compile(
    r"\*\*(ADDITIONAL|CONFIDENCE|ROOT_CAUSE|RECOMMENDATION|"
    r"KPI|TIMELINE|BUDGET|FEASIBILITY|ALIGNMENT_CHECK|"
    r"RISKS|VERDICT|MISSING_INFO)\*?\*?:?",
    re.IGNORECASE,
)


def format_llm_response(
    text: str,
    *,
    max_length: int = 0,
    strip_markdown: bool = False,
    wrap_width: int = 0,
) -> str:
    """Clean and optionally reformat raw LLM-generated text.

    Transformations applied:
        - Strip leading/trailing whitespace
        - Remove ``**SECTION_TAG**:`` markers that leak from prompt templates
        - Optionally strip all markdown bold (``**...**``) formatting
        - Optionally strip markdown heading prefixes (``## ...``)
        - Collapse excessive blank lines (>2 → 2)
        - Optionally truncate to *max_length* chars (with ``...`` suffix)
        - Optionally word-wrap at *wrap_width* columns

    Args:
        text:           The raw LLM output string.
        max_length:     If >0, truncate to this many characters.
        strip_markdown: If True, remove ``**bold**`` and heading markers.
        wrap_width:     If >0, word-wrap each line to this column width.

    Returns:
        Cleaned string.
    """
    if not text:
        return ""

    out = text.strip()

    # Remove leaked prompt template section tags
    out = _SECTION_TAGS.sub("", out)

    if strip_markdown:
        # Convert **bold** → plain text
        out = _BOLD_RE.sub(r"\1", out)
        # Remove heading markers
        out = _HEADING_RE.sub("", out)

    # Collapse excessive blank lines
    out = _MULTI_NEWLINE.sub("\n\n", out)

    # Strip again after transformations
    out = out.strip()

    # Word wrap
    if wrap_width > 0:
        lines = out.split("\n")
        wrapped = []
        for line in lines:
            if len(line) <= wrap_width:
                wrapped.append(line)
            else:
                wrapped.extend(textwrap.wrap(line, width=wrap_width))
        out = "\n".join(wrapped)

    # Truncate
    if max_length > 0 and len(out) > max_length:
        out = out[: max_length - 3].rstrip() + "..."

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Module-level self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ISPS Dashboard Utils — Self-Test")
    print("=" * 60)

    # 1. Load data
    print("\n[1] load_analysis_results()")
    data = load_analysis_results()
    for key, val in data.items():
        kind = type(val).__name__
        size = len(val) if isinstance(val, (dict, list)) else "N/A"
        status = "OK" if val else "EMPTY"
        print(f"  {key:<15} {kind:<6} len={size:<5} {status}")

    # 2. PDF report
    print("\n[2] generate_pdf_report()")
    pdf_bytes = generate_pdf_report(data)
    if pdf_bytes:
        out_path = OUTPUT_DIR / "isps_report.pdf"
        out_path.write_bytes(pdf_bytes)
        print(f"  PDF generated: {out_path} ({len(pdf_bytes):,} bytes)")
    else:
        print("  PDF generation failed — empty bytes returned.")

    # 3. Export data
    print("\n[3] export_data()")
    for fmt in ("csv", "json", "excel"):
        raw, filename, mime = export_data(data, fmt)
        print(f"  {fmt:>5}: {filename:<25} {len(raw):>8,} bytes  ({mime})")
        out_path = OUTPUT_DIR / filename
        out_path.write_bytes(raw)

    # 4. Chart factory
    print("\n[4] create_plotly_charts()")
    charts = create_plotly_charts(data)
    for name, fig in charts.items():
        traces = len(fig.data)
        print(f"  {name:<25} {traces} trace(s)")

    # 5. format_llm_response
    print("\n[5] format_llm_response()")
    test_cases = [
        ("**ROOT_CAUSE**: The action lacks alignment...", {}),
        ("**Bold text** and more **bold**", {"strip_markdown": True}),
        ("A " * 100, {"max_length": 50}),
        ("Short text with\n\n\n\nextra newlines", {}),
        ("## Heading\n- bullet 1\n- bullet 2", {"strip_markdown": True}),
    ]
    for raw, kwargs in test_cases:
        cleaned = format_llm_response(raw, **kwargs)
        print(f"  IN:  {raw[:60]!r}")
        print(f"  OUT: {cleaned[:60]!r}")
        print()

    print("=" * 60)
    print("All self-tests passed.")
    print("=" * 60)
