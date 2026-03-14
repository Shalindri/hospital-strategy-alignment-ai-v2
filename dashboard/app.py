"""
Healthcare Strategy Aligner — Single-Page Streamlit App.

Upload Strategic Plan and Action Plan PDFs, run the full analysis pipeline,
and view all results on one page — no sidebar navigation needed.

Launch::

    streamlit run dashboard/app.py

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so absolute imports (dashboard.*, src.*)
# work regardless of how Streamlit launches the app.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import format_llm_response, generate_pdf_report

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Hide sidebar */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* Theme */
    :root {
        --primary: #1565C0;
        --primary-light: #42A5F5;
        --secondary: #2E7D32;
        --accent: #FF8F00;
        --bg-card: #F8FAFB;
        --text-dark: #1A237E;
    }
    .main .block-container { max-width: 1500px; padding-top: 0.5rem; }

    /* ── Header bar ── */
    .header-bar {
        background: linear-gradient(135deg, #1565C0 0%, #1A237E 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        text-align: center;
    }
    .header-bar h1 {
        color: white !important;
        font-size: 1.5rem !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    h2, h3 { color: var(--text-dark) !important; }

    /* ── Toolbar area ── */
    .toolbar-container {
        background: var(--bg-card);
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }

    /* Compact upload in toolbar */
    [data-testid="stFileUploader"] label p { font-size: 0.75rem !important; }
    [data-testid="stFileUploader"] section { padding: 0.2rem !important; }
    [data-testid="stFileUploader"] { margin-bottom: 0 !important; }

    /* Processing label */
    .processing-label {
        text-align: center;
        color: #F57F17;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #F1F8E9 100%);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .metric-card h4 { margin: 0 0 0.1rem 0; color: var(--primary); font-size: 0.72rem; }
    .metric-card .value { font-size: 1.3rem; font-weight: 700; color: var(--text-dark); }
    .metric-card .sub { font-size: 0.68rem; color: #666; }

    /* Badges */
    .badge-orphan  { background: #FFCDD2; color: #B71C1C; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-weak    { background: #FFE0B2; color: #E65100; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-gap     { background: #E1BEE7; color: #6A1B9A; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-good    { background: #C8E6C9; color: #1B5E20; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-high    { background: #C8E6C9; color: #1B5E20; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-medium  { background: #FFF9C4; color: #F57F17; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-low     { background: #FFCDD2; color: #B71C1C; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }

    /* Trace step */
    .trace-step {
        border-left: 3px solid var(--primary-light);
        padding: 0.5rem 0 0.5rem 1rem;
        margin-bottom: 0.8rem;
        background: var(--bg-card);
        border-radius: 0 8px 8px 0;
    }

    /* ── Stepper progress bar ── */
    .stepper { display: flex; align-items: flex-start; justify-content: space-between; padding: 0.4rem 0; }
    .stepper .step { flex: 1; text-align: center; position: relative; }
    .stepper .step .circle {
        width: 26px; height: 26px; border-radius: 50%;
        background: #E0E0E0; color: #999;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700;
        border: 2px solid #BDBDBD;
        transition: all 0.3s;
        position: relative; z-index: 2;
    }
    .stepper .step .label { font-size: 0.6rem; color: #888; margin-top: 3px; line-height: 1.1; }
    /* Connector line between circles */
    .stepper .step:not(:last-child)::after {
        content: ''; position: absolute; top: 13px;
        left: calc(50% + 16px); right: calc(-50% + 16px);
        height: 3px; background: #E0E0E0; z-index: 1;
    }
    /* Active step */
    .stepper .step.active .circle {
        background: #FFF9C4; color: #F57F17;
        border-color: #F57F17;
        box-shadow: 0 0 0 3px rgba(245,127,23,0.2);
        animation: pulse 1.5s infinite;
    }
    .stepper .step.active .label { color: #F57F17; font-weight: 600; }
    @keyframes pulse { 0%,100% { box-shadow: 0 0 0 3px rgba(245,127,23,0.2); } 50% { box-shadow: 0 0 0 6px rgba(245,127,23,0.1); } }
    /* Completed step */
    .stepper .step.done .circle {
        background: #4CAF50; color: white;
        border-color: #388E3C;
    }
    .stepper .step.done .label { color: #2E7D32; font-weight: 600; }
    .stepper .step.done:not(:last-child)::after { background: #4CAF50; }

    /* ── Output placeholder ── */
    .output-placeholder {
        background: var(--bg-card);
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 3rem 2rem;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #999;
        font-size: 1.1rem;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Stepper HTML builder
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    ("1", "PDF Extract"),
    ("2", "Alignment"),
    ("3", "Ontology"),
    ("4", "KG Build"),
    ("5", "RAG"),
]


def _build_stepper_html(current_step: int = 0) -> str:
    """Build HTML for the milestone stepper.

    Args:
        current_step: 0 = not started, 1-6 = that step is active,
                      7 = all done.
    """
    parts = ['<div class="stepper">']
    for idx, (num, label) in enumerate(PIPELINE_STEPS):
        step_num = idx + 1
        if step_num < current_step:
            cls = "step done"
            icon = "&#10003;"  # checkmark
        elif step_num == current_step:
            cls = "step active"
            icon = num
        else:
            cls = "step"
            icon = num
        parts.append(
            f'<div class="{cls}">'
            f'<div class="circle">{icon}</div>'
            f'<div class="label">{label}</div>'
            f'</div>'
        )
    parts.append('</div>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def confidence_badge(conf: str) -> str:
    cls = conf.lower() if conf.lower() in ("high", "medium", "low") else "medium"
    return f'<span class="badge-{cls}">{conf}</span>'


def make_metric_card(title: str, value: str, subtitle: str = "") -> str:
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="metric-card">'
        f'<h4>{title}</h4>'
        f'<div class="value">{value}</div>'
        f'{sub_html}</div>'
    )


# ═══════════════════════════════════════════════════════════════════════
# Render: Sync Analysis
# ═══════════════════════════════════════════════════════════════════════

def _render_sync_analysis(data: dict) -> None:
    alignment = data["alignment"]

    col1, col2 = st.columns([1, 2])
    with col1:
        score = alignment.get("overall_score", 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            title={"text": "Overall Synchronization", "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1565C0"},
                "steps": [
                    {"range": [0, 45], "color": "#FFCDD2"},
                    {"range": [45, 60], "color": "#FFF9C4"},
                    {"range": [60, 75], "color": "#C8E6C9"},
                    {"range": [75, 100], "color": "#81C784"},
                ],
                "threshold": {
                    "line": {"color": "#B71C1C", "width": 3},
                    "thickness": 0.8, "value": 45,
                },
            },
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Strategy-wise Alignment")
        obj_data = alignment.get("objective_alignments", [])
        if obj_data:
            df_obj = pd.DataFrame([{
                "Objective": f"{o['code']}: {o['name'][:30]}",
                "Mean Similarity": o["mean_similarity"],
                "Max Similarity": o["max_similarity"],
                "Coverage": o["coverage_score"],
            } for o in obj_data])
            fig = px.bar(df_obj, x="Objective",
                         y=["Mean Similarity", "Max Similarity"],
                         barmode="group",
                         color_discrete_sequence=["#42A5F5", "#1565C0"])
            fig.update_layout(height=280, margin=dict(t=20, b=20),
                              yaxis_title="Cosine Similarity",
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width='stretch')

    st.divider()

    # Heatmap
    st.subheader("Alignment Matrix Heatmap")
    matrix = alignment.get("alignment_matrix", [])
    row_labels = alignment.get("matrix_row_labels", [])
    col_labels = alignment.get("matrix_col_labels", [])

    if matrix:
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"Act {c}" for c in col_labels],
            y=[f"Obj {r}" for r in row_labels],
            colorscale=[
                [0, "#FFCDD2"], [0.45, "#FFE082"],
                [0.6, "#A5D6A7"], [1, "#1B5E20"],
            ],
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}", textfont={"size": 9},
            hovertemplate="Objective %{y}<br>Action %{x}<br>Score: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(height=300, margin=dict(t=20, b=20),
                          xaxis_title="Action Items", yaxis_title="Strategic Objectives")
        st.plotly_chart(fig, width='stretch')

    # Histogram + action table
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Distribution")
        if matrix:
            all_scores = [s for row in matrix for s in row]
            fig = px.histogram(x=all_scores, nbins=25,
                               labels={"x": "Cosine Similarity", "y": "Count"},
                               color_discrete_sequence=["#42A5F5"])
            fig.add_vline(x=0.45, line_dash="dash", line_color="red",
                          annotation_text="Fair threshold")
            fig.add_vline(x=0.60, line_dash="dash", line_color="green",
                          annotation_text="Good threshold")
            fig.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Action Alignment Details")
        action_data = alignment.get("action_alignments", [])
        if action_data:
            df_actions = pd.DataFrame([{
                "Action": f"{a['action_number']}. {a['title'][:40]}",
                "Declared": a["declared_objective"],
                "Best": a["best_objective"],
                "Score": a["best_score"],
                "Class": a["classification"],
                "Orphan": "Yes" if a["is_orphan"] else "",
            } for a in action_data])
            st.dataframe(df_actions, width='stretch', height=300,
                         hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Render: Recommendations
# ═══════════════════════════════════════════════════════════════════════

def _render_recommendations(data: dict) -> None:
    rag = data["rag"]
    gaps_data = data["gaps"]

    if not rag:
        st.info("RAG recommendations were not generated. Run the full pipeline to see results here.")
        return

    tab1, tab2, tab3 = st.tabs(["Poorly Aligned Actions", "Missing Actions", "Strategic Gaps"])

    with tab1:
        improvements = rag.get("improvements", [])
        if not improvements:
            st.info("No improvement recommendations available.")
        else:
            st.markdown(f"**{len(improvements)} actions** identified for improvement")
            for imp in improvements:
                with st.expander(
                    f"Action {imp['action_number']}: {imp['action_title'][:55]} "
                    f"— Score: {imp['alignment_score']:.3f}", expanded=False,
                ):
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        st.metric("Alignment Score", f"{imp['alignment_score']:.3f}")
                    with cols[1]:
                        st.metric("Declared Objective", imp["declared_objective"])
                    with cols[2]:
                        st.markdown(f"Confidence: {confidence_badge(imp.get('confidence', 'MEDIUM'))}",
                                    unsafe_allow_html=True)
                    if imp.get("modified_description"):
                        st.markdown("**Suggested Description:**")
                        desc = format_llm_response(imp["modified_description"], max_length=500)
                        st.markdown(f"> {desc}")
                    if imp.get("strategic_linkage"):
                        linkage = format_llm_response(imp["strategic_linkage"], max_length=300)
                        st.markdown(f"**Strategic Linkage:** {linkage}")

    with tab2:
        suggestions = rag.get("new_action_suggestions", [])
        if not suggestions:
            st.info("No new action suggestions available.")
        else:
            st.markdown(f"**{len(suggestions)} new actions** suggested to fill gaps")
            for sug in suggestions:
                with st.expander(
                    f"[Obj {sug['objective_code']}] {sug.get('title', 'Untitled')[:55]}",
                    expanded=False,
                ):
                    if sug.get("description"):
                        st.markdown(f"**Description:** {sug['description'][:400]}")
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown(f"**Owner:** {sug.get('owner', 'N/A')}")
                    with cols[1]:
                        st.markdown(f"**Timeline:** {sug.get('timeline', 'N/A')}")
                    with cols[2]:
                        st.markdown(f"**Budget:** {sug.get('budget_estimate', 'N/A')}")
                    if sug.get("kpis"):
                        st.markdown("**KPIs:**")
                        for kpi in sug["kpis"][:4]:
                            st.markdown(f"- {kpi}")

    with tab3:
        uncovered = gaps_data.get("uncovered_strategy_concepts", [])
        weak = gaps_data.get("weak_actions", [])
        if uncovered:
            st.subheader(f"Uncovered Strategy Concepts ({len(uncovered)})")
            for gap in uncovered:
                st.markdown(
                    f'<span class="badge-gap">{gap["concept_id"]}</span> '
                    f'— Goals: {", ".join(gap.get("related_strategy_goals", []))}',
                    unsafe_allow_html=True,
                )
                st.caption(gap.get("note", ""))
        if weak:
            st.subheader(f"Weakly Aligned Actions ({len(weak)})")
            df_weak = pd.DataFrame([{
                "Action": w["action_id"],
                "Best Concept": w.get("best_concept", ""),
                "Score": w.get("best_score", 0),
                "Note": w.get("note", "")[:60],
            } for w in weak])
            st.dataframe(df_weak, width='stretch', hide_index=True)
        if not uncovered and not weak:
            st.info("No strategic gaps detected.")


# ═══════════════════════════════════════════════════════════════════════
# Render: Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════

def _render_knowledge_graph(data: dict) -> None:
    kg = data["kg"]
    if not kg:
        st.info("Knowledge graph was not built. Run the full pipeline to see results here.")
        return

    insights = kg.get("insights", {})

    cols = st.columns(4)
    with cols[0]:
        st.metric("Nodes", insights.get("node_count", len(kg.get("nodes", []))))
    with cols[1]:
        st.metric("Edges", insights.get("edge_count", len(kg.get("edges", kg.get("links", [])))))
    with cols[2]:
        st.metric("Communities", insights.get("community_count", 0))
    with cols[3]:
        st.metric("Isolated Actions", len(insights.get("isolated_actions", [])))

    st.divider()

    threshold = st.slider("Minimum edge weight to display", 0.0, 1.0, 0.45, 0.05,
                           key="kg_threshold")

    nodes = kg.get("nodes", [])
    links = kg.get("edges", kg.get("links", []))

    if not nodes:
        st.info("No graph nodes found.")
        return

    filtered_links = [l for l in links if l.get("weight", 1.0) >= threshold]
    node_map = {n["id"]: i for i, n in enumerate(nodes)}

    np.random.seed(42)
    n = len(nodes)
    pos = np.random.randn(n, 2) * 2

    for _ in range(50):
        for link in filtered_links:
            src_idx = node_map.get(link.get("source"))
            tgt_idx = node_map.get(link.get("target"))
            if src_idx is not None and tgt_idx is not None:
                diff = pos[tgt_idx] - pos[src_idx]
                dist = max(np.linalg.norm(diff), 0.1)
                force = diff / dist * 0.05
                pos[src_idx] += force
                pos[tgt_idx] -= force

    color_map = {
        "StrategyObjective": "#4285F4", "StrategyGoal": "#81D4FA",
        "OntologyConcept": "#9C27B0", "Action": "#4CAF50",
        "KPI": "#FFEB3B", "Stakeholder": "#FF9800", "TimelineQuarter": "#9E9E9E",
    }

    show_types = st.multiselect(
        "Node types to display", list(color_map.keys()),
        default=["StrategyObjective", "Action", "OntologyConcept"],
        key="kg_node_types",
    )

    visible = set()
    for i, node in enumerate(nodes):
        if node.get("node_type") in show_types:
            visible.add(i)
    for link in filtered_links:
        src = node_map.get(link.get("source"))
        tgt = node_map.get(link.get("target"))
        if src in visible or tgt in visible:
            if src is not None:
                visible.add(src)
            if tgt is not None:
                visible.add(tgt)

    edge_x, edge_y = [], []
    for link in filtered_links:
        src = node_map.get(link.get("source"))
        tgt = node_map.get(link.get("target"))
        if src in visible and tgt in visible:
            edge_x += [pos[src][0], pos[tgt][0], None]
            edge_y += [pos[src][1], pos[tgt][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#CCCCCC"), hoverinfo="none",
    ))

    for ntype in show_types:
        idx_list = [i for i, nd in enumerate(nodes)
                    if nd.get("node_type") == ntype and i in visible]
        if not idx_list:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in idx_list],
            y=[pos[i][1] for i in idx_list],
            mode="markers+text",
            marker=dict(
                size=[min(nodes[i].get("size", 10) * 1.5, 40) for i in idx_list],
                color=color_map.get(ntype, "#999"),
                line=dict(width=1, color="white"),
            ),
            text=[nodes[i].get("label", "")[:20] for i in idx_list],
            textposition="top center", textfont=dict(size=7),
            name=ntype,
            hovertext=[
                f"{nodes[i].get('label', '')}<br>Type: {ntype}<br>"
                f"Score: {nodes[i].get('alignment_score', 'N/A')}"
                for i in idx_list
            ],
            hoverinfo="text",
        ))

    fig.update_layout(
        height=600, showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=20, b=40, l=20, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bridge Nodes")
        bridge = insights.get("bridge_nodes", [])
        if bridge:
            for bn in bridge[:5]:
                st.markdown(
                    f"**{bn.get('label', '')[:40]}** ({bn.get('node_type', '')}) — "
                    f"betweenness: {bn.get('betweenness', 0):.4f}"
                )
        else:
            st.caption("No significant bridge nodes detected.")
    with col2:
        st.subheader("Suggested New Connections")
        suggestions = insights.get("new_connections", [])
        if suggestions:
            for sug in suggestions[:5]:
                st.markdown(
                    f"**{sug.get('concept', '')}** <- "
                    f"{sug.get('suggested_action', '')} "
                    f"(conf: {sug.get('confidence', 0):.3f})"
                )
        else:
            st.caption("No connection suggestions available.")


# ═══════════════════════════════════════════════════════════════════════
# Render: Ontology Browser
# ═══════════════════════════════════════════════════════════════════════

def _render_ontology(data: dict) -> None:
    mappings = data["mappings"]
    gaps_data = data["gaps"]

    if not mappings:
        st.info("Ontology mappings were not computed. Run the full pipeline to see results here.")
        return

    meta = mappings.get("metadata", {})
    st.markdown(
        f"**Model:** {meta.get('embedding_model', 'N/A')} | "
        f"**Concepts:** {meta.get('total_concepts', 0)} | "
        f"**Threshold:** {meta.get('mapping_threshold', 0)}"
    )

    concept_tree: dict[str, list[str]] = {}
    concept_labels: dict[str, str] = {}
    concept_action_count: dict[str, int] = {}

    for section in ("action_mappings", "strategy_mappings"):
        for item in mappings.get(section, []):
            for m in item.get("mappings", []):
                parent = m.get("parent_concept", "")
                cid = m["concept_id"]
                concept_labels[cid] = m.get("concept_label", cid)
                if parent and parent != cid:
                    concept_tree.setdefault(parent, [])
                    if cid not in concept_tree[parent]:
                        concept_tree[parent].append(cid)
                if section == "action_mappings":
                    concept_action_count[cid] = concept_action_count.get(cid, 0) + 1

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Concept Hierarchy")
        uncovered_ids = {g["concept_id"] for g in gaps_data.get("uncovered_strategy_concepts", [])}
        for parent in sorted(concept_tree.keys()):
            count = concept_action_count.get(parent, 0)
            st.markdown(f"### {concept_labels.get(parent, parent)} ({count} action links)")
            for child in sorted(concept_tree[parent]):
                child_count = concept_action_count.get(child, 0)
                if child in uncovered_ids:
                    badge = '<span class="badge-orphan">UNCOVERED</span>'
                elif child_count > 0:
                    badge = f'<span class="badge-good">{child_count} actions</span>'
                else:
                    badge = '<span class="badge-weak">0 actions</span>'
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;{concept_labels.get(child, child)} {badge}",
                    unsafe_allow_html=True,
                )

    with col2:
        st.subheader("Coverage Summary")
        total_concepts = len(concept_labels)
        covered = sum(1 for c in concept_labels if concept_action_count.get(c, 0) > 0)
        coverage_pct = covered / max(total_concepts, 1)

        st.metric("Total Concepts", total_concepts)
        st.metric("Covered by Actions", covered)
        st.metric("Coverage Rate", f"{coverage_pct:.0%}")
        st.metric("Uncovered (Strategy)", len(uncovered_ids))

        sunburst_data = []
        for parent in concept_tree:
            for child in concept_tree[parent]:
                sunburst_data.append({
                    "parent": concept_labels.get(parent, parent),
                    "child": concept_labels.get(child, child),
                    "count": concept_action_count.get(child, 0),
                })
        if sunburst_data:
            df_sun = pd.DataFrame(sunburst_data)
            fig = px.sunburst(df_sun, path=["parent", "child"], values="count",
                              color="count",
                              color_continuous_scale=["#FFCDD2", "#C8E6C9", "#1B5E20"])
            fig.update_layout(height=400, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# Render: Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════

def _render_evaluation(data: dict) -> None:
    alignment = data["alignment"]

    orphan_detected = set(alignment.get("orphan_actions", []))
    poorly_aligned = set(alignment.get("poorly_aligned_actions", []))

    st.subheader("Alignment Detection Summary")

    methods = {
        "Embedding Similarity (Orphan Detection)": orphan_detected,
        "Embedding Similarity (Poor Alignment)": poorly_aligned,
    }

    gaps_data = data["gaps"]
    weak_actions = set()
    for w in gaps_data.get("weak_actions", []):
        act_id = w.get("action_id", "")
        num_str = act_id.replace("action_", "")
        if num_str.isdigit():
            weak_actions.add(int(num_str))
    if weak_actions:
        methods["Ontology Mapping (Weak Actions)"] = weak_actions

    if weak_actions:
        combined = orphan_detected | weak_actions
        methods["Combined (Embedding + Ontology)"] = combined

    det_rows = []
    for method_name, detected in methods.items():
        det_rows.append({
            "Method": method_name,
            "Flagged Actions": len(detected),
            "Actions": ", ".join(str(a) for a in sorted(detected)) if detected else "None",
        })
    df_det = pd.DataFrame(det_rows)
    st.dataframe(df_det, width='stretch', hide_index=True)

    st.subheader("Cross-Method Agreement")
    all_flagged = set()
    for detected in methods.values():
        all_flagged |= detected
    if all_flagged:
        agreement_rows = []
        for act_num in sorted(all_flagged):
            flagged_by = [name for name, det in methods.items() if act_num in det]
            agreement_rows.append({
                "Action": act_num,
                "Flagged By": len(flagged_by),
                "Methods": ", ".join(m.split("(")[0].strip() for m in flagged_by),
            })
        df_agree = pd.DataFrame(agreement_rows)
        st.dataframe(df_agree, width='stretch', hide_index=True)
    else:
        st.success("No misalignment detected by any method.")

    misaligned_details = alignment.get("mismatched_actions", [])
    if misaligned_details:
        st.divider()
        st.subheader("Declared vs Best Objective Mismatches")
        df_mis = pd.DataFrame([{
            "Action": m["action_number"],
            "Title": m["title"][:45],
            "Declared": m["declared_objective"],
            "Best Match": m["best_objective"],
            "Best Score": f"{m['best_score']:.3f}",
        } for m in misaligned_details])
        st.dataframe(df_mis, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════

def _run_upload_analysis(strategic_file, action_file, stepper_placeholder) -> None:
    """Process uploaded PDFs and run the full 6-step pipeline.

    Updates the stepper_placeholder (an st.empty()) at each milestone.
    """
    from src.pdf_processor import (
        extract_strategic_plan_from_pdf,
        extract_action_plan_from_pdf,
    )
    from src.dynamic_analyzer import run_dynamic_analysis
    from dashboard.pipeline_runner import (
        run_dynamic_ontology, run_dynamic_kg,
        run_dynamic_rag,
    )

    def _update(step: int) -> None:
        stepper_placeholder.markdown(_build_stepper_html(step), unsafe_allow_html=True)

    try:
        # Step 1: PDF extraction
        _update(1)
        strategic_bytes = strategic_file.read()
        strategic_data = extract_strategic_plan_from_pdf(strategic_bytes)
        action_bytes = action_file.read()
        action_data = extract_action_plan_from_pdf(action_bytes)

        # Step 2: Alignment
        _update(2)
        report = run_dynamic_analysis(strategic_data, action_data)
        st.session_state["upload_report"] = report

        # Step 3: Ontology
        _update(3)
        mappings, gaps = run_dynamic_ontology(report)
        st.session_state["dynamic_mappings"] = mappings
        st.session_state["dynamic_gaps"] = gaps

        # Step 4: Knowledge Graph
        _update(4)
        kg = run_dynamic_kg(report, mappings)
        st.session_state["dynamic_kg"] = kg

        # Step 5: RAG
        _update(5)
        rag = run_dynamic_rag(report)
        st.session_state["dynamic_rag"] = rag

        # All done
        _update(6)
        st.session_state["pipeline_done"] = True
        st.session_state.pop("pipeline_running", None)
        st.rerun()

    except ValueError as e:
        st.session_state.pop("pipeline_running", None)
        st.error(f"Parsing error: {e}")
    except ConnectionError:
        st.session_state.pop("pipeline_running", None)
        st.error("Cannot connect to the LLM API. Check your API key.")
    except Exception as e:
        st.session_state.pop("pipeline_running", None)
        st.error(f"Analysis failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Healthcare Strategy Aligner",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Header bar ──
    st.markdown(
        '<div class="header-bar"><h1>Healthcare Strategy Aligner</h1></div>',
        unsafe_allow_html=True,
    )

    pipeline_done = st.session_state.get("pipeline_done", False)

    # ── Toolbar: uploaders | button | stepper | download — one row ──
    st.markdown('<div class="toolbar-container">', unsafe_allow_html=True)

    strategic_file = None
    action_file = None
    run_clicked = False

    if not pipeline_done:
        # Layout: [uploader1] [uploader2] [button] [stepper] [empty]
        c_up1, c_up2, c_btn, c_step, c_dl = st.columns(
            [1.2, 1.2, 1, 2.5, 1], gap="medium",
        )

        @st.cache_data(ttl=60, show_spinner=False)
        def _check_llm():
            from src.pdf_processor import check_llm_available
            return check_llm_available()

        llm_ok = _check_llm()

        with c_up1:
            if not llm_ok:
                st.error("OpenAI API key not set.")
            else:
                strategic_file = st.file_uploader(
                    "strategy.pdf", type=["pdf"], key="strategic_pdf",
                    label_visibility="collapsed",
                )

        with c_up2:
            if llm_ok:
                action_file = st.file_uploader(
                    "action.pdf", type=["pdf"], key="action_pdf",
                    label_visibility="collapsed",
                )

        with c_btn:
            if strategic_file and action_file:
                if st.session_state.get("pipeline_running"):
                    st.button(
                        "Start Analysis", type="primary",
                        width='stretch', disabled=True,
                    )
                else:
                    run_clicked = st.button(
                        "Start Analysis", type="primary",
                        width='stretch',
                    )
            else:
                st.button(
                    "Start Analysis", type="primary",
                    width='stretch', disabled=True,
                )

        stepper_area = c_step.empty()
        if st.session_state.get("pipeline_running"):
            stepper_area.markdown(_build_stepper_html(0), unsafe_allow_html=True)
            c_step.markdown(
                '<div class="processing-label">PROCESSING...</div>',
                unsafe_allow_html=True,
            )
        else:
            stepper_area.markdown(_build_stepper_html(0), unsafe_allow_html=True)

        with c_dl:
            st.button(
                "Download Report as PDF",
                width='stretch', disabled=True,
            )
    else:
        # Pipeline done: [summary] [button] [stepper] [download]
        c_info, c_btn, c_step, c_dl = st.columns(
            [2, 1, 2.5, 1.5], gap="medium",
        )

        report = st.session_state["upload_report"]

        with c_info:
            i1, i2, i3, i4, i5 = st.columns(5)
            with i1:
                st.metric("Score", f"{report['overall_score']:.0%}")
            with i2:
                st.metric("Actions", len(report["action_alignments"]))
            with i3:
                st.metric("Orphans", len(report.get("orphan_actions", [])))
            with i4:
                st.metric("Poor", len(report.get("poorly_aligned_actions", [])))
            with i5:
                st.metric("Good", len(report.get("well_aligned_actions", [])))

        with c_btn:
            if st.button("New Analysis", type="secondary", width='stretch'):
                for key in list(st.session_state.keys()):
                    if key in ("upload_report", "pipeline_done", "pipeline_running") \
                            or key.startswith("dynamic_"):
                        del st.session_state[key]
                st.rerun()

        stepper_area = c_step.empty()
        stepper_area.markdown(_build_stepper_html(6), unsafe_allow_html=True)

        with c_dl:
            from dashboard.data_adapter import build_data_dict
            data = build_data_dict(report, dict(st.session_state))
            pdf_bytes = generate_pdf_report(data)
            if pdf_bytes:
                st.download_button(
                    "Download Report as PDF", data=pdf_bytes,
                    file_name="isps_alignment_report.pdf",
                    mime="application/pdf", width='stretch',
                )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Trigger analysis OUTSIDE columns so stepper updates ──
    if run_clicked and strategic_file and action_file:
        st.session_state["pipeline_running"] = True
        _run_upload_analysis(strategic_file, action_file, stepper_area)

    # ── Tabs + Output ─────────────────────────────────────────────
    if pipeline_done:
        from dashboard.data_adapter import build_data_dict
        report = st.session_state["upload_report"]
        data = build_data_dict(report, dict(st.session_state))

        tabs = st.tabs([
            "Sync Analysis", "Recommendations", "Knowledge Graph",
            "Ontology", "Evaluation",
        ])
        with tabs[0]:
            _render_sync_analysis(data)
        with tabs[1]:
            _render_recommendations(data)
        with tabs[2]:
            _render_knowledge_graph(data)
        with tabs[3]:
            _render_ontology(data)
        with tabs[4]:
            _render_evaluation(data)
    else:
        tabs = st.tabs([
            "Sync Analysis", "Recommendations", "Knowledge Graph",
            "Ontology", "Evaluation",
        ])
        for tab in tabs:
            with tab:
                st.markdown(
                    '<div class="output-placeholder">Output</div>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
