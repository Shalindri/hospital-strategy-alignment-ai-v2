"""
app.py
------
Streamlit dashboard for ISPS — 5 tabs covering all coursework sections.

How it fits in the pipeline:
  All src/ modules --> [this module] --> renders in the browser

Run it with:
  streamlit run dashboard/app.py

Tab map:
  1. Synchronization  --> alignment scores, heatmap, per-action table
  2. Recommendations  --> RAG suggestions, new action proposals
  3. Knowledge Graph  --> interactive pyvis HTML
  4. Ontology         --> concept mappings per objective/action
  5. Evaluation       --> Precision, Recall, F1, AUC, confusion matrix
"""

# --- Standard library ---
import json
import os
import sys

# Make project root importable (needed when launching from dashboard/ subfolder)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Third-party ---
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# --- Local ---
from src.config import (
    STRATEGIC_PLAN_FILE, ACTION_PLAN_FILE, OUTPUTS_DIR,
    THRESHOLD_EXCELLENT, THRESHOLD_GOOD, THRESHOLD_FAIR,
)
from src import (
    alignment_scorer,
    vector_store,
    ontology_mapper,
    knowledge_graph,
    rag_engine,
    agent_reasoner,
)
from tests import evaluation

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ISPS — Strategic Plan Synchronization",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 ISPS — Intelligent Strategic Plan Synchronization")
st.caption("MSc Information Retrieval Coursework · Hospital Strategy Alignment AI")


# =============================================================================
# DATA LOADING HELPERS
# Cached so heavy computation only runs once per Streamlit session.
# =============================================================================

@st.cache_data(show_spinner="Loading data files...")
def load_data() -> tuple:
    """
    Load strategic plan objectives and action plan items from JSON files,
    normalising them into the {id, title, description} format used by all modules.

    Returns:
        tuple: (objectives list, actions list)
    """
    def _load_json(path):
        """Open a JSON file and return its contents."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    sp_data = _load_json(STRATEGIC_PLAN_FILE)
    ap_data = _load_json(ACTION_PLAN_FILE)

    raw_objectives = sp_data.get("objectives", [])
    objectives = []
    for obj in raw_objectives:
        if "id" in obj:
            objectives.append(obj)
        elif "code" in obj:
            # Convert old format to simplified format
            objectives.append({
                "id":          obj["code"],
                "title":       obj.get("name", obj["code"]),
                "description": obj.get("goal_statement", ""),
            })

    raw_actions = ap_data.get("actions", [])
    actions = []
    for act in raw_actions:
        if "id" in act:
            actions.append(act)
        elif "action_number" in act:
            actions.append({
                "id":          str(act["action_number"]),
                "title":       act.get("title", str(act["action_number"])),
                "description": act.get("description", ""),
            })

    return objectives, actions


@st.cache_data(show_spinner="Running alignment scoring (this takes ~30 seconds)...")
def get_alignment_result(_objectives: list) -> dict:
    """
    Run the alignment scorer and return results.
    The leading underscore in _objectives tells Streamlit not to hash this arg.

    Args:
        _objectives (list): List of objective dicts.

    Returns:
        dict: Full alignment result from alignment_scorer.run_alignment().
    """
    return alignment_scorer.run_alignment(_objectives)


# =============================================================================
# COLOUR HELPER
# =============================================================================

def tier_colour(tier: str) -> str:
    """Return a hex colour for each tier label."""
    return {
        "Excellent": "#2e8b57",   # dark green
        "Good":      "#4a90d9",   # blue
        "Fair":      "#f5a623",   # amber
        "Poor":      "#d0021b",   # red
    }.get(tier, "#888888")


# =============================================================================
# TAB 1 — SYNCHRONIZATION
# =============================================================================

def tab_synchronization(alignment_result: dict, objectives: list, actions: list):
    """
    Show overall score gauge, objectives x actions heatmap, and per-action table.

    Args:
        alignment_result (dict): Full alignment result.
        objectives (list): Objective dicts.
        actions (list): Action dicts.
    """
    st.info(
        "**Synchronization Overview** \n\n"
        "This tab measures how well the Annual Action Plan supports each Strategic Objective. "
        "The gauge shows the overall alignment score (0 = no alignment, 1 = perfect). "
        "The heatmap shows every objective-action pair — hover to see exact scores. "
        "The table below lists each action's best-matching objective and its tier label. "
        "Actions highlighted in red are 'orphans' — they have no meaningful link "
        "to any strategic objective and need the most attention from leadership."
    )

    overall = alignment_result["overall_score"]
    matrix  = alignment_result["matrix"]
    classifications = alignment_result["classifications"]
    orphan_ids = {act["id"] for act in alignment_result["orphan_actions"]}

    # --- Gauge chart ---
    col1, col2 = st.columns([1, 2])
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = round(overall * 100, 1),
            title = {"text": "Overall Alignment Score (%)"},
            delta = {"reference": 60},   # reference: "Good" threshold * 100
            gauge = {
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#4a90d9"},
                "steps": [
                    {"range": [0, 42],  "color": "#ffcccc"},   # Poor
                    {"range": [42, 60], "color": "#fff3cc"},   # Fair
                    {"range": [60, 75], "color": "#cce5ff"},   # Good
                    {"range": [75, 100],"color": "#ccffcc"},   # Excellent
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 3},
                    "value": 75
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30, b=0))
        st.plotly_chart(fig_gauge, width='stretch')

        # Quick stats
        tier_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
        for c in classifications:
            tier_counts[c["tier"]] += 1
        for tier, count in tier_counts.items():
            colour = tier_colour(tier)
            st.markdown(
                f'<span style="color:{colour}">■</span> **{tier}**: {count} pairs',
                unsafe_allow_html=True
            )

    # --- Heatmap ---
    with col2:
        # Use full objective titles as row labels so the heatmap is self-explanatory
        obj_labels = [f"{obj['id']}: {obj['title']}" for obj in objectives]
        act_ids    = [act["id"] for act in actions]

        # matrix is (n_objectives x n_actions) — build DataFrame
        import numpy as np
        mat_np = np.array(matrix)
        df_heat = pd.DataFrame(mat_np, index=obj_labels, columns=act_ids)

        fig_heat = px.imshow(
            df_heat,
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            aspect="auto",
            title="Alignment Heatmap (objectives × actions)",
            labels={"color": "Score"},
        )
        fig_heat.update_layout(height=350, margin=dict(t=40, b=0))
        st.plotly_chart(fig_heat, width='stretch')

    # --- Per-action summary table ---
    st.subheader("Per-Action Alignment Summary")

    # Build a table: action | title | best_score | tier | best_objective | orphan
    action_lookup = {act["id"]: act["title"] for act in actions}
    obj_lookup    = {obj["id"]: obj["title"] for obj in objectives}

    # Find best score & objective for each action
    best_per_action: dict = {}
    for c in classifications:
        aid = c["action_id"]
        if aid not in best_per_action or c["score"] > best_per_action[aid]["score"]:
            best_per_action[aid] = {
                "score":    c["score"],
                "tier":     c["tier"],
                "best_obj": c["objective_id"],
            }

    # Find which objective has the highest average score across all actions
    obj_avg_scores = {}
    for obj in objectives:
        oid = obj["id"]
        scores = [c["score"] for c in classifications if c["objective_id"] == oid]
        obj_avg_scores[oid] = round(sum(scores) / len(scores), 3) if scores else 0.0
    best_objective_id = max(obj_avg_scores, key=obj_avg_scores.get)

    rows = []
    for act in actions:
        aid   = act["id"]
        info  = best_per_action.get(aid, {"score": 0.0, "tier": "Poor", "best_obj": ""})
        best_obj_id    = info["best_obj"]
        best_obj_title = obj_lookup.get(best_obj_id, best_obj_id)
        rows.append({
            "Action ID":      aid,
            "Action Title":   act["title"],
            "Best Score":     round(info["score"], 3),
            "Tier":           info["tier"],
            "Best Objective": f"⭐ {best_obj_id}: {best_obj_title}" if best_obj_id == best_objective_id else f"{best_obj_id}: {best_obj_title}",
            "Orphan":         "⚠️ Yes" if aid in orphan_ids else "",
        })

    df_table = pd.DataFrame(rows)

    # Colour the Tier column
    def colour_tier(val):
        colours = {
            "Excellent": "background-color: #ccffcc",
            "Good":      "background-color: #cce5ff",
            "Fair":      "background-color: #fff3cc",
            "Poor":      "background-color: #ffcccc",
        }
        return colours.get(val, "")

    st.dataframe(
        df_table.style.applymap(colour_tier, subset=["Tier"]),
        width='stretch', height=400
    )

    if orphan_ids:
        st.warning(
            f"⚠️ {len(orphan_ids)} orphan action(s) detected — "
            "these score below {THRESHOLD_FAIR} against every strategic objective. "
            "See the Recommendations tab for improvement suggestions."
        )


# =============================================================================
# TAB 2 — RECOMMENDATIONS
# =============================================================================

def tab_recommendations(alignment_result: dict):
    """
    Show RAG improvement suggestions and new action proposals.

    Args:
        alignment_result (dict): Full alignment result.
    """
    st.info(
        "**AI-Powered Recommendations** \n\n"
        "For each poorly-aligned action (best score below 'Good'), the system: \n"
        "1. Retrieves the 3 most relevant strategic objectives from the vector database \n"
        "2. Asks GPT-4o-mini to suggest 3 specific improvements \n\n"
        "For each objective with no 'Excellent'-tier action, the system proposes a "
        "brand-new action to close the gap. \n\n"
        "Use these suggestions as a starting point for your planning discussions."
    )

    if st.button("Generate Recommendations (calls OpenAI API)"):
        with st.spinner("Running RAG engine... this may take 1-2 minutes..."):
            try:
                # First ensure objectives are embedded in ChromaDB
                vector_store.embed_and_store(
                    alignment_result["objectives"],
                    "strategic_objectives"
                )
                rag_result = rag_engine.run_rag_suggestions(alignment_result)
                st.session_state["rag_result"] = rag_result
            except Exception as e:
                st.error(f"RAG engine error: {e}")
                return

    rag_result = st.session_state.get("rag_result")
    if not rag_result:
        st.caption("Click the button above to generate recommendations.")
        return

    # --- Improvement suggestions ---
    st.subheader("Improvement Suggestions for Poorly-Aligned Actions")
    improvements = rag_result.get("improvement_suggestions", {})

    if improvements:
        action_lookup = {act["id"]: act for act in alignment_result["actions"]}
        for act_id, suggestion in improvements.items():
            act = action_lookup.get(act_id, {"title": act_id})
            with st.expander(f"[{act_id}] {act['title']}", expanded=False):
                st.markdown(suggestion)
    else:
        st.success("All actions are sufficiently aligned — no improvements needed!")

    # --- New action proposals ---
    st.subheader("Proposed New Actions for Uncovered Objectives")
    proposals = rag_result.get("new_action_proposals", {})

    if proposals:
        obj_lookup = {obj["id"]: obj for obj in alignment_result["objectives"]}
        for obj_id, proposal in proposals.items():
            obj = obj_lookup.get(obj_id, {"title": obj_id})
            with st.expander(f"[{obj_id}] {obj['title']}", expanded=False):
                st.info("📋 **Proposed New Action:**")
                st.markdown(proposal)
    else:
        st.success("All strategic objectives have at least one Excellent-tier action!")


# =============================================================================
# TAB 3 — KNOWLEDGE GRAPH
# =============================================================================

def tab_knowledge_graph(alignment_result: dict, objectives: list, actions: list):
    """
    Embed the pyvis interactive HTML knowledge graph in Streamlit.

    Args:
        alignment_result (dict): Full alignment result.
        objectives (list): Objective dicts.
        actions (list): Action dicts.
    """
    st.info(
        "**Knowledge Graph** \n\n"
        "Each node is a strategic objective (blue) or action plan item (green). "
        "Edges connect objectives to the actions that support them, with edge thickness "
        "proportional to alignment score. Gold nodes are 'bridge nodes' — highly "
        "connected items that are central to the plan's coherence. \n\n"
        "Drag nodes to explore, scroll to zoom, hover for details."
    )

    html_path = os.path.join(OUTPUTS_DIR, "knowledge_graph.html")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("**Legend:**")
        st.markdown("🔵 Strategic Objective")
        st.markdown("🟢 Action Plan Item")
        st.markdown("🟡 Bridge Node (high centrality)")

        regenerate = st.button("Regenerate Graph")

    with col1:
        if regenerate or not os.path.exists(html_path):
            with st.spinner("Building knowledge graph..."):
                try:
                    knowledge_graph.run_knowledge_graph(objectives, actions, alignment_result)
                except Exception as e:
                    st.error(f"Knowledge graph error: {e}")
                    return

        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=700, scrolling=True)
        else:
            st.warning("Knowledge graph HTML not found. Click 'Regenerate Graph'.")


# =============================================================================
# TAB 4 — ONTOLOGY
# =============================================================================

def tab_ontology(alignment_result: dict, objectives: list, actions: list):
    """
    Show healthcare concept mappings for all objectives and actions.

    Args:
        alignment_result (dict): Full alignment result.
        objectives (list): Objective dicts.
        actions (list): Action dicts.
    """
    st.info(
        "**Ontology Mapping** \n\n"
        "Each objective and action is classified into a healthcare concept class "
        "(ClinicalCare, PatientSafety, Finance, etc.) using keyword matching. \n\n"
        "The coverage chart shows whether the Action Plan covers all the same concept "
        "areas as the Strategic Plan. Gaps (concepts in strategy but absent from actions) "
        "indicate areas where the plan needs strengthening."
    )

    with st.spinner("Running ontology mapper..."):
        try:
            mappings = ontology_mapper.run_ontology_mapping(objectives, actions)
        except Exception as e:
            st.error(f"Ontology mapper error: {e}")
            return

    obj_ids = {obj["id"] for obj in objectives}
    act_ids = {act["id"] for act in actions}
    obj_lookup = {obj["id"]: obj["title"] for obj in objectives}
    act_lookup = {act["id"]: act["title"] for act in actions}

    # Build table rows
    rows = []
    for item_id, concept in mappings.items():
        if item_id in obj_ids:
            item_type = "Objective"
            title = obj_lookup.get(item_id, item_id)
        else:
            item_type = "Action"
            title = act_lookup.get(item_id, item_id)

        rows.append({
            "ID":      item_id,
            "Type":    item_type,
            "Title":   title[:60],
            "Concept": concept,
        })

    df_ont = pd.DataFrame(rows)

    # Show table
    st.subheader("Item-to-Concept Mappings")
    st.dataframe(df_ont, width='stretch', height=350)

    # Coverage comparison chart
    st.subheader("Concept Coverage: Strategy vs Action Plan")

    obj_concepts = [mappings[obj["id"]] for obj in objectives if obj["id"] in mappings]
    act_concepts = [mappings[act["id"]] for act in actions    if act["id"] in mappings]

    all_concepts = sorted(set(obj_concepts + act_concepts))

    from collections import Counter
    obj_counts = Counter(obj_concepts)
    act_counts = Counter(act_concepts)

    df_cov = pd.DataFrame({
        "Concept":     all_concepts,
        "Strategy":    [obj_counts.get(c, 0) for c in all_concepts],
        "Action Plan": [act_counts.get(c, 0) for c in all_concepts],
    })

    fig_cov = px.bar(
        df_cov.melt(id_vars="Concept", var_name="Plan", value_name="Count"),
        x="Concept", y="Count", color="Plan", barmode="group",
        title="Concept Coverage Comparison",
        color_discrete_map={"Strategy": "#4a90d9", "Action Plan": "#5bad72"},
    )
    fig_cov.update_layout(height=350, xaxis_tickangle=-30)
    st.plotly_chart(fig_cov, width='stretch')

    # Highlight gaps
    gaps = [c for c in all_concepts
            if obj_counts.get(c, 0) > 0 and act_counts.get(c, 0) == 0]
    if gaps:
        st.warning(
            f"⚠️ Coverage gap: The strategic plan includes items in "
            f"**{', '.join(gaps)}** but no action plan items map to these concepts."
        )
    else:
        st.success("All strategic concept areas are covered by the action plan.")

    ttl_path = os.path.join(OUTPUTS_DIR, "ontology.ttl")
    if os.path.exists(ttl_path):
        st.caption(f"Turtle ontology file saved at: `{ttl_path}`")


# =============================================================================
# TAB 5 — EVALUATION
# =============================================================================

def tab_evaluation(objectives: list):
    """
    Show Precision, Recall, F1, AUC, and confusion matrix vs ground truth.

    Args:
        objectives (list): Objective dicts (used to run alignment scorer).
    """
    st.info(
        "**System Evaluation** \n\n"
        "We compare the ISPS predictions against a 58-pair human-annotated ground truth "
        "dataset. Each pair is a (strategic objective, action plan item) combination "
        "with a human-assigned alignment score. \n\n"
        "Precision answers: *'Of the pairs the system marked as aligned, how many truly were?'* \n"
        "Recall answers: *'Of all truly aligned pairs, how many did the system find?'* \n"
        "F1 balances both. AUC measures ranking quality (1.0 = perfect, 0.5 = random)."
    )

    if st.button("Run Evaluation (compares against ground truth)"):
        with st.spinner("Running evaluation..."):
            try:
                metrics = evaluation.run_evaluation(objectives)
                st.session_state["eval_metrics"] = metrics
            except Exception as e:
                st.error(f"Evaluation error: {e}")
                return

    # Try to load from saved file if not yet in session
    if "eval_metrics" not in st.session_state:
        saved_path = os.path.join(OUTPUTS_DIR, "evaluation_results.json")
        if os.path.exists(saved_path):
            with open(saved_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Build a metrics dict compatible with our display code
            st.session_state["eval_metrics"] = {
                "precision":          saved.get("precision") or 0,
                "recall":             saved.get("recall") or 0,
                "f1":                 saved.get("f1") or 0,
                "auc":                saved.get("auc") or 0,
                "pearson_r":          saved.get("pearson_r") or 0,
                "pearson_p":          saved.get("pearson_p") or 0,
                "confusion_matrix":   saved.get("confusion_matrix") or [[0,0],[0,0]],
                "y_true_binary":      [],
                "y_pred_binary":      [],
                "optimal_threshold":  saved.get("optimal_threshold"),
                "optimal_f1":         saved.get("optimal_f1"),
                "sweep_table":        None,  # not stored in JSON — re-run to get it
            }
            st.caption("Showing previously saved evaluation results. "
                       "Click 'Run Evaluation' to recompute.")

    metrics = st.session_state.get("eval_metrics")
    if not metrics:
        st.caption("Click 'Run Evaluation' above to compute metrics.")
        return

    # --- Metric cards ---
    st.subheader("Evaluation Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("Recall",    f"{metrics['recall']:.3f}")
    col3.metric("F1 Score",  f"{metrics['f1']:.3f}")
    col4.metric("AUC (ROC)", f"{metrics['auc']:.3f}")
    col5.metric("Pearson r", f"{metrics['pearson_r']:.3f}")

    # --- Confusion matrix heatmap ---
    st.subheader("Confusion Matrix")
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

    if len(cm) >= 2:
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Not Aligned", "Predicted Aligned"],
            y=["Actual Not Aligned",    "Actual Aligned"],
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, width='stretch')

    # --- Interpretation note ---
    f1  = metrics["f1"]
    auc = metrics["auc"]

    # Show optimal threshold hint if the evaluation swept thresholds
    opt_thresh = metrics.get("optimal_threshold")
    opt_f1     = metrics.get("optimal_f1")

    if opt_thresh and opt_thresh != THRESHOLD_FAIR and opt_f1 and opt_f1 > f1 + 0.01:
        st.info(
            f"**Threshold tuning suggestion:** The sweep found that "
            f"`THRESHOLD_FAIR = {opt_thresh}` gives a better F1 of **{opt_f1:.3f}** "
            f"(current threshold {THRESHOLD_FAIR} gives F1={f1:.3f}). "
            f"Update `THRESHOLD_FAIR` in [src/config.py](src/config.py) to improve results."
        )

    # AUC is a threshold-independent measure of ranking quality — use it as primary signal
    if auc >= 0.85:
        st.success(
            f"**Strong performance** — AUC={auc:.3f}, F1={f1:.3f}. "
            f"The model correctly ranks aligned pairs above non-aligned ones."
        )
    elif auc >= 0.70:
        st.info(
            f"**Good ranking quality** — AUC={auc:.3f}, F1={f1:.3f}. "
            f"The model separates aligned and non-aligned pairs reasonably well. "
            f"F1 can be improved by tuning `THRESHOLD_FAIR` in config.py "
            f"(see the sweep suggestion above)."
        )
    else:
        st.warning(
            f"**Low AUC ({auc:.3f})** — the model struggles to rank aligned pairs above "
            f"non-aligned ones. This suggests the embedding model may not capture enough "
            f"domain-specific meaning for this dataset."
        )

    # --- Threshold sweep chart ---
    sweep_table = metrics.get("sweep_table")
    if sweep_table:
        st.subheader("Threshold Sweep")
        st.caption(
            "This chart shows how F1, Precision, and Recall change as we vary the "
            "classification threshold. The peak of the F1 curve is the optimal threshold."
        )
        df_sweep = pd.DataFrame(sweep_table)
        fig_sweep = px.line(
            df_sweep.melt(id_vars="threshold", var_name="Metric", value_name="Score"),
            x="threshold", y="Score", color="Metric",
            title="F1 / Precision / Recall vs Threshold",
            color_discrete_map={"f1": "#4a90d9", "precision": "#5bad72", "recall": "#f5a623"},
        )
        # Mark the current threshold
        fig_sweep.add_vline(
            x=THRESHOLD_FAIR, line_dash="dash", line_color="red",
            annotation_text=f"Current ({THRESHOLD_FAIR})", annotation_position="top right"
        )
        if opt_thresh and opt_thresh != THRESHOLD_FAIR:
            fig_sweep.add_vline(
                x=opt_thresh, line_dash="dot", line_color="green",
                annotation_text=f"Optimal ({opt_thresh})", annotation_position="top left"
            )
        fig_sweep.update_layout(height=350, xaxis_title="THRESHOLD_FAIR value",
                                yaxis_range=[0, 1])
        st.plotly_chart(fig_sweep, width='stretch')

    saved_path = os.path.join(OUTPUTS_DIR, "evaluation_results.json")
    if os.path.exists(saved_path):
        st.caption(f"Results saved at: `{saved_path}`")


# =============================================================================
# MAIN — CHECK DATA FILES, THEN RENDER ALL TABS
# =============================================================================

def main():
    """Entry point: check data, load, run alignment, render 5 tabs."""

    # Guard: check that data files exist before proceeding
    if not os.path.exists(STRATEGIC_PLAN_FILE) or not os.path.exists(ACTION_PLAN_FILE):
        st.error(
            "**Data files not found.**\n\n"
            f"Expected:\n"
            f"- `{STRATEGIC_PLAN_FILE}`\n"
            f"- `{ACTION_PLAN_FILE}`\n\n"
            "Please run `src/pdf_processor.py` to extract data from your PDFs first."
        )
        st.stop()

    # Load data
    try:
        objectives, actions = load_data()
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.stop()

    # Run alignment scoring
    try:
        alignment_result = get_alignment_result(tuple(
            (obj["id"], obj["title"]) for obj in objectives
        ))
        # get_alignment_result takes a hashable arg — we pass a tuple of (id, title) pairs
        # and re-run internally. Simpler: just run directly since caching is by arg hash.
    except Exception:
        alignment_result = None

    # The @st.cache_data trick above doesn't work because the arg isn't the full list.
    # Let's use session_state instead for the alignment result.
    if "alignment_result" not in st.session_state:
        with st.spinner("Running alignment scoring..."):
            try:
                st.session_state["alignment_result"] = alignment_scorer.run_alignment(objectives)
            except Exception as e:
                st.error(f"Alignment scorer error: {e}")
                st.stop()

    alignment_result = st.session_state["alignment_result"]

    # Render the 5 tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Synchronization",
        "💡 Recommendations",
        "🕸️ Knowledge Graph",
        "🔬 Ontology",
        "📈 Evaluation",
    ])

    with tab1:
        tab_synchronization(alignment_result, objectives, actions)

    with tab2:
        tab_recommendations(alignment_result)

    with tab3:
        tab_knowledge_graph(alignment_result, objectives, actions)

    with tab4:
        tab_ontology(alignment_result, objectives, actions)

    with tab5:
        tab_evaluation(objectives)


if __name__ == "__main__":
    main()
