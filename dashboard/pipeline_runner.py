"""
On-demand pipeline runners for dynamic analysis.

Each function runs a single pipeline stage against uploaded data
and returns the result dict in the same format as the static JSON files.

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dashboard.data_adapter import build_data_dict

logger = logging.getLogger("pipeline_runner")


# ──────────────────────────────────────────────────────────────────────
# Tier 2: Ontology Mapping (no LLM, uses sentence-transformers)
# ──────────────────────────────────────────────────────────────────────

def run_dynamic_ontology(upload_report: dict[str, Any]) -> tuple[dict, dict]:
    """Run ontology mapping on uploaded data.

    Returns:
        Tuple of (mappings_dict, gaps_dict).
    """
    from src.ontology_mapper import OntologyMapper

    mapper = OntologyMapper()

    # Convert uploaded objectives to the format OntologyMapper expects
    strategic_data = upload_report.get("strategic_data", {})
    action_data = upload_report.get("action_data", {})

    objectives = []
    for obj in strategic_data.get("objectives", []):
        objectives.append({
            "code": obj.get("code", ""),
            "name": obj.get("name", ""),
            "goal_statement": obj.get("goal_statement", ""),
            "goals": [
                {"id": g.get("id", ""), "description": g.get("description", "")}
                for g in obj.get("strategic_goals", [])
            ],
            "kpis": [
                k.get("KPI", k) if isinstance(k, dict) else str(k)
                for k in obj.get("kpis", [])
            ],
        })

    actions = []
    for act in action_data.get("actions", []):
        num = act.get("action_number", 0)
        actions.append({
            "number": num,
            "action_number": num,
            "title": act.get("title", ""),
            "objective_code": act.get("strategic_objective_code", ""),
            "strategic_objective_code": act.get("strategic_objective_code", ""),
            "strategic_objective_name": act.get("strategic_objective_name", ""),
            "description": act.get("description", ""),
            "outcome": act.get("expected_outcome", ""),
            "expected_outcome": act.get("expected_outcome", ""),
            "owner": act.get("action_owner", ""),
            "action_owner": act.get("action_owner", ""),
            "timeline": act.get("timeline", ""),
            "quarters": act.get("quarters", []),
            "budget": f"LKR {act.get('budget_lkr_millions', 0)}M",
            "budget_lkr_millions": act.get("budget_lkr_millions", 0),
            "kpis": act.get("kpis", []),
            "keywords": act.get("keywords", []),
        })

    # Inject data directly, skip file parsing
    mapper.strategic_objectives = objectives
    mapper.actions = actions

    # Run pipeline steps (skip parse, skip file export)
    mapper.build_ontology_schema()
    mapper.map_all()
    mapper.detect_gaps()
    mapper.build_rdf_instances()

    # Extract mappings dict (same format as export_mappings)
    mappings_data = {
        "metadata": {
            "embedding_model": mapper.embedding_model_name
                if hasattr(mapper, "embedding_model_name") else "all-MiniLM-L6-v2",
            "mapping_threshold": mapper.threshold,
            "total_concepts": len(mapper.concept_embeddings)
                if hasattr(mapper, "concept_embeddings") else 0,
        },
        "action_mappings": [
            {
                "item_id": am.item_id,
                "item_type": am.item_type,
                "top_level_areas": am.top_level_areas,
                "is_multi_area": am.is_multi_area,
                "mappings": [asdict(m) for m in am.mappings],
            }
            for am in mapper.action_mappings
        ],
        "strategy_mappings": [
            {
                "item_id": sm.item_id,
                "item_type": sm.item_type,
                "top_level_areas": sm.top_level_areas,
                "is_multi_area": sm.is_multi_area,
                "mappings": [asdict(m) for m in sm.mappings],
            }
            for sm in mapper.strategy_mappings
        ],
    }

    # Extract gaps dict (same format as export_gaps)
    gaps_data = {
        "uncovered_strategy_concepts": mapper.gap_report.uncovered_concepts,
        "weak_actions": mapper.gap_report.weak_actions,
        "conflicting_actions": mapper.gap_report.conflicting_actions,
        "summary": {
            "uncovered_concept_count": len(mapper.gap_report.uncovered_concepts),
            "weak_action_count": len(mapper.gap_report.weak_actions),
            "conflicting_action_count": len(mapper.gap_report.conflicting_actions),
        },
    }

    logger.info("Dynamic ontology: %d action mappings, %d strategy mappings, %d gaps",
                len(mappings_data["action_mappings"]),
                len(mappings_data["strategy_mappings"]),
                len(gaps_data["weak_actions"]))
    return mappings_data, gaps_data


# ──────────────────────────────────────────────────────────────────────
# Tier 2: Knowledge Graph (no LLM, uses networkx)
# ──────────────────────────────────────────────────────────────────────

def run_dynamic_kg(
    upload_report: dict[str, Any],
    mappings: dict[str, Any],
) -> dict[str, Any]:
    """Build knowledge graph from uploaded data.

    Args:
        upload_report: The alignment report.
        mappings: Ontology mappings (from run_dynamic_ontology).

    Returns:
        KG dict in node-link JSON format with insights.
    """
    from networkx.readwrite import json_graph
    from src.knowledge_graph import KnowledgeGraphBuilder

    kg = KnowledgeGraphBuilder.__new__(KnowledgeGraphBuilder)
    # Initialize attributes manually (skip __init__ which reads files)
    import networkx as nx
    from src.knowledge_graph import GraphInsights
    kg.G = nx.DiGraph()
    kg.insights = GraphInsights()

    # Inject data
    data = build_data_dict(upload_report)
    kg._strategic = upload_report.get("strategic_data", {})
    kg._actions = upload_report.get("action_data", {})
    kg._alignment = data["alignment"]
    kg._mappings = mappings

    # Build graph (skip _load_data)
    kg._add_strategy_nodes()
    kg._add_action_nodes()
    kg._add_ontology_concept_nodes()
    kg._add_alignment_edges()
    kg._add_ontology_edges()
    kg._normalise_sizes()
    kg.compute_metrics()
    kg.identify_bottlenecks()
    kg.suggest_new_connections()

    # Export to dict (same format as export_json)
    result = json_graph.node_link_data(kg.G)
    result["insights"] = {
        "node_count": kg.insights.node_count,
        "edge_count": kg.insights.edge_count,
        "community_count": len(kg.insights.communities),
        "isolated_actions": kg.insights.isolated_actions,
        "isolated_strategies": getattr(kg.insights, "isolated_strategies", []),
        "bridge_nodes": kg.insights.bridge_nodes[:5],
        "bottlenecks": kg.insights.bottlenecks[:5],
        "new_connections": kg.insights.new_connections[:5],
    }

    logger.info("Dynamic KG: %d nodes, %d edges",
                result["insights"]["node_count"],
                result["insights"]["edge_count"])
    return result


# ──────────────────────────────────────────────────────────────────────
# Tier 3: RAG Recommendations (requires OpenAI LLM)
# ──────────────────────────────────────────────────────────────────────

def run_dynamic_rag(upload_report: dict[str, Any]) -> dict[str, Any]:
    """Run RAG recommendation engine on uploaded data.

    Returns:
        RAG results dict with improvements and suggestions.
    """
    import shutil
    from src.rag_engine import RAGEngine
    from src.vector_store import VectorStore

    temp_dir = tempfile.mkdtemp(prefix="isps_rag_")
    try:
        # Create temporary vector store with uploaded data
        vs = VectorStore(chroma_dir=temp_dir)
        _embed_data_into_vs(vs, upload_report)

        # Create RAG engine with our vector store
        engine = RAGEngine(vector_store=vs)

        # Override loaded data with uploaded data
        engine._strategic_data = upload_report.get("strategic_data", {})
        engine._action_data = upload_report.get("action_data", {})
        data = build_data_dict(upload_report)
        engine._alignment_report = data["alignment"]

        # Run RAG pipeline
        results = engine.run()
        return asdict(results)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
# Tier 3: Agent Reasoner (requires OpenAI LLM)
# ──────────────────────────────────────────────────────────────────────

def run_dynamic_agent(
    upload_report: dict[str, Any],
    mappings: dict[str, Any],
    kg_data: dict[str, Any],
) -> tuple[dict, dict]:
    """Run agent reasoner on uploaded data.

    Args:
        upload_report: The alignment report.
        mappings: Ontology mappings.
        kg_data: Knowledge graph data.

    Returns:
        Tuple of (agent_recs_dict, agent_trace_dict).
    """
    import shutil
    from src.agent_reasoner import AgentReasoner, ISSUE_SCORE_THRESHOLD
    from src.config import OPENAI_MODEL
    from src.vector_store import VectorStore

    temp_dir = tempfile.mkdtemp(prefix="isps_agent_")
    try:
        # Create temporary vector store
        vs = VectorStore(chroma_dir=temp_dir)
        _embed_data_into_vs(vs, upload_report)

        # Create agent (skip __init__ to avoid file reads)
        agent = AgentReasoner.__new__(AgentReasoner)
        from src.config import MAX_ITERATIONS
        agent.max_iterations = MAX_ITERATIONS

        # Inject data
        data = build_data_dict(upload_report)
        agent._alignment = data["alignment"]
        agent._mappings = mappings
        agent._strategic = upload_report.get("strategic_data", {})
        agent._actions = upload_report.get("action_data", {})
        agent._kg_data = kg_data
        agent._vs = vs

        # Initialize LLM (provider selected via .env)
        from src.config import get_llm, LLM_TEMPERATURE
        agent._llm = get_llm(temperature=LLM_TEMPERATURE)

        # Build lookup indices
        agent._action_by_num = {
            a["action_number"]: a for a in agent._actions.get("actions", [])
        }
        agent._objective_by_code = {
            o["code"]: o for o in agent._strategic.get("objectives", [])
        }
        agent._alignment_by_action = {
            aa["action_number"]: aa
            for aa in agent._alignment.get("action_alignments", [])
        }
        agent._mapping_by_action = {
            m["item_id"]: m for m in mappings.get("action_mappings", [])
        }
        agent._mapping_by_goal = {
            m["item_id"]: m for m in mappings.get("strategy_mappings", [])
        }

        # Initialize state
        agent._issues = []
        agent._recommendations = []
        agent._traces = []
        agent._tool_call_count = 0

        # Run agent
        recommendations = agent.run()

        # Build output dicts (same format as save_recommendations/save_trace)
        agent_recs = {
            "metadata": {
                "agent_model": OPENAI_MODEL,
                "max_iterations": agent.max_iterations,
                "issue_threshold": ISSUE_SCORE_THRESHOLD,
                "total_issues_diagnosed": len(agent._issues),
                "total_recommendations": len(agent._recommendations),
                "total_impact_score": round(
                    sum(r.impact_score for r in agent._recommendations), 4
                ),
            },
            "recommendations": [asdict(r) for r in agent._recommendations],
            "diagnosed_issues_summary": [
                {
                    "issue_id": i.issue_id,
                    "type": i.issue_type,
                    "priority_score": i.priority_score,
                    "affected_objectives": i.affected_objectives,
                    "addressed": i.issue_id in {r.issue_id for r in agent._recommendations},
                }
                for i in agent._issues[:15]
            ],
        }

        agent_trace = {
            "metadata": {
                "total_iterations": len(agent._traces),
                "agent_model": OPENAI_MODEL,
            },
            "traces": [asdict(t) for t in agent._traces],
        }

        logger.info("Dynamic agent: %d issues diagnosed, %d recommendations",
                    len(agent._issues), len(agent._recommendations))
        return agent_recs, agent_trace

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
# Shared helper: embed uploaded data into a VectorStore
# ──────────────────────────────────────────────────────────────────────

def _embed_data_into_vs(
    vs: "VectorStore",
    upload_report: dict[str, Any],
) -> None:
    """Embed objectives and actions into a VectorStore instance.

    Reuses the same logic as dynamic_analyzer.py.
    """
    from src.dynamic_analyzer import (
        _build_objective_text_generic,
        _build_action_text_generic,
    )
    from src.vector_store import STRATEGIC_COLLECTION, ACTION_COLLECTION

    strategic_data = upload_report.get("strategic_data", {})
    action_data = upload_report.get("action_data", {})

    # Embed objectives
    obj_texts, obj_ids, obj_metas = [], [], []
    for obj in strategic_data.get("objectives", []):
        text = _build_objective_text_generic(obj)
        code = obj.get("code", chr(65 + len(obj_ids)))
        obj_texts.append(text)
        obj_ids.append(f"obj_{code}")
        obj_metas.append({
            "code": code,
            "name": obj.get("name", f"Objective {code}"),
            "goal_statement": obj.get("goal_statement", "")[:500],
            "num_goals": len(obj.get("strategic_goals", [])),
            "num_kpis": len(obj.get("kpis", [])),
            "keywords": ", ".join(obj.get("keywords", [])),
        })

    if obj_texts:
        vs.embed_documents(
            texts=obj_texts,
            collection_name=STRATEGIC_COLLECTION,
            ids=obj_ids,
            metadatas=obj_metas,
        )

    # Embed actions
    act_texts, act_ids, act_metas = [], [], []
    for act in action_data.get("actions", []):
        text = _build_action_text_generic(act)
        num = act.get("action_number", len(act_ids) + 1)
        act_texts.append(text)
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

    if act_texts:
        vs.embed_documents(
            texts=act_texts,
            collection_name=ACTION_COLLECTION,
            ids=act_ids,
            metadatas=act_metas,
        )
