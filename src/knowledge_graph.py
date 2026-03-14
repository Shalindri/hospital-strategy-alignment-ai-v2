"""
Knowledge Graph for Hospital Strategy--Action Plan Alignment System (ISPS).

This module builds and analyses a directed, weighted, typed knowledge graph
that integrates data from all upstream pipeline stages:

*  Parsed strategic objectives and action items (JSON).
*  Embedding-based alignment scores (alignment report).
*  Ontology concept mappings (mappings.json).

The resulting graph captures the full ecosystem of relationships among
strategic objectives, goals, ontology concepts, actions, KPIs,
stakeholders, and timeline quarters --- enabling rich structural
analysis (centrality, community detection, critical-path search) that
complements the vector-similarity and ontology-mapping layers.

Node types
----------
=================  ==========  ===================================
Type               Colour      Description
=================  ==========  ===================================
StrategyObjective  ``#4285F4``  Top-level objectives (A--E)
StrategyGoal       ``#81D4FA``  Individual goals (A1, A2 ...)
OntologyConcept    ``#9C27B0``  Mid-/top-level ontology concepts
Action             ``#4CAF50``  Operational actions (1--25)
KPI                ``#FFEB3B``  Key performance indicators
Stakeholder        ``#FF9800``  Action owners / responsible parties
TimelineQuarter    ``#9E9E9E``  Q1--Q4 2025
=================  ==========  ===================================

Exports
-------
*  ``outputs/strategy_action_kg.gexf``    -- for Gephi
*  ``outputs/strategy_action_kg.graphml`` -- XML interchange
*  ``outputs/strategy_action_kg.json``    -- node-link JSON for dashboard

Typical usage::

    from src.knowledge_graph import KnowledgeGraphBuilder

    kg = KnowledgeGraphBuilder()
    kg.build()
    kg.export_all()

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("knowledge_graph")

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

STRATEGIC_JSON = DATA_DIR / "strategic_plan.json"
ACTION_JSON = DATA_DIR / "action_plan.json"
ALIGNMENT_JSON = DATA_DIR / "alignment_report.json"
MAPPINGS_JSON = OUTPUT_DIR / "mappings.json"

GEXF_OUT = OUTPUT_DIR / "strategy_action_kg.gexf"
GRAPHML_OUT = OUTPUT_DIR / "strategy_action_kg.graphml"
JSON_OUT = OUTPUT_DIR / "strategy_action_kg.json"
PNG_OUT = OUTPUT_DIR / "strategy_action_kg.png"

# Node-type colour palette (hex strings for Gephi / dashboard)
NODE_COLOURS: dict[str, str] = {
    "StrategyObjective": "#4285F4",
    "StrategyGoal": "#81D4FA",
    "OntologyConcept": "#9C27B0",
    "Action": "#4CAF50",
    "KPI": "#FFEB3B",
    "Stakeholder": "#FF9800",
    "TimelineQuarter": "#9E9E9E",
}

# Alignment thresholds (from shared config)
from src.config import THRESHOLD_GOOD, THRESHOLD_FAIR  # noqa: E402

# Deterministic layout seed
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data-class for analysis results
# ---------------------------------------------------------------------------

@dataclass
class GraphInsights:
    """Container for knowledge-graph analytics.

    Attributes:
        node_count:          Total nodes in the graph.
        edge_count:          Total edges in the graph.
        degree_centrality:   Top-k nodes by degree centrality (per type).
        betweenness_centrality: Top-k nodes by betweenness centrality.
        communities:         List of community sets from modularity detection.
        isolated_strategies: Objectives with no action above FAIR threshold.
        isolated_actions:    Actions with no valid ontology mapping.
        bridge_nodes:        Nodes with highest betweenness (cross-area connectors).
        bottlenecks:         Top-k bottleneck nodes with explanations.
        critical_paths:      Cached critical-path results (populated on demand).
        new_connections:     Suggested new edges with evidence.
    """
    node_count: int = 0
    edge_count: int = 0
    degree_centrality: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    betweenness_centrality: list[tuple[str, float]] = field(default_factory=list)
    communities: list[list[str]] = field(default_factory=list)
    isolated_strategies: list[str] = field(default_factory=list)
    isolated_actions: list[str] = field(default_factory=list)
    bridge_nodes: list[dict[str, Any]] = field(default_factory=list)
    bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    critical_paths: list[dict[str, Any]] = field(default_factory=list)
    new_connections: list[dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# KnowledgeGraphBuilder
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeGraphBuilder:
    """Constructs and analyses the strategy--action knowledge graph.

    The builder loads data from the upstream pipeline JSON files, creates
    a ``networkx.DiGraph`` with typed / coloured / weighted nodes and
    edges, computes structural metrics, and exports to multiple formats.

    Args:
        strategic_json: Path to parsed strategic plan JSON.
        action_json:    Path to parsed action plan JSON.
        alignment_json: Path to alignment report JSON.
        mappings_json:  Path to ontology mappings JSON.
    """

    def __init__(
        self,
        strategic_json: Path = STRATEGIC_JSON,
        action_json: Path = ACTION_JSON,
        alignment_json: Path = ALIGNMENT_JSON,
        mappings_json: Path = MAPPINGS_JSON,
    ) -> None:
        self.strategic_json = strategic_json
        self.action_json = action_json
        self.alignment_json = alignment_json
        self.mappings_json = mappings_json

        self.G: nx.DiGraph = nx.DiGraph()
        self.insights = GraphInsights()

        # Raw data (loaded in _load_data)
        self._strategic: dict[str, Any] = {}
        self._actions: dict[str, Any] = {}
        self._alignment: dict[str, Any] = {}
        self._mappings: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load all upstream JSON files into memory."""
        with open(self.strategic_json, encoding="utf-8") as f:
            self._strategic = json.load(f)
        with open(self.action_json, encoding="utf-8") as f:
            self._actions = json.load(f)
        with open(self.alignment_json, encoding="utf-8") as f:
            self._alignment = json.load(f)
        with open(self.mappings_json, encoding="utf-8") as f:
            self._mappings = json.load(f)
        logger.info("Loaded all upstream JSON data.")

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _add_strategy_nodes(self) -> None:
        """Add StrategyObjective and StrategyGoal nodes."""
        for obj in self._strategic.get("objectives", []):
            code = obj["code"]
            obj_id = f"obj_{code}"

            self.G.add_node(
                obj_id,
                label=f"Objective {code}: {obj['name']}",
                node_type="StrategyObjective",
                colour=NODE_COLOURS["StrategyObjective"],
                priority=1.0,
                size=30.0,
            )

            # Strategic goals
            for goal in obj.get("strategic_goals", []):
                goal_id = f"goal_{goal['id']}"
                self.G.add_node(
                    goal_id,
                    label=f"{goal['id']}: {goal['description'][:60]}",
                    node_type="StrategyGoal",
                    colour=NODE_COLOURS["StrategyGoal"],
                    size=18.0,
                )
                self.G.add_edge(
                    obj_id, goal_id,
                    weight=1.0,
                    edge_type="has_goal",
                    created_by="structure",
                )

            # KPIs for the objective
            for kpi in obj.get("kpis", []):
                kpi_name = kpi.get("KPI", "")
                if not kpi_name:
                    continue
                kpi_id = f"kpi_obj_{code}_{self._sanitise(kpi_name)}"
                self.G.add_node(
                    kpi_id,
                    label=kpi_name[:60],
                    node_type="KPI",
                    colour=NODE_COLOURS["KPI"],
                    size=10.0,
                )
                self.G.add_edge(
                    obj_id, kpi_id,
                    weight=1.0,
                    edge_type="has_kpi",
                    created_by="structure",
                )

        logger.info("Added %d strategy nodes (objectives + goals + KPIs).",
                     sum(1 for _, d in self.G.nodes(data=True)
                         if d.get("node_type", "").startswith("Strategy")
                         or d.get("node_type") == "KPI"))

    def _add_action_nodes(self) -> None:
        """Add Action, KPI, Stakeholder, and TimelineQuarter nodes."""
        # Look up alignment scores for sizing
        action_scores: dict[int, float] = {}
        for aa in self._alignment.get("action_alignments", []):
            action_scores[aa["action_number"]] = aa["best_score"]

        for action in self._actions.get("actions", []):
            num = action["action_number"]
            act_id = f"action_{num}"
            score = action_scores.get(num, 0.0)
            budget = action.get("budget_lkr_millions", 1.0) or 1.0

            # Node size: alignment_score * log(budget+1), normalised later
            raw_size = score * math.log(budget + 1, 10)

            self.G.add_node(
                act_id,
                label=f"Action {num}: {action['title'][:50]}",
                node_type="Action",
                colour=NODE_COLOURS["Action"],
                alignment_score=round(score, 4),
                budget=budget,
                timeline=action.get("timeline", ""),
                size=round(raw_size * 10, 2),
            )

            # Action -> KPIs
            for i, kpi_text in enumerate(action.get("kpis", []), 1):
                kpi_id = f"kpi_act_{num}_{i}"
                self.G.add_node(
                    kpi_id,
                    label=kpi_text[:60],
                    node_type="KPI",
                    colour=NODE_COLOURS["KPI"],
                    size=8.0,
                )
                self.G.add_edge(
                    act_id, kpi_id,
                    weight=1.0,
                    edge_type="has_kpi",
                    created_by="structure",
                )

            # Action -> Stakeholder (owner)
            owner = action.get("action_owner", "")
            if owner:
                owner_id = f"owner_{self._sanitise(owner)}"
                if owner_id not in self.G:
                    self.G.add_node(
                        owner_id,
                        label=owner[:50],
                        node_type="Stakeholder",
                        colour=NODE_COLOURS["Stakeholder"],
                        size=14.0,
                    )
                self.G.add_edge(
                    act_id, owner_id,
                    weight=1.0,
                    edge_type="has_owner",
                    created_by="structure",
                )

            # Action -> TimelineQuarter
            for q in action.get("quarters", []):
                q_id = f"quarter_{q}_2025"
                if q_id not in self.G:
                    self.G.add_node(
                        q_id,
                        label=f"{q} 2025",
                        node_type="TimelineQuarter",
                        colour=NODE_COLOURS["TimelineQuarter"],
                        size=12.0,
                    )
                self.G.add_edge(
                    act_id, q_id,
                    weight=1.0,
                    edge_type="has_timeline",
                    created_by="structure",
                )

        logger.info("Added action nodes, KPIs, stakeholders, timeline quarters.")

    def _add_ontology_concept_nodes(self) -> None:
        """Add OntologyConcept nodes from mappings and link them."""
        concept_set: set[str] = set()

        for section_key in ("action_mappings", "strategy_mappings"):
            for item in self._mappings.get(section_key, []):
                for m in item.get("mappings", []):
                    cid = m["concept_id"]
                    if cid not in concept_set:
                        concept_set.add(cid)
                        self.G.add_node(
                            f"concept_{cid}",
                            label=m.get("concept_label", cid),
                            node_type="OntologyConcept",
                            colour=NODE_COLOURS["OntologyConcept"],
                            parent_concept=m.get("parent_concept", ""),
                            size=15.0,  # will be resized by degree later
                        )

        logger.info("Added %d ontology concept nodes.", len(concept_set))

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def _add_alignment_edges(self) -> None:
        """Add StrategyObjective -> Action edges from alignment scores.

        Only edges above ``THRESHOLD_FAIR`` (0.45) are included to keep
        the graph readable.  The edge weight equals the cosine-similarity
        alignment score.
        """
        matrix = self._alignment.get("alignment_matrix", [])
        row_labels = self._alignment.get("matrix_row_labels", [])
        col_labels = self._alignment.get("matrix_col_labels", [])

        count = 0
        for i, obj_code in enumerate(row_labels):
            for j, act_num in enumerate(col_labels):
                score = matrix[i][j]
                if score >= THRESHOLD_FAIR:
                    self.G.add_edge(
                        f"obj_{obj_code}",
                        f"action_{act_num}",
                        weight=round(score, 4),
                        edge_type="alignment",
                        created_by="alignment",
                        evidence=f"cosine_similarity={score:.4f}",
                        method="embedding+keyword",
                    )
                    count += 1

        logger.info("Added %d alignment edges (threshold >= %.2f).",
                     count, THRESHOLD_FAIR)

    def _add_ontology_edges(self) -> None:
        """Add Action -> OntologyConcept and StrategyGoal -> OntologyConcept edges.

        Edge weight is the ``final_score`` from the hybrid ontology mapping.
        """
        count = 0

        # Action -> Concept
        for item in self._mappings.get("action_mappings", []):
            item_id = item["item_id"]  # e.g. "action_1"
            for m in item.get("mappings", []):
                concept_node = f"concept_{m['concept_id']}"
                if concept_node in self.G and item_id in self.G:
                    kws = ", ".join(m.get("matched_keywords", [])[:5])
                    self.G.add_edge(
                        item_id, concept_node,
                        weight=round(m["final_score"], 4),
                        edge_type="ontology_mapping",
                        created_by="ontology",
                        evidence=f"keywords=[{kws}]; score={m['final_score']:.3f}",
                    )
                    count += 1

        # StrategyGoal -> Concept
        for item in self._mappings.get("strategy_mappings", []):
            item_id = item["item_id"]  # e.g. "goal_A1"
            for m in item.get("mappings", []):
                concept_node = f"concept_{m['concept_id']}"
                if concept_node in self.G and item_id in self.G:
                    kws = ", ".join(m.get("matched_keywords", [])[:5])
                    self.G.add_edge(
                        item_id, concept_node,
                        weight=round(m["final_score"], 4),
                        edge_type="ontology_mapping",
                        created_by="ontology",
                        evidence=f"keywords=[{kws}]; score={m['final_score']:.3f}",
                    )
                    count += 1

        logger.info("Added %d ontology-mapping edges.", count)

    # ------------------------------------------------------------------
    # Node size normalisation
    # ------------------------------------------------------------------

    def _normalise_sizes(self) -> None:
        """Resize OntologyConcept nodes by degree centrality.

        Action nodes are already sized by ``alignment_score * log(budget+1)``.
        Concept nodes are resized proportionally to their degree so
        well-connected concepts appear larger.
        """
        deg = dict(self.G.degree())
        concept_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "OntologyConcept"
        ]
        if concept_nodes:
            max_deg = max(deg.get(n, 1) for n in concept_nodes)
            for n in concept_nodes:
                normalised = deg.get(n, 1) / max(max_deg, 1)
                self.G.nodes[n]["size"] = round(10 + 30 * normalised, 2)

    # ------------------------------------------------------------------
    # Graph metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> GraphInsights:
        """Compute centrality, community, and structural metrics.

        Returns:
            :class:`GraphInsights` populated with all computed analytics.

        Algorithm notes:
        - **Degree centrality**: fraction of nodes each node is connected to.
          High degree centrality means a node is a hub.
        - **Betweenness centrality**: fraction of all shortest paths that
          pass through a node.  High betweenness = bridge / bottleneck.
        - **Community detection**: Greedy modularity on the *undirected*
          projection.  Groups densely interconnected nodes into clusters.
        """
        ins = self.insights
        ins.node_count = self.G.number_of_nodes()
        ins.edge_count = self.G.number_of_edges()

        # ── Degree centrality by type ──────────────────────────────────
        dc = nx.degree_centrality(self.G)
        type_groups: dict[str, list[tuple[str, float]]] = {}
        for n, score in dc.items():
            ntype = self.G.nodes[n].get("node_type", "Unknown")
            type_groups.setdefault(ntype, []).append((n, round(score, 4)))
        for ntype in type_groups:
            type_groups[ntype].sort(key=lambda x: x[1], reverse=True)
            type_groups[ntype] = type_groups[ntype][:10]
        ins.degree_centrality = type_groups

        # ── Betweenness centrality ─────────────────────────────────────
        bc = nx.betweenness_centrality(self.G, weight="weight", seed=RANDOM_SEED)
        bc_sorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:20]
        ins.betweenness_centrality = [(n, round(s, 4)) for n, s in bc_sorted]

        # ── Community detection (greedy modularity) ────────────────────
        # NetworkX's greedy_modularity_communities works on undirected graphs.
        undirected = self.G.to_undirected()
        try:
            communities = nx.community.greedy_modularity_communities(
                undirected, weight="weight",
            )
            ins.communities = [sorted(list(c)) for c in communities]
        except Exception as exc:
            logger.warning("Community detection failed: %s", exc)
            ins.communities = []

        # ── Isolated strategies ────────────────────────────────────────
        for obj_align in self._alignment.get("objective_alignments", []):
            code = obj_align["code"]
            obj_id = f"obj_{code}"
            # Check if any action edge has weight >= THRESHOLD_FAIR
            action_edges = [
                (u, v, d) for u, v, d in self.G.out_edges(obj_id, data=True)
                if d.get("edge_type") == "alignment"
            ]
            if not action_edges:
                ins.isolated_strategies.append(obj_id)

        # ── Isolated actions (no ontology mapping edges) ───────────────
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") == "Action":
                ontology_edges = [
                    1 for _, v, ed in self.G.out_edges(n, data=True)
                    if ed.get("edge_type") == "ontology_mapping"
                ]
                if not ontology_edges:
                    ins.isolated_actions.append(n)

        # ── Bridge nodes (top betweenness among actions + concepts) ────
        relevant_types = {"Action", "OntologyConcept", "StrategyGoal"}
        bridge_candidates = [
            (n, s) for n, s in bc_sorted
            if self.G.nodes[n].get("node_type") in relevant_types
        ][:10]
        for n, score in bridge_candidates:
            ntype = self.G.nodes[n].get("node_type", "")
            in_types = set()
            out_types = set()
            for u, _ in self.G.in_edges(n):
                in_types.add(self.G.nodes[u].get("node_type", ""))
            for _, v in self.G.out_edges(n):
                out_types.add(self.G.nodes[v].get("node_type", ""))

            ins.bridge_nodes.append({
                "node_id": n,
                "label": self.G.nodes[n].get("label", n),
                "node_type": ntype,
                "betweenness": score,
                "connects_in": sorted(in_types),
                "connects_out": sorted(out_types),
            })

        logger.info(
            "Metrics computed: %d nodes, %d edges, %d communities, "
            "%d isolated actions, %d bridge nodes.",
            ins.node_count, ins.edge_count, len(ins.communities),
            len(ins.isolated_actions), len(ins.bridge_nodes),
        )
        return ins

    # ------------------------------------------------------------------
    # Bottleneck & critical-path analysis
    # ------------------------------------------------------------------

    def find_critical_paths(self, strategy_id: str, kpi_id: str,
                            top_k: int = 3) -> list[dict[str, Any]]:
        """Find top-k highest-weight paths from a strategy to a KPI node.

        Uses Dijkstra on *inverse weights* (``1 / weight``) so that
        high-weight edges are preferred.  Returns the ``top_k`` shortest
        (i.e. highest-original-weight) paths.

        Args:
            strategy_id: Source node (e.g. ``"obj_A"``).
            kpi_id:      Target node (e.g. ``"kpi_act_6_1"``).
            top_k:       Number of paths to return.

        Returns:
            List of dicts with ``path``, ``total_weight``, ``edges``.
        """
        if strategy_id not in self.G or kpi_id not in self.G:
            return [{"error": f"Node '{strategy_id}' or '{kpi_id}' not in graph."}]

        # Build inverse-weight graph for shortest-path = highest-weight
        inv_G = nx.DiGraph()
        for u, v, d in self.G.edges(data=True):
            w = d.get("weight", 1.0)
            inv_G.add_edge(u, v, weight=1.0 / max(w, 0.01), original_weight=w)

        results: list[dict[str, Any]] = []
        try:
            paths = list(nx.shortest_simple_paths(
                inv_G, strategy_id, kpi_id, weight="weight",
            ))
            for path in paths[:top_k]:
                edges = []
                total_w = 0.0
                for i in range(len(path) - 1):
                    ow = inv_G[path[i]][path[i + 1]].get("original_weight", 1.0)
                    total_w += ow
                    edges.append({
                        "from": path[i],
                        "to": path[i + 1],
                        "weight": round(ow, 4),
                    })
                results.append({
                    "path": path,
                    "total_weight": round(total_w, 4),
                    "hop_count": len(path) - 1,
                    "edges": edges,
                })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            results.append({"error": f"No path from '{strategy_id}' to '{kpi_id}'."})

        return results

    def identify_bottlenecks(self, top_k: int = 10) -> list[dict[str, Any]]:
        """Identify bottleneck nodes with highest betweenness centrality.

        A bottleneck is a node that many shortest paths pass through.
        If such a node fails or is removed, connectivity between large
        parts of the graph is disrupted.

        Args:
            top_k: Number of top bottleneck nodes to return.

        Returns:
            List of dicts with ``node_id``, ``betweenness``,
            ``in_degree``, ``out_degree``, ``explanation``.
        """
        bc = nx.betweenness_centrality(self.G, weight="weight", seed=RANDOM_SEED)
        sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_k]

        bottlenecks: list[dict[str, Any]] = []
        for node_id, score in sorted_bc:
            if score < 0.001:
                continue
            ndata = self.G.nodes[node_id]
            ntype = ndata.get("node_type", "Unknown")
            in_deg = self.G.in_degree(node_id)
            out_deg = self.G.out_degree(node_id)

            # Build explanation based on type and connectivity
            if ntype == "Stakeholder":
                explanation = (
                    f"Stakeholder '{ndata.get('label', node_id)}' owns multiple "
                    f"actions (in-degree={in_deg}). If this stakeholder is "
                    f"overloaded, dependent actions may stall."
                )
            elif ntype == "Action":
                explanation = (
                    f"Action '{ndata.get('label', node_id)[:50]}' connects "
                    f"{in_deg} strategies/concepts to {out_deg} downstream "
                    f"KPIs/owners. It serves as a critical execution bridge."
                )
            elif ntype == "OntologyConcept":
                explanation = (
                    f"Concept '{ndata.get('label', node_id)}' links multiple "
                    f"strategies and actions (degree={in_deg + out_deg}). "
                    f"Weak coverage here creates a strategic gap."
                )
            elif ntype == "TimelineQuarter":
                explanation = (
                    f"Quarter '{ndata.get('label', node_id)}' has {in_deg} "
                    f"concurrent actions. Resource contention risk is high."
                )
            else:
                explanation = (
                    f"Node '{node_id}' (type={ntype}) has betweenness "
                    f"{score:.4f}, in-degree={in_deg}, out-degree={out_deg}."
                )

            bottlenecks.append({
                "node_id": node_id,
                "label": ndata.get("label", node_id),
                "node_type": ntype,
                "betweenness": round(score, 4),
                "in_degree": in_deg,
                "out_degree": out_deg,
                "explanation": explanation,
            })

        self.insights.bottlenecks = bottlenecks
        return bottlenecks

    def suggest_new_connections(self, top_k: int = 10) -> list[dict[str, Any]]:
        """Suggest new action--concept connections to fill coverage gaps.

        For each ontology concept that appears in strategy mappings but
        has no action edge, find the most semantically similar actions
        (using alignment scores as proxy) and suggest them as candidates.

        Args:
            top_k: Maximum number of suggestions to return.

        Returns:
            List of dicts with ``concept``, ``suggested_action``,
            ``confidence``, ``evidence``.
        """
        suggestions: list[dict[str, Any]] = []

        # Find concepts linked to strategy goals but not to actions
        strategy_concepts: set[str] = set()
        action_concepts: set[str] = set()

        for n, d in self.G.nodes(data=True):
            if d.get("node_type") != "OntologyConcept":
                continue
            concept_node = n
            for u, _ in self.G.in_edges(concept_node):
                utype = self.G.nodes[u].get("node_type", "")
                if utype == "StrategyGoal":
                    strategy_concepts.add(concept_node)
                elif utype == "Action":
                    action_concepts.add(concept_node)

        uncovered = strategy_concepts - action_concepts

        # For each uncovered concept, find closest actions by alignment score
        action_alignments = {
            aa["action_number"]: aa
            for aa in self._alignment.get("action_alignments", [])
        }

        for concept_node in sorted(uncovered):
            concept_label = self.G.nodes[concept_node].get("label", concept_node)

            # Find which objectives this concept relates to via goals
            related_objectives: set[str] = set()
            for u, _ in self.G.in_edges(concept_node):
                if self.G.nodes[u].get("node_type") == "StrategyGoal":
                    # Goal -> parent objective
                    for pu, _ in self.G.in_edges(u):
                        if self.G.nodes[pu].get("node_type") == "StrategyObjective":
                            related_objectives.add(pu.replace("obj_", ""))

            # Score actions by their alignment to related objectives
            candidates: list[tuple[int, float]] = []
            for act_num, aa in action_alignments.items():
                avg_score = np.mean([
                    aa["similarities"].get(obj_code, 0.0)
                    for obj_code in related_objectives
                ]) if related_objectives else 0.0
                # Exclude actions already connected to this concept
                act_id = f"action_{act_num}"
                if not self.G.has_edge(act_id, concept_node):
                    candidates.append((act_num, float(avg_score)))

            candidates.sort(key=lambda x: x[1], reverse=True)

            for act_num, conf in candidates[:2]:
                if conf < 0.30:
                    continue
                aa = action_alignments[act_num]
                suggestions.append({
                    "concept": concept_label,
                    "concept_node": concept_node,
                    "suggested_action": f"action_{act_num}",
                    "action_title": aa.get("title", ""),
                    "confidence": round(conf, 4),
                    "related_objectives": sorted(related_objectives),
                    "evidence": (
                        f"Action {act_num} has avg alignment {conf:.3f} to "
                        f"objectives [{', '.join(sorted(related_objectives))}] "
                        f"which need coverage for concept '{concept_label}'."
                    ),
                })
                if len(suggestions) >= top_k:
                    break
            if len(suggestions) >= top_k:
                break

        self.insights.new_connections = suggestions
        return suggestions

    # ------------------------------------------------------------------
    # Full build pipeline
    # ------------------------------------------------------------------

    def build(self) -> nx.DiGraph:
        """Execute the full graph-building pipeline.

        Steps:
            1. Load upstream JSON data.
            2. Add strategy nodes (objectives, goals, KPIs).
            3. Add action nodes (actions, KPIs, owners, quarters).
            4. Add ontology concept nodes.
            5. Add alignment edges (objective -> action).
            6. Add ontology-mapping edges.
            7. Normalise node sizes.
            8. Compute all metrics.
            9. Identify bottlenecks and suggest connections.

        Returns:
            The constructed ``nx.DiGraph``.
        """
        logger.info("=" * 60)
        logger.info("KNOWLEDGE GRAPH — Starting build")
        logger.info("=" * 60)

        self._load_data()
        self._add_strategy_nodes()
        self._add_action_nodes()
        self._add_ontology_concept_nodes()
        self._add_alignment_edges()
        self._add_ontology_edges()
        self._normalise_sizes()
        self.compute_metrics()
        self.identify_bottlenecks()
        self.suggest_new_connections()

        logger.info("=" * 60)
        logger.info("KNOWLEDGE GRAPH BUILD COMPLETE")
        logger.info("  Nodes: %d", self.G.number_of_nodes())
        logger.info("  Edges: %d", self.G.number_of_edges())
        logger.info("=" * 60)

        return self.G

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def export_gexf(self, filepath: Path = GEXF_OUT) -> Path:
        """Export graph to GEXF format (Gephi compatible).

        Args:
            filepath: Output file path.

        Returns:
            The path written to.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(self.G, str(filepath))
        logger.info("Exported GEXF to %s.", filepath)
        return filepath

    def export_graphml(self, filepath: Path = GRAPHML_OUT) -> Path:
        """Export graph to GraphML format.

        Args:
            filepath: Output file path.

        Returns:
            The path written to.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.G, str(filepath))
        logger.info("Exported GraphML to %s.", filepath)
        return filepath

    def export_json(self, filepath: Path = JSON_OUT) -> Path:
        """Export graph to node-link JSON format for web dashboards.

        Args:
            filepath: Output file path.

        Returns:
            The path written to.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.G)

        # Also embed insights summary
        data["insights"] = {
            "node_count": self.insights.node_count,
            "edge_count": self.insights.edge_count,
            "community_count": len(self.insights.communities),
            "isolated_actions": self.insights.isolated_actions,
            "isolated_strategies": self.insights.isolated_strategies,
            "bridge_nodes": self.insights.bridge_nodes[:5],
            "bottlenecks": self.insights.bottlenecks[:5],
            "new_connections": self.insights.new_connections[:5],
        }

        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Exported node-link JSON to %s.", filepath)
        return filepath

    def export_all(self) -> dict[str, str]:
        """Export to all three formats.

        Returns:
            Dict mapping format name to file path.
        """
        return {
            "gexf": str(self.export_gexf()),
            "graphml": str(self.export_graphml()),
            "json": str(self.export_json()),
        }

    # ------------------------------------------------------------------
    # Visualization helper (optional, fails gracefully)
    # ------------------------------------------------------------------

    def render_png(self, filepath: Path = PNG_OUT) -> Path | None:
        """Render a spring-layout PNG of the graph (optional).

        Requires ``matplotlib``.  If unavailable or in a headless
        environment, the function returns ``None`` without raising.

        Args:
            filepath: Output PNG path.

        Returns:
            Path written to, or ``None`` if rendering failed.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping PNG render.")
            return None

        filepath.parent.mkdir(parents=True, exist_ok=True)

        pos = nx.spring_layout(self.G, seed=RANDOM_SEED, k=0.5, iterations=80)

        # Colour map from node attributes
        node_colours = [
            self.G.nodes[n].get("colour", "#CCCCCC") for n in self.G.nodes()
        ]
        node_sizes = [
            self.G.nodes[n].get("size", 10) * 3 for n in self.G.nodes()
        ]

        fig, ax = plt.subplots(1, 1, figsize=(24, 18))
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            node_color=node_colours, node_size=node_sizes, alpha=0.85,
        )
        nx.draw_networkx_edges(
            self.G, pos, ax=ax,
            edge_color="#CCCCCC", arrows=True, arrowsize=8, alpha=0.4,
        )

        # Labels only for key node types
        label_types = {"StrategyObjective", "Action", "OntologyConcept"}
        labels = {
            n: self.G.nodes[n].get("label", n)[:25]
            for n in self.G.nodes()
            if self.G.nodes[n].get("node_type") in label_types
        }
        nx.draw_networkx_labels(
            self.G, pos, labels, ax=ax, font_size=6, font_weight="bold",
        )

        ax.set_title("Strategy-Action Knowledge Graph",
                      fontsize=14, fontweight="bold")
        ax.axis("off")

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colour, markersize=10, label=ntype)
            for ntype, colour in NODE_COLOURS.items()
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        fig.tight_layout()
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Rendered PNG to %s.", filepath)
        return filepath

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitise(text: str) -> str:
        """Convert arbitrary text to a safe node-ID fragment."""
        return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())[:40]


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    kg = KnowledgeGraphBuilder()
    kg.build()
    exports = kg.export_all()

    # Optional PNG
    png_path = kg.render_png()

    # ── Summary ────────────────────────────────────────────────────
    ins = kg.insights
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH — SUMMARY")
    print("=" * 70)
    print(f"\n  Nodes : {ins.node_count}")
    print(f"  Edges : {ins.edge_count}")
    print(f"  Communities detected : {len(ins.communities)}")
    print(f"  Isolated strategies  : {ins.isolated_strategies}")
    print(f"  Isolated actions     : {ins.isolated_actions}")

    # ── Degree centrality highlights ───────────────────────────────
    print("\n--- DEGREE CENTRALITY (top per type) ---")
    for ntype, entries in ins.degree_centrality.items():
        if entries:
            top = entries[0]
            print(f"  {ntype:<22}  {top[0]:<30}  {top[1]:.4f}")

    # ── Betweenness centrality ─────────────────────────────────────
    print("\n--- BETWEENNESS CENTRALITY (top 10) ---")
    for node_id, score in ins.betweenness_centrality[:10]:
        label = kg.G.nodes[node_id].get("label", node_id)[:40]
        ntype = kg.G.nodes[node_id].get("node_type", "?")
        print(f"  {score:.4f}  {ntype:<22}  {label}")

    # ── Communities ────────────────────────────────────────────────
    print(f"\n--- COMMUNITIES ({len(ins.communities)}) ---")
    for i, comm in enumerate(ins.communities[:8]):
        key_nodes = [n for n in comm if kg.G.nodes.get(n, {}).get("node_type")
                     in ("StrategyObjective", "Action", "OntologyConcept")][:5]
        print(f"  Community {i}: {len(comm)} nodes — key: {key_nodes}")

    # ── Bottlenecks ────────────────────────────────────────────────
    print("\n--- BOTTLENECKS (top 5) ---")
    for bn in ins.bottlenecks[:5]:
        print(f"  {bn['node_id']:<30}  btw={bn['betweenness']:.4f}")
        print(f"    → {bn['explanation'][:100]}")

    # ── Suggested new connections ──────────────────────────────────
    print("\n--- SUGGESTED NEW CONNECTIONS ---")
    for sug in ins.new_connections[:5]:
        print(f"  {sug['concept']:<25} ← {sug['suggested_action']} "
              f"(conf={sug['confidence']:.3f})")
        print(f"    {sug['evidence'][:100]}")

    # ── Critical path demo ─────────────────────────────────────────
    print("\n--- CRITICAL PATH DEMO (obj_B -> kpi_act_9_1) ---")
    paths = kg.find_critical_paths("obj_B", "kpi_act_9_1", top_k=2)
    for p in paths:
        if "error" in p:
            print(f"  {p['error']}")
        else:
            route = " -> ".join(p["path"])
            print(f"  Weight={p['total_weight']:.3f}  Hops={p['hop_count']}  "
                  f"Path: {route}")

    # ── Export paths ───────────────────────────────────────────────
    print("\n--- EXPORTS ---")
    for fmt, path in exports.items():
        print(f"  {fmt}: {path}")
    if png_path:
        print(f"  png: {png_path}")
