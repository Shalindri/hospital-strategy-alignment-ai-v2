"""
knowledge_graph.py — Build a NetworkX directed graph and export as interactive PyVis HTML.
"""

import os

import networkx as nx
from pyvis.network import Network

from src.config import THRESHOLD_FAIR, OUTPUTS_DIR

COLOUR_OBJECTIVE = "#4A90D9"
COLOUR_ACTION    = "#5BAD72"
COLOUR_BRIDGE    = "#F5A623"
BRIDGE_THRESHOLD = 0.3


def build_graph(objectives: list, actions: list, alignment_result: dict) -> nx.DiGraph:
    """
    Build a directed graph: objective → action for pairs scoring ≥ THRESHOLD_FAIR.

    Edge weight = alignment score.
    """
    print("📌 Building knowledge graph...")

    G = nx.DiGraph()

    for obj in objectives:
        G.add_node(obj["id"], label=obj["title"][:40], title=obj["title"],
                   description=obj.get("description", ""), node_type="objective")

    for act in actions:
        G.add_node(act["id"], label=act["title"][:40], title=act["title"],
                   description=act.get("description", ""), node_type="action")

    for c in alignment_result["classifications"]:
        if c["score"] >= THRESHOLD_FAIR:
            G.add_edge(c["objective_id"], c["action_id"], weight=c["score"], tier=c["tier"])

    print(f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def compute_centrality(graph: nx.DiGraph) -> dict:
    """Compute and return degree centrality for all nodes."""
    print("📌 Computing centrality...")
    centrality = nx.degree_centrality(graph)

    top5 = sorted(centrality.items(), key=lambda x: -x[1])[:5]
    print("   Top 5:", [(n, f"{s:.3f}") for n, s in top5])
    print("✅ Centrality done.")
    return centrality


def identify_bridge_nodes(centrality: dict) -> list:
    """Return node IDs with centrality ≥ BRIDGE_THRESHOLD."""
    bridges = [n for n, s in centrality.items() if s >= BRIDGE_THRESHOLD]
    print(f"   Bridge nodes: {bridges}")
    return bridges


def export_html(graph: nx.DiGraph, centrality: dict, output_path: str) -> str:
    """Export the graph as an interactive HTML file using PyVis."""
    print(f"📌 Exporting HTML → {output_path}...")

    bridge_nodes = identify_bridge_nodes(centrality)

    net = Network(height="700px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#333333")

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200}
      }
    }
    """)

    for node_id, data in graph.nodes(data=True):
        node_type = data.get("node_type", "action")
        c_score   = centrality.get(node_id, 0)

        if node_id in bridge_nodes:
            colour = COLOUR_BRIDGE
        elif node_type == "objective":
            colour = COLOUR_OBJECTIVE
        else:
            colour = COLOUR_ACTION

        tooltip = (
            f"<b>{data.get('title', node_id)}</b><br>"
            f"Type: {node_type}<br>"
            f"Centrality: {c_score:.3f}<br>"
            f"{data.get('description', '')[:120]}..."
        )

        net.add_node(str(node_id), label=str(node_id), title=tooltip,
                     color=colour, size=20 + int(c_score * 40))

    for src, dst, edge_data in graph.edges(data=True):
        score = edge_data.get("weight", 0)
        net.add_edge(str(src), str(dst),
                     title=f"Score: {score:.3f} ({edge_data.get('tier', '')})",
                     width=1 + score * 4, color="#aaaaaa")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net.save_graph(output_path)
    print(f"✅ HTML saved: {output_path}")
    return output_path


def run_knowledge_graph(objectives: list, actions: list, alignment_result: dict) -> tuple:
    """Full pipeline: build graph → centrality → HTML export. Returns (graph, centrality)."""
    print("\n" + "="*60)
    print("  RUNNING KNOWLEDGE GRAPH BUILDER")
    print("="*60)

    graph      = build_graph(objectives, actions, alignment_result)
    centrality = compute_centrality(graph)
    export_html(graph, centrality, os.path.join(OUTPUTS_DIR, "knowledge_graph.html"))

    print("\n✅ Knowledge graph complete.")
    return graph, centrality
