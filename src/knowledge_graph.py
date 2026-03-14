"""
knowledge_graph.py
------------------
Build a NetworkX directed graph of objectives → actions, then export as interactive HTML.

How it fits in the pipeline:
  alignment_scorer → ontology_mapper → [this module] → dashboard

Beginner note:
  A "knowledge graph" is a network where:
    - Nodes = entities (objectives, actions)
    - Edges = relationships between them (with weights = alignment scores)

  We use NetworkX (a Python graph library) to build the graph, then pyvis
  to export it as an interactive HTML page where you can drag, zoom, and
  hover over nodes in your browser.

  "Bridge nodes" are nodes with high centrality — they connect many other
  nodes and are therefore strategically important.
"""

# --- Standard library ---
import os

# --- Third-party ---
import networkx as nx              # graph data structure and algorithms
from pyvis.network import Network  # interactive HTML graph visualisation

# --- Local ---
from src.config import THRESHOLD_FAIR, OUTPUTS_DIR

# Node colours (chosen to be visually distinct and accessible)
COLOUR_OBJECTIVE = "#4A90D9"   # blue  — strategic objectives
COLOUR_ACTION    = "#5BAD72"   # green — action items
COLOUR_BRIDGE    = "#F5A623"   # gold  — high-centrality "bridge" nodes

# Centrality threshold above which a node is considered a "bridge node"
BRIDGE_CENTRALITY_THRESHOLD = 0.3


# =============================================================================
# STEP 1: BUILD THE DIRECTED GRAPH
# =============================================================================

def build_graph(objectives: list, actions: list, alignment_result: dict) -> nx.DiGraph:
    """
    Build a directed graph: objective → action (when alignment score ≥ THRESHOLD_FAIR).

    Nodes:
      - Each objective gets a node with type="objective"
      - Each action gets a node with type="action"

    Edges:
      - Directed: objective_id → action_id
      - Only added when score >= THRESHOLD_FAIR (we skip weak connections)
      - Edge weight = alignment score; tier label stored as edge attribute

    Args:
        objectives (list): List of objective dicts (id, title, description).
        actions (list): List of action dicts (id, title, description).
        alignment_result (dict): Output of alignment_scorer.run_alignment().

    Returns:
        nx.DiGraph: The constructed directed graph.
    """
    print("📌 Step 1: Building knowledge graph...")

    G = nx.DiGraph()

    # Add objective nodes
    for obj in objectives:
        G.add_node(
            obj["id"],
            label       = obj["title"][:40],   # short label for display
            title       = obj["title"],         # full title (shown on hover in pyvis)
            description = obj.get("description", ""),
            node_type   = "objective",
        )

    # Add action nodes
    for act in actions:
        G.add_node(
            act["id"],
            label       = act["title"][:40],
            title       = act["title"],
            description = act.get("description", ""),
            node_type   = "action",
        )

    # Add edges for sufficiently aligned pairs
    edges_added = 0
    for classification in alignment_result["classifications"]:
        score = classification["score"]
        tier  = classification["tier"]

        # Only add edges for Fair/Good/Excellent alignment
        if score >= THRESHOLD_FAIR:
            G.add_edge(
                classification["objective_id"],
                classification["action_id"],
                weight = score,
                tier   = tier,
            )
            edges_added += 1

    print(f"✅ Step 1 complete. Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges "
          f"(from {len(alignment_result['classifications'])} pairs, "
          f"threshold ≥ {THRESHOLD_FAIR}).")
    return G


# =============================================================================
# STEP 2: COMPUTE CENTRALITY
# =============================================================================

def compute_centrality(graph: nx.DiGraph) -> dict:
    """
    Compute degree centrality for all nodes.

    Degree centrality = (number of edges connected) / (max possible edges).
    A score of 1.0 means the node is connected to every other node.
    High centrality = important "bridge" node in the network.

    Args:
        graph (nx.DiGraph): The knowledge graph.

    Returns:
        dict: {node_id: centrality_score (float 0.0 to 1.0)}
    """
    print("📌 Step 2: Computing node centrality...")

    # degree_centrality counts both in-edges and out-edges for directed graphs
    centrality = nx.degree_centrality(graph)

    # Print the top 5 most central nodes
    top_nodes = sorted(centrality.items(), key=lambda x: -x[1])[:5]
    print("   Top 5 most central nodes:")
    for node_id, score in top_nodes:
        print(f"     {node_id}: {score:.3f}")

    print("✅ Step 2 complete.")
    return centrality


# =============================================================================
# STEP 3: IDENTIFY BRIDGE NODES
# =============================================================================

def identify_bridge_nodes(centrality: dict,
                           threshold: float = BRIDGE_CENTRALITY_THRESHOLD) -> list:
    """
    Find nodes with centrality above the threshold.

    Bridge nodes are strategically important because they connect many
    objectives to many actions (or vice versa).

    Args:
        centrality (dict): {node_id: centrality_score}.
        threshold (float): Nodes scoring above this are bridge nodes. Default 0.3.

    Returns:
        list: Node IDs that are bridge nodes.
    """
    bridge_nodes = [node for node, score in centrality.items()
                    if score >= threshold]

    print(f"   Bridge nodes (centrality ≥ {threshold}): {bridge_nodes}")
    return bridge_nodes


# =============================================================================
# STEP 4: EXPORT INTERACTIVE HTML
# =============================================================================

def export_html(graph: nx.DiGraph, centrality: dict, output_path: str) -> str:
    """
    Export the graph as an interactive HTML page using pyvis.

    Node colours:
      Gold  = bridge node (high centrality)
      Blue  = strategic objective
      Green = action item

    Args:
        graph (nx.DiGraph): The knowledge graph.
        centrality (dict): {node_id: centrality_score}.
        output_path (str): Where to save the .html file.

    Returns:
        str: Path to the saved HTML file.
    """
    print(f"📌 Step 3: Exporting interactive HTML to {output_path}...")

    bridge_nodes = identify_bridge_nodes(centrality)

    # Create a pyvis Network
    # directed=True draws arrows on the edges
    net = Network(
        height     = "700px",
        width      = "100%",
        directed   = True,
        bgcolor    = "#ffffff",
        font_color = "#333333",
    )

    # Improve physics so nodes spread out nicely
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

    # Add nodes with colour and size based on type and centrality
    for node_id, data in graph.nodes(data=True):
        node_type = data.get("node_type", "action")
        c_score   = centrality.get(node_id, 0)

        # Determine colour
        if node_id in bridge_nodes:
            colour = COLOUR_BRIDGE
        elif node_type == "objective":
            colour = COLOUR_OBJECTIVE
        else:
            colour = COLOUR_ACTION

        # Larger nodes = higher centrality
        size = 20 + int(c_score * 40)

        # Tooltip shown on hover
        tooltip = (
            f"<b>{data.get('title', node_id)}</b><br>"
            f"Type: {node_type}<br>"
            f"Centrality: {c_score:.3f}<br>"
            f"{data.get('description', '')[:120]}..."
        )

        net.add_node(
            str(node_id),
            label = str(node_id),    # short ID shown inside the node
            title = tooltip,         # HTML tooltip on hover
            color = colour,
            size  = size,
        )

    # Add edges, with thickness proportional to alignment score
    for src, dst, edge_data in graph.edges(data=True):
        score = edge_data.get("weight", 0)
        tier  = edge_data.get("tier", "Fair")
        net.add_edge(
            str(src),
            str(dst),
            title = f"Score: {score:.3f} ({tier})",
            width = 1 + score * 4,       # thicker edge = higher score
            color = "#aaaaaa",
        )

    # Save as HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net.save_graph(output_path)

    print(f"✅ Step 3 complete. HTML graph saved to {output_path}")
    return output_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_knowledge_graph(objectives: list, actions: list,
                         alignment_result: dict) -> tuple:
    """
    Full pipeline: build graph → centrality → HTML export → return results.

    Args:
        objectives (list): Objective dicts.
        actions (list): Action dicts.
        alignment_result (dict): Output of alignment_scorer.run_alignment().

    Returns:
        tuple: (nx.DiGraph, dict centrality)
    """
    print("\n" + "="*60)
    print("  RUNNING KNOWLEDGE GRAPH BUILDER")
    print("="*60)

    graph       = build_graph(objectives, actions, alignment_result)
    centrality  = compute_centrality(graph)

    html_path = os.path.join(OUTPUTS_DIR, "knowledge_graph.html")
    export_html(graph, centrality, html_path)

    print("\n✅ Knowledge graph complete.")
    return graph, centrality
