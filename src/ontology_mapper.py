"""
ontology_mapper.py
------------------
Build a lightweight OWL ontology and map objectives/actions to concept classes.

How it fits in the pipeline:
  alignment_scorer → [this module] → knowledge_graph / dashboard

Beginner note:
  An "ontology" is a formal, machine-readable way of defining concepts
  and their relationships. We use RDF/OWL — the W3C standard — expressed
  in "Turtle" format (.ttl). Think of it like a spreadsheet where each row
  says "this thing IS-A that concept".

  We map each objective and action to a healthcare concept class using
  keyword counting — no ML needed! The concept whose keywords appear most
  often in the item's text wins.
"""

# --- Standard library ---
import os
from collections import Counter

# --- Third-party ---
from rdflib import Graph, Namespace, RDF, OWL, RDFS, Literal

# --- Local ---
from src.config import OUTPUTS_DIR

# Our custom RDF namespace — every ISPS concept lives under this base URI
ISPS_NS = Namespace("http://isps.hospital.org/ontology#")

# =============================================================================
# CONCEPT KEYWORD DICTIONARY
# Each key is a healthcare concept class name.
# Each value is a list of keywords that strongly suggest that class.
# =============================================================================
CONCEPT_KEYWORDS: dict = {
    "ClinicalCare": [
        "clinical", "patient care", "treatment", "nursing",
        "medical", "therapy", "diagnostic", "ward", "surgery", "care delivery",
    ],
    "PatientSafety": [
        "safety", "infection", "risk", "incident", "quality",
        "adverse", "error", "protocol", "accreditation", "compliance",
    ],
    "OperationalEfficiency": [
        "efficiency", "process", "workflow", "operational",
        "capacity", "throughput", "turnaround", "bed", "occupancy", "lean",
    ],
    "Finance": [
        "finance", "budget", "cost", "revenue", "expenditure",
        "funding", "investment", "profit", "financial", "procurement",
    ],
    "HumanResources": [
        "staff", "workforce", "training", "recruitment",
        "human resource", "employee", "retention", "skill", "competency",
    ],
    "Technology": [
        "technology", "digital", "system", "software", "it",
        "data", "ehr", "electronic", "automation", "infrastructure",
    ],
    "CommunityEngagement": [
        "community", "outreach", "awareness", "public",
        "stakeholder", "partnership", "engagement", "social",
    ],
}


# =============================================================================
# STEP 1: BUILD THE OWL ONTOLOGY GRAPH
# =============================================================================

def build_ontology() -> Graph:
    """
    Create an RDF/OWL graph that defines our healthcare concept classes.

    Each class is declared as an OWL.Class with an RDFS label.
    A common parent class (HospitalConcept) is added so all concepts share
    a common ancestor — making the ontology more expressive.

    Returns:
        rdflib.Graph: Ontology with all concept classes defined.
    """
    print("📌 Step 1: Building OWL ontology...")

    g = Graph()

    # Bind short namespace prefixes so the Turtle file is human-readable.
    # Instead of <http://isps.hospital.org/ontology#ClinicalCare> we get isps:ClinicalCare
    g.bind("isps", ISPS_NS)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)

    # Declare the ontology itself
    g.add((ISPS_NS["ISPSOntology"], RDF.type, OWL.Ontology))

    # Add a generic parent class so all concepts share a common ancestor
    g.add((ISPS_NS["HospitalConcept"], RDF.type,   OWL.Class))
    g.add((ISPS_NS["HospitalConcept"], RDFS.label, Literal("HospitalConcept")))

    # Add each concept class as an OWL Class with a label
    for concept_name in CONCEPT_KEYWORDS:
        concept_uri = ISPS_NS[concept_name]
        g.add((concept_uri, RDF.type,         OWL.Class))
        g.add((concept_uri, RDFS.label,       Literal(concept_name)))
        g.add((concept_uri, RDFS.subClassOf,  ISPS_NS["HospitalConcept"]))

    print(f"✅ Step 1 complete. Ontology has {len(g)} triples, "
          f"{len(CONCEPT_KEYWORDS)} concept classes.")
    return g


# =============================================================================
# STEP 2: MAP A SINGLE ITEM TO ITS BEST CONCEPT
# =============================================================================

def map_item_to_concept(title: str, description: str) -> str:
    """
    Classify one item (objective or action) into the best concept class.

    Strategy: count how many keywords from each concept appear in the
    combined text. The concept with the most hits wins.

    Args:
        title (str): The item's title.
        description (str): The item's description.

    Returns:
        str: Best matching concept name, or "General" if no keywords match.
    """
    # Lowercase for case-insensitive matching
    combined_text = (title + " " + description).lower()

    best_concept = "General"
    best_count   = 0

    for concept_name, keywords in CONCEPT_KEYWORDS.items():
        # Count occurrences of each keyword in the text
        count = sum(1 for kw in keywords if kw in combined_text)

        if count > best_count:
            best_count   = count
            best_concept = concept_name

    return best_concept


# =============================================================================
# STEP 3: MAP ALL OBJECTIVES AND ACTIONS
# =============================================================================

def map_all_items(objectives: list, actions: list) -> dict:
    """
    Map every objective and action to a healthcare concept class.

    Args:
        objectives (list): Objective dicts (id, title, description).
        actions (list): Action dicts (id, title, description).

    Returns:
        dict: {item_id: concept_class_name} for all objectives and actions.
    """
    print("📌 Step 2: Mapping items to ontology concepts...")

    mappings = {}

    for obj in objectives:
        concept = map_item_to_concept(obj["title"], obj.get("description", ""))
        mappings[obj["id"]] = concept
        print(f"   Objective {str(obj['id']):>4} → {concept}")

    for act in actions:
        concept = map_item_to_concept(act["title"], act.get("description", ""))
        mappings[act["id"]] = concept
        print(f"   Action    {str(act['id']):>4} → {concept}")

    print(f"✅ Step 2 complete. Mapped {len(mappings)} items.")
    return mappings


# =============================================================================
# STEP 4: ADD INSTANCES AND EXPORT AS TURTLE
# =============================================================================

def export_ontology(graph: Graph, mappings: dict) -> str:
    """
    Add one RDF instance triple per mapped item, then serialise to Turtle (.ttl).

    An "instance" triple says: item X is of type concept Y.
    Example: isps:O1 rdf:type isps:ClinicalCare

    Args:
        graph (Graph): The base ontology from build_ontology().
        mappings (dict): {item_id: concept_name} from map_all_items().

    Returns:
        str: File path of the saved .ttl file.
    """
    print("📌 Step 3: Adding instance triples and exporting Turtle...")

    for item_id, concept_name in mappings.items():
        # Replace any characters that would break a URI (spaces, slashes, etc.)
        safe_id = str(item_id).replace(" ", "_")
        graph.add((
            ISPS_NS[safe_id],       # the item — e.g. isps:O1
            RDF.type,               # is a
            ISPS_NS[concept_name],  # the concept — e.g. isps:ClinicalCare
        ))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ttl_path = os.path.join(OUTPUTS_DIR, "ontology.ttl")

    graph.serialize(destination=ttl_path, format="turtle")

    print(f"✅ Step 3 complete. Ontology saved to {ttl_path} "
          f"({len(graph)} total triples).")
    return ttl_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_ontology_mapping(objectives: list, actions: list) -> dict:
    """
    Full pipeline: build ontology → map all items → export Turtle → return mappings.

    Args:
        objectives (list): Objective dicts.
        actions (list): Action dicts.

    Returns:
        dict: {item_id: concept_class_name} for all objectives and actions.
    """
    print("\n" + "="*60)
    print("  RUNNING ONTOLOGY MAPPER")
    print("="*60)

    graph    = build_ontology()
    mappings = map_all_items(objectives, actions)
    export_ontology(graph, mappings)

    # Print a coverage summary
    print("\n📊 Concept coverage summary:")
    concept_counts = Counter(mappings.values())
    for concept, count in sorted(concept_counts.items(), key=lambda x: -x[1]):
        print(f"   {concept}: {count} items")

    print("\n✅ Ontology mapping complete.")
    return mappings
