"""
ontology_mapper.py — Build an OWL ontology and map objectives/actions to concept classes.
"""

import os
from collections import Counter

from rdflib import Graph, Namespace, RDF, OWL, RDFS, Literal

from src.config import OUTPUTS_DIR

ISPS_NS = Namespace("http://isps.hospital.org/ontology#")

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


def build_ontology() -> Graph:
    """Create an RDF/OWL graph with healthcare concept classes."""
    print("📌 Building OWL ontology...")

    g = Graph()
    g.bind("isps", ISPS_NS)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)

    g.add((ISPS_NS["ISPSOntology"],   RDF.type,   OWL.Ontology))
    g.add((ISPS_NS["HospitalConcept"], RDF.type,   OWL.Class))
    g.add((ISPS_NS["HospitalConcept"], RDFS.label, Literal("HospitalConcept")))

    for concept_name in CONCEPT_KEYWORDS:
        uri = ISPS_NS[concept_name]
        g.add((uri, RDF.type,        OWL.Class))
        g.add((uri, RDFS.label,      Literal(concept_name)))
        g.add((uri, RDFS.subClassOf, ISPS_NS["HospitalConcept"]))

    print(f"✅ Ontology: {len(g)} triples, {len(CONCEPT_KEYWORDS)} classes.")
    return g


def map_item_to_concept(title: str, description: str) -> str:
    """Map one item to the concept class with the most keyword matches."""
    combined = (title + " " + description).lower()
    best_concept, best_count = "General", 0

    for concept, keywords in CONCEPT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in combined)
        if count > best_count:
            best_count, best_concept = count, concept

    return best_concept


def map_all_items(objectives: list, actions: list) -> dict:
    """Map every objective and action to a concept class. Returns {id: concept}."""
    print("📌 Mapping items to ontology concepts...")
    mappings = {}

    for obj in objectives:
        concept = map_item_to_concept(obj["title"], obj.get("description", ""))
        mappings[obj["id"]] = concept
        print(f"   Objective {str(obj['id']):>4} → {concept}")

    for act in actions:
        concept = map_item_to_concept(act["title"], act.get("description", ""))
        mappings[act["id"]] = concept
        print(f"   Action    {str(act['id']):>4} → {concept}")

    print(f"✅ Mapped {len(mappings)} items.")
    return mappings


def export_ontology(graph: Graph, mappings: dict) -> str:
    """Add instance triples and save ontology as Turtle (.ttl)."""
    print("📌 Exporting ontology to Turtle...")

    for item_id, concept_name in mappings.items():
        safe_id = str(item_id).replace(" ", "_")
        graph.add((ISPS_NS[safe_id], RDF.type, ISPS_NS[concept_name]))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ttl_path = os.path.join(OUTPUTS_DIR, "ontology.ttl")
    graph.serialize(destination=ttl_path, format="turtle")

    print(f"✅ Ontology saved: {ttl_path} ({len(graph)} triples).")
    return ttl_path


def run_ontology_mapping(objectives: list, actions: list) -> dict:
    """Full pipeline: build ontology → map items → export Turtle → return mappings."""
    print("\n" + "="*60)
    print("  RUNNING ONTOLOGY MAPPER")
    print("="*60)

    graph    = build_ontology()
    mappings = map_all_items(objectives, actions)
    export_ontology(graph, mappings)

    print("\n📊 Concept coverage:")
    for concept, count in sorted(Counter(mappings.values()).items(), key=lambda x: -x[1]):
        print(f"   {concept}: {count}")

    print("\n✅ Ontology mapping complete.")
    return mappings
