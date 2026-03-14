"""
Ontology Mapper for Hospital Strategy--Action Plan Alignment System (ISPS).

This module implements an ontology-based mapping layer that connects the
Nawaloka Hospital Negombo Strategic Plan (2026--2030) and Annual Action
Plan (2025) through a lightweight healthcare-strategy ontology expressed
in RDF/OWL.

Architecture
------------
::

    strategic_plan.md ──┐
                        ├── Parsers ── Structured dicts ─┐
    action_plan.md ─────┘                                │
                                                         ▼
    Ontology Definition (RDFLib) ────────────────► Hybrid Mapper
      - 10 top-level concepts                      ├─ Keyword matching
      - ~40 mid-level concepts                     └─ Embedding similarity
      - Object properties                                │
                                                         ▼
                                                  Mapping scores
                                                         │
                                            ┌────────────┼────────────┐
                                            ▼            ▼            ▼
                                       mappings.json  gaps.json  ontology.ttl

Hybrid scoring
--------------
Each action/goal is mapped to ontology concepts via two signals:

*  **Keyword score** (0--1): fraction of a concept's curated keywords
   found in the item text.
*  **Embedding score** (0--1): cosine similarity between the item's
   sentence-transformer embedding and the concept description embedding.

These are combined as::

    final_score = 0.6 * embedding_score + 0.4 * keyword_score

A mapping is considered *valid* when ``final_score >= 0.55``.

Outputs
-------
*  ``outputs/ontology.ttl``  -- full RDF graph in Turtle format.
*  ``outputs/mappings.json`` -- per-item concept mappings with scores
   and evidence.
*  ``outputs/gaps.json``     -- uncovered strategy concepts and weakly
   aligned actions.

Typical usage::

    from src.ontology_mapper import OntologyMapper

    mapper = OntologyMapper()
    mapper.run()

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ontology_mapper")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

STRATEGIC_PLAN_FILE = DATA_DIR / "strategic_plan.md"
ACTION_PLAN_FILE = DATA_DIR / "action_plan.md"

ONTOLOGY_TTL = OUTPUT_DIR / "ontology.ttl"
MAPPINGS_JSON = OUTPUT_DIR / "mappings.json"
GAPS_JSON = OUTPUT_DIR / "gaps.json"

# Embedding model (same as vector_store.py for consistency)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Hybrid scoring weights
EMBEDDING_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4
MAPPING_THRESHOLD = 0.55

# RDF namespaces
ISPS = Namespace("http://isps.nawaloka.lk/ontology#")
ISPS_DATA = Namespace("http://isps.nawaloka.lk/data#")

# ---------------------------------------------------------------------------
# Ontology concept definitions
# ---------------------------------------------------------------------------

# Top-level concepts with descriptions and mid-level children
ONTOLOGY_CONCEPTS: dict[str, dict[str, Any]] = {
    "PatientCare": {
        "label": "Patient Care",
        "description": (
            "Clinical care delivery, patient safety, treatment outcomes, "
            "patient satisfaction, and infection control across all service lines."
        ),
        "children": {
            "EmergencyServices": {
                "label": "Emergency Services",
                "description": "24/7 accident and emergency department, trauma care, ambulance services.",
            },
            "InpatientCare": {
                "label": "Inpatient Care",
                "description": "Hospital admission, bed capacity, ward management, bed occupancy.",
            },
            "OutpatientCare": {
                "label": "Outpatient Care",
                "description": "OPD consultations, specialist channelling, patient wait times.",
            },
            "SurgicalServices": {
                "label": "Surgical Services",
                "description": "Operating theatres, day surgery, surgical site infection rates.",
            },
            "InfectionControl": {
                "label": "Infection Control",
                "description": "Hospital-acquired infection prevention, HAI rate, sterilisation protocols.",
            },
            "PatientExperience": {
                "label": "Patient Experience",
                "description": "Patient satisfaction scores, NPS, complaints management, patient feedback.",
            },
        },
    },
    "DigitalHealth": {
        "label": "Digital Health",
        "description": (
            "Health information technology, electronic health records, "
            "telemedicine, data analytics, and digital transformation."
        ),
        "children": {
            "EHR": {
                "label": "Electronic Health Records",
                "description": "EHR system implementation, clinical documentation, HL7 FHIR interoperability.",
            },
            "Telemedicine": {
                "label": "Telemedicine",
                "description": "Virtual consultations, remote specialist access, telehealth platform.",
            },
            "DataAnalytics": {
                "label": "Data Analytics",
                "description": "Hospital analytics, operational intelligence, clinical outcome benchmarking.",
            },
            "Cybersecurity": {
                "label": "Cybersecurity",
                "description": "Information security, ISO 27001, data protection, PDPA compliance, vulnerability assessment.",
            },
            "AI_AssistedDiagnostics": {
                "label": "AI-Assisted Diagnostics",
                "description": "AI-powered radiology, pathology screening, predictive analytics, machine learning.",
            },
        },
    },
    "Operations": {
        "label": "Operations",
        "description": (
            "Hospital operational management, process efficiency, supply chain, "
            "quality assurance, and accreditation."
        ),
        "children": {
            "QualityAssurance": {
                "label": "Quality Assurance",
                "description": "JCI accreditation, NABH standards, clinical governance, quality indicators.",
            },
            "SupplyChain": {
                "label": "Supply Chain",
                "description": "Procurement, equipment vendor management, medical supplies, inventory.",
            },
            "FacilityManagement": {
                "label": "Facility Management",
                "description": "Building maintenance, MEP systems, housekeeping, utility management.",
            },
        },
    },
    "Finance": {
        "label": "Finance",
        "description": (
            "Financial planning, budgeting, revenue growth, capital investment, "
            "ROI analysis, and cost management."
        ),
        "children": {
            "RevenueGrowth": {
                "label": "Revenue Growth",
                "description": "Revenue targets, service line revenue, financial performance, CAGR.",
            },
            "CapitalInvestment": {
                "label": "Capital Investment",
                "description": "Capital expenditure, construction funding, equipment procurement budget.",
            },
            "CostManagement": {
                "label": "Cost Management",
                "description": "Budget allocation, cost control, operational expenditure management.",
            },
        },
    },
    "WorkforceHR": {
        "label": "Workforce & HR",
        "description": (
            "Workforce planning, recruitment, retention, training, professional "
            "development, and employee engagement."
        ),
        "children": {
            "Recruitment": {
                "label": "Recruitment",
                "description": "Staff recruitment, specialist hiring, nursing workforce expansion.",
            },
            "Training": {
                "label": "Training & CPD",
                "description": "Continuing professional development, clinical skills workshops, simulation training.",
            },
            "Retention": {
                "label": "Staff Retention",
                "description": "Staff turnover, retention strategy, employee satisfaction, remuneration packages.",
            },
            "LeadershipDevelopment": {
                "label": "Leadership Development",
                "description": "Management pipeline, leadership programme, internal promotion, mentoring.",
            },
            "SpecialistFellowships": {
                "label": "Specialist Fellowships",
                "description": "Training fellowships in cardiac care, nephrology, fertility medicine, geriatric care.",
            },
        },
    },
    "ResearchInnovation": {
        "label": "Research & Innovation",
        "description": (
            "Clinical research, innovation, academic partnerships, peer-reviewed "
            "publications, and clinical trials."
        ),
        "children": {
            "ClinicalResearch": {
                "label": "Clinical Research",
                "description": "Clinical research unit, observational studies, clinical trials, ethics review.",
            },
            "AcademicPartnerships": {
                "label": "Academic Partnerships",
                "description": "University collaboration, MOU, student placements, co-supervision.",
            },
            "Publications": {
                "label": "Publications",
                "description": "Peer-reviewed papers, research output, clinical symposium.",
            },
            "InnovationProgrammes": {
                "label": "Innovation Programmes",
                "description": "Innovation awards, novel treatment protocols, staff innovation proposals.",
            },
        },
    },
    "CommunityRegionalHealth": {
        "label": "Community & Regional Health",
        "description": (
            "Community health programmes, preventive care, screening, "
            "regional partnerships, and public health outreach."
        ),
        "children": {
            "CorporateWellness": {
                "label": "Corporate Wellness",
                "description": "Corporate wellness partnerships, employer health packages, executive health checks.",
            },
            "ScreeningPrograms": {
                "label": "Screening Programmes",
                "description": "Community health screening, diabetes, hypertension, cardiovascular risk screening.",
            },
            "PreventiveCare": {
                "label": "Preventive Care",
                "description": "Preventive health, health education, women's health, cancer screening.",
            },
        },
    },
    "InternationalMedicalTourism": {
        "label": "International & Medical Tourism",
        "description": (
            "Medical tourism, international patients, Maldives outreach, "
            "concierge services, and cross-border healthcare."
        ),
        "children": {
            "MaldivesOutreach": {
                "label": "Maldives Outreach",
                "description": "Maldives partnership, visiting specialists to Male and Addu City, Aasandha reimbursement.",
            },
            "ConciergeServices": {
                "label": "Concierge Services",
                "description": "Medical tourism concierge unit, patient journey coordination, visa assistance.",
            },
            "AirlineHotelPartnerships": {
                "label": "Airline & Hotel Partnerships",
                "description": "SriLankan Airlines medical travel, hotel recovery packages, airport coordination.",
            },
            "Accreditation": {
                "label": "International Accreditation",
                "description": "JCI accreditation, international quality standards, accreditation roadmap.",
            },
        },
    },
    "InfrastructureExpansion": {
        "label": "Infrastructure Expansion",
        "description": (
            "Physical infrastructure development, construction, bed expansion, "
            "facility upgrades, and land acquisition."
        ),
        "children": {
            "LandAcquisition": {
                "label": "Land Acquisition",
                "description": "Adjacent land purchase, due diligence, environmental assessment, town planning.",
            },
            "BedExpansion": {
                "label": "Bed Expansion",
                "description": "Capacity expansion from 75 to 150 beds, ward construction, phased building.",
            },
            "FacilityUpgrade": {
                "label": "Facility Upgrade",
                "description": "Building refurbishment, equipment installation, laboratory upgrade, fit-out.",
            },
        },
    },
    "ClinicalServices": {
        "label": "Clinical Services",
        "description": (
            "Specialist clinical service lines including cardiology, nephrology, "
            "fertility, aesthetics, elderly care, and daycare."
        ),
        "children": {
            "Cardiology": {
                "label": "Cardiology",
                "description": "Cardiac catheterisation laboratory, interventional cardiology, cardiac surgeon.",
            },
            "Nephrology": {
                "label": "Nephrology",
                "description": "Nephrology dialysis unit, haemodialysis, peritoneal dialysis, water treatment.",
            },
            "Fertility": {
                "label": "Fertility Centre",
                "description": "IVF cycles, fertility centre, embryology laboratory, cryopreservation, reproductive health.",
            },
            "AestheticMedicine": {
                "label": "Aesthetic Medicine",
                "description": "Aesthetic and cosmetic medical services, dermatological aesthetics, laser treatments.",
            },
            "ElderlyCare": {
                "label": "Elderly Care",
                "description": "Elderly care unit, geriatric nursing, long-term residential care, occupational therapy.",
            },
            "DaycareServices": {
                "label": "Daycare Services",
                "description": "Daycare medical centre, day surgery, outpatient procedures, daily throughput.",
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Curated keyword dictionary per concept (for deterministic matching)
# ---------------------------------------------------------------------------
CONCEPT_KEYWORDS: dict[str, list[str]] = {
    # Top-level
    "PatientCare": [
        "patient care", "patient safety", "clinical care", "treatment outcome",
        "patient satisfaction", "nps", "patient experience",
    ],
    "DigitalHealth": [
        "digital health", "digital transformation", "health information",
        "health technology", "information system", "it infrastructure",
    ],
    "Operations": [
        "operations", "operational", "process efficiency", "quality assurance",
        "accreditation", "governance",
    ],
    "Finance": [
        "finance", "financial", "budget", "revenue", "capital investment",
        "cost", "roi", "expenditure", "funding",
    ],
    "WorkforceHR": [
        "workforce", "human resource", "hr", "staffing", "staff",
        "employee", "personnel",
    ],
    "ResearchInnovation": [
        "research", "innovation", "clinical research", "publication",
        "clinical trial", "peer-reviewed",
    ],
    "CommunityRegionalHealth": [
        "community", "regional health", "community health", "public health",
        "outreach", "screening programme",
    ],
    "InternationalMedicalTourism": [
        "medical tourism", "international patient", "maldives",
        "concierge", "cross-border",
    ],
    "InfrastructureExpansion": [
        "infrastructure", "expansion", "construction", "building",
        "land acquisition", "bed capacity", "facility",
    ],
    "ClinicalServices": [
        "clinical service", "specialty", "specialist", "service line",
    ],
    # Mid-level
    "EmergencyServices": ["emergency", "accident", "trauma", "ambulance", "a&e"],
    "InpatientCare": [
        "inpatient", "bed occupancy", "admission", "ward", "bed capacity",
        "150 bed", "75 bed",
    ],
    "OutpatientCare": ["outpatient", "opd", "channelling", "wait time"],
    "SurgicalServices": [
        "surgery", "surgical", "operating theatre", "day surgery",
        "surgical site infection",
    ],
    "InfectionControl": [
        "infection control", "hai", "hospital-acquired infection",
        "sterilisation", "infection rate",
    ],
    "PatientExperience": [
        "patient satisfaction", "nps", "patient feedback", "complaint",
        "patient experience",
    ],
    "EHR": [
        "ehr", "electronic health record", "health record", "hl7", "fhir",
        "clinical documentation",
    ],
    "Telemedicine": [
        "telemedicine", "telehealth", "virtual consultation",
        "remote consultation", "teleconsultation",
    ],
    "DataAnalytics": [
        "data analytics", "analytics", "dashboard", "benchmarking",
        "operational intelligence", "predictive",
    ],
    "Cybersecurity": [
        "cybersecurity", "cyber security", "information security",
        "iso 27001", "data protection", "pdpa", "vulnerability",
        "encryption", "penetration testing",
    ],
    "AI_AssistedDiagnostics": [
        "ai-assisted", "ai-powered", "artificial intelligence",
        "machine learning", "ai diagnostic", "ai radiology",
        "ai pathology", "predictive scoring",
    ],
    "QualityAssurance": [
        "jci", "nabh", "accreditation", "quality indicator",
        "clinical governance", "quality assurance",
    ],
    "SupplyChain": [
        "supply chain", "procurement", "vendor", "inventory",
        "medical supplies", "equipment vendor",
    ],
    "FacilityManagement": [
        "facility management", "maintenance", "mep", "housekeeping",
        "utility",
    ],
    "RevenueGrowth": [
        "revenue growth", "revenue target", "cagr", "revenue projection",
        "financial performance",
    ],
    "CapitalInvestment": [
        "capital investment", "capital expenditure", "capex",
        "construction funding", "investment plan",
    ],
    "CostManagement": [
        "cost management", "budget allocation", "cost control",
        "operational expenditure",
    ],
    "Recruitment": [
        "recruitment", "recruit", "hiring", "appointment",
        "workforce expansion", "staffing plan",
    ],
    "Training": [
        "training", "cpd", "continuing professional development",
        "workshop", "simulation", "e-learning", "skills",
    ],
    "Retention": [
        "retention", "turnover", "staff satisfaction",
        "remuneration", "incentive", "engagement",
    ],
    "LeadershipDevelopment": [
        "leadership", "management pipeline", "leadership programme",
        "internal promotion", "mentoring",
    ],
    "SpecialistFellowships": [
        "fellowship", "specialist training", "training fellowship",
    ],
    "ClinicalResearch": [
        "clinical research", "observational study", "clinical trial",
        "ethics review", "ethics committee", "erc",
    ],
    "AcademicPartnerships": [
        "university", "academic partnership", "mou", "student placement",
        "collaborative research", "university of kelaniya",
    ],
    "Publications": [
        "publication", "peer-reviewed", "paper", "symposium",
        "research output",
    ],
    "InnovationProgrammes": [
        "innovation award", "novel treatment", "innovation proposal",
    ],
    "CorporateWellness": [
        "corporate wellness", "employer", "executive health",
        "wellness package", "corporate partner",
    ],
    "ScreeningPrograms": [
        "screening", "screening camp", "diabetes screening",
        "hypertension screening", "cardiovascular screening",
        "health screening",
    ],
    "PreventiveCare": [
        "preventive", "health education", "women's health",
        "cancer screening", "preventive health",
    ],
    "MaldivesOutreach": [
        "maldives", "male", "addu city", "aasandha", "maldivian",
        "visiting specialist",
    ],
    "ConciergeServices": [
        "concierge", "medical tourism concierge", "patient journey",
        "visa assistance", "airport pickup",
    ],
    "AirlineHotelPartnerships": [
        "srilankan airlines", "hotel partnership", "hotel recovery",
        "airline", "airport medical",
    ],
    "Accreditation": [
        "jci accreditation", "nabh accreditation", "international accreditation",
        "accreditation roadmap",
    ],
    "LandAcquisition": [
        "land acquisition", "adjacent land", "land valuation",
        "environmental clearance", "town planning",
    ],
    "BedExpansion": [
        "bed expansion", "150 bed", "75 to 150", "bed capacity",
        "ward construction", "phased construction",
    ],
    "FacilityUpgrade": [
        "refurbishment", "upgrade", "fit-out", "laboratory upgrade",
        "equipment installation", "renovation",
    ],
    "Cardiology": [
        "cardiac", "cardiology", "cardiac catheterisation",
        "interventional cardiologist", "cardiac surgeon", "angiography",
    ],
    "Nephrology": [
        "nephrology", "dialysis", "haemodialysis", "peritoneal dialysis",
        "water treatment", "renal",
    ],
    "Fertility": [
        "fertility", "ivf", "iui", "embryology", "cryopreservation",
        "reproductive", "fertility centre",
    ],
    "AestheticMedicine": [
        "aesthetic", "cosmetic", "dermatological", "laser treatment",
        "anti-ageing", "cosmetic medicine",
    ],
    "ElderlyCare": [
        "elderly care", "geriatric", "geriatric nursing",
        "long-term residential", "occupational therapy", "ageing",
    ],
    "DaycareServices": [
        "daycare", "day care", "daycare medical centre",
        "day surgery", "daily throughput",
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConceptMapping:
    """A single concept mapping for an item (action or strategy goal).

    Attributes:
        concept_id:       Ontology concept identifier (e.g. ``"Cardiology"``).
        concept_label:    Human-readable label.
        parent_concept:   Top-level parent concept (e.g. ``"ClinicalServices"``).
        keyword_score:    Normalised keyword match score (0--1).
        embedding_score:  Cosine similarity from embeddings (0--1).
        final_score:      Weighted combination of keyword and embedding scores.
        matched_keywords: List of keywords that matched.
        evidence_text:    Relevant text snippet(s) that drove the mapping.
    """
    concept_id: str
    concept_label: str
    parent_concept: str
    keyword_score: float = 0.0
    embedding_score: float = 0.0
    final_score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    evidence_text: str = ""


@dataclass
class ItemMappingResult:
    """Full mapping result for a single action or strategy item.

    Attributes:
        item_id:        Unique identifier (e.g. ``"action_8"`` or ``"goal_A1"``).
        item_type:      ``"action"`` or ``"strategy_goal"``.
        item_text:      Composite text used for matching.
        mappings:       List of valid :class:`ConceptMapping` entries
                        (``final_score >= threshold``).
        all_mappings:   Complete list including below-threshold mappings.
        top_level_areas: Set of distinct top-level concepts with valid mappings.
        is_multi_area:  ``True`` if mapped to >=2 unrelated top-level concepts.
    """
    item_id: str
    item_type: str
    item_text: str
    mappings: list[ConceptMapping] = field(default_factory=list)
    all_mappings: list[ConceptMapping] = field(default_factory=list)
    top_level_areas: list[str] = field(default_factory=list)
    is_multi_area: bool = False


@dataclass
class GapReport:
    """Gap and misalignment detection results.

    Attributes:
        uncovered_concepts:  Strategy concepts with no supporting actions.
        weak_actions:        Actions mapping only below threshold.
        conflicting_actions: Actions mapped to multiple unrelated top-level areas.
    """
    uncovered_concepts: list[dict[str, Any]] = field(default_factory=list)
    weak_actions: list[dict[str, Any]] = field(default_factory=list)
    conflicting_actions: list[dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Lightweight markdown parsers
# ═══════════════════════════════════════════════════════════════════════

def _extract_section(text: str, start_heading: str,
                     stop_headings: list[str]) -> str:
    """Extract a section of markdown between headings.

    Args:
        text:           Full markdown text.
        start_heading:  Regex-safe heading text marking section start.
        stop_headings:  List of regex-safe heading texts where extraction
                        stops (exclusive).

    Returns:
        The extracted substring, or an empty string if not found.
    """
    start_pattern = re.compile(
        rf"^#{{1,6}}\s+{start_heading}",
        re.MULTILINE | re.IGNORECASE,
    )
    start_match = start_pattern.search(text)
    if not start_match:
        return ""

    for stop in stop_headings:
        stop_pattern = re.compile(
            rf"^#{{1,6}}\s+{stop}",
            re.MULTILINE | re.IGNORECASE,
        )
        stop_match = stop_pattern.search(text, start_match.end())
        if stop_match:
            return text[start_match.start():stop_match.start()]

    return text[start_match.start():]


def parse_strategic_plan(filepath: Path) -> list[dict[str, Any]]:
    """Parse strategic objectives from the strategic plan markdown.

    Extracts each of the five strategic objectives (A--E) with their
    goal statement, strategic goals, and KPIs as structured dicts.

    Args:
        filepath: Path to ``strategic_plan.md``.

    Returns:
        List of dicts, each with keys: ``code``, ``name``,
        ``goal_statement``, ``goals`` (list of dicts with ``id``,
        ``description``), ``kpis`` (list of strings).
    """
    text = filepath.read_text(encoding="utf-8")
    objectives: list[dict[str, Any]] = []

    # Derive objective patterns from the strategic plan JSON if available
    obj_patterns: list[tuple[str, str]] = []
    strategic_json = filepath.parent / "strategic_plan.json"
    if strategic_json.exists():
        import json as _json
        with open(strategic_json, encoding="utf-8") as _f:
            _sp = _json.load(_f)
        for _obj in _sp.get("objectives", []):
            obj_patterns.append((_obj["code"], _obj["name"]))
    if not obj_patterns:
        logger.warning("No strategic_plan.json found; cannot derive objective patterns.")

    stop_headings = [
        r"Strategic Objective [A-E]",
        r"\d+\.\s+International",
        r"\d+\.\s+Digital Transformation",
        r"\d+\.\s+Financial",
    ]

    for code, name in obj_patterns:
        section = _extract_section(
            text,
            rf"Strategic Objective {code}:\s*{re.escape(name)}",
            stop_headings,
        )
        if not section:
            logger.warning("Could not find section for Objective %s.", code)
            continue

        # Goal statement
        goal_stmt_match = re.search(
            r"\*\*Goal Statement:\*\*\s*(.+?)(?:\n|$)", section
        )
        goal_statement = goal_stmt_match.group(1).strip() if goal_stmt_match else ""

        # Strategic goals (numbered list)
        goals: list[dict[str, str]] = []
        goal_pattern = re.compile(
            rf"\d+\.\s+\*\*({code}\d+)\*\*\s*[—–-]\s*(.+?)(?:\n|$)"
        )
        for match in goal_pattern.finditer(section):
            goals.append({
                "id": match.group(1).strip(),
                "description": match.group(2).strip(),
            })

        # KPIs from table
        kpis: list[str] = []
        kpi_section = _extract_section(section, "KPIs", ["Timeline", "Responsible"])
        if kpi_section:
            for line in kpi_section.splitlines():
                line = line.strip()
                if "|" in line and not re.match(r"^\|[\s\-:|]+\|$", line):
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    if len(cells) >= 1 and cells[0] and cells[0].lower() != "kpi":
                        kpis.append(cells[0].strip("*").strip())

        objectives.append({
            "code": code,
            "name": name,
            "goal_statement": goal_statement,
            "goals": goals,
            "kpis": kpis,
        })

    logger.info("Parsed %d strategic objectives.", len(objectives))
    return objectives


def parse_action_plan(filepath: Path) -> list[dict[str, Any]]:
    """Parse actions from the action plan markdown.

    Extracts all 25 actions with their structured fields.

    Args:
        filepath: Path to ``action_plan.md``.

    Returns:
        List of dicts, each with keys: ``number``, ``title``,
        ``objective_code``, ``description``, ``owner``, ``timeline``,
        ``budget``, ``outcome``, ``kpis``.
    """
    text = filepath.read_text(encoding="utf-8")
    actions: list[dict[str, Any]] = []

    # Split into action blocks using "#### Action N:" pattern
    action_blocks = re.split(r"(?=####\s+Action\s+\d+:)", text)

    for block in action_blocks:
        # Match action header
        header_match = re.match(
            r"####\s+Action\s+(\d+):\s*(.+?)(?:\n|$)", block
        )
        if not header_match:
            continue

        number = int(header_match.group(1))
        title = header_match.group(2).strip()

        # Objective code
        obj_match = re.search(
            r"\*\*Strategic Objective:\*\*\s*([A-E])\s*[—–-]", block
        )
        objective_code = obj_match.group(1) if obj_match else ""

        # Description
        desc_match = re.search(
            r"\*\*Description:\*\*\s*(.+?)(?=\*\*Action Owner|\*\*Timeline|\*\*Budget|\Z)",
            block, re.DOTALL,
        )
        description = desc_match.group(1).strip() if desc_match else ""

        # Owner
        owner_match = re.search(r"\*\*Action Owner:\*\*\s*(.+?)(?:\n|$)", block)
        owner = owner_match.group(1).strip() if owner_match else ""

        # Timeline
        timeline_match = re.search(r"\*\*Timeline:\*\*\s*(.+?)(?:\n|$)", block)
        timeline = timeline_match.group(1).strip() if timeline_match else ""

        # Budget
        budget_match = re.search(r"\*\*Budget Allocation:\*\*\s*(.+?)(?:\n|$)", block)
        budget = budget_match.group(1).strip() if budget_match else ""

        # Expected outcome
        outcome_match = re.search(
            r"\*\*Expected Outcome:\*\*\s*(.+?)(?=\*\*KPI|\Z)",
            block, re.DOTALL,
        )
        outcome = outcome_match.group(1).strip() if outcome_match else ""

        # KPIs (bullet list)
        kpis: list[str] = []
        kpi_section_match = re.search(
            r"\*\*KPIs?:\*\*\s*\n((?:\s*-\s*.+\n?)+)", block
        )
        if kpi_section_match:
            for line in kpi_section_match.group(1).strip().splitlines():
                cleaned = re.sub(r"^\s*-\s*", "", line).strip()
                if cleaned and not re.match(r"^-+$", cleaned):
                    kpis.append(cleaned)

        actions.append({
            "number": number,
            "title": title,
            "objective_code": objective_code,
            "description": description,
            "owner": owner,
            "timeline": timeline,
            "budget": budget,
            "outcome": outcome,
            "kpis": kpis,
        })

    logger.info("Parsed %d actions from action plan.", len(actions))
    return actions


# ═══════════════════════════════════════════════════════════════════════
# OntologyMapper class
# ═══════════════════════════════════════════════════════════════════════

class OntologyMapper:
    """Hybrid ontology-based mapper for strategy--action alignment.

    Combines a curated healthcare-strategy RDF ontology with keyword
    matching and sentence-transformer embedding similarity to map
    strategic goals and action items to ontology concepts, detect
    coverage gaps, and export the full knowledge graph in Turtle format.

    Args:
        embedding_model_name: Sentence-Transformers model identifier.
        mapping_threshold:    Minimum ``final_score`` for a valid mapping.
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        mapping_threshold: float = MAPPING_THRESHOLD,
    ) -> None:
        self.threshold = mapping_threshold

        # Load embedding model
        logger.info("Loading embedding model '%s' ...", embedding_model_name)
        self.model = SentenceTransformer(embedding_model_name)

        # RDF graph
        self.graph = Graph()
        self.graph.bind("isps", ISPS)
        self.graph.bind("isps_data", ISPS_DATA)
        self.graph.bind("owl", OWL)

        # Internal state (populated by run())
        self.strategic_objectives: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.concept_embeddings: dict[str, np.ndarray] = {}
        self.action_mappings: list[ItemMappingResult] = []
        self.strategy_mappings: list[ItemMappingResult] = []
        self.gap_report: GapReport = GapReport()

    # ------------------------------------------------------------------
    # 1) Ontology definition (RDF)
    # ------------------------------------------------------------------

    def build_ontology_schema(self) -> None:
        """Define ontology classes, properties, and concept hierarchy in RDF.

        Creates:
        - Top-level OWL classes for each domain concept.
        - Mid-level sub-classes under each top-level concept.
        - Object and data properties for linking instances.
        """
        g = self.graph

        # ── Object properties ──────────────────────────────────────────
        properties = {
            "supportsObjective": "Links an action to the strategic objective it supports.",
            "implementsGoal": "Links an action to the specific strategic goal it implements.",
            "relatedToConcept": "Links an item (action/goal) to an ontology concept.",
            "hasKPI": "Associates a KPI with an objective or action.",
            "hasOwner": "Specifies the responsible stakeholder for an action.",
            "hasTimeline": "Specifies the timeline for an action or objective.",
            "hasBudget": "Specifies the budget allocation for an action.",
            "hasEvidenceText": "Stores evidence text supporting a mapping.",
        }
        for prop_name, comment in properties.items():
            prop_uri = ISPS[prop_name]
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            g.add((prop_uri, RDFS.label, Literal(prop_name)))
            g.add((prop_uri, RDFS.comment, Literal(comment)))

        # ── Data properties ────────────────────────────────────────────
        data_props = {
            "hasScore": ("Mapping score (0-1).", XSD.float),
            "hasKeywordScore": ("Keyword match score (0-1).", XSD.float),
            "hasEmbeddingScore": ("Embedding similarity score (0-1).", XSD.float),
            "hasBudgetValue": ("Budget in LKR millions.", XSD.float),
            "hasActionNumber": ("Action sequential number.", XSD.integer),
            "hasObjectiveCode": ("Strategic objective letter code.", XSD.string),
        }
        for prop_name, (comment, datatype) in data_props.items():
            prop_uri = ISPS[prop_name]
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            g.add((prop_uri, RDFS.label, Literal(prop_name)))
            g.add((prop_uri, RDFS.comment, Literal(comment)))
            g.add((prop_uri, RDFS.range, datatype))

        # ── Classes (top-level + mid-level) ────────────────────────────
        for concept_id, concept_data in ONTOLOGY_CONCEPTS.items():
            cls_uri = ISPS[concept_id]
            g.add((cls_uri, RDF.type, OWL.Class))
            g.add((cls_uri, RDFS.label, Literal(concept_data["label"])))
            g.add((cls_uri, RDFS.comment, Literal(concept_data["description"])))

            for child_id, child_data in concept_data.get("children", {}).items():
                child_uri = ISPS[child_id]
                g.add((child_uri, RDF.type, OWL.Class))
                g.add((child_uri, RDFS.subClassOf, cls_uri))
                g.add((child_uri, RDFS.label, Literal(child_data["label"])))
                g.add((child_uri, RDFS.comment, Literal(child_data["description"])))

        logger.info(
            "Ontology schema built: %d triples.",
            len(self.graph),
        )

    # ------------------------------------------------------------------
    # 2) Concept embedding precomputation
    # ------------------------------------------------------------------

    def _precompute_concept_embeddings(self) -> None:
        """Encode all concept descriptions into embeddings for reuse."""
        texts: list[str] = []
        ids: list[str] = []

        for concept_id, concept_data in ONTOLOGY_CONCEPTS.items():
            ids.append(concept_id)
            texts.append(f"{concept_data['label']}. {concept_data['description']}")
            for child_id, child_data in concept_data.get("children", {}).items():
                ids.append(child_id)
                texts.append(f"{child_data['label']}. {child_data['description']}")

        embeddings = self.model.encode(texts, normalize_embeddings=True,
                                       show_progress_bar=False)
        for cid, emb in zip(ids, embeddings):
            self.concept_embeddings[cid] = emb

        logger.info("Precomputed embeddings for %d concepts.", len(ids))

    # ------------------------------------------------------------------
    # 3) Keyword matching
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_score(text: str, concept_id: str) -> tuple[float, list[str]]:
        """Compute normalised keyword match score for one concept.

        Args:
            text:       The item text (lowercased externally).
            concept_id: Key into ``CONCEPT_KEYWORDS``.

        Returns:
            Tuple of (score 0--1, list of matched keywords).
        """
        keywords = CONCEPT_KEYWORDS.get(concept_id, [])
        if not keywords:
            return 0.0, []
        matched = [kw for kw in keywords if kw in text]
        return len(matched) / len(keywords), matched

    # ------------------------------------------------------------------
    # 4) Embedding similarity
    # ------------------------------------------------------------------

    def _embedding_score(self, text_embedding: np.ndarray,
                         concept_id: str) -> float:
        """Cosine similarity between item embedding and concept embedding.

        Both embeddings are already L2-normalised, so dot product = cosine.
        """
        concept_emb = self.concept_embeddings.get(concept_id)
        if concept_emb is None:
            return 0.0
        score = float(np.dot(text_embedding, concept_emb))
        return max(0.0, score)  # clamp negatives

    # ------------------------------------------------------------------
    # 5) Hybrid mapping for a single item
    # ------------------------------------------------------------------

    def _map_item(self, item_id: str, item_type: str,
                  item_text: str) -> ItemMappingResult:
        """Map one item to all ontology concepts using hybrid scoring.

        Args:
            item_id:   Unique identifier.
            item_type: ``"action"`` or ``"strategy_goal"``.
            item_text: Composite text for matching.

        Returns:
            :class:`ItemMappingResult` with scored concept mappings.
        """
        text_lower = item_text.lower()
        text_embedding = self.model.encode(
            item_text, normalize_embeddings=True, show_progress_bar=False,
        )

        all_mappings: list[ConceptMapping] = []

        # Determine parent for each concept
        concept_parent: dict[str, str] = {}
        for top_id, top_data in ONTOLOGY_CONCEPTS.items():
            concept_parent[top_id] = top_id  # top-level is its own parent
            for child_id in top_data.get("children", {}):
                concept_parent[child_id] = top_id

        for concept_id in self.concept_embeddings:
            kw_score, matched_kws = self._keyword_score(text_lower, concept_id)
            emb_score = self._embedding_score(text_embedding, concept_id)
            final = EMBEDDING_WEIGHT * emb_score + KEYWORD_WEIGHT * kw_score

            # Build evidence snippet (first 200 chars of item text)
            evidence = item_text[:200].strip()
            if len(item_text) > 200:
                evidence += "..."

            # Look up label
            concept_label = concept_id
            for top_id, top_data in ONTOLOGY_CONCEPTS.items():
                if top_id == concept_id:
                    concept_label = top_data["label"]
                    break
                for child_id, child_data in top_data.get("children", {}).items():
                    if child_id == concept_id:
                        concept_label = child_data["label"]
                        break

            all_mappings.append(ConceptMapping(
                concept_id=concept_id,
                concept_label=concept_label,
                parent_concept=concept_parent.get(concept_id, concept_id),
                keyword_score=round(kw_score, 4),
                embedding_score=round(emb_score, 4),
                final_score=round(final, 4),
                matched_keywords=matched_kws,
                evidence_text=evidence,
            ))

        # Sort by final_score descending
        all_mappings.sort(key=lambda m: m.final_score, reverse=True)

        # Valid mappings above threshold
        valid = [m for m in all_mappings if m.final_score >= self.threshold]

        # Determine top-level areas
        top_areas = list(dict.fromkeys(m.parent_concept for m in valid))
        is_multi = len(top_areas) >= 2

        return ItemMappingResult(
            item_id=item_id,
            item_type=item_type,
            item_text=item_text,
            mappings=valid,
            all_mappings=all_mappings,
            top_level_areas=top_areas,
            is_multi_area=is_multi,
        )

    # ------------------------------------------------------------------
    # 6) Map all items
    # ------------------------------------------------------------------

    def map_all(self) -> None:
        """Map all strategic goals and actions to ontology concepts."""
        self._precompute_concept_embeddings()

        # Map strategic goals
        for obj in self.strategic_objectives:
            for goal in obj.get("goals", []):
                goal_text = (
                    f"{goal['id']}: {goal['description']}. "
                    f"Objective: {obj['name']}. {obj['goal_statement']}"
                )
                result = self._map_item(
                    item_id=f"goal_{goal['id']}",
                    item_type="strategy_goal",
                    item_text=goal_text,
                )
                self.strategy_mappings.append(result)

        logger.info("Mapped %d strategic goals.", len(self.strategy_mappings))

        # Map actions
        for action in self.actions:
            kpi_text = "; ".join(action.get("kpis", []))
            action_text = (
                f"Action {action['number']}: {action['title']}. "
                f"{action['description']} "
                f"Expected outcome: {action.get('outcome', '')}. "
                f"KPIs: {kpi_text}"
            )
            result = self._map_item(
                item_id=f"action_{action['number']}",
                item_type="action",
                item_text=action_text,
            )
            self.action_mappings.append(result)

        logger.info("Mapped %d actions.", len(self.action_mappings))

    # ------------------------------------------------------------------
    # 7) Gap & misalignment detection
    # ------------------------------------------------------------------

    def detect_gaps(self) -> GapReport:
        """Detect uncovered concepts, weak actions, and conflicts.

        Returns:
            A :class:`GapReport` with three lists.
        """
        report = GapReport()

        # ── Uncovered strategy concepts ────────────────────────────────
        # For each strategic goal, check if any action maps to the
        # same concept with a valid score.
        strategy_concept_ids: set[str] = set()
        for sm in self.strategy_mappings:
            for m in sm.mappings:
                strategy_concept_ids.add(m.concept_id)

        action_concept_ids: set[str] = set()
        for am in self.action_mappings:
            for m in am.mappings:
                action_concept_ids.add(m.concept_id)

        uncovered = strategy_concept_ids - action_concept_ids
        for cid in sorted(uncovered):
            # Find which strategy goals use this concept
            related_goals = [
                sm.item_id for sm in self.strategy_mappings
                if any(m.concept_id == cid for m in sm.mappings)
            ]
            report.uncovered_concepts.append({
                "concept_id": cid,
                "related_strategy_goals": related_goals,
                "note": f"No action maps to concept '{cid}' above threshold.",
            })

        # ── Weak / poorly aligned actions ──────────────────────────────
        for am in self.action_mappings:
            if not am.mappings:
                # No valid mappings at all
                best = am.all_mappings[0] if am.all_mappings else None
                report.weak_actions.append({
                    "action_id": am.item_id,
                    "best_concept": best.concept_id if best else "none",
                    "best_score": best.final_score if best else 0.0,
                    "note": "No concept mapping above threshold.",
                })
            elif all(m.parent_concept == am.mappings[0].parent_concept
                     for m in am.mappings):
                # Only maps within one generic top-level area - check if
                # all scores are marginal
                avg_score = sum(m.final_score for m in am.mappings) / len(am.mappings)
                if avg_score < 0.60:
                    report.weak_actions.append({
                        "action_id": am.item_id,
                        "best_concept": am.mappings[0].concept_id,
                        "best_score": am.mappings[0].final_score,
                        "average_score": round(avg_score, 4),
                        "note": "Marginal mapping — only weakly aligned to generic concepts.",
                    })

        # ── Conflicting mappings ───────────────────────────────────────
        for am in self.action_mappings:
            if am.is_multi_area:
                report.conflicting_actions.append({
                    "action_id": am.item_id,
                    "top_level_areas": am.top_level_areas,
                    "note": (
                        f"Mapped to {len(am.top_level_areas)} unrelated top-level areas: "
                        f"{', '.join(am.top_level_areas)}. Review for coherence."
                    ),
                })

        self.gap_report = report
        logger.info(
            "Gap detection: %d uncovered concepts, %d weak actions, %d conflicting.",
            len(report.uncovered_concepts),
            len(report.weak_actions),
            len(report.conflicting_actions),
        )
        return report

    # ------------------------------------------------------------------
    # 8) Explainability
    # ------------------------------------------------------------------

    def explain_mapping(self, item_id: str,
                        item_type: str = "action") -> dict[str, Any]:
        """Generate a human-readable explanation for an item's mapping.

        Args:
            item_id:   The item identifier (e.g. ``"action_8"``).
            item_type: ``"action"`` or ``"strategy_goal"``.

        Returns:
            Dict with keys: ``item_id``, ``top_concepts`` (with scores),
            ``matched_keywords``, ``evidence_snippets``, ``explanation``.
        """
        source = (
            self.action_mappings if item_type == "action"
            else self.strategy_mappings
        )
        result = next((r for r in source if r.item_id == item_id), None)
        if result is None:
            return {"error": f"Item '{item_id}' not found in {item_type} mappings."}

        top_concepts = [
            {
                "concept": m.concept_id,
                "label": m.concept_label,
                "parent": m.parent_concept,
                "final_score": m.final_score,
                "keyword_score": m.keyword_score,
                "embedding_score": m.embedding_score,
            }
            for m in result.mappings[:5]
        ]

        all_keywords: list[str] = []
        for m in result.mappings[:5]:
            all_keywords.extend(m.matched_keywords)
        unique_keywords = list(dict.fromkeys(all_keywords))

        evidence_snippets = [m.evidence_text for m in result.mappings[:3]]

        # Build natural-language explanation
        if not result.mappings:
            explanation = (
                f"'{item_id}' has no concept mapping above the threshold "
                f"({self.threshold}). This suggests the item may be misaligned "
                f"with the defined ontology or represents an activity outside "
                f"the strategic framework."
            )
        else:
            top = result.mappings[0]
            area_list = ", ".join(result.top_level_areas[:3])
            explanation = (
                f"'{item_id}' is primarily mapped to '{top.concept_label}' "
                f"(score: {top.final_score:.2f}) under the '{top.parent_concept}' "
                f"domain. "
            )
            if len(result.mappings) > 1:
                second = result.mappings[1]
                explanation += (
                    f"It also relates to '{second.concept_label}' "
                    f"(score: {second.final_score:.2f}). "
                )
            if unique_keywords:
                explanation += (
                    f"Key evidence terms: {', '.join(unique_keywords[:6])}. "
                )
            if result.is_multi_area:
                explanation += (
                    f"Note: this item spans multiple domains ({area_list}), "
                    f"which may warrant review for focus."
                )

        return {
            "item_id": item_id,
            "item_type": item_type,
            "top_concepts": top_concepts,
            "matched_keywords": unique_keywords,
            "evidence_snippets": evidence_snippets,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    # 9) RDF graph population with instances
    # ------------------------------------------------------------------

    def build_rdf_instances(self) -> None:
        """Add data instances (objectives, goals, actions, KPIs) to the RDF graph.

        Creates individual RDF resources for each strategic objective,
        strategic goal, action item, and KPI, then links them to ontology
        concepts via ``relatedToConcept``, ``supportsObjective``, and
        ``implementsGoal`` properties.
        """
        g = self.graph

        # ── Strategic objective instances ──────────────────────────────
        for obj in self.strategic_objectives:
            obj_uri = ISPS_DATA[f"objective_{obj['code']}"]
            g.add((obj_uri, RDF.type, ISPS["StrategicObjective"]))
            g.add((obj_uri, RDFS.label, Literal(f"Objective {obj['code']}: {obj['name']}")))
            g.add((obj_uri, ISPS["hasObjectiveCode"], Literal(obj["code"])))

            # Goals
            for goal in obj.get("goals", []):
                goal_uri = ISPS_DATA[f"goal_{goal['id']}"]
                g.add((goal_uri, RDF.type, ISPS["StrategicGoal"]))
                g.add((goal_uri, RDFS.label, Literal(f"{goal['id']}: {goal['description'][:80]}")))
                g.add((goal_uri, ISPS["supportsObjective"], obj_uri))

                # Link goal to mapped concepts
                goal_mapping = next(
                    (sm for sm in self.strategy_mappings
                     if sm.item_id == f"goal_{goal['id']}"),
                    None,
                )
                if goal_mapping:
                    for m in goal_mapping.mappings[:3]:
                        g.add((goal_uri, ISPS["relatedToConcept"], ISPS[m.concept_id]))
                        g.add((goal_uri, ISPS["hasScore"],
                               Literal(m.final_score, datatype=XSD.float)))

            # KPIs
            for i, kpi_text in enumerate(obj.get("kpis", []), 1):
                kpi_uri = ISPS_DATA[f"kpi_{obj['code']}_{i}"]
                g.add((kpi_uri, RDF.type, ISPS["KPI"]))
                g.add((kpi_uri, RDFS.label, Literal(kpi_text[:100])))
                g.add((kpi_uri, ISPS["hasKPI"], obj_uri))

        # ── Action instances ───────────────────────────────────────────
        for action in self.actions:
            act_uri = ISPS_DATA[f"action_{action['number']}"]
            g.add((act_uri, RDF.type, ISPS["ActionItem"]))
            g.add((act_uri, RDFS.label,
                   Literal(f"Action {action['number']}: {action['title'][:60]}")))
            g.add((act_uri, ISPS["hasActionNumber"],
                   Literal(action["number"], datatype=XSD.integer)))

            # Link to declared objective
            obj_uri = ISPS_DATA[f"objective_{action['objective_code']}"]
            g.add((act_uri, ISPS["supportsObjective"], obj_uri))

            # Owner
            if action.get("owner"):
                g.add((act_uri, ISPS["hasOwner"], Literal(action["owner"])))

            # Timeline
            if action.get("timeline"):
                g.add((act_uri, ISPS["hasTimeline"], Literal(action["timeline"])))

            # Budget
            budget_match = re.search(r"LKR\s*(\d+)M", action.get("budget", ""))
            if budget_match:
                g.add((act_uri, ISPS["hasBudgetValue"],
                       Literal(float(budget_match.group(1)), datatype=XSD.float)))

            # Link to mapped concepts
            action_mapping = next(
                (am for am in self.action_mappings
                 if am.item_id == f"action_{action['number']}"),
                None,
            )
            if action_mapping:
                for m in action_mapping.mappings[:5]:
                    concept_uri = ISPS[m.concept_id]
                    g.add((act_uri, ISPS["relatedToConcept"], concept_uri))

                    # Also link action → goals that share the same concept
                    for sm in self.strategy_mappings:
                        if any(sm_m.concept_id == m.concept_id
                               for sm_m in sm.mappings[:3]):
                            goal_uri = ISPS_DATA[sm.item_id]
                            g.add((act_uri, ISPS["implementsGoal"], goal_uri))

            # KPIs for the action
            for i, kpi_text in enumerate(action.get("kpis", []), 1):
                kpi_uri = ISPS_DATA[f"action_kpi_{action['number']}_{i}"]
                g.add((kpi_uri, RDF.type, ISPS["KPI"]))
                g.add((kpi_uri, RDFS.label, Literal(kpi_text[:100])))
                g.add((kpi_uri, ISPS["hasKPI"], act_uri))

        # ── Meta-classes for instance types ────────────────────────────
        for cls_name, comment in [
            ("StrategicObjective", "A high-level strategic objective (A-E)."),
            ("StrategicGoal", "A specific goal under a strategic objective."),
            ("ActionItem", "An operational action item from the action plan."),
            ("KPI", "A key performance indicator."),
        ]:
            cls_uri = ISPS[cls_name]
            g.add((cls_uri, RDF.type, OWL.Class))
            g.add((cls_uri, RDFS.label, Literal(cls_name)))
            g.add((cls_uri, RDFS.comment, Literal(comment)))

        logger.info("RDF instances built. Total triples: %d.", len(self.graph))

    # ------------------------------------------------------------------
    # 10) Export helpers
    # ------------------------------------------------------------------

    def export_turtle(self, filepath: Path | None = None) -> Path:
        """Serialise the RDF graph to Turtle format.

        Args:
            filepath: Output path. Defaults to ``outputs/ontology.ttl``.

        Returns:
            The path written to.
        """
        filepath = filepath or ONTOLOGY_TTL
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(filepath), format="turtle")
        logger.info("Ontology exported to %s (%d triples).",
                     filepath, len(self.graph))
        return filepath

    def export_mappings(self, filepath: Path | None = None) -> Path:
        """Export action and strategy mappings to JSON.

        Args:
            filepath: Output path. Defaults to ``outputs/mappings.json``.

        Returns:
            The path written to.
        """
        filepath = filepath or MAPPINGS_JSON
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "embedding_model": EMBEDDING_MODEL_NAME,
                "mapping_threshold": self.threshold,
                "embedding_weight": EMBEDDING_WEIGHT,
                "keyword_weight": KEYWORD_WEIGHT,
                "total_concepts": len(self.concept_embeddings),
            },
            "action_mappings": [
                {
                    "item_id": am.item_id,
                    "item_type": am.item_type,
                    "top_level_areas": am.top_level_areas,
                    "is_multi_area": am.is_multi_area,
                    "mappings": [asdict(m) for m in am.mappings],
                }
                for am in self.action_mappings
            ],
            "strategy_mappings": [
                {
                    "item_id": sm.item_id,
                    "item_type": sm.item_type,
                    "top_level_areas": sm.top_level_areas,
                    "is_multi_area": sm.is_multi_area,
                    "mappings": [asdict(m) for m in sm.mappings],
                }
                for sm in self.strategy_mappings
            ],
        }

        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        logger.info("Mappings exported to %s.", filepath)
        return filepath

    def export_gaps(self, filepath: Path | None = None) -> Path:
        """Export gap and misalignment analysis to JSON.

        Args:
            filepath: Output path. Defaults to ``outputs/gaps.json``.

        Returns:
            The path written to.
        """
        filepath = filepath or GAPS_JSON
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "uncovered_strategy_concepts": self.gap_report.uncovered_concepts,
            "weak_actions": self.gap_report.weak_actions,
            "conflicting_actions": self.gap_report.conflicting_actions,
            "summary": {
                "uncovered_concept_count": len(self.gap_report.uncovered_concepts),
                "weak_action_count": len(self.gap_report.weak_actions),
                "conflicting_action_count": len(self.gap_report.conflicting_actions),
            },
        }

        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        logger.info("Gaps exported to %s.", filepath)
        return filepath

    # ------------------------------------------------------------------
    # 11) Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full ontology mapping pipeline.

        Steps:
            1. Parse strategic plan and action plan from markdown.
            2. Build ontology schema in RDF.
            3. Map all items to concepts (hybrid scoring).
            4. Detect gaps and misalignments.
            5. Build RDF instances.
            6. Export all outputs.

        Returns:
            Summary dict with counts and file paths.
        """
        logger.info("=" * 60)
        logger.info("ONTOLOGY MAPPER — Starting pipeline")
        logger.info("=" * 60)

        # Step 1: Parse
        self.strategic_objectives = parse_strategic_plan(STRATEGIC_PLAN_FILE)
        self.actions = parse_action_plan(ACTION_PLAN_FILE)

        # Step 2: Build ontology schema
        self.build_ontology_schema()

        # Step 3: Map all items
        self.map_all()

        # Step 4: Detect gaps
        self.detect_gaps()

        # Step 5: Build RDF instances
        self.build_rdf_instances()

        # Step 6: Export
        ttl_path = self.export_turtle()
        mappings_path = self.export_mappings()
        gaps_path = self.export_gaps()

        summary = {
            "strategic_objectives_parsed": len(self.strategic_objectives),
            "actions_parsed": len(self.actions),
            "strategy_goals_mapped": len(self.strategy_mappings),
            "actions_mapped": len(self.action_mappings),
            "total_rdf_triples": len(self.graph),
            "uncovered_concepts": len(self.gap_report.uncovered_concepts),
            "weak_actions": len(self.gap_report.weak_actions),
            "conflicting_actions": len(self.gap_report.conflicting_actions),
            "outputs": {
                "ontology_ttl": str(ttl_path),
                "mappings_json": str(mappings_path),
                "gaps_json": str(gaps_path),
            },
        }

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("  Objectives parsed : %d", summary["strategic_objectives_parsed"])
        logger.info("  Actions parsed    : %d", summary["actions_parsed"])
        logger.info("  Goals mapped      : %d", summary["strategy_goals_mapped"])
        logger.info("  Actions mapped    : %d", summary["actions_mapped"])
        logger.info("  RDF triples       : %d", summary["total_rdf_triples"])
        logger.info("  Uncovered concepts: %d", summary["uncovered_concepts"])
        logger.info("  Weak actions      : %d", summary["weak_actions"])
        logger.info("  Conflicting maps  : %d", summary["conflicting_actions"])
        logger.info("=" * 60)

        return summary


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mapper = OntologyMapper()
    summary = mapper.run()

    # ── Print mapping highlights ───────────────────────────────────
    print("\n" + "=" * 70)
    print("ONTOLOGY MAPPING — RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nStrategic objectives: {summary['strategic_objectives_parsed']}")
    print(f"Actions parsed:       {summary['actions_parsed']}")
    print(f"Strategy goals mapped: {summary['strategy_goals_mapped']}")
    print(f"Actions mapped:       {summary['actions_mapped']}")
    print(f"RDF triples:          {summary['total_rdf_triples']}")

    # ── Action mapping details ─────────────────────────────────────
    print("\n--- ACTION MAPPINGS (top concept per action) ---")
    for am in mapper.action_mappings:
        if am.mappings:
            top = am.mappings[0]
            areas = ", ".join(am.top_level_areas[:3])
            flag = " [MULTI-AREA]" if am.is_multi_area else ""
            print(
                f"  {am.item_id:<12}  "
                f"{top.concept_label:<30}  "
                f"score={top.final_score:.3f}  "
                f"areas=[{areas}]{flag}"
            )
        else:
            print(f"  {am.item_id:<12}  ** NO VALID MAPPING **")

    # ── Gap report ─────────────────────────────────────────────────
    print(f"\n--- GAP REPORT ---")
    print(f"Uncovered strategy concepts: {summary['uncovered_concepts']}")
    for gap in mapper.gap_report.uncovered_concepts:
        print(f"  - {gap['concept_id']} (goals: {gap['related_strategy_goals']})")

    print(f"\nWeak / poorly aligned actions: {summary['weak_actions']}")
    for wa in mapper.gap_report.weak_actions:
        print(f"  - {wa['action_id']}: best={wa['best_concept']} "
              f"(score={wa['best_score']:.3f})")

    print(f"\nConflicting (multi-area) actions: {summary['conflicting_actions']}")
    for ca in mapper.gap_report.conflicting_actions:
        print(f"  - {ca['action_id']}: areas={ca['top_level_areas']}")

    # ── Explainability demo ────────────────────────────────────────
    print("\n--- EXPLAINABILITY DEMO ---")
    for demo_id in ["action_8", "action_19", "action_1", "action_9"]:
        expl = mapper.explain_mapping(demo_id, "action")
        print(f"\n  {demo_id}:")
        print(f"    {expl.get('explanation', 'N/A')}")

    # ── Output paths ───────────────────────────────────────────────
    print(f"\n--- OUTPUT FILES ---")
    for name, path in summary["outputs"].items():
        print(f"  {name}: {path}")
