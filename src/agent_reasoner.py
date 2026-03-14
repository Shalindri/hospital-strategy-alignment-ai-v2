"""
Agent Reasoner for Hospital Strategy--Action Plan Alignment System (ISPS).

This module implements an agentic reasoning layer that diagnoses
synchronization issues between the Strategic Plan and Action Plan,
gathers evidence from the vector database and ontology, generates
improvement recommendations via a local LLM, and self-critiques
its proposals before finalising them.

Agent workflow (Plan → Act → Reflect)
-------------------------------------
The agent executes a bounded loop (``max_iterations=5``)::

    ┌──────────────┐
    │   DIAGNOSE   │  Read sync report → score & rank issues
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ INVESTIGATE  │  Build evidence plan → call tools
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   REASON     │  LLM proposes recommendations (evidence-grounded)
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  CRITIQUE    │  LLM checks feasibility, alignment, measurability
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   REFINE     │  Revise based on critique; early-stop if marginal
    └──────────────┘

Tool interfaces
---------------
The agent has four typed tools:

*  ``vector_search``      — semantic retrieval from ChromaDB
*  ``ontology_lookup``    — concept expansion from ontology mappings
*  ``calculate_impact``   — budget / time / risk estimation
*  ``validate_alignment`` — score a suggestion against an objective

Outputs
-------
*  ``outputs/agent_recommendations.json`` — final recommendations
*  ``outputs/agent_trace.json``           — structured reasoning trace

Typical usage::

    from src.agent_reasoner import AgentReasoner

    agent = AgentReasoner()
    agent.run()

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_llm

from src.vector_store import (
    VectorStore,
    STRATEGIC_COLLECTION,
    ACTION_COLLECTION,
)

# ---------------------------------------------------------------------------
# Logging — dual handler: console + file
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("agent_reasoner")
logger.setLevel(logging.INFO)
logger.propagate = False

# Console handler
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_ch)

    # File handler
    _fh = logging.FileHandler(LOG_DIR / "agent_reasoner.log", mode="w",
                               encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_fh)

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ALIGNMENT_JSON = DATA_DIR / "alignment_report.json"
MAPPINGS_JSON = OUTPUT_DIR / "mappings.json"
KG_JSON = OUTPUT_DIR / "strategy_action_kg.json"
STRATEGIC_JSON = DATA_DIR / "strategic_plan.json"
ACTION_JSON = DATA_DIR / "action_plan.json"

RECOMMENDATIONS_OUT = OUTPUT_DIR / "agent_recommendations.json"
TRACE_OUT = OUTPUT_DIR / "agent_trace.json"

# LLM configuration (set in src/config.py via .env)
from src.config import (
    OPENAI_MODEL, LLM_TEMPERATURE,
    MAX_RETRIES, RETRY_DELAYS, MAX_ITERATIONS,
)
MAX_TOOL_CALLS_PER_ITERATION = 8
ISSUE_SCORE_THRESHOLD = 0.15
MARGINAL_IMPROVEMENT_THRESHOLD = 0.05

# Alignment thresholds (from shared config)
from src.config import ORPHAN_THRESHOLD  # noqa: E402
LOW_COVERAGE_THRESHOLD = 0.25
WEAK_MAPPING_THRESHOLD = 0.55

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Objective metadata (populated dynamically from loaded data)
from src.synchronization_analyzer import OBJECTIVE_NAMES  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocSnippet:
    """A retrieved document snippet from the vector database.

    Attributes:
        id:     Document ID (e.g. ``"action_6"``, ``"obj_A"``).
        source: Collection name the snippet came from.
        text:   The document text content.
        score:  Cosine similarity score (0--1).
    """
    id: str
    source: str
    text: str
    score: float


@dataclass
class DiagnosedIssue:
    """A diagnosed synchronization issue with priority scoring.

    Attributes:
        issue_id:              Unique identifier (e.g. ``"orphan_action_8"``).
        issue_type:            Category: ``"orphan_action"``,
                               ``"under_supported_strategy"``,
                               ``"conflicting_mapping"``, ``"kpi_mismatch"``.
        severity:              Severity score (0--1).
        confidence:            Confidence in the diagnosis (0--1).
        business_impact:       Estimated business impact (0--1).
        priority_score:        ``severity * confidence * business_impact``.
        affected_objectives:   List of objective codes impacted.
        description:           Human-readable description.
        evidence_ids:          Snippet IDs supporting the diagnosis.
        action_numbers:        Affected action numbers.
    """
    issue_id: str
    issue_type: str
    severity: float
    confidence: float
    business_impact: float
    priority_score: float
    affected_objectives: list[str]
    description: str
    evidence_ids: list[str] = field(default_factory=list)
    action_numbers: list[int] = field(default_factory=list)


@dataclass
class Recommendation:
    """A concrete improvement recommendation.

    Attributes:
        rec_id:            Unique recommendation identifier.
        issue_id:          The diagnosed issue this addresses.
        issue_type:        Issue category.
        what_to_change:    Summary of the proposed change.
        why:               Rationale grounded in evidence.
        actions:           List of specific change actions.
        kpis:              Measurable KPIs for the recommendation.
        estimated_budget:  Budget estimate in LKR millions.
        estimated_timeline: Timeline estimate (e.g. ``"Q2-Q3 2025"``).
        impact_score:      Estimated impact (0--1).
        affected_objectives: Objective codes.
        evidence_ids:      Snippet IDs used as evidence.
        confidence:        Agent confidence (``"HIGH"``/``"MEDIUM"``/``"LOW"``).
    """
    rec_id: str
    issue_id: str
    issue_type: str
    what_to_change: str
    why: str
    actions: list[str] = field(default_factory=list)
    kpis: list[str] = field(default_factory=list)
    estimated_budget: float = 0.0
    estimated_timeline: str = ""
    impact_score: float = 0.0
    affected_objectives: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    confidence: str = "MEDIUM"


@dataclass
class CritiqueSummary:
    """Self-critique results for a recommendation.

    Attributes:
        feasibility:      Assessment of resource / time feasibility.
        alignment_check:  Whether the recommendation strengthens alignment.
        kpi_measurability: Whether KPIs are concrete and measurable.
        risks:            Identified risks.
        missing_info:     Information gaps that limit confidence.
        accepted:         Whether the recommendation passes critique.
        revision_notes:   Changes made after critique.
    """
    feasibility: str = ""
    alignment_check: str = ""
    kpi_measurability: str = ""
    risks: str = ""
    missing_info: str = ""
    accepted: bool = True
    revision_notes: str = ""


@dataclass
class TraceEntry:
    """Structured trace entry for one iteration of the agent loop.

    Attributes:
        iteration:          Iteration number (1-based).
        issue_detected:     The diagnosed issue being addressed.
        investigation_plan: What evidence was sought.
        evidence_used:      Snippet IDs gathered.
        tool_calls:         Log of tool invocations.
        reasoning_summary:  3--6 bullet points of reasoning.
        recommendation:     The proposed recommendation.
        critique_summary:   Self-critique results.
        final_decision:     Accepted changes.
    """
    iteration: int
    issue_detected: dict[str, Any] = field(default_factory=dict)
    investigation_plan: list[str] = field(default_factory=list)
    evidence_used: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: list[str] = field(default_factory=list)
    recommendation: dict[str, Any] = field(default_factory=dict)
    critique_summary: dict[str, Any] = field(default_factory=dict)
    final_decision: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DIAGNOSE_AND_CRITIQUE_PROMPT = """You are a hospital strategy consultant analyzing synchronization issues
between a hospital's strategic plan and action plan. You must both recommend AND self-critique.

ISSUE TYPE: {issue_type}
ISSUE DESCRIPTION: {issue_description}

AFFECTED OBJECTIVES: {affected_objectives}

RELEVANT STRATEGIC CONTEXT:
{strategic_context}

RELEVANT ACTION CONTEXT:
{action_context}

ONTOLOGY CONTEXT:
{ontology_context}

TASK: Analyze this issue, provide recommendations, then critically evaluate them yourself.
Format your response EXACTLY as shown below (including the separator line):

ROOT_CAUSE: [1-2 sentences explaining why this misalignment exists]

RECOMMENDATION_1: [Concise description of change]
RECOMMENDATION_2: [Concise description of change]
RECOMMENDATION_3: [Concise description of change if applicable]

NEW_KPIS:
- [Measurable KPI 1]
- [Measurable KPI 2]
- [Measurable KPI 3]

TIMELINE: [Suggested timeline for implementing changes]

BUDGET_ESTIMATE: [Estimated additional budget needed in LKR millions, or "No additional budget"]

EXPECTED_OUTCOME: [2-3 sentences describing the expected improvement]

CONFIDENCE: [HIGH, MEDIUM, or LOW]

--- SELF-CRITIQUE ---

FEASIBILITY: [Is this achievable given typical hospital resources? 1-2 sentences]

ALIGNMENT_CHECK: [Does this genuinely strengthen strategic alignment? 1-2 sentences]

KPI_MEASURABILITY: [Are the proposed KPIs concrete and measurable? 1-2 sentences]

RISKS: [Key risks of implementing this change. 1-2 sentences]

MISSING_INFO: [What additional information would improve this recommendation? 1-2 sentences]

VERDICT: [ACCEPT, REVISE, or REJECT]

REVISION_NOTES: [If REVISE: specific changes needed. If ACCEPT or REJECT: brief justification]"""


# ═══════════════════════════════════════════════════════════════════════
# AgentReasoner class
# ═══════════════════════════════════════════════════════════════════════

class AgentReasoner:
    """Agentic reasoning layer for strategy--action alignment improvement.

    The agent follows a Plan → Act → Reflect loop:

    1. **Diagnose** — identify and score synchronization issues.
    2. **Investigate** — gather evidence using typed tools.
    3. **Reason** — generate recommendations via LLM.
    4. **Critique** — self-evaluate feasibility and alignment.
    5. **Refine** — revise and finalise.

    Args:
        max_iterations:  Maximum agent loop iterations.
    """

    def __init__(
        self,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.max_iterations = max_iterations

        # Load upstream data
        self._alignment = self._load_json(ALIGNMENT_JSON)
        self._mappings = self._load_json(MAPPINGS_JSON)
        self._strategic = self._load_json(STRATEGIC_JSON)
        self._actions = self._load_json(ACTION_JSON)
        self._kg_data = self._load_json(KG_JSON) if KG_JSON.exists() else {}

        # Initialise vector store
        logger.info("Initialising vector store ...")
        self._vs = VectorStore()

        # Initialise LLM (provider selected via .env)
        logger.info("Initialising LLM ...")
        self._llm = get_llm(temperature=LLM_TEMPERATURE)

        # Build lookup indices
        self._action_by_num: dict[int, dict] = {
            a["action_number"]: a for a in self._actions.get("actions", [])
        }
        self._objective_by_code: dict[str, dict] = {
            o["code"]: o for o in self._strategic.get("objectives", [])
        }
        self._alignment_by_action: dict[int, dict] = {
            aa["action_number"]: aa
            for aa in self._alignment.get("action_alignments", [])
        }
        self._mapping_by_action: dict[str, dict] = {
            m["item_id"]: m for m in self._mappings.get("action_mappings", [])
        }
        self._mapping_by_goal: dict[str, dict] = {
            m["item_id"]: m for m in self._mappings.get("strategy_mappings", [])
        }

        # State
        self._issues: list[DiagnosedIssue] = []
        self._recommendations: list[Recommendation] = []
        self._traces: list[TraceEntry] = []
        self._tool_call_count: int = 0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(filepath: Path) -> dict[str, Any]:
        """Load a JSON file, returning empty dict if missing."""
        if not filepath.exists():
            logger.warning("File not found: %s", filepath)
            return {}
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Tool 1: vector_search
    # ------------------------------------------------------------------

    def vector_search(self, query: str, k: int = 5,
                      collection: str = "both") -> list[DocSnippet]:
        """Semantic search against the vector database.

        Args:
            query:      Free-text search query.
            k:          Number of results per collection.
            collection: ``"strategic"``, ``"action"``, or ``"both"``.

        Returns:
            List of :class:`DocSnippet` ordered by score descending.
        """
        self._tool_call_count += 1
        snippets: list[DocSnippet] = []

        collections = []
        if collection in ("strategic", "both"):
            collections.append(STRATEGIC_COLLECTION)
        if collection in ("action", "both"):
            collections.append(ACTION_COLLECTION)

        for coll in collections:
            try:
                results = self._vs.search_similar(query, coll, top_k=k)
                for doc_id, score, text in zip(
                    results.get("ids", []),
                    results.get("scores", []),
                    results.get("documents", []),
                ):
                    snippets.append(DocSnippet(
                        id=doc_id,
                        source=coll,
                        text=text[:300],
                        score=round(score, 4),
                    ))
            except Exception as exc:
                logger.warning("vector_search failed for '%s': %s", coll, exc)

        snippets.sort(key=lambda s: s.score, reverse=True)
        logger.debug("vector_search('%s') → %d snippets.", query[:40], len(snippets))
        return snippets

    # ------------------------------------------------------------------
    # Tool 2: ontology_lookup
    # ------------------------------------------------------------------

    def ontology_lookup(self, concept: str) -> dict[str, Any]:
        """Look up an ontology concept from cached mappings.

        Searches both action and strategy mappings for concept references,
        returning the concept's metadata and related concepts.

        Args:
            concept: Concept name or ID (e.g. ``"Cardiology"``).

        Returns:
            Dict with ``concept_id``, ``label``, ``description``,
            ``related_concepts``, ``keywords``.
        """
        self._tool_call_count += 1
        concept_lower = concept.lower().replace(" ", "")

        # Search through all mappings for matching concept
        for section in ("action_mappings", "strategy_mappings"):
            for item in self._mappings.get(section, []):
                for m in item.get("mappings", []):
                    cid = m.get("concept_id", "")
                    if (cid.lower().replace("_", "") == concept_lower or
                            m.get("concept_label", "").lower().replace(" ", "") == concept_lower):
                        # Find related concepts in same parent
                        parent = m.get("parent_concept", "")
                        related = set()
                        for s2 in ("action_mappings", "strategy_mappings"):
                            for item2 in self._mappings.get(s2, []):
                                for m2 in item2.get("mappings", []):
                                    if m2.get("parent_concept") == parent and m2["concept_id"] != cid:
                                        related.add(m2["concept_id"])

                        return {
                            "concept_id": cid,
                            "label": m.get("concept_label", cid),
                            "parent_concept": parent,
                            "related_concepts": sorted(related),
                            "keywords": m.get("matched_keywords", []),
                            "found": True,
                        }

        return {"concept_id": concept, "found": False, "related_concepts": [], "keywords": []}

    # ------------------------------------------------------------------
    # Tool 3: calculate_impact
    # ------------------------------------------------------------------

    def calculate_impact(self, suggestion: dict[str, Any]) -> dict[str, Any]:
        """Estimate the impact of a recommendation.

        Uses a rule-based model considering the number of affected
        objectives, budget proportionality, and alignment improvement
        potential.

        Args:
            suggestion: Dict with ``affected_objectives``, ``actions``,
                        ``estimated_budget``, ``estimated_timeline``.

        Returns:
            Dict with ``impact_score``, ``cost_estimate_LKR``,
            ``time_to_deliver_weeks``, ``risk_level``, ``rationale``.
        """
        self._tool_call_count += 1

        n_objectives = len(suggestion.get("affected_objectives", []))
        n_actions = len(suggestion.get("actions", []))
        budget = suggestion.get("estimated_budget", 0.0) or 0.0

        # Impact score: weighted combination
        obj_factor = min(n_objectives / 5.0, 1.0) * 0.4
        action_factor = min(n_actions / 4.0, 1.0) * 0.3
        budget_factor = min(budget / 50.0, 1.0) * 0.3  # normalised to 50M LKR
        impact_score = round(obj_factor + action_factor + budget_factor, 3)

        # Time estimate from timeline string
        timeline = suggestion.get("estimated_timeline", "")
        quarters = re.findall(r"Q[1-4]", timeline)
        weeks = len(quarters) * 13 if quarters else 13

        # Risk assessment
        if budget > 30:
            risk_level = "HIGH"
        elif budget > 15 or n_objectives > 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        rationale = (
            f"Impacts {n_objectives} objective(s) with {n_actions} change action(s). "
            f"Budget of LKR {budget:.0f}M is "
            f"{'significant' if budget > 20 else 'moderate' if budget > 10 else 'modest'}."
        )

        return {
            "impact_score": impact_score,
            "cost_estimate_LKR": budget,
            "time_to_deliver_weeks": weeks,
            "risk_level": risk_level,
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    # Tool 4: validate_alignment
    # ------------------------------------------------------------------

    def validate_alignment(self, suggestion_text: str,
                           target_objective: str) -> dict[str, Any]:
        """Validate that a suggestion aligns with a target objective.

        Encodes the suggestion text and compares it against the
        strategic objective embedding using cosine similarity.

        Args:
            suggestion_text:   Description of the suggested change.
            target_objective:  Objective code (e.g. ``"A"``).

        Returns:
            Dict with ``alignment_score``, ``evidence_ids``.
        """
        self._tool_call_count += 1

        # Search strategic objectives for the most relevant match
        results = self._vs.search_similar(
            suggestion_text, STRATEGIC_COLLECTION, top_k=5,
        )

        target_score = 0.0
        evidence_ids: list[str] = []
        for doc_id, score in zip(results.get("ids", []),
                                  results.get("scores", [])):
            if target_objective.lower() in doc_id.lower():
                target_score = score
            evidence_ids.append(doc_id)

        return {
            "alignment_score": round(target_score, 4),
            "best_match_id": results["ids"][0] if results.get("ids") else "",
            "best_match_score": round(results["scores"][0], 4) if results.get("scores") else 0.0,
            "evidence_ids": evidence_ids,
        }

    # ------------------------------------------------------------------
    # LLM invocation with cache + retry
    # ------------------------------------------------------------------

    def _invoke_llm(self, prompt_text: str) -> str:
        """Call the LLM with retry logic.

        Args:
            prompt_text: The fully formatted prompt string.

        Returns:
            The LLM response text.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info("LLM call attempt %d/%d ...",
                            attempt + 1, MAX_RETRIES)
                response = self._llm.invoke(prompt_text)
                return response
            except Exception as exc:
                last_error = exc
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning("LLM call failed (attempt %d): %s. Retrying in %.1fs ...",
                               attempt + 1, str(exc)[:120], delay)
                time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts: {last_error}"
        )

    # ══════════════════════════════════════════════════════════════════
    # Phase A: DIAGNOSE
    # ══════════════════════════════════════════════════════════════════

    def diagnose(self) -> list[DiagnosedIssue]:
        """Identify and score synchronization issues.

        Scans the alignment report and ontology mappings for four
        issue types:

        1. **Orphan actions** — actions with no strategic link above
           threshold.
        2. **Under-supported strategies** — objectives with low
           action coverage.
        3. **Conflicting mappings** — actions mapped to unrelated
           top-level ontology areas.
        4. **KPI mismatch** — actions missing measurable KPIs or
           with KPIs disconnected from strategic KPIs.

        Returns:
            Sorted list of :class:`DiagnosedIssue` by priority_score
            descending.
        """
        issues: list[DiagnosedIssue] = []

        # ── 1. Orphan actions ──────────────────────────────────────────
        for aa in self._alignment.get("action_alignments", []):
            if aa.get("is_orphan", False) or aa.get("best_score", 1.0) < ORPHAN_THRESHOLD:
                num = aa["action_number"]
                action = self._action_by_num.get(num, {})
                budget = action.get("budget_lkr_millions", 0.0) or 0.0
                # Business impact scales with budget
                biz_impact = min(budget / 25.0, 1.0)
                severity = 1.0 - aa.get("best_score", 0.0)

                issues.append(DiagnosedIssue(
                    issue_id=f"orphan_action_{num}",
                    issue_type="orphan_action",
                    severity=round(severity, 3),
                    confidence=0.90,
                    business_impact=round(biz_impact, 3),
                    priority_score=round(severity * 0.90 * biz_impact, 4),
                    affected_objectives=[aa.get("declared_objective", "")],
                    description=(
                        f"Action {num} ('{action.get('title', '?')}') has a best "
                        f"alignment score of {aa.get('best_score', 0):.3f}, below the "
                        f"threshold of {ORPHAN_THRESHOLD}. Budget: LKR {budget:.0f}M."
                    ),
                    action_numbers=[num],
                ))

        # ── 2. Under-supported strategies ──────────────────────────────
        for oa in self._alignment.get("objective_alignments", []):
            coverage = oa.get("coverage_score", 1.0)
            if coverage < LOW_COVERAGE_THRESHOLD:
                code = oa["code"]
                gap_actions = oa.get("gap_actions", [])
                severity = 1.0 - coverage
                biz_impact = 0.8  # all strategic objectives are high-impact

                issues.append(DiagnosedIssue(
                    issue_id=f"under_supported_{code}",
                    issue_type="under_supported_strategy",
                    severity=round(severity, 3),
                    confidence=0.85,
                    business_impact=biz_impact,
                    priority_score=round(severity * 0.85 * biz_impact, 4),
                    affected_objectives=[code],
                    description=(
                        f"Objective {code} ('{OBJECTIVE_NAMES.get(code, '')}') has "
                        f"only {coverage:.0%} action coverage. Gap actions: {gap_actions}."
                    ),
                    action_numbers=gap_actions,
                ))

        # ── 3. Conflicting mappings ────────────────────────────────────
        for am in self._mappings.get("action_mappings", []):
            if am.get("is_multi_area", False):
                areas = am.get("top_level_areas", [])
                num_str = am["item_id"].replace("action_", "")
                num = int(num_str) if num_str.isdigit() else 0
                action = self._action_by_num.get(num, {})

                issues.append(DiagnosedIssue(
                    issue_id=f"conflicting_{am['item_id']}",
                    issue_type="conflicting_mapping",
                    severity=0.6,
                    confidence=0.75,
                    business_impact=0.5,
                    priority_score=round(0.6 * 0.75 * 0.5, 4),
                    affected_objectives=[action.get("strategic_objective_code", "")],
                    description=(
                        f"{am['item_id']} maps to multiple unrelated ontology areas: "
                        f"{', '.join(areas)}. May indicate lack of strategic focus."
                    ),
                    action_numbers=[num] if num else [],
                ))

        # ── 4. KPI mismatch ────────────────────────────────────────────
        for aa in self._alignment.get("action_alignments", []):
            num = aa["action_number"]
            action = self._action_by_num.get(num, {})
            action_kpis = action.get("kpis", [])
            declared_obj = aa.get("declared_objective", "")
            obj = self._objective_by_code.get(declared_obj, {})
            obj_kpis = [k.get("KPI", "") for k in obj.get("kpis", [])]

            # Check if action KPIs have any overlap with objective KPIs
            action_kpi_text = " ".join(action_kpis).lower()
            matched_obj_kpis = [
                k for k in obj_kpis
                if any(word in action_kpi_text
                       for word in k.lower().split() if len(word) > 3)
            ]

            if len(action_kpis) < 2 or (obj_kpis and not matched_obj_kpis):
                severity = 0.5 if not matched_obj_kpis else 0.3
                issues.append(DiagnosedIssue(
                    issue_id=f"kpi_mismatch_{num}",
                    issue_type="kpi_mismatch",
                    severity=severity,
                    confidence=0.70,
                    business_impact=0.4,
                    priority_score=round(severity * 0.70 * 0.4, 4),
                    affected_objectives=[declared_obj],
                    description=(
                        f"Action {num} KPIs ({len(action_kpis)} found) do not align "
                        f"with Objective {declared_obj} KPIs. "
                        f"Matched: {len(matched_obj_kpis)}/{len(obj_kpis)} strategic KPIs."
                    ),
                    action_numbers=[num],
                ))

        # Sort by priority score descending (deterministic tiebreak on id)
        issues.sort(key=lambda i: (-i.priority_score, i.issue_id))
        self._issues = issues

        logger.info("Diagnosed %d issues. Top-3 scores: %s",
                     len(issues),
                     [f"{i.issue_id}={i.priority_score:.4f}" for i in issues[:3]])
        return issues

    # ══════════════════════════════════════════════════════════════════
    # Phase B: INVESTIGATE (evidence gathering)
    # ══════════════════════════════════════════════════════════════════

    def _gather_evidence(self, issue: DiagnosedIssue) -> tuple[
        list[DocSnippet], dict[str, Any], list[dict[str, Any]]
    ]:
        """Gather evidence for a diagnosed issue using tools.

        Args:
            issue: The issue to investigate.

        Returns:
            Tuple of (snippets, ontology_info, tool_call_log).
        """
        tool_calls: list[dict[str, Any]] = []
        all_snippets: list[DocSnippet] = []

        # ── Vector search for relevant strategic context ───────────────
        if issue.action_numbers:
            action = self._action_by_num.get(issue.action_numbers[0], {})
            query = f"{action.get('title', '')} {action.get('description', '')[:150]}"
        else:
            obj_code = (
                issue.affected_objectives[0]
                if issue.affected_objectives
                else next(iter(self._objective_by_code), "A")
            )
            obj = self._objective_by_code.get(obj_code, {})
            query = obj.get("goal_statement", OBJECTIVE_NAMES.get(obj_code, ""))

        snippets = self.vector_search(query[:200], k=3, collection="both")
        all_snippets.extend(snippets)
        tool_calls.append({
            "tool": "vector_search",
            "query": query[:100],
            "results": len(snippets),
        })

        # ── Ontology lookup ────────────────────────────────────────────
        ontology_info: dict[str, Any] = {}
        if issue.action_numbers:
            act_id = f"action_{issue.action_numbers[0]}"
            mapping = self._mapping_by_action.get(act_id, {})
            concepts = [m["concept_id"] for m in mapping.get("mappings", [])[:2]]
            if not concepts:
                # Fallback: search by objective
                for obj_code in issue.affected_objectives:
                    for gm in self._mappings.get("strategy_mappings", []):
                        if obj_code in gm.get("item_id", ""):
                            concepts.extend(
                                m["concept_id"] for m in gm.get("mappings", [])[:1]
                            )
                            break

            if concepts:
                ontology_info = self.ontology_lookup(concepts[0])
                tool_calls.append({
                    "tool": "ontology_lookup",
                    "concept": concepts[0],
                    "found": ontology_info.get("found", False),
                })

        return all_snippets, ontology_info, tool_calls

    # ══════════════════════════════════════════════════════════════════
    # Phase C: REASON (LLM-based recommendation generation)
    # ══════════════════════════════════════════════════════════════════

    def _generate_and_critique(
        self, issue: DiagnosedIssue, snippets: list[DocSnippet],
        ontology_info: dict[str, Any],
    ) -> tuple[Recommendation, CritiqueSummary]:
        """Generate a recommendation AND self-critique in a single LLM call.

        Args:
            issue:         The issue to address.
            snippets:      Retrieved evidence snippets.
            ontology_info: Ontology concept metadata.

        Returns:
            Tuple of (Recommendation, CritiqueSummary).
        """
        # Build context strings
        strategic_ctx = "\n".join(
            f"- [{s.id}] (score={s.score:.3f}): {s.text[:200]}"
            for s in snippets if s.source == STRATEGIC_COLLECTION
        ) or "No strategic context retrieved."

        action_ctx = "\n".join(
            f"- [{s.id}] (score={s.score:.3f}): {s.text[:200]}"
            for s in snippets if s.source == ACTION_COLLECTION
        ) or "No action context retrieved."

        ontology_ctx = "Not available."
        if ontology_info.get("found"):
            related = ", ".join(ontology_info.get("related_concepts", [])[:5])
            keywords = ", ".join(ontology_info.get("keywords", [])[:8])
            ontology_ctx = (
                f"Concept: {ontology_info.get('label', '?')} "
                f"(parent: {ontology_info.get('parent_concept', '?')})\n"
                f"Related concepts: {related}\n"
                f"Keywords: {keywords}"
            )

        prompt = DIAGNOSE_AND_CRITIQUE_PROMPT.format(
            issue_type=issue.issue_type,
            issue_description=issue.description,
            affected_objectives=", ".join(
                f"{c} ({OBJECTIVE_NAMES.get(c, '')})" for c in issue.affected_objectives
            ),
            strategic_context=strategic_ctx,
            action_context=action_ctx,
            ontology_context=ontology_ctx,
        )

        response = self._invoke_llm(prompt)

        # Split response into recommendation and critique parts
        parts = response.split("--- SELF-CRITIQUE ---")
        rec_text = parts[0]
        critique_text = parts[1] if len(parts) > 1 else ""

        rec = self._parse_recommendation(rec_text, issue, snippets)
        critique = self._parse_critique(critique_text) if critique_text.strip() else CritiqueSummary(
            feasibility="Not evaluated",
            alignment_check="Not evaluated",
            kpi_measurability="Not evaluated",
            risks="Not evaluated",
            missing_info="Not evaluated",
            accepted=True,
            revision_notes="Self-critique section missing; accepting by default.",
        )
        return rec, critique

    def _parse_recommendation(
        self, raw: str, issue: DiagnosedIssue, snippets: list[DocSnippet],
    ) -> Recommendation:
        """Parse LLM response into a structured Recommendation.

        Args:
            raw:      Raw LLM output text.
            issue:    The originating issue.
            snippets: Evidence snippets for citation.

        Returns:
            Populated :class:`Recommendation`.
        """
        def _extract(label: str) -> str:
            pattern = rf"{label}:\s*(.+?)(?=\n(?:ROOT_CAUSE|RECOMMENDATION_\d|NEW_KPIS|TIMELINE|BUDGET_ESTIMATE|EXPECTED_OUTCOME|CONFIDENCE):|\Z)"
            m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        def _extract_list(label: str) -> list[str]:
            block = _extract(label)
            items = []
            for line in block.splitlines():
                cleaned = re.sub(r"^\s*[-•*]\s*", "", line).strip()
                if cleaned and len(cleaned) > 3:
                    items.append(cleaned)
            return items

        root_cause = _extract("ROOT_CAUSE")
        recs = []
        for i in range(1, 5):
            r = _extract(f"RECOMMENDATION_{i}")
            if r:
                recs.append(r)
        kpis = _extract_list("NEW_KPIS")
        timeline = _extract("TIMELINE")
        budget_raw = _extract("BUDGET_ESTIMATE")
        outcome = _extract("EXPECTED_OUTCOME")
        confidence_raw = _extract("CONFIDENCE").upper()

        # Parse budget number
        budget_num = 0.0
        budget_match = re.search(r"(\d+(?:\.\d+)?)", budget_raw)
        if budget_match:
            budget_num = float(budget_match.group(1))

        # Confidence
        if "HIGH" in confidence_raw:
            confidence = "HIGH"
        elif "LOW" in confidence_raw:
            confidence = "LOW"
        else:
            confidence = "MEDIUM"

        evidence_ids = [s.id for s in snippets[:5]]

        return Recommendation(
            rec_id=f"rec_{issue.issue_id}",
            issue_id=issue.issue_id,
            issue_type=issue.issue_type,
            what_to_change=root_cause or issue.description,
            why=outcome or root_cause,
            actions=recs,
            kpis=kpis,
            estimated_budget=budget_num,
            estimated_timeline=timeline,
            impact_score=0.0,  # filled by calculate_impact
            affected_objectives=issue.affected_objectives,
            evidence_ids=evidence_ids,
            confidence=confidence,
        )

    # ══════════════════════════════════════════════════════════════════
    # Phase D: CRITIQUE
    # ══════════════════════════════════════════════════════════════════

    def _critique_recommendation(self, rec: Recommendation) -> CritiqueSummary:
        """Self-critique a recommendation via LLM.

        Args:
            rec: The recommendation to evaluate.

        Returns:
            :class:`CritiqueSummary` with evaluation results.
        """
        prompt = CRITIQUE_PROMPT.format(
            issue_description=rec.what_to_change,
            affected_objectives=", ".join(
                f"{c} ({OBJECTIVE_NAMES.get(c, '')})" for c in rec.affected_objectives
            ),
            recommendations="\n".join(f"- {a}" for a in rec.actions),
            kpis="\n".join(f"- {k}" for k in rec.kpis),
            budget=f"{rec.estimated_budget:.0f}",
            timeline=rec.estimated_timeline or "Not specified",
        )

        response = self._invoke_llm(prompt)
        return self._parse_critique(response)

    def _parse_critique(self, raw: str) -> CritiqueSummary:
        """Parse LLM critique response into a CritiqueSummary."""
        def _extract(label: str) -> str:
            pattern = rf"{label}:\s*(.+?)(?=\n(?:FEASIBILITY|ALIGNMENT_CHECK|KPI_MEASURABILITY|RISKS|MISSING_INFO|VERDICT|REVISION_NOTES):|\Z)"
            m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        verdict = _extract("VERDICT").upper()
        accepted = "REJECT" not in verdict

        return CritiqueSummary(
            feasibility=_extract("FEASIBILITY"),
            alignment_check=_extract("ALIGNMENT_CHECK"),
            kpi_measurability=_extract("KPI_MEASURABILITY"),
            risks=_extract("RISKS"),
            missing_info=_extract("MISSING_INFO"),
            accepted=accepted,
            revision_notes=_extract("REVISION_NOTES"),
        )

    # ══════════════════════════════════════════════════════════════════
    # Phase E: REFINE (apply critique, compute impact)
    # ══════════════════════════════════════════════════════════════════

    def _refine_recommendation(self, rec: Recommendation,
                               critique: CritiqueSummary) -> Recommendation:
        """Refine a recommendation based on critique feedback.

        If the critique suggests revisions, append them to the
        recommendation's actions. Then compute impact score via the
        ``calculate_impact`` tool.

        Args:
            rec:      The original recommendation.
            critique: Critique results.

        Returns:
            The refined :class:`Recommendation`.
        """
        # Apply revision notes if any
        if critique.revision_notes and len(critique.revision_notes) > 10:
            rec.actions.append(f"[Revised] {critique.revision_notes[:200]}")

        # If critique mentioned risk, downgrade confidence
        if critique.risks and "high" in critique.risks.lower():
            if rec.confidence == "HIGH":
                rec.confidence = "MEDIUM"

        # Compute impact
        impact = self.calculate_impact({
            "affected_objectives": rec.affected_objectives,
            "actions": rec.actions,
            "estimated_budget": rec.estimated_budget,
            "estimated_timeline": rec.estimated_timeline,
        })
        rec.impact_score = impact["impact_score"]

        # Validate alignment for primary objective
        if rec.affected_objectives:
            alignment = self.validate_alignment(
                " ".join(rec.actions[:3]),
                rec.affected_objectives[0],
            )
            rec.evidence_ids.extend(alignment.get("evidence_ids", []))
            # Deduplicate evidence IDs
            rec.evidence_ids = list(dict.fromkeys(rec.evidence_ids))

        return rec

    # ══════════════════════════════════════════════════════════════════
    # Main agent loop
    # ══════════════════════════════════════════════════════════════════

    def run(self) -> list[Recommendation]:
        """Execute the full agent reasoning pipeline.

        Steps per iteration:
            1. Diagnose (first iteration only, or refresh if needed).
            2. Pick the next highest-priority unaddressed issue.
            3. Investigate — gather evidence.
            4. Reason — generate recommendation via LLM.
            5. Critique — self-evaluate.
            6. Refine — apply critique and compute impact.
            7. Check early-stop conditions.

        Returns:
            List of finalised :class:`Recommendation` objects.
        """
        logger.info("=" * 60)
        logger.info("AGENT REASONER — Starting pipeline")
        logger.info("  Max iterations:  %d", self.max_iterations)
        logger.info("  Issue threshold:  %.3f", ISSUE_SCORE_THRESHOLD)
        logger.info("=" * 60)

        # Phase A: Diagnose all issues
        self.diagnose()

        if not self._issues:
            logger.info("No issues diagnosed. Pipeline complete.")
            return []

        prev_total_impact = 0.0
        addressed_ids: set[str] = set()

        for iteration in range(1, self.max_iterations + 1):
            logger.info("─── Iteration %d/%d ───", iteration, self.max_iterations)
            self._tool_call_count = 0

            # Pick next unaddressed issue above threshold
            issue = None
            for candidate in self._issues:
                if (candidate.issue_id not in addressed_ids and
                        candidate.priority_score >= ISSUE_SCORE_THRESHOLD):
                    issue = candidate
                    break

            if issue is None:
                logger.info("No more issues above threshold. Stopping early.")
                break

            logger.info("Addressing: %s (score=%.4f, type=%s)",
                        issue.issue_id, issue.priority_score, issue.issue_type)
            addressed_ids.add(issue.issue_id)

            trace = TraceEntry(iteration=iteration)
            trace.issue_detected = {
                "issue_id": issue.issue_id,
                "type": issue.issue_type,
                "severity": issue.severity,
                "confidence": issue.confidence,
                "business_impact": issue.business_impact,
                "priority_score": issue.priority_score,
                "impacted_objectives": issue.affected_objectives,
            }

            # Phase B: Build investigation plan
            trace.investigation_plan = [
                f"Search vector DB for context related to {issue.issue_type}",
                f"Look up ontology concepts for affected actions: {issue.action_numbers}",
                f"Validate alignment of any proposed changes against objectives: {issue.affected_objectives}",
            ]

            # Phase B: Gather evidence
            snippets, ontology_info, tool_calls = self._gather_evidence(issue)
            trace.evidence_used = [s.id for s in snippets]
            trace.tool_calls.extend(tool_calls)

            # Guardrail: check tool call limit
            if self._tool_call_count >= MAX_TOOL_CALLS_PER_ITERATION:
                logger.warning("Tool call limit reached (%d). Proceeding with available evidence.",
                               MAX_TOOL_CALLS_PER_ITERATION)

            # Phase C+D: Reason + Critique in a single LLM call
            rec, critique = self._generate_and_critique(issue, snippets, ontology_info)
            trace.tool_calls.append({"tool": "llm_invoke", "purpose": "generate_and_critique"})

            trace.reasoning_summary = [
                f"Identified {issue.issue_type}: {issue.description[:100]}",
                f"Retrieved {len(snippets)} evidence snippets from vector DB",
                f"Ontology lookup: {'found' if ontology_info.get('found') else 'not found'}",
                f"Generated {len(rec.actions)} recommended changes",
                f"Proposed {len(rec.kpis)} new KPIs",
                f"Initial confidence: {rec.confidence}",
            ]
            trace.critique_summary = asdict(critique)

            # Phase E: Refine
            if critique.accepted:
                rec = self._refine_recommendation(rec, critique)
                trace.final_decision = {
                    "status": "ACCEPTED",
                    "impact_score": rec.impact_score,
                    "confidence": rec.confidence,
                    "changes_count": len(rec.actions),
                }
                self._recommendations.append(rec)
                logger.info("Recommendation ACCEPTED: %s (impact=%.3f)",
                            rec.rec_id, rec.impact_score)
            else:
                trace.final_decision = {
                    "status": "REJECTED",
                    "reason": critique.revision_notes or "Failed critique evaluation.",
                }
                logger.info("Recommendation REJECTED: %s", rec.rec_id)

            trace.recommendation = asdict(rec)
            self._traces.append(trace)

            # ── Early-stop: marginal improvement check ─────────────────
            current_total_impact = sum(r.impact_score for r in self._recommendations)
            improvement = current_total_impact - prev_total_impact
            prev_total_impact = current_total_impact

            if iteration > 1 and improvement < MARGINAL_IMPROVEMENT_THRESHOLD:
                logger.info(
                    "Marginal improvement (%.4f) below threshold (%.4f). Stopping.",
                    improvement, MARGINAL_IMPROVEMENT_THRESHOLD,
                )
                break

        logger.info("=" * 60)
        logger.info("AGENT REASONER — Pipeline complete")
        logger.info("  Issues diagnosed:     %d", len(self._issues))
        logger.info("  Issues addressed:     %d", len(addressed_ids))
        logger.info("  Recommendations made: %d", len(self._recommendations))
        logger.info("  Total impact score:   %.3f",
                     sum(r.impact_score for r in self._recommendations))
        logger.info("=" * 60)

        return self._recommendations

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def save_recommendations(self, filepath: Path = RECOMMENDATIONS_OUT) -> Path:
        """Export recommendations to JSON.

        Args:
            filepath: Output file path.

        Returns:
            Path written to.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "agent_model": OPENAI_MODEL,
                "max_iterations": self.max_iterations,
                "issue_threshold": ISSUE_SCORE_THRESHOLD,
                "total_issues_diagnosed": len(self._issues),
                "total_recommendations": len(self._recommendations),
                "total_impact_score": round(
                    sum(r.impact_score for r in self._recommendations), 4
                ),
            },
            "recommendations": [asdict(r) for r in self._recommendations],
            "diagnosed_issues_summary": [
                {
                    "issue_id": i.issue_id,
                    "type": i.issue_type,
                    "priority_score": i.priority_score,
                    "affected_objectives": i.affected_objectives,
                    "addressed": i.issue_id in {r.issue_id for r in self._recommendations},
                }
                for i in self._issues[:15]
            ],
        }

        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Recommendations saved to %s.", filepath)
        return filepath

    def save_trace(self, filepath: Path = TRACE_OUT) -> Path:
        """Export the structured reasoning trace to JSON.

        Args:
            filepath: Output file path.

        Returns:
            Path written to.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "total_iterations": len(self._traces),
                "agent_model": OPENAI_MODEL,
            },
            "traces": [asdict(t) for t in self._traces],
        }

        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Reasoning trace saved to %s.", filepath)
        return filepath

    def save_all(self) -> dict[str, str]:
        """Export both recommendations and trace.

        Returns:
            Dict mapping output name to file path.
        """
        return {
            "recommendations": str(self.save_recommendations()),
            "trace": str(self.save_trace()),
        }


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    agent = AgentReasoner()
    recommendations = agent.run()
    exports = agent.save_all()

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGENT REASONER — RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nIssues diagnosed:     {len(agent._issues)}")
    print(f"Recommendations made: {len(recommendations)}")
    total_impact = sum(r.impact_score for r in recommendations)
    print(f"Total impact score:   {total_impact:.3f}")

    # ── Diagnosed issues (top 10) ──────────────────────────────────
    print("\n--- TOP DIAGNOSED ISSUES ---")
    for i, issue in enumerate(agent._issues[:10], 1):
        print(f"  {i}. {issue.issue_id:<30} "
              f"type={issue.issue_type:<25} "
              f"score={issue.priority_score:.4f}")

    # ── Recommendations ────────────────────────────────────────────
    print("\n--- RECOMMENDATIONS ---")
    for rec in recommendations:
        print(f"\n  [{rec.rec_id}]  ({rec.confidence})")
        print(f"    Issue:    {rec.issue_type}")
        print(f"    Obj:      {', '.join(rec.affected_objectives)}")
        print(f"    Impact:   {rec.impact_score:.3f}")
        print(f"    Budget:   LKR {rec.estimated_budget:.0f}M")
        print(f"    Timeline: {rec.estimated_timeline}")
        print(f"    Actions:")
        for a in rec.actions[:4]:
            print(f"      - {a[:100]}")
        print(f"    KPIs:")
        for k in rec.kpis[:3]:
            print(f"      - {k[:80]}")
        print(f"    Evidence: {rec.evidence_ids[:5]}")

    # ── Trace summary ──────────────────────────────────────────────
    print("\n--- REASONING TRACE ---")
    for trace in agent._traces:
        status = trace.final_decision.get("status", "?")
        print(f"  Iteration {trace.iteration}: "
              f"{trace.issue_detected.get('issue_id', '?'):<30} "
              f"→ {status}")

    # ── Export paths ───────────────────────────────────────────────
    print("\n--- EXPORTS ---")
    for name, path in exports.items():
        print(f"  {name}: {path}")
    print(f"  log: {LOG_DIR / 'agent_reasoner.log'}")
