"""
RAG Engine for Hospital Strategy–Action Plan Alignment System (ISPS).

This module implements a Retrieval-Augmented Generation (RAG) pipeline
that takes the alignment analysis results (from
:mod:`synchronization_analyzer`) and uses an LLM (OpenAI GPT-4o-mini)
to generate:

1. **Improvement recommendations** for poorly-aligned actions — concrete
   suggestions to strengthen the link between an operational action and
   its intended strategic objective.
2. **New action suggestions** for strategic objectives with low coverage
   — filling gaps where the current action plan leaves a strategic goal
   under-addressed.

Architecture
------------
::

    ┌────────────────┐     ┌──────────────┐     ┌──────────────────┐
    │ Alignment      │────▶│ ChromaDB     │────▶│ Prompt           │
    │ Report         │     │ Retrieval    │     │ Construction     │
    └────────────────┘     └──────────────┘     └───────┬──────────┘
                                                        │
                                                ┌───────▼──────────┐
                                                │ OpenAI LLM       │
                                                │ (GPT-4o-mini)    │
                                                └───────┬──────────┘
                                                        │
                                                ┌───────▼──────────┐
                                                │ Response Parser  │
                                                │                  │
                                                └──────────────────┘

Retry
-----
LLM calls are wrapped with exponential-backoff retry (3 attempts,
1s / 2s / 4s delays).

Typical usage::

    from src.rag_engine import RAGEngine

    engine = RAGEngine()
    results = engine.run()
    engine.save_results()

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

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import get_llm, OPENAI_MODEL
from src.vector_store import VectorStore, STRATEGIC_COLLECTION
from src.synchronization_analyzer import (
    SynchronizationAnalyzer,
    SynchronizationReport,
    THRESHOLD_FAIR,
    OBJECTIVE_NAMES,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_JSON = DATA_DIR / "rag_recommendations.json"

# LLM configuration (provider set in src/config.py via .env)
LLM_TEMPERATURE = 0.3          # Low temperature for deterministic, focused output

# Retry configuration
from src.config import MAX_RETRIES, RETRY_DELAYS

# Thresholds
POOR_ALIGNMENT_THRESHOLD = 0.50   # Actions below this get improvement recs
LOW_COVERAGE_THRESHOLD = 0.25     # Objectives below this coverage get new suggestions

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

IMPROVEMENT_PROMPT = PromptTemplate.from_template(
    """You are a hospital strategic planning consultant analyzing the alignment between
a hospital's strategic plan and its operational action plan.

STRATEGIC OBJECTIVE ({objective_code}): {objective_name}
Goal Statement: {goal_statement}

Key Strategic Goals:
{strategic_goals}

Key KPIs:
{kpis}

CURRENT ACTION (Action {action_number}): {action_title}
Description: {action_description}
Action Owner: {action_owner}
Timeline: {action_timeline}
Budget: {action_budget}
Current KPIs: {action_kpis}
Expected Outcome: {expected_outcome}

ALIGNMENT ANALYSIS:
- Current alignment score: {alignment_score:.2f} (classified as {classification})
- Best-matching objective: {best_objective} (score: {best_score:.2f})
- Declared objective: {declared_objective}

RELEVANT STRATEGIC CONTEXT (retrieved from vector database):
{retrieved_context}

TASK: Provide specific, actionable recommendations to improve this action's alignment
with the strategic plan. Structure your response EXACTLY as follows:

MODIFIED_DESCRIPTION: [Write a revised action description that explicitly connects to the strategic objective. Keep it concise — 2-3 sentences.]

ADDITIONAL_KPIS:
- [KPI 1 with measurable target]
- [KPI 2 with measurable target]
- [KPI 3 with measurable target]

TIMELINE_ADJUSTMENTS: [Specific timeline changes if needed, or "No change required" if current timeline is appropriate.]

RESOURCE_REALLOCATION: [Specific budget or staffing changes if needed, or "No change required" if current allocation is appropriate.]

STRATEGIC_LINKAGE: [One sentence explaining how the improved action directly supports the strategic objective.]

CONFIDENCE: [HIGH, MEDIUM, or LOW — how confident you are that these changes would meaningfully improve alignment.]"""
)

GAP_ACTION_PROMPT = PromptTemplate.from_template(
    """You are a hospital strategic planning consultant. A strategic objective has low
action coverage — meaning the current action plan does not sufficiently address it.

STRATEGIC OBJECTIVE ({objective_code}): {objective_name}
Goal Statement: {goal_statement}

Key Strategic Goals:
{strategic_goals}

Key KPIs:
{kpis}

Timeline Milestones:
{timeline}

CURRENT ACTIONS UNDER THIS OBJECTIVE:
{current_actions}

COVERAGE ANALYSIS:
- Coverage score: {coverage_score:.1%}
- Actions aligned (similarity >= 0.45): {aligned_count}
- Gap actions (declared but poorly aligned): {gap_actions}

TASK: Suggest {num_suggestions} NEW concrete actions that would strengthen this objective's
coverage in the action plan. For each action, structure your response EXACTLY as follows:

NEW_ACTION_1:
TITLE: [Concise action title]
DESCRIPTION: [2-3 sentence operational description]
OWNER: [Responsible stakeholder/department]
TIMELINE: [Quarter-based timeline for 2025]
BUDGET: [Estimated budget in LKR millions]
KPIS:
- [Measurable KPI 1]
- [Measurable KPI 2]
STRATEGIC_LINKAGE: [How this action directly supports the objective]

NEW_ACTION_2:
TITLE: [Concise action title]
DESCRIPTION: [2-3 sentence operational description]
OWNER: [Responsible stakeholder/department]
TIMELINE: [Quarter-based timeline for 2025]
BUDGET: [Estimated budget in LKR millions]
KPIS:
- [Measurable KPI 1]
- [Measurable KPI 2]
STRATEGIC_LINKAGE: [How this action directly supports the objective]

{extra_action_block}"""
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActionImprovement:
    """LLM-generated improvement recommendation for a single action.

    Attributes:
        action_number:        The action being improved.
        action_title:         Original action title.
        declared_objective:   Objective code declared in the plan.
        alignment_score:      Original cosine similarity to best objective.
        modified_description: LLM-suggested revised description.
        additional_kpis:      List of new KPIs to add.
        timeline_adjustments: Suggested timeline changes.
        resource_reallocation: Suggested budget/staffing changes.
        strategic_linkage:    Explanation of strategic connection.
        confidence:           LLM self-assessed confidence (HIGH/MEDIUM/LOW).
        confidence_score:     Numeric confidence (1.0/0.7/0.4).
    """
    action_number: int
    action_title: str
    declared_objective: str
    alignment_score: float
    modified_description: str = ""
    additional_kpis: list[str] = field(default_factory=list)
    timeline_adjustments: str = ""
    resource_reallocation: str = ""
    strategic_linkage: str = ""
    confidence: str = ""
    confidence_score: float = 0.0


@dataclass
class NewActionSuggestion:
    """LLM-generated new action to fill a strategic coverage gap.

    Attributes:
        objective_code:   The objective this action supports.
        suggestion_index: 1-based index among suggestions for this objective.
        title:            Suggested action title.
        description:      Operational description.
        owner:            Responsible stakeholder.
        timeline:         Quarter-based timeline.
        budget_estimate:  Estimated budget string.
        kpis:             List of suggested KPIs.
        strategic_linkage: How this action supports the objective.
    """
    objective_code: str
    suggestion_index: int
    title: str = ""
    description: str = ""
    owner: str = ""
    timeline: str = ""
    budget_estimate: str = ""
    kpis: list[str] = field(default_factory=list)
    strategic_linkage: str = ""


@dataclass
class RAGResults:
    """Complete RAG engine output.

    Attributes:
        improvements:         List of action improvement recommendations.
        new_action_suggestions: List of new action suggestions per objective.
        summary:              Aggregate statistics.
        model_used:           LLM model identifier.
    """
    improvements: list[dict] = field(default_factory=list)
    new_action_suggestions: list[dict] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    model_used: str = OPENAI_MODEL


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_improvement_response(raw: str) -> dict[str, Any]:
    """Parse a structured improvement response from the LLM.

    Extracts fields delimited by known labels (MODIFIED_DESCRIPTION,
    ADDITIONAL_KPIS, etc.) using regex.  Gracefully handles missing or
    malformed sections.

    Args:
        raw: The raw LLM output string.

    Returns:
        A dict with keys matching :class:`ActionImprovement` fields.
    """
    def _extract(label: str, text: str) -> str:
        """Extract text following a label until the next known label or end."""
        pattern = rf"{label}:\s*(.+?)(?=\n(?:MODIFIED_DESCRIPTION|ADDITIONAL_KPIS|TIMELINE_ADJUSTMENTS|RESOURCE_REALLOCATION|STRATEGIC_LINKAGE|CONFIDENCE):|\Z)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _extract_list(label: str, text: str) -> list[str]:
        """Extract a bullet-list section."""
        block = _extract(label, text)
        items = []
        for line in block.splitlines():
            cleaned = re.sub(r"^\s*[-•*]\s*", "", line).strip()
            if cleaned and len(cleaned) > 3:
                items.append(cleaned)
        return items

    confidence_raw = _extract("CONFIDENCE", raw).upper().strip()
    confidence_map = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}
    # Find the first match in the confidence string
    confidence_label = "MEDIUM"
    confidence_score = 0.7
    for label, score in confidence_map.items():
        if label in confidence_raw:
            confidence_label = label
            confidence_score = score
            break

    return {
        "modified_description": _extract("MODIFIED_DESCRIPTION", raw),
        "additional_kpis": _extract_list("ADDITIONAL_KPIS", raw),
        "timeline_adjustments": _extract("TIMELINE_ADJUSTMENTS", raw),
        "resource_reallocation": _extract("RESOURCE_REALLOCATION", raw),
        "strategic_linkage": _extract("STRATEGIC_LINKAGE", raw),
        "confidence": confidence_label,
        "confidence_score": confidence_score,
    }


def _parse_gap_response(raw: str, objective_code: str) -> list[dict[str, Any]]:
    """Parse new-action suggestions from the LLM.

    Splits the response on ``NEW_ACTION_N:`` delimiters and extracts
    structured fields from each block.

    Args:
        raw:             The raw LLM output string.
        objective_code:  The objective code these suggestions belong to.

    Returns:
        A list of dicts, one per suggested action.
    """
    # Split on NEW_ACTION_N: markers
    blocks = re.split(r"NEW_ACTION_\d+:", raw, flags=re.IGNORECASE)
    suggestions: list[dict[str, Any]] = []

    for idx, block in enumerate(blocks[1:], start=1):  # skip preamble
        def _field(label: str) -> str:
            pattern = rf"{label}:\s*(.+?)(?=\n(?:TITLE|DESCRIPTION|OWNER|TIMELINE|BUDGET|KPIS|STRATEGIC_LINKAGE|NEW_ACTION):|\Z)"
            m = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        kpis_block = _field("KPIS")
        kpis = [
            re.sub(r"^\s*[-•*]\s*", "", ln).strip()
            for ln in kpis_block.splitlines()
            if ln.strip() and len(ln.strip()) > 3
        ]

        suggestions.append({
            "objective_code": objective_code,
            "suggestion_index": idx,
            "title": _field("TITLE"),
            "description": _field("DESCRIPTION"),
            "owner": _field("OWNER"),
            "timeline": _field("TIMELINE"),
            "budget_estimate": _field("BUDGET"),
            "kpis": kpis,
            "strategic_linkage": _field("STRATEGIC_LINKAGE"),
        })

    return suggestions


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """Retrieval-Augmented Generation engine for alignment recommendations.

    This class orchestrates the full RAG pipeline:

    1. Loads the alignment report to identify poorly-aligned actions
       and under-covered objectives.
    2. Retrieves relevant strategic context from ChromaDB for each
       target action/objective.
    3. Constructs structured prompts and queries the local LLM.
    4. Parses LLM responses into typed data structures.

    Attributes:
        vs:         The :class:`VectorStore` instance for retrieval.
        llm:        The LangChain OpenAI LLM wrapper.
        results:    The most recent :class:`RAGResults` (after :meth:`run`).

    Example::

        engine = RAGEngine()
        results = engine.run()
        print(f"Generated {len(results.improvements)} improvements")
        engine.save_results()
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        model_name: str = OPENAI_MODEL,
    ) -> None:
        """Initialise the RAG engine with LLM and vector store.

        Args:
            vector_store: An existing :class:`VectorStore`. If ``None``,
                          a new one is created with default paths.
            model_name:   OpenAI model identifier (default
                          ``"gpt-4o-mini"``).
        """
        self.vs = vector_store or VectorStore()

        # ── Initialise LLM via config factory ─────────────────────────
        logger.info("Initialising LLM ...")
        self.llm = get_llm(temperature=LLM_TEMPERATURE)
        self.parser = StrOutputParser()
        self.model_name = model_name

        # ── Load data ────────────────────────────────────────────────
        self._strategic_data = self._load_json(DATA_DIR / "strategic_plan.json")
        self._action_data = self._load_json(DATA_DIR / "action_plan.json")
        self._alignment_report = self._load_json(DATA_DIR / "alignment_report.json")

        self.results: RAGResults | None = None
        logger.info("RAGEngine initialised.")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load a JSON file, returning empty dict on failure."""
        if not path.exists():
            logger.warning("JSON not found: %s", path)
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # LLM invocation with retry
    # ------------------------------------------------------------------

    def _invoke_llm(self, prompt_text: str) -> str:
        """Call the LLM with retry logic.

        Invokes the LLM with exponential-backoff retry (up to 3
        attempts).

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
                logger.info(
                    "LLM call attempt %d/%d …",
                    attempt + 1,
                    MAX_RETRIES,
                )
                response = self.llm.invoke(prompt_text)
                return response

            except Exception as exc:
                last_error = exc
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs …",
                    attempt + 1,
                    MAX_RETRIES,
                    str(exc)[:120],
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def _retrieve_strategic_context(
        self, action_text: str, top_k: int = 3
    ) -> str:
        """Retrieve the most relevant strategic-objective text for an action.

        Uses the vector store's semantic search to find the top-k
        strategic objectives most similar to the action's composite text.

        Args:
            action_text: The action's title + description composite.
            top_k:       Number of objectives to retrieve.

        Returns:
            A formatted string of retrieved objective summaries.
        """
        results = self.vs.search_similar(
            query=action_text,
            collection_name=STRATEGIC_COLLECTION,
            top_k=top_k,
        )

        context_parts: list[str] = []
        for doc_id, score, doc, meta in zip(
            results["ids"],
            results["scores"],
            results["documents"],
            results["metadatas"],
        ):
            code = meta.get("code", "?")
            name = meta.get("name", "")
            # Truncate document to keep prompt within context window
            snippet = doc[:600]
            context_parts.append(
                f"[Objective {code}: {name} — similarity: {score:.3f}]\n{snippet}"
            )

        return "\n\n".join(context_parts) if context_parts else "No relevant context found."

    # ------------------------------------------------------------------
    # Objective data helpers
    # ------------------------------------------------------------------

    def _get_objective_data(self, code: str) -> dict[str, Any]:
        """Look up a strategic objective by code from the parsed JSON.

        Args:
            code: Objective letter (A–E).

        Returns:
            The objective dict, or a minimal fallback if not found.
        """
        for obj in self._strategic_data.get("objectives", []):
            if obj.get("code") == code:
                return obj
        return {"code": code, "name": OBJECTIVE_NAMES.get(code, code)}

    def _get_action_data(self, action_number: int) -> dict[str, Any]:
        """Look up an action item by number from the parsed JSON.

        Args:
            action_number: The 1-based action number.

        Returns:
            The action dict, or a minimal fallback if not found.
        """
        for act in self._action_data.get("actions", []):
            if act.get("action_number") == action_number:
                return act
        return {"action_number": action_number, "title": f"Action {action_number}"}

    def _format_goals(self, obj: dict) -> str:
        """Format strategic goals as a numbered list."""
        goals = obj.get("strategic_goals", [])
        if not goals:
            return "Not available."
        return "\n".join(
            f"  {g['id']}: {g['description']}" for g in goals
        )

    def _format_kpis(self, obj: dict) -> str:
        """Format KPIs as a bulleted list."""
        kpis = obj.get("kpis", [])
        if not kpis:
            return "Not available."
        return "\n".join(
            f"  - {k.get('KPI', k) if isinstance(k, dict) else k}"
            for k in kpis
        )

    def _format_timeline(self, obj: dict) -> str:
        """Format timeline milestones."""
        timeline = obj.get("timeline", [])
        if not timeline:
            return "Not available."
        return "\n".join(
            f"  {t.get('Phase', 'Phase')}: {t.get('Milestones', '')[:150]}"
            for t in timeline
        )

    # ------------------------------------------------------------------
    # Improvement recommendations
    # ------------------------------------------------------------------

    def generate_improvements(self) -> list[ActionImprovement]:
        """Generate improvement recommendations for poorly-aligned actions.

        Iterates over all actions whose best alignment score is below
        :data:`POOR_ALIGNMENT_THRESHOLD`, retrieves relevant strategic
        context from the vector store, constructs a detailed prompt, and
        queries the LLM for structured recommendations.

        Returns:
            A list of :class:`ActionImprovement` objects.
        """
        report = self._alignment_report
        if not report:
            raise RuntimeError("No alignment report found. Run SynchronizationAnalyzer first.")

        improvements: list[ActionImprovement] = []
        action_alignments = report.get("action_alignments", [])

        # Filter to poorly-aligned actions
        poor_actions = [
            aa for aa in action_alignments
            if aa["best_score"] < POOR_ALIGNMENT_THRESHOLD
        ]

        logger.info(
            "Generating improvements for %d poorly-aligned actions "
            "(threshold < %.2f).",
            len(poor_actions),
            POOR_ALIGNMENT_THRESHOLD,
        )

        for i, aa in enumerate(poor_actions, 1):
            act_num = aa["action_number"]
            act_data = self._get_action_data(act_num)
            declared_code = aa["declared_objective"]
            obj_data = self._get_objective_data(declared_code)

            logger.info(
                "[%d/%d] Action %d: %s (score=%.4f)",
                i, len(poor_actions), act_num, aa["title"][:40], aa["best_score"],
            )

            # ── Retrieve strategic context ───────────────────────────
            action_text = f"{act_data.get('title', '')}. {act_data.get('description', '')}"
            retrieved_context = self._retrieve_strategic_context(action_text)

            # ── Build prompt ─────────────────────────────────────────
            prompt_text = IMPROVEMENT_PROMPT.format(
                objective_code=declared_code,
                objective_name=OBJECTIVE_NAMES.get(declared_code, declared_code),
                goal_statement=obj_data.get("goal_statement", "Not available."),
                strategic_goals=self._format_goals(obj_data),
                kpis=self._format_kpis(obj_data),
                action_number=act_num,
                action_title=aa["title"],
                action_description=act_data.get("description", "Not available."),
                action_owner=aa.get("action_owner", "Not specified"),
                action_timeline=act_data.get("timeline", "Not specified"),
                action_budget=act_data.get("budget_raw", "Not specified"),
                action_kpis="; ".join(act_data.get("kpis", [])) or "None specified",
                expected_outcome=act_data.get("expected_outcome", "Not specified"),
                alignment_score=aa["best_score"],
                classification=aa["classification"],
                best_objective=aa["best_objective"],
                best_score=aa["best_score"],
                declared_objective=declared_code,
                retrieved_context=retrieved_context,
            )

            # ── Invoke LLM ──────────────────────────────────────────
            raw_response = self._invoke_llm(prompt_text)
            parsed = _parse_improvement_response(raw_response)

            improvement = ActionImprovement(
                action_number=act_num,
                action_title=aa["title"],
                declared_objective=declared_code,
                alignment_score=aa["best_score"],
                **parsed,
            )
            improvements.append(improvement)
            logger.info(
                "  → Confidence: %s | KPIs suggested: %d",
                improvement.confidence,
                len(improvement.additional_kpis),
            )

        logger.info("Generated %d improvement recommendations.", len(improvements))
        return improvements

    # ------------------------------------------------------------------
    # New action suggestions for coverage gaps
    # ------------------------------------------------------------------

    def generate_gap_actions(self) -> list[NewActionSuggestion]:
        """Generate new action suggestions for under-covered objectives.

        Identifies strategic objectives whose coverage score is below
        :data:`LOW_COVERAGE_THRESHOLD` and generates 2–3 new action
        suggestions for each using the LLM.

        Returns:
            A list of :class:`NewActionSuggestion` objects.
        """
        report = self._alignment_report
        if not report:
            raise RuntimeError("No alignment report found.")

        all_suggestions: list[NewActionSuggestion] = []
        obj_alignments = report.get("objective_alignments", [])

        # Filter to under-covered objectives
        low_coverage = [
            oa for oa in obj_alignments
            if oa["coverage_score"] < LOW_COVERAGE_THRESHOLD
        ]

        logger.info(
            "Generating gap-filling actions for %d under-covered objectives "
            "(threshold < %.0f%%).",
            len(low_coverage),
            LOW_COVERAGE_THRESHOLD * 100,
        )

        for oa in low_coverage:
            code = oa["code"]
            obj_data = self._get_objective_data(code)

            # Determine how many suggestions (2 for moderate gaps, 3 for severe)
            num_suggestions = 3 if oa["coverage_score"] < 0.15 else 2

            # Format current actions under this objective
            current_action_texts: list[str] = []
            for aa in report.get("action_alignments", []):
                if aa["declared_objective"] == code:
                    current_action_texts.append(
                        f"  Action {aa['action_number']}: {aa['title']} "
                        f"(alignment: {aa['best_score']:.3f})"
                    )
            current_actions_str = "\n".join(current_action_texts) or "  None declared."

            # Extra action block for 3-suggestion case
            extra_block = ""
            if num_suggestions >= 3:
                extra_block = (
                    "NEW_ACTION_3:\n"
                    "TITLE: [Concise action title]\n"
                    "DESCRIPTION: [2-3 sentence operational description]\n"
                    "OWNER: [Responsible stakeholder/department]\n"
                    "TIMELINE: [Quarter-based timeline for 2025]\n"
                    "BUDGET: [Estimated budget in LKR millions]\n"
                    "KPIS:\n"
                    "- [Measurable KPI 1]\n"
                    "- [Measurable KPI 2]\n"
                    "STRATEGIC_LINKAGE: [How this action directly supports the objective]"
                )

            logger.info(
                "Objective %s (%s): coverage=%.1f%%, generating %d suggestions.",
                code, oa["name"], oa["coverage_score"] * 100, num_suggestions,
            )

            prompt_text = GAP_ACTION_PROMPT.format(
                objective_code=code,
                objective_name=OBJECTIVE_NAMES.get(code, code),
                goal_statement=obj_data.get("goal_statement", "Not available."),
                strategic_goals=self._format_goals(obj_data),
                kpis=self._format_kpis(obj_data),
                timeline=self._format_timeline(obj_data),
                current_actions=current_actions_str,
                coverage_score=oa["coverage_score"],
                aligned_count=oa["aligned_action_count"],
                gap_actions=oa.get("gap_actions", []),
                num_suggestions=num_suggestions,
                extra_action_block=extra_block,
            )

            # ── Invoke LLM ──────────────────────────────────────────
            raw_response = self._invoke_llm(prompt_text)
            suggestions = _parse_gap_response(raw_response, code)

            for s in suggestions:
                all_suggestions.append(NewActionSuggestion(**s))

            logger.info(
                "  → Generated %d new action suggestions for Obj %s.",
                len(suggestions), code,
            )

        logger.info(
            "Generated %d total gap-filling suggestions.",
            len(all_suggestions),
        )
        return all_suggestions

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> RAGResults:
        """Execute the full RAG pipeline.

        Generates both improvement recommendations and gap-filling
        action suggestions, then packages everything into a
        :class:`RAGResults` object.

        Returns:
            The complete :class:`RAGResults`.
        """
        logger.info("=" * 60)
        logger.info("RAG Engine — Starting full pipeline")
        logger.info("=" * 60)

        improvements = self.generate_improvements()
        gap_actions = self.generate_gap_actions()

        # ── Summary statistics ───────────────────────────────────────
        confidence_scores = [
            imp.confidence_score for imp in improvements
        ]
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        summary = {
            "total_improvements": len(improvements),
            "total_gap_suggestions": len(gap_actions),
            "avg_confidence_score": round(avg_confidence, 2),
            "confidence_distribution": {
                "HIGH": sum(1 for c in confidence_scores if c >= 0.9),
                "MEDIUM": sum(1 for c in confidence_scores if 0.5 <= c < 0.9),
                "LOW": sum(1 for c in confidence_scores if c < 0.5),
            },
            "objectives_with_gaps": list(set(
                s.objective_code for s in gap_actions
            )),
            "model_used": self.model_name,
            "poor_alignment_threshold": POOR_ALIGNMENT_THRESHOLD,
            "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
        }

        self.results = RAGResults(
            improvements=[asdict(imp) for imp in improvements],
            new_action_suggestions=[asdict(s) for s in gap_actions],
            summary=summary,
            model_used=self.model_name,
        )

        logger.info("RAG pipeline complete: %s", summary)
        return self.results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, output_path: Path | str = RESULTS_JSON) -> Path:
        """Save RAG results to a JSON file.

        Args:
            output_path: Destination file path.

        Returns:
            The resolved output path.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        if self.results is None:
            raise RuntimeError("No results to save. Call run() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self.results), fh, indent=2, ensure_ascii=False)

        size_kb = output_path.stat().st_size / 1024
        logger.info("Results saved: %s (%.1f KB)", output_path, size_kb)
        return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the RAG engine and print a formatted summary.

    This entry point ensures prerequisite data exists, runs the full
    RAG pipeline, saves results, and prints a human-readable report.
    """
    logger.info("=" * 60)
    logger.info("ISPS RAG Engine — Starting")
    logger.info("=" * 60)

    # ── Ensure prerequisites ─────────────────────────────────────────
    alignment_path = DATA_DIR / "alignment_report.json"
    if not alignment_path.exists():
        logger.info("Alignment report not found — running analyzer first …")
        vs = VectorStore()
        stats = vs.collection_stats()
        if stats["strategic_objectives"]["count"] == 0:
            vs.build_from_json()
        analyzer = SynchronizationAnalyzer(vector_store=vs)
        analyzer.analyze()
        analyzer.save_report()
        engine = RAGEngine(vector_store=vs)
    else:
        engine = RAGEngine()

    # ── Run pipeline ─────────────────────────────────────────────────
    results = engine.run()
    engine.save_results()

    # ── Print report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RAG RECOMMENDATIONS REPORT")
    print("  AI-Generated Improvement Suggestions")
    print("=" * 70)

    print(f"\n{'SUMMARY':─^70}")
    s = results.summary
    print(f"  Model                : {s['model_used']}")
    print(f"  Improvements generated: {s['total_improvements']}")
    print(f"  Gap suggestions      : {s['total_gap_suggestions']}")
    print(f"  Avg confidence       : {s['avg_confidence_score']:.2f}")
    print(f"  Confidence dist.     : {s['confidence_distribution']}")

    print(f"\n{'ACTION IMPROVEMENTS':─^70}")
    for imp in results.improvements:
        print(f"\n  Action {imp['action_number']}: {imp['action_title'][:50]}")
        print(f"  Original alignment: {imp['alignment_score']:.4f} | "
              f"Confidence: {imp['confidence']} ({imp['confidence_score']:.1f})")
        if imp["modified_description"]:
            desc = imp["modified_description"][:120]
            print(f"  Revised description: {desc}…")
        if imp["additional_kpis"]:
            print(f"  New KPIs:")
            for kpi in imp["additional_kpis"][:3]:
                print(f"    + {kpi[:80]}")
        if imp["strategic_linkage"]:
            print(f"  Linkage: {imp['strategic_linkage'][:100]}")

    print(f"\n{'NEW ACTION SUGGESTIONS':─^70}")
    current_obj = ""
    for sug in results.new_action_suggestions:
        if sug["objective_code"] != current_obj:
            current_obj = sug["objective_code"]
            print(f"\n  Objective {current_obj}: "
                  f"{OBJECTIVE_NAMES.get(current_obj, current_obj)}")
        print(f"\n    Suggestion {sug['suggestion_index']}:")
        print(f"      Title    : {sug['title'][:60]}")
        if sug["description"]:
            print(f"      Desc     : {sug['description'][:100]}…")
        if sug["owner"]:
            print(f"      Owner    : {sug['owner']}")
        if sug["timeline"]:
            print(f"      Timeline : {sug['timeline']}")
        if sug["budget_estimate"]:
            print(f"      Budget   : {sug['budget_estimate']}")
        if sug["kpis"]:
            for kpi in sug["kpis"][:2]:
                print(f"      KPI      : {kpi[:70]}")

    print(f"\n  Results saved to: {RESULTS_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
