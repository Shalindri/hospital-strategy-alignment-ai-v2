"""
Synchronization Analyzer for Hospital Strategy–Action Plan Alignment System.

This module computes alignment (synchronization) scores between the
hospital's five strategic objectives and 25 action-plan items using the
384-dimensional cosine-similarity embeddings stored in ChromaDB.

It produces three levels of analysis:

1. **Overall synchronization** — a single aggregate score that
   summarises how well the action plan as a whole supports the
   strategic plan.
2. **Strategy-wise synchronization** — per-objective alignment metrics,
   including top-aligned actions, gap identification, and coverage
   breadth.
3. **Alignment matrix** — a full 5 × 25 matrix of pairwise cosine
   similarities that powers the dashboard heatmap.

The output is a comprehensive report dict that is also serialised to
``data/alignment_report.json`` for consumption by the Streamlit
dashboard and the evaluation harness.

Scoring thresholds
------------------
These thresholds were calibrated against the ``all-MiniLM-L6-v2``
model's typical similarity range for hospital-domain documents (where
unrelated pairs score 0.05–0.20 and strongly related pairs score
0.45–0.65):

==========  ================  ==========================================
Range       Classification    Interpretation
==========  ================  ==========================================
≥ 0.75      Excellent         Near-direct operationalisation of strategy
0.60–0.74   Good              Clear strategic support
0.45–0.59   Fair              Partial or indirect alignment
< 0.45      Poor              Weak or no meaningful alignment
==========  ================  ==========================================

An action is flagged as an **orphan** when its highest similarity to
*any* strategic objective falls below 0.45 — meaning it cannot be
convincingly linked to any part of the five-year strategy.

Typical usage::

    from src.synchronization_analyzer import SynchronizationAnalyzer

    analyzer = SynchronizationAnalyzer()
    report   = analyzer.analyze()
    analyzer.save_report()

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.vector_store import (
    VectorStore,
    STRATEGIC_COLLECTION,
    ACTION_COLLECTION,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("synchronization_analyzer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_JSON = DATA_DIR / "alignment_report.json"

# ── Classification thresholds (from shared config) ──────────────────
from src.config import (  # noqa: E402
    THRESHOLD_EXCELLENT,
    THRESHOLD_GOOD,
    THRESHOLD_FAIR,
    ORPHAN_THRESHOLD,
)

# Objective metadata — default display names.
# These are used as fallbacks; the system derives names from the loaded
# strategic plan data when available.
OBJECTIVE_NAMES: dict[str, str] = {}

STRATEGIC_JSON = DATA_DIR / "strategic_plan.json"
ACTION_JSON = DATA_DIR / "action_plan.json"


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class ActionAlignment:
    """Alignment profile for a single action item against all objectives.

    Attributes:
        action_id:          ChromaDB document ID (e.g. ``"action_1"``).
        action_number:      Integer action number (1–25).
        title:              Short action title.
        declared_objective: The objective code declared in the action plan
                            (the label assigned by the plan author).
        action_owner:       Responsible stakeholder.
        budget_lkr_millions: Budget in LKR millions.
        similarities:       Dict mapping objective code → cosine similarity.
        best_objective:     Code of the objective with highest similarity.
        best_score:         Highest cosine similarity across all objectives.
        classification:     One of ``"Excellent"``, ``"Good"``, ``"Fair"``,
                            ``"Poor"``.
        is_orphan:          ``True`` if ``best_score < ORPHAN_THRESHOLD``.
        declared_match:     ``True`` if ``best_objective == declared_objective``.
    """
    action_id: str
    action_number: int
    title: str
    declared_objective: str
    action_owner: str
    budget_lkr_millions: float
    similarities: dict[str, float]
    best_objective: str = ""
    best_score: float = 0.0
    classification: str = ""
    is_orphan: bool = False
    declared_match: bool = True


@dataclass
class ObjectiveAlignment:
    """Alignment profile for a single strategic objective.

    Attributes:
        code:                Objective letter (A–E).
        name:                Objective display name.
        top_actions:         List of ``(action_number, similarity)`` tuples,
                             top-5 by similarity.
        mean_similarity:     Mean cosine similarity across all 25 actions.
        max_similarity:      Highest cosine similarity to any action.
        aligned_action_count: Number of actions with similarity ≥ FAIR.
        declared_action_count: Actions whose plan-declared objective matches.
        coverage_score:      Fraction of actions scoring ≥ FAIR (0–1).
        gap_actions:         Actions declared under this objective but
                             scoring below FAIR — possible execution gaps.
    """
    code: str
    name: str
    top_actions: list[tuple[int, float]] = field(default_factory=list)
    mean_similarity: float = 0.0
    max_similarity: float = 0.0
    aligned_action_count: int = 0
    declared_action_count: int = 0
    coverage_score: float = 0.0
    gap_actions: list[int] = field(default_factory=list)


@dataclass
class SynchronizationReport:
    """Top-level report aggregating all alignment analysis.

    Attributes:
        overall_score:             Weighted mean of per-action best scores.
        overall_classification:    Classification of the overall score.
        mean_similarity:           Unweighted mean of the full 5 × 25 matrix.
        std_similarity:            Standard deviation of the full matrix.
        median_similarity:         Median of the full matrix.
        distribution:              Count of actions per classification band.
        objective_alignments:      Per-objective analysis (list of 5).
        action_alignments:         Per-action analysis (list of 25).
        well_aligned_actions:      Actions classified Good or Excellent.
        poorly_aligned_actions:    Actions classified Poor.
        orphan_actions:            Actions where best_score < ORPHAN_THRESHOLD.
        mismatched_actions:        Actions where the semantically-best
                                   objective differs from the declared one.
        alignment_matrix:          5 × 25 similarity matrix as nested list.
        matrix_row_labels:         Objective codes (row order).
        matrix_col_labels:         Action numbers (column order).
    """
    overall_score: float = 0.0
    overall_classification: str = ""
    mean_similarity: float = 0.0
    std_similarity: float = 0.0
    median_similarity: float = 0.0
    distribution: dict[str, int] = field(default_factory=dict)
    objective_alignments: list[dict] = field(default_factory=list)
    action_alignments: list[dict] = field(default_factory=list)
    well_aligned_actions: list[int] = field(default_factory=list)
    poorly_aligned_actions: list[int] = field(default_factory=list)
    orphan_actions: list[int] = field(default_factory=list)
    mismatched_actions: list[dict] = field(default_factory=list)
    alignment_matrix: list[list[float]] = field(default_factory=list)
    matrix_row_labels: list[str] = field(default_factory=list)
    matrix_col_labels: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_score(score: float) -> str:
    """Map a cosine-similarity score to a human-readable classification.

    Args:
        score: Cosine similarity in the range [−1, 1] (typically 0–0.8
               for hospital-domain text with all-MiniLM-L6-v2).

    Returns:
        One of ``"Excellent"``, ``"Good"``, ``"Fair"``, ``"Poor"``.
    """
    if score >= THRESHOLD_EXCELLENT:
        return "Excellent"
    if score >= THRESHOLD_GOOD:
        return "Good"
    if score >= THRESHOLD_FAIR:
        return "Fair"
    return "Poor"


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class SynchronizationAnalyzer:
    """Computes and reports strategy–action alignment synchronization.

    This is the core analytical engine of the ISPS system.  It pulls
    pre-computed embeddings from ChromaDB (via :class:`VectorStore`),
    builds the full pairwise similarity matrix, and derives alignment
    metrics at the overall, per-objective, and per-action levels.

    The analyzer also loads the original JSON data to enrich results
    with metadata (titles, owners, budgets, declared objectives) that
    the embeddings alone do not carry.

    Attributes:
        vs:           The :class:`VectorStore` instance.
        report:       The most recent :class:`SynchronizationReport`
                      (populated after calling :meth:`analyze`).

    Example::

        analyzer = SynchronizationAnalyzer()
        report   = analyzer.analyze()

        print(f"Overall: {report.overall_score:.3f} "
              f"({report.overall_classification})")
        print(f"Orphans: {report.orphan_actions}")
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        """Initialise the analyzer.

        Args:
            vector_store: An existing :class:`VectorStore` instance.
                          If ``None``, a new one is created with
                          default paths.
        """
        self.vs = vector_store or VectorStore()
        self.report: SynchronizationReport | None = None

        # Load original JSON for metadata enrichment
        self._action_data = self._load_json(ACTION_JSON)
        self._strategic_data = self._load_json(STRATEGIC_JSON)

        # Build objective name lookup from loaded data
        self._populate_objective_names()

        logger.info("SynchronizationAnalyzer initialised.")

    def _populate_objective_names(self) -> None:
        """Populate the module-level OBJECTIVE_NAMES from loaded strategic data."""
        global OBJECTIVE_NAMES
        for obj in self._strategic_data.get("objectives", []):
            code = obj.get("code", "")
            name = obj.get("name", "")
            if code and name:
                OBJECTIVE_NAMES[code] = name

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load a JSON file or return an empty dict on failure.

        Args:
            path: Absolute path to the JSON file.

        Returns:
            Parsed JSON as a dictionary.
        """
        if not path.exists():
            logger.warning("JSON not found: %s", path)
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _build_similarity_matrix(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Compute the full objectives × actions cosine-similarity matrix.

        Retrieves all embeddings from both ChromaDB collections and
        computes pairwise cosine similarities using a single matrix
        multiplication (since vectors are already L2-normalised).

        Returns:
            A 3-tuple of:

            - **matrix** — ``np.ndarray`` of shape ``(n_objectives, n_actions)``
              containing cosine similarities.
            - **obj_ids** — Ordered list of objective document IDs
              (e.g. ``["obj_A", "obj_B", …]``).
            - **act_ids** — Ordered list of action document IDs
              (e.g. ``["action_1", "action_2", …]``).

        Raises:
            RuntimeError: If either collection is empty (indicates
                          that :meth:`VectorStore.build_from_json` has
                          not been called).
        """
        # ── Retrieve all embeddings from ChromaDB ────────────────────
        obj_data = self.vs.get_all_embeddings(STRATEGIC_COLLECTION)
        act_data = self.vs.get_all_embeddings(ACTION_COLLECTION)

        if not obj_data["ids"] or not act_data["ids"]:
            raise RuntimeError(
                "One or both collections are empty. "
                "Run VectorStore.build_from_json() first."
            )

        # Convert to numpy arrays - cosine similarity = dot product.
        obj_emb = np.array(obj_data["embeddings"])  # (5, 384)
        act_emb = np.array(act_data["embeddings"])  # (25, 384)

        matrix = obj_emb @ act_emb.T
        matrix = np.clip(matrix, -1.0, 1.0)

        logger.info(
            "Embedding shapes: objectives %s, actions %s",
            obj_emb.shape,
            act_emb.shape,
        )

        logger.info(
            "Similarity matrix: shape=%s, min=%.4f, max=%.4f, mean=%.4f",
            matrix.shape,
            matrix.min(),
            matrix.max(),
            matrix.mean(),
        )

        return matrix, obj_data["ids"], act_data["ids"]

    def _build_action_lookup(self) -> dict[int, dict[str, Any]]:
        """Create a lookup table from action number to action JSON data.

        Returns:
            Dict mapping ``action_number`` → full action dict from
            ``action_plan.json``.
        """
        actions = self._action_data.get("actions", [])
        return {a["action_number"]: a for a in actions}

    def _build_objective_lookup(self) -> dict[str, dict[str, Any]]:
        """Create a lookup table from objective code to objective JSON data.

        Returns:
            Dict mapping objective code (e.g. ``"A"``) → full objective
            dict from ``strategic_plan.json``.
        """
        objectives = self._strategic_data.get("objectives", [])
        return {o["code"]: o for o in objectives}

    def analyze(self) -> SynchronizationReport:
        """Run the full synchronization analysis.

        This is the primary method.  It:

        1. Builds the 5 × 25 cosine-similarity matrix.
        2. Computes per-action alignment profiles.
        3. Computes per-objective alignment profiles.
        4. Derives aggregate statistics and classifications.
        5. Identifies orphan and mismatched actions.
        6. Packages everything into a :class:`SynchronizationReport`.

        Returns:
            The completed :class:`SynchronizationReport`.
        """
        logger.info("=" * 60)
        logger.info("Starting synchronization analysis")
        logger.info("=" * 60)

        # ── Step 1: Build similarity matrix ──────────────────────────
        matrix, obj_ids, act_ids = self._build_similarity_matrix()

        # Map IDs to codes / numbers for readability
        #   obj_ids: ["obj_A", "obj_B", …]  → ["A", "B", …]
        #   act_ids: ["action_1", "action_2", …] → [1, 2, …]
        obj_codes = [oid.replace("obj_", "") for oid in obj_ids]
        act_numbers = [int(aid.replace("action_", "")) for aid in act_ids]

        # Build index mappings for matrix row/column lookups
        obj_idx = {code: i for i, code in enumerate(obj_codes)}
        act_idx = {num: j for j, num in enumerate(act_numbers)}

        action_lookup = self._build_action_lookup()

        # ── Step 2: Per-action alignment ─────────────────────────────
        action_alignments: list[ActionAlignment] = []

        for j, act_num in enumerate(act_numbers):
            act_meta = action_lookup.get(act_num, {})
            title = act_meta.get("title", f"Action {act_num}")
            declared = act_meta.get("strategic_objective_code", "")
            owner = act_meta.get("action_owner", "")
            budget = act_meta.get("budget_lkr_millions", 0.0)

            # Similarity to each objective
            sims = {
                code: round(float(matrix[i, j]), 4)
                for code, i in obj_idx.items()
            }

            best_code = max(sims, key=sims.get)
            best_score = sims[best_code]
            classification = _classify_score(best_score)

            aa = ActionAlignment(
                action_id=f"action_{act_num}",
                action_number=act_num,
                title=title,
                declared_objective=declared,
                action_owner=owner,
                budget_lkr_millions=budget,
                similarities=sims,
                best_objective=best_code,
                best_score=round(best_score, 4),
                classification=classification,
                is_orphan=best_score < ORPHAN_THRESHOLD,
                declared_match=(best_code == declared),
            )
            action_alignments.append(aa)

        logger.info("Computed alignment for %d actions.", len(action_alignments))

        # ── Step 3: Per-objective alignment ──────────────────────────
        objective_alignments: list[ObjectiveAlignment] = []

        for code in obj_codes:
            i = obj_idx[code]
            row = matrix[i, :]  # similarities to all actions

            # Top-5 actions by similarity
            top5_indices = np.argsort(row)[::-1][:5]
            top_actions = [
                (act_numbers[j], round(float(row[j]), 4))
                for j in top5_indices
            ]

            # Actions scoring at or above FAIR threshold
            aligned_mask = row >= THRESHOLD_FAIR
            aligned_count = int(aligned_mask.sum())

            # Actions declared under this objective
            declared_nums = [
                a.action_number
                for a in action_alignments
                if a.declared_objective == code
            ]

            # Gap actions: declared under this objective but scoring < FAIR
            gap_actions = []
            for act_num in declared_nums:
                j = act_idx.get(act_num)
                if j is not None and row[j] < THRESHOLD_FAIR:
                    gap_actions.append(act_num)

            oa = ObjectiveAlignment(
                code=code,
                name=OBJECTIVE_NAMES.get(code, code),
                top_actions=top_actions,
                mean_similarity=round(float(row.mean()), 4),
                max_similarity=round(float(row.max()), 4),
                aligned_action_count=aligned_count,
                declared_action_count=len(declared_nums),
                coverage_score=round(aligned_count / len(act_numbers), 4),
                gap_actions=gap_actions,
            )
            objective_alignments.append(oa)

        logger.info("Computed alignment for %d objectives.", len(objective_alignments))

        # ── Step 4: Aggregate statistics ─────────────────────────────
        flat_matrix = matrix.flatten()
        best_scores = np.array([aa.best_score for aa in action_alignments])

        # Overall score: mean of each action's best-objective similarity.
        # This represents "on average, how well is each action connected
        # to its closest strategic objective?"
        overall_score = round(float(best_scores.mean()), 4)
        overall_class = _classify_score(overall_score)

        # Distribution: how many actions fall into each band
        distribution = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
        for aa in action_alignments:
            distribution[aa.classification] += 1

        # ── Step 5: Identify special categories ──────────────────────
        well_aligned = sorted(
            a.action_number
            for a in action_alignments
            if a.classification in ("Good", "Excellent")
        )
        poorly_aligned = sorted(
            a.action_number
            for a in action_alignments
            if a.classification == "Poor"
        )
        orphans = sorted(
            a.action_number
            for a in action_alignments
            if a.is_orphan
        )
        mismatched = [
            {
                "action_number": a.action_number,
                "title": a.title,
                "declared_objective": a.declared_objective,
                "best_objective": a.best_objective,
                "best_score": a.best_score,
                "declared_score": a.similarities.get(a.declared_objective, 0.0),
            }
            for a in action_alignments
            if not a.declared_match
        ]

        # ── Step 6: Assemble report ──────────────────────────────────
        report = SynchronizationReport(
            overall_score=overall_score,
            overall_classification=overall_class,
            mean_similarity=round(float(flat_matrix.mean()), 4),
            std_similarity=round(float(flat_matrix.std()), 4),
            median_similarity=round(float(np.median(flat_matrix)), 4),
            distribution=distribution,
            objective_alignments=[asdict(oa) for oa in objective_alignments],
            action_alignments=[asdict(aa) for aa in action_alignments],
            well_aligned_actions=well_aligned,
            poorly_aligned_actions=poorly_aligned,
            orphan_actions=orphans,
            mismatched_actions=mismatched,
            alignment_matrix=np.round(matrix, 4).tolist(),
            matrix_row_labels=obj_codes,
            matrix_col_labels=act_numbers,
        )

        self.report = report

        logger.info("Overall synchronization: %.4f (%s)", overall_score, overall_class)
        logger.info("Distribution: %s", distribution)
        logger.info("Well-aligned actions: %s", well_aligned)
        logger.info("Poorly-aligned actions: %s", poorly_aligned)
        logger.info("Orphan actions: %s", orphans)
        logger.info("Mismatched declarations: %d actions", len(mismatched))

        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, output_path: Path | str = REPORT_JSON) -> Path:
        """Serialise the analysis report to a JSON file.

        The JSON structure is designed for direct consumption by the
        Streamlit dashboard.  It includes the alignment matrix,
        per-action and per-objective breakdowns, and all visualization
        metadata.

        Args:
            output_path: Destination file path.

        Returns:
            The resolved ``Path`` of the written file.

        Raises:
            RuntimeError: If :meth:`analyze` has not been called yet.
        """
        if self.report is None:
            raise RuntimeError("No report to save. Call analyze() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self.report)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        size_kb = output_path.stat().st_size / 1024
        logger.info("Report saved: %s (%.1f KB)", output_path, size_kb)
        return output_path

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def get_visualization_data(self) -> dict[str, Any]:
        """Return a dashboard-ready dict with all visualization payloads.

        The returned structure includes pre-formatted data for:

        - Heatmap (alignment matrix with labels)
        - Radar chart (per-objective mean similarity)
        - Bar chart (per-action best scores with classification colours)
        - Summary cards (overall score, orphan count, etc.)
        - Scatter data (budget vs. alignment)

        Returns:
            A dict consumable by the Streamlit/Plotly dashboard.

        Raises:
            RuntimeError: If :meth:`analyze` has not been called yet.
        """
        if self.report is None:
            raise RuntimeError(
                "No report available. Call analyze() first."
            )
        r = self.report

        # ── Heatmap payload ──────────────────────────────────────────
        heatmap = {
            "z": r.alignment_matrix,
            "x_labels": [f"Action {n}" for n in r.matrix_col_labels],
            "y_labels": [
                f"{code}: {OBJECTIVE_NAMES.get(code, code)}"
                for code in r.matrix_row_labels
            ],
            "x_ids": r.matrix_col_labels,
            "y_ids": r.matrix_row_labels,
        }

        # ── Radar chart payload (per-objective mean similarity) ──────
        radar = {
            "categories": [
                OBJECTIVE_NAMES.get(oa["code"], oa["code"])
                for oa in r.objective_alignments
            ],
            "values": [
                oa["mean_similarity"]
                for oa in r.objective_alignments
            ],
        }

        # ── Bar chart payload (per-action best score) ────────────────
        classification_colours = {
            "Excellent": "#2ecc71",
            "Good": "#27ae60",
            "Fair": "#f39c12",
            "Poor": "#e74c3c",
        }
        bar_chart = {
            "actions": [
                f"A{aa['action_number']}"
                for aa in r.action_alignments
            ],
            "scores": [aa["best_score"] for aa in r.action_alignments],
            "colours": [
                classification_colours[aa["classification"]]
                for aa in r.action_alignments
            ],
            "titles": [aa["title"] for aa in r.action_alignments],
            "classifications": [
                aa["classification"] for aa in r.action_alignments
            ],
        }

        # ── Summary cards ────────────────────────────────────────────
        summary = {
            "overall_score": r.overall_score,
            "overall_classification": r.overall_classification,
            "total_actions": len(r.action_alignments),
            "well_aligned_count": len(r.well_aligned_actions),
            "poorly_aligned_count": len(r.poorly_aligned_actions),
            "orphan_count": len(r.orphan_actions),
            "mismatch_count": len(r.mismatched_actions),
            "mean_similarity": r.mean_similarity,
            "std_similarity": r.std_similarity,
            "distribution": r.distribution,
        }

        # ── Scatter: budget vs alignment ─────────────────────────────
        scatter = {
            "action_numbers": [
                aa["action_number"] for aa in r.action_alignments
            ],
            "budgets": [
                aa["budget_lkr_millions"] for aa in r.action_alignments
            ],
            "scores": [aa["best_score"] for aa in r.action_alignments],
            "titles": [aa["title"] for aa in r.action_alignments],
            "classifications": [
                aa["classification"] for aa in r.action_alignments
            ],
        }

        return {
            "heatmap": heatmap,
            "radar": radar,
            "bar_chart": bar_chart,
            "summary": summary,
            "scatter": scatter,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the synchronization analysis and print a formatted report.

    This entry point builds the vector store (if needed), runs the
    full analysis, saves the JSON report, and prints a human-readable
    summary to stdout.
    """
    logger.info("=" * 60)
    logger.info("ISPS Synchronization Analyzer — Starting")
    logger.info("=" * 60)

    # ── Ensure vector store is populated ──────────────────────────────
    vs = VectorStore()
    stats = vs.collection_stats()
    if stats[STRATEGIC_COLLECTION]["count"] == 0:
        logger.info("Collections empty — building from JSON …")
        vs.build_from_json()

    # ── Run analysis ─────────────────────────────────────────────────
    analyzer = SynchronizationAnalyzer(vector_store=vs)
    report = analyzer.analyze()
    analyzer.save_report()

    # ── Print formatted report ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SYNCHRONIZATION ANALYSIS REPORT")
    print("  Strategy vs. Action Plan Alignment")
    print("=" * 70)

    print(f"\n{'OVERALL SYNCHRONIZATION':─^70}")
    print(f"  Score          : {report.overall_score:.4f}")
    print(f"  Classification : {report.overall_classification}")
    print(f"  Matrix mean    : {report.mean_similarity:.4f} "
          f"(σ = {report.std_similarity:.4f})")
    print(f"  Matrix median  : {report.median_similarity:.4f}")
    print(f"\n  Distribution:")
    for cls, count in report.distribution.items():
        bar = "█" * count
        print(f"    {cls:<10} : {count:>2}  {bar}")

    print(f"\n{'PER-OBJECTIVE ALIGNMENT':─^70}")
    for oa in report.objective_alignments:
        print(f"\n  [{oa['code']}] {oa['name']}")
        print(f"      Mean sim     : {oa['mean_similarity']:.4f}")
        print(f"      Max sim      : {oa['max_similarity']:.4f}")
        print(f"      Aligned acts : {oa['aligned_action_count']} / {len(report.action_alignments)}"
              f"  (coverage: {oa['coverage_score']:.1%})")
        print(f"      Declared acts: {oa['declared_action_count']}")
        print(f"      Top-5 actions:")
        for act_num, sim in oa["top_actions"]:
            act_info = next(
                (a for a in report.action_alignments
                 if a["action_number"] == act_num),
                {},
            )
            title = act_info.get("title", "")[:42]
            print(f"        Action {act_num:>2}  sim={sim:.4f}  {title}")
        if oa["gap_actions"]:
            print(f"      ⚠ Gap actions (declared but < {THRESHOLD_FAIR}): "
                  f"{oa['gap_actions']}")

    print(f"\n{'PER-ACTION ALIGNMENT':─^70}")
    print(f"  {'#':>3}  {'Title':<44} {'Decl':>4} {'Best':>4} "
          f"{'Score':>6} {'Class':<10} {'Match'}")
    print(f"  {'─'*3}  {'─'*44} {'─'*4} {'─'*4} {'─'*6} {'─'*10} {'─'*5}")
    for aa in report.action_alignments:
        match_icon = "✓" if aa["declared_match"] else "✗"
        orphan_flag = " [ORPHAN]" if aa["is_orphan"] else ""
        print(
            f"  {aa['action_number']:>3}  "
            f"{aa['title'][:44]:<44} "
            f"{aa['declared_objective']:>4} "
            f"{aa['best_objective']:>4} "
            f"{aa['best_score']:>6.4f} "
            f"{aa['classification']:<10} "
            f"{match_icon}{orphan_flag}"
        )

    if report.orphan_actions:
        print(f"\n{'ORPHAN ACTIONS':─^70}")
        print(f"  Actions not meaningfully aligned to any objective "
              f"(best score < {ORPHAN_THRESHOLD}):")
        for act_num in report.orphan_actions:
            aa = next(
                a for a in report.action_alignments
                if a["action_number"] == act_num
            )
            print(f"    Action {act_num:>2}: {aa['title'][:50]}"
                  f"  (best={aa['best_score']:.4f} → Obj {aa['best_objective']})")

    if report.mismatched_actions:
        print(f"\n{'DECLARATION MISMATCHES':─^70}")
        print(f"  Actions where semantic best-fit differs from declared objective:")
        for m in report.mismatched_actions:
            print(f"    Action {m['action_number']:>2}: declared={m['declared_objective']}"
                  f"  semantic={m['best_objective']}"
                  f"  (declared_sim={m['declared_score']:.4f}"
                  f"  best_sim={m['best_score']:.4f})")

    print(f"\n{'ALIGNMENT MATRIX (heatmap)':─^70}")
    # Print compact matrix
    header = "       " + "".join(f"A{n:<4}" for n in report.matrix_col_labels)
    print(header)
    for i, code in enumerate(report.matrix_row_labels):
        row_str = f"  {code}  | "
        for val in report.alignment_matrix[i]:
            if val >= THRESHOLD_GOOD:
                row_str += f"{val:.2f} "
            elif val >= THRESHOLD_FAIR:
                row_str += f"{val:.2f} "
            else:
                row_str += f"{val:.2f} "
        print(row_str)

    print(f"\n  Report saved to: {REPORT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
