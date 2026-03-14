"""
Ground Truth Labeller for Hospital Strategy–Action Plan Alignment.

Creates a human-annotated ground truth dataset of strategy–action alignment
labels for evaluating the ISPS system.  Each labelled pair records a
strategic objective (A–E) and an action (1–25) together with an alignment
score on a four-point scale:

    1.0  — Strongly Aligned   (directly implements the objective)
    0.7  — Aligned            (clearly supports the objective)
    0.4  — Weakly Aligned     (tangential or indirect relationship)
    0.0  — Not Aligned        (no meaningful relationship)

Usage
-----
Interactive mode (review / relabel every pair one by one)::

    python -m tests.create_ground_truth --interactive

Batch mode (accept pre-seeded expert labels, write immediately)::

    python -m tests.create_ground_truth

Review-only mode (show pre-seeded labels, prompt to edit any)::

    python -m tests.create_ground_truth --review

Output
------
``tests/ground_truth.json`` — list of dicts, each with::

    {
        "objective_code":  "A",
        "objective_name":  "Patient Care Excellence",
        "action_number":   3,
        "action_title":    "Cardiac Catheterisation Laboratory ...",
        "declared_objective": "A",
        "alignment_label": 1.0,
        "label_text":      "Strongly Aligned",
        "rationale":       "Free-text explanation of why this label ...",
        "is_declared_pair": true
    }

Labelling Criteria
------------------
Apply these guidelines consistently:

**Strongly Aligned (1.0):**
    The action *directly implements* a goal or KPI of the objective.
    Completing this action *necessarily* advances the objective.
    Examples: a cardiology lab for Patient Care; an EHR for Digital Health.

**Aligned (0.7):**
    The action *clearly supports* the objective but is not the primary
    mechanism.  It contributes to the objective's broader intent.
    Examples: a nursing expansion plan supports Patient Care; a CPD
    programme supports Workforce Development.

**Weakly Aligned (0.4):**
    The action has only a *tangential or indirect* relationship.
    One could argue a connection, but it requires multiple logical steps.
    Examples: land acquisition is weakly related to Patient Care (enables
    future capacity); cybersecurity is weakly related to Research.

**Not Aligned (0.0):**
    No meaningful connection between the action and the objective.
    The action belongs to a completely different strategic domain.
    Examples: a sports tournament has nothing to do with Research; a
    pharmacy outlet has nothing to do with Digital Health.

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_FILE = TESTS_DIR / "ground_truth.json"

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, float] = {
    "1": 1.0,    # Strongly Aligned
    "2": 0.7,    # Aligned
    "3": 0.4,    # Weakly Aligned
    "4": 0.0,    # Not Aligned
    "s": 1.0,
    "a": 0.7,
    "w": 0.4,
    "n": 0.0,
}

SCORE_TO_TEXT: dict[float, str] = {
    1.0: "Strongly Aligned",
    0.7: "Aligned",
    0.4: "Weakly Aligned",
    0.0: "Not Aligned",
}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_objectives() -> list[dict[str, Any]]:
    """Load strategic objectives from the processed JSON."""
    path = DATA_DIR / "strategic_plan.json"
    with open(path, encoding="utf-8") as f:
        sp = json.load(f)
    return sp.get("objectives", [])


def load_actions() -> list[dict[str, Any]]:
    """Load action items from the processed JSON."""
    path = DATA_DIR / "action_plan.json"
    with open(path, encoding="utf-8") as f:
        ap = json.load(f)
    return ap.get("actions", [])


# ---------------------------------------------------------------------------
# Pre-seeded expert labels (50+ pairs)
#
# These were assigned by a domain expert who read both the strategic plan
# and the action plan in full.  The interactive CLI allows overriding any
# of these during a review session.
#
# Pair selection strategy:
#   - All 25 declared (action → its own objective) pairs
#   - 10 high-cross-alignment pairs (action scores well on *another* obj)
#   - 10 clearly-not-aligned pairs (for negative class balance)
#   - 10 borderline / interesting pairs
#   Total: 55 pairs ≥ 50 for statistical significance
# ---------------------------------------------------------------------------

PRE_SEEDED: list[dict[str, Any]] = [
    # ══════════════════════════════════════════════════════════════════
    # DECLARED PAIRS (action → its own declared objective, 25 pairs)
    # ══════════════════════════════════════════════════════════════════

    # Objective A: Patient Care Excellence
    {"obj": "A", "act": 1,  "score": 0.4,
     "rationale": "Land acquisition enables future hospital expansion but does not directly deliver patient care."},
    {"obj": "A", "act": 2,  "score": 0.7,
     "rationale": "An architectural master plan for 150-bed expansion directly supports capacity for patient care."},
    {"obj": "A", "act": 3,  "score": 1.0,
     "rationale": "A cardiac catheterisation lab is a core clinical service that directly delivers patient care excellence."},
    {"obj": "A", "act": 4,  "score": 1.0,
     "rationale": "An elderly care unit pilot directly implements a new patient care service line."},
    {"obj": "A", "act": 5,  "score": 0.7,
     "rationale": "Aesthetic medicine feasibility supports service diversification under patient care."},
    {"obj": "A", "act": 6,  "score": 1.0,
     "rationale": "Fertility centre lab upgrade directly enhances clinical capability for patient care."},
    {"obj": "A", "act": 7,  "score": 1.0,
     "rationale": "Nephrology service development is a core clinical service advancing patient care excellence."},
    {"obj": "A", "act": 8,  "score": 0.0,
     "rationale": "Administrative office refurbishment has no clinical component — it is a facilities/admin action misaligned with Patient Care."},

    # Objective B: Digital Health Transformation
    {"obj": "B", "act": 9,  "score": 1.0,
     "rationale": "EHR vendor selection is the cornerstone action for digital health transformation."},
    {"obj": "B", "act": 10, "score": 1.0,
     "rationale": "IT infrastructure upgrade directly enables the digital health platform."},
    {"obj": "B", "act": 11, "score": 1.0,
     "rationale": "Telemedicine platform is a primary digital health service delivery mechanism."},
    {"obj": "B", "act": 12, "score": 0.7,
     "rationale": "Cybersecurity supports digital health infrastructure but is an enabler, not a direct service."},

    # Objective C: Research & Innovation
    {"obj": "C", "act": 13, "score": 1.0,
     "rationale": "Ethics review committee is foundational infrastructure for conducting clinical research."},
    {"obj": "C", "act": 14, "score": 1.0,
     "rationale": "University research partnership MOU directly establishes research collaboration."},
    {"obj": "C", "act": 15, "score": 1.0,
     "rationale": "Research coordinator appointment directly enables research programme operations."},

    # Objective D: Workforce Development
    {"obj": "D", "act": 16, "score": 1.0,
     "rationale": "Specialist clinician recruitment is a core workforce development action."},
    {"obj": "D", "act": 17, "score": 1.0,
     "rationale": "Nursing workforce expansion directly builds the hospital's human capital."},
    {"obj": "D", "act": 18, "score": 1.0,
     "rationale": "CPD programme launch directly implements continuous professional development."},
    {"obj": "D", "act": 19, "score": 0.0,
     "rationale": "A cricket tournament is a social/recreational activity with no meaningful connection to systematic workforce development."},

    # Objective E: Community & Regional Health Expansion
    {"obj": "E", "act": 20, "score": 1.0,
     "rationale": "Maldives partnership scoping directly extends regional health reach."},
    {"obj": "E", "act": 21, "score": 0.7,
     "rationale": "Medical tourism concierge supports regional expansion but is commercially focused rather than community health."},
    {"obj": "E", "act": 22, "score": 1.0,
     "rationale": "Community preventive health screening is the most direct community health action."},
    {"obj": "E", "act": 23, "score": 0.7,
     "rationale": "Corporate wellness partnerships extend community engagement but serve a commercial niche."},
    {"obj": "E", "act": 24, "score": 0.0,
     "rationale": "Executive lounge furniture upgrade is an internal amenity with no community health impact — misaligned."},
    {"obj": "E", "act": 25, "score": 0.0,
     "rationale": "Retail pharmacy outlet expansion is a commercial retail action with no direct community health programme link — misaligned."},

    # ══════════════════════════════════════════════════════════════════
    # CROSS-OBJECTIVE HIGH-ALIGNMENT PAIRS (10 pairs)
    # Actions that score well on objectives other than their declared one
    # ══════════════════════════════════════════════════════════════════
    {"obj": "A", "act": 17, "score": 0.7,
     "rationale": "Nursing expansion directly improves patient care delivery capacity."},
    {"obj": "A", "act": 16, "score": 0.7,
     "rationale": "Specialist clinician recruitment enhances patient care quality."},
    {"obj": "E", "act": 11, "score": 0.7,
     "rationale": "Telemedicine enables regional/remote care delivery, supporting community expansion."},
    {"obj": "A", "act": 9,  "score": 0.4,
     "rationale": "EHR improves clinical workflows but is primarily a digital health action."},
    {"obj": "D", "act": 14, "score": 0.4,
     "rationale": "University partnership creates faculty development opportunities tangentially."},
    {"obj": "B", "act": 20, "score": 0.4,
     "rationale": "Maldives partnership may use telemedicine links but is primarily a regional expansion action."},
    {"obj": "A", "act": 10, "score": 0.4,
     "rationale": "IT infrastructure supports clinical systems but is primarily a digital health enabler."},
    {"obj": "D", "act": 9,  "score": 0.0,
     "rationale": "EHR vendor selection is a technology procurement action, not workforce development."},
    {"obj": "C", "act": 18, "score": 0.4,
     "rationale": "CPD may include research skills training, a tangential link to Research & Innovation."},
    {"obj": "E", "act": 5,  "score": 0.4,
     "rationale": "Aesthetic medicine could attract medical tourists, tangentially supporting regional expansion."},

    # ══════════════════════════════════════════════════════════════════
    # CLEARLY NOT-ALIGNED PAIRS (10 pairs — negative class)
    # ══════════════════════════════════════════════════════════════════
    {"obj": "C", "act": 8,  "score": 0.0,
     "rationale": "Office refurbishment has zero connection to research and innovation."},
    {"obj": "D", "act": 8,  "score": 0.0,
     "rationale": "Office refurbishment does not develop workforce capability."},
    {"obj": "B", "act": 19, "score": 0.0,
     "rationale": "A cricket tournament has nothing to do with digital health transformation."},
    {"obj": "C", "act": 19, "score": 0.0,
     "rationale": "A sports tournament has no relation to clinical research or innovation."},
    {"obj": "E", "act": 19, "score": 0.0,
     "rationale": "A staff social event does not advance community or regional health."},
    {"obj": "B", "act": 24, "score": 0.0,
     "rationale": "Lounge furniture has no digital health component."},
    {"obj": "C", "act": 24, "score": 0.0,
     "rationale": "Lounge decor has no research or innovation component."},
    {"obj": "D", "act": 24, "score": 0.0,
     "rationale": "Furniture upgrade does not develop the workforce."},
    {"obj": "B", "act": 25, "score": 0.0,
     "rationale": "A retail pharmacy outlet has no digital health transformation component."},
    {"obj": "C", "act": 25, "score": 0.0,
     "rationale": "A pharmacy retail outlet has no research or innovation component."},

    # ══════════════════════════════════════════════════════════════════
    # BORDERLINE / INTERESTING PAIRS (15 pairs)
    # ══════════════════════════════════════════════════════════════════
    {"obj": "D", "act": 15, "score": 0.4,
     "rationale": "Research coordinator role contributes to workforce development tangentially — it is primarily a research action."},
    {"obj": "A", "act": 18, "score": 0.4,
     "rationale": "CPD programme may improve clinical skills, weakly supporting patient care quality."},
    {"obj": "A", "act": 22, "score": 0.4,
     "rationale": "Community screening connects to patient care through early detection but is primarily a community action."},
    {"obj": "C", "act": 3,  "score": 0.4,
     "rationale": "A cath lab enables cardiology research tangentially but is primarily a care delivery action."},
    {"obj": "A", "act": 11, "score": 0.7,
     "rationale": "Telemedicine improves patient access to care, supporting patient care excellence."},
    {"obj": "D", "act": 13, "score": 0.4,
     "rationale": "Ethics committee requires trained staff but is primarily a research governance action."},
    {"obj": "B", "act": 3,  "score": 0.4,
     "rationale": "A cath lab requires digital monitoring equipment, a tangential digital health link."},
    {"obj": "E", "act": 4,  "score": 0.4,
     "rationale": "Elderly care pilot serves the local community tangentially but is primarily a patient care action."},
    {"obj": "D", "act": 21, "score": 0.0,
     "rationale": "Medical tourism concierge is a service unit, not a workforce development initiative."},
    {"obj": "A", "act": 20, "score": 0.4,
     "rationale": "Maldives partnership may refer patients for treatment, weakly supporting patient care volume."},
    {"obj": "C", "act": 9,  "score": 0.4,
     "rationale": "EHR enables research data collection, a tangential but real connection to Research & Innovation."},
    {"obj": "E", "act": 22, "score": 1.0,
     "rationale": "Community screening is the most direct community health expansion action (also declared under E)."},
    {"obj": "B", "act": 22, "score": 0.4,
     "rationale": "Community screening could feed data into digital health systems, a tangential link."},
    {"obj": "D", "act": 17, "score": 1.0,
     "rationale": "Nursing workforce expansion is a core workforce development action (also declared under D)."},
    {"obj": "A", "act": 25, "score": 0.0,
     "rationale": "A retail pharmacy outlet does not deliver patient care — it is a commercial retail operation."},
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_ground_truth(
    objectives: list[dict],
    actions: list[dict],
    pre_seeded: list[dict],
) -> list[dict[str, Any]]:
    """Convert pre-seeded shorthand into full ground truth records.

    Enriches each pre-seeded entry with objective/action metadata
    from the upstream JSON files.
    """
    obj_lookup = {o["code"]: o for o in objectives}
    act_lookup = {a["action_number"]: a for a in actions}

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for seed in pre_seeded:
        obj_code = seed["obj"]
        act_num = seed["act"]
        key = (obj_code, act_num)
        if key in seen:
            continue  # skip duplicates
        seen.add(key)

        obj = obj_lookup.get(obj_code, {})
        act = act_lookup.get(act_num, {})

        records.append({
            "objective_code":    obj_code,
            "objective_name":    obj.get("name", ""),
            "action_number":     act_num,
            "action_title":      act.get("title", ""),
            "declared_objective": act.get("strategic_objective_code", ""),
            "alignment_label":   seed["score"],
            "label_text":        SCORE_TO_TEXT[seed["score"]],
            "rationale":         seed["rationale"],
            "is_declared_pair":  act.get("strategic_objective_code", "") == obj_code,
        })

    return records


def print_summary(records: list[dict]) -> None:
    """Print a summary of the ground truth dataset."""
    print(f"\nGround Truth Summary")
    print("=" * 60)
    print(f"Total labelled pairs: {len(records)}")

    # Label distribution
    from collections import Counter
    label_counts = Counter(r["alignment_label"] for r in records)
    print("\nLabel Distribution:")
    for score in sorted(label_counts.keys(), reverse=True):
        text = SCORE_TO_TEXT[score]
        count = label_counts[score]
        bar = "#" * count
        print(f"  {score:.1f} ({text:<18}) : {count:>3}  {bar}")

    # By objective
    print("\nPairs per Objective:")
    obj_counts = Counter(r["objective_code"] for r in records)
    for code in sorted(obj_counts.keys()):
        print(f"  Objective {code}: {obj_counts[code]} pairs")

    # Declared vs cross-objective
    declared = sum(1 for r in records if r["is_declared_pair"])
    cross = len(records) - declared
    print(f"\nDeclared pairs:       {declared}")
    print(f"Cross-objective pairs: {cross}")

    # Misaligned ground truth check
    misaligned_acts = set()
    for r in records:
        if r["is_declared_pair"] and r["alignment_label"] == 0.0:
            misaligned_acts.add(r["action_number"])
    print(f"\nIntentionally misaligned (declared but score=0.0): "
          f"{sorted(misaligned_acts)}")


def interactive_label(
    objectives: list[dict],
    actions: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Run an interactive CLI session to label or relabel pairs.

    For each pair the annotator can:
        - Press Enter to accept the pre-seeded label (if any)
        - Enter 1/2/3/4 (or s/a/w/n) to assign a new label
        - Enter 'q' to stop early and save progress
        - Enter 'r' to provide a rationale

    Returns the updated record list.
    """
    obj_lookup = {o["code"]: o for o in objectives}
    act_lookup = {a["action_number"]: a for a in actions}
    existing_map = {
        (r["objective_code"], r["action_number"]): r for r in existing
    }

    print("\n" + "=" * 65)
    print("  ISPS Ground Truth Interactive Labeller")
    print("=" * 65)
    print("""
Labelling Scale:
  [1/s] Strongly Aligned (1.0) — directly implements the objective
  [2/a] Aligned          (0.7) — clearly supports the objective
  [3/w] Weakly Aligned   (0.4) — tangential / indirect relationship
  [4/n] Not Aligned      (0.0) — no meaningful relationship

Controls:
  Enter    → accept current label (shown in brackets)
  1-4 / s/a/w/n → assign label
  r        → add/edit rationale for current pair
  q        → quit and save progress
""")

    updated = list(existing)
    updated_map = dict(existing_map)

    # Build the pair list: existing pairs first, then add any missing
    pair_keys: list[tuple[str, int]] = []
    for r in existing:
        pair_keys.append((r["objective_code"], r["action_number"]))

    total = len(pair_keys)
    for idx, (obj_code, act_num) in enumerate(pair_keys):
        obj = obj_lookup.get(obj_code, {})
        act = act_lookup.get(act_num, {})
        rec = updated_map.get((obj_code, act_num), {})

        current_label = rec.get("alignment_label")
        current_text = SCORE_TO_TEXT.get(current_label, "?")
        current_rationale = rec.get("rationale", "")

        print(f"\n─── Pair {idx + 1}/{total} ─────────────────────────────")
        print(f"  Objective {obj_code}: {obj.get('name', '?')}")
        if obj.get("goal_statement"):
            print(f"    Goal: {obj['goal_statement'][:90]}...")
        print(f"  Action #{act_num}: {act.get('title', '?')}")
        if act.get("description"):
            print(f"    Desc: {act['description'][:90]}...")
        print(f"  Declared objective: {act.get('strategic_objective_code', '?')}")
        if current_label is not None:
            print(f"  Current label: {current_label} ({current_text})")
        if current_rationale:
            print(f"  Rationale: {current_rationale[:80]}")

        while True:
            prompt = f"  Label [{current_label}]: " if current_label is not None else "  Label: "
            try:
                choice = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Saving and exiting...")
                return updated

            if choice == "q":
                print("  Saving and exiting...")
                return updated

            if choice == "r":
                try:
                    rationale = input("  Rationale: ").strip()
                except (EOFError, KeyboardInterrupt):
                    return updated
                if rationale:
                    rec["rationale"] = rationale
                    # Update in the list too
                    for r in updated:
                        if r["objective_code"] == obj_code and r["action_number"] == act_num:
                            r["rationale"] = rationale
                continue

            if choice == "" and current_label is not None:
                # Accept existing label
                break

            if choice in LABEL_MAP:
                new_score = LABEL_MAP[choice]
                new_text = SCORE_TO_TEXT[new_score]

                # Update the record
                found = False
                for r in updated:
                    if r["objective_code"] == obj_code and r["action_number"] == act_num:
                        r["alignment_label"] = new_score
                        r["label_text"] = new_text
                        found = True
                        break

                if not found:
                    new_rec = {
                        "objective_code":    obj_code,
                        "objective_name":    obj.get("name", ""),
                        "action_number":     act_num,
                        "action_title":      act.get("title", ""),
                        "declared_objective": act.get("strategic_objective_code", ""),
                        "alignment_label":   new_score,
                        "label_text":        new_text,
                        "rationale":         "",
                        "is_declared_pair":  act.get("strategic_objective_code", "") == obj_code,
                    }
                    updated.append(new_rec)
                    updated_map[(obj_code, act_num)] = new_rec

                print(f"  → {new_score} ({new_text})")
                break

            print("  Invalid input. Use 1-4, s/a/w/n, r, q, or Enter.")

    return updated


def review_mode(records: list[dict]) -> list[dict]:
    """Display all pre-seeded labels and let the user edit any."""
    print("\n" + "=" * 65)
    print("  ISPS Ground Truth — Review Mode")
    print("=" * 65)
    print(f"  {len(records)} pre-seeded labels loaded.\n")

    for i, r in enumerate(records):
        declared_marker = "*" if r["is_declared_pair"] else " "
        print(
            f"  {i + 1:>3}. {declared_marker} "
            f"Obj {r['objective_code']} × Act {r['action_number']:>2} "
            f"→ {r['alignment_label']:.1f} ({r['label_text']:<18}) "
            f"| {r['rationale'][:50]}"
        )

    print(f"\n  * = declared pair (action's own objective)")
    print(f"\nEnter pair number to edit, or 'done' to save all:")

    while True:
        try:
            choice = input("  Edit #: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice in ("done", "d", "q", ""):
            break

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(records):
                r = records[idx]
                print(f"\n    Obj {r['objective_code']}: {r['objective_name']}")
                print(f"    Act #{r['action_number']}: {r['action_title']}")
                print(f"    Current: {r['alignment_label']} ({r['label_text']})")
                print(f"    Rationale: {r['rationale']}")
                print("    [1] Strongly Aligned  [2] Aligned  "
                      "[3] Weakly Aligned  [4] Not Aligned")
                try:
                    new_label = input("    New label: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if new_label in LABEL_MAP:
                    r["alignment_label"] = LABEL_MAP[new_label]
                    r["label_text"] = SCORE_TO_TEXT[r["alignment_label"]]
                    try:
                        new_rat = input("    New rationale (Enter to keep): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if new_rat:
                        r["rationale"] = new_rat
                    print(f"    → Updated to {r['alignment_label']} ({r['label_text']})")
            else:
                print(f"    Invalid index. Use 1-{len(records)}.")
        else:
            print("    Enter a number or 'done'.")

    return records


def save_ground_truth(records: list[dict], path: Path) -> None:
    """Write ground truth records to JSON."""
    # Sort by objective then action for readability
    records.sort(key=lambda r: (r["objective_code"], r["action_number"]))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(records)} labelled pairs to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or review ground truth labels for strategy–action alignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Labelling Scale:
  1.0  Strongly Aligned   — directly implements the objective
  0.7  Aligned            — clearly supports the objective
  0.4  Weakly Aligned     — tangential / indirect relationship
  0.0  Not Aligned        — no meaningful relationship

Examples:
  python -m tests.create_ground_truth                 # batch mode (accept pre-seeded)
  python -m tests.create_ground_truth --interactive   # label each pair interactively
  python -m tests.create_ground_truth --review        # review and optionally edit
        """,
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactively label each pair (shows context, asks for input).",
    )
    parser.add_argument(
        "--review", "-r",
        action="store_true",
        help="Show all pre-seeded labels and allow selective editing.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output path for ground truth JSON (default: {OUTPUT_FILE}).",
    )
    args = parser.parse_args()

    # Load upstream data
    print("Loading strategic plan and action plan...")
    objectives = load_objectives()
    actions = load_actions()
    print(f"  {len(objectives)} objectives, {len(actions)} actions loaded.")

    # Build initial records from pre-seeded labels
    records = build_ground_truth(objectives, actions, PRE_SEEDED)

    # Remove duplicates that share the same (obj, act) key
    seen: set[tuple[str, int]] = set()
    deduped = []
    for r in records:
        key = (r["objective_code"], r["action_number"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    records = deduped

    print(f"  {len(records)} pre-seeded pairs prepared.")

    # Mode dispatch
    if args.interactive:
        records = interactive_label(objectives, actions, records)
    elif args.review:
        records = review_mode(records)
    else:
        print("\nBatch mode: accepting all pre-seeded expert labels.")

    # Summary
    print_summary(records)

    # Save
    save_ground_truth(records, args.output)

    # Validation
    if len(records) >= 50:
        print(f"\nStatistical significance check: {len(records)} >= 50 pairs. PASS")
    else:
        print(f"\nWARNING: Only {len(records)} pairs. Add more for statistical "
              f"significance (target: 50+).")


if __name__ == "__main__":
    main()
