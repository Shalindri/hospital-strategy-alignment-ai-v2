"""
agent_reasoner.py
-----------------
Agentic AI reasoning layer: a 3-iteration Plan → Act → Reflect loop.

How it fits in the pipeline:
  rag_engine → [this module] → dashboard

Beginner note:
  An "agentic" system iteratively reasons, acts, and self-evaluates.
  We implement 3 iterations using plain Python + OpenAI API calls:

    Iteration 1 — PLAN:    LLM reads all orphan/poor actions and decides
                           which 3 are the most critical to fix.
    Iteration 2 — ACT:     For each critical action, call rag_engine to
                           retrieve context and generate improvement suggestions.
    Iteration 3 — REFLECT: LLM reads each suggestion, scores it 1–10,
                           and notes whether more work is needed.

  No extra frameworks needed — just functions and for loops!
"""

# --- Standard library ---
import json

# --- Local ---
from src.config import get_openai_client, OPENAI_MODEL, THRESHOLD_GOOD
from src import rag_engine, vector_store
from src.config import OBJECTIVES_COLLECTION


# =============================================================================
# ITERATION 1 — PLAN
# =============================================================================

def plan_iteration(alignment_result: dict) -> dict:
    """
    PLAN: Ask the LLM which poor/orphan actions are most critical to fix.

    We pass a summary of all poorly-aligned actions to the LLM and ask it
    to prioritise. This mirrors what a human consultant would do first —
    triage, then act.

    Args:
        alignment_result (dict): Output from alignment_scorer.run_alignment().

    Returns:
        dict: {
            "critical_action_ids": [list of str action IDs],
            "reasoning": str  (LLM's explanation of why these were chosen)
        }
    """
    print("\n" + "="*60)
    print("  === PLAN ===")
    print("="*60)
    print("📌 Asking LLM to prioritise which poor actions to fix...\n")

    classifications = alignment_result["classifications"]
    actions         = alignment_result["actions"]

    # Build a lookup: action_id → best score across all objectives
    best_scores: dict = {}
    for c in classifications:
        act_id = c["action_id"]
        if act_id not in best_scores or c["score"] > best_scores[act_id]:
            best_scores[act_id] = c["score"]

    # Collect all actions scoring below THRESHOLD_GOOD
    poor_actions = [
        act for act in actions
        if best_scores.get(act["id"], 0.0) < THRESHOLD_GOOD
    ]

    # Also include explicit orphans
    orphan_ids = {act["id"] for act in alignment_result.get("orphan_actions", [])}

    if not poor_actions:
        print("   ✅ No poor/orphan actions found. Nothing to prioritise.")
        return {
            "critical_action_ids": [],
            "reasoning":           "All actions are well-aligned (Good or better).",
        }

    # Build a human-readable summary for the LLM
    action_lines = []
    for act in poor_actions:
        score = best_scores.get(act["id"], 0.0)
        tag   = " [ORPHAN]" if act["id"] in orphan_ids else ""
        action_lines.append(
            f"- [{act['id']}]{tag} \"{act['title']}\" (best alignment score: {score:.3f})"
        )

    action_summary = "\n".join(action_lines)

    prompt = f"""You are a hospital strategy consultant reviewing an annual action plan
that is partially misaligned with the hospital's strategic plan.

Here are the poorly-aligned action plan items (sorted by alignment score):
{action_summary}

Task: Identify the 3 MOST CRITICAL actions to improve first. Consider which
actions, if improved, would have the greatest strategic impact.

Respond ONLY with valid JSON (no extra text):
{{
  "critical_action_ids": ["A1", "A5", "A12"],
  "reasoning": "Brief explanation of why these 3 were chosen."
}}"""

    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model       = OPENAI_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.2,
        )
        response_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if the LLM added them
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        plan_result = json.loads(response_text)

    except json.JSONDecodeError:
        # If the LLM returns invalid JSON, fall back to the top 3 by score
        print("   ⚠️  LLM returned invalid JSON. Falling back to top 3 worst actions.")
        sorted_poor = sorted(poor_actions, key=lambda a: best_scores.get(a["id"], 0.0))
        plan_result = {
            "critical_action_ids": [a["id"] for a in sorted_poor[:3]],
            "reasoning":           "Fallback: selected the 3 actions with lowest alignment scores.",
        }
    except Exception as e:
        print(f"   ❌ OpenAI API error during PLAN: {e}")
        sorted_poor = sorted(poor_actions, key=lambda a: best_scores.get(a["id"], 0.0))
        plan_result = {
            "critical_action_ids": [a["id"] for a in sorted_poor[:3]],
            "reasoning":           f"Fallback due to API error: {str(e)[:100]}",
        }

    print(f"   Critical actions selected: {plan_result['critical_action_ids']}")
    print(f"   Reasoning: {plan_result['reasoning']}")
    return plan_result


# =============================================================================
# ITERATION 2 — ACT
# =============================================================================

def act_iteration(critical_action_ids: list, alignment_result: dict) -> dict:
    """
    ACT: For each critical action, use RAG to generate improvement suggestions.

    This is where we actually DO something — retrieve context from ChromaDB
    and ask the LLM to write specific improvements for each critical action.

    Args:
        critical_action_ids (list): Action IDs from the PLAN step.
        alignment_result (dict): Full alignment results for context.

    Returns:
        dict: {action_id: suggestion_text}
    """
    print("\n" + "="*60)
    print("  === ACT ===")
    print("="*60)
    print(f"📌 Generating improvements for {len(critical_action_ids)} critical action(s)...\n")

    # Build a lookup: action_id → action dict
    action_lookup = {act["id"]: act for act in alignment_result["actions"]}

    suggestions: dict = {}

    for act_id in critical_action_ids:
        act = action_lookup.get(act_id)
        if act is None:
            print(f"   ⚠️  Action [{act_id}] not found in alignment result. Skipping.")
            continue

        print(f"\n   → Acting on [{act_id}]: {act['title'][:55]}...")

        # Retrieve top-3 relevant objectives from ChromaDB
        query_text  = act["title"] + " " + act.get("description", "")
        top_results = vector_store.query(
            text            = query_text,
            collection_name = OBJECTIVES_COLLECTION,
            top_k           = 3,
        )

        if not top_results:
            print(f"   ⚠️  ChromaDB has no objectives. Generating without retrieval context.")
            top_results = []

        try:
            suggestion = rag_engine.get_improvement_suggestion(act, top_results)
            suggestions[act_id] = suggestion
            print(f"   ✅ Suggestion ready for [{act_id}].")
            # Show a short preview
            preview = suggestion[:200].replace("\n", " ")
            print(f"   Preview: {preview}...")
        except RuntimeError as e:
            print(f"   ❌ Error for [{act_id}]: {e}")
            suggestions[act_id] = f"Could not generate suggestion: {e}"

    return suggestions


# =============================================================================
# ITERATION 3 — REFLECT
# =============================================================================

def reflect_iteration(suggestions: dict, alignment_result: dict) -> dict:
    """
    REFLECT: LLM scores each suggestion 1–10 and notes if more work is needed.

    This models how a human expert would review AI-generated recommendations
    before presenting them to decision-makers.

    Args:
        suggestions (dict): {action_id: suggestion_text} from the ACT step.
        alignment_result (dict): Full alignment context.

    Returns:
        dict: {action_id: {"score": int (1-10), "note": str, "needs_more_work": bool}}
    """
    print("\n" + "="*60)
    print("  === REFLECT ===")
    print("="*60)
    print("📌 LLM self-evaluating the quality of its suggestions...\n")

    if not suggestions:
        print("   No suggestions to reflect on.")
        return {}

    # Build a numbered list of all suggestions for the LLM to review
    suggestion_lines = []
    act_ids = list(suggestions.keys())

    for act_id, suggestion_text in suggestions.items():
        suggestion_lines.append(
            f"Action [{act_id}]:\n{suggestion_text[:400]}"
        )

    suggestions_block = "\n\n---\n\n".join(suggestion_lines)

    prompt = f"""You are a senior hospital strategy consultant reviewing AI-generated
improvement recommendations for poorly-aligned action plan items.

For each recommendation below, evaluate:
1. Specificity (is it concrete, not vague?)
2. Strategic relevance (does it connect to hospital strategy?)
3. Feasibility (is it realistic for a hospital to implement?)

Score each recommendation from 1 (poor) to 10 (excellent).

Recommendations to review:
{suggestions_block}

Respond ONLY with valid JSON (no extra text):
{{
  "reflections": [
    {{
      "action_id": "A1",
      "score": 8,
      "note": "Brief explanation of strengths and weaknesses",
      "needs_more_work": false
    }}
  ]
}}"""

    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model       = OPENAI_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.2,
        )
        response_text = response.choices[0].message.content.strip()

        # Strip markdown code fences
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        parsed = json.loads(response_text)
        reflections_list = parsed.get("reflections", [])

    except (json.JSONDecodeError, Exception) as e:
        print(f"   ⚠️  Could not parse LLM reflection response: {e}")
        # Fallback: give every suggestion a neutral score
        reflections_list = [
            {"action_id": aid, "score": 5, "note": "Auto-scored due to parse error.",
             "needs_more_work": False}
            for aid in act_ids
        ]

    # Convert list to dict keyed by action_id
    reflection_result: dict = {}
    for r in reflections_list:
        aid = r.get("action_id", "unknown")
        reflection_result[aid] = {
            "score":          r.get("score", 5),
            "note":           r.get("note", ""),
            "needs_more_work": r.get("needs_more_work", False),
        }
        score  = reflection_result[aid]["score"]
        note   = reflection_result[aid]["note"][:80]
        flag   = " ⚠️ needs work" if reflection_result[aid]["needs_more_work"] else ""
        print(f"   [{aid}] Score: {score}/10{flag}  — {note}")

    return reflection_result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_agent_reasoning(alignment_result: dict) -> dict:
    """
    Run the full 3-iteration Plan → Act → Reflect agentic loop.

    Args:
        alignment_result (dict): Output from alignment_scorer.run_alignment().

    Returns:
        dict: {
            "plan":    {"critical_action_ids": [...], "reasoning": str},
            "act":     {action_id: suggestion_text},
            "reflect": {action_id: {"score": int, "note": str, "needs_more_work": bool}},
            "summary": str  (high-level human-readable summary)
        }
    """
    print("\n" + "="*60)
    print("  RUNNING AGENT REASONER (Plan → Act → Reflect)")
    print("="*60)

    # Iteration 1 — PLAN
    plan_result = plan_iteration(alignment_result)

    # Iteration 2 — ACT
    act_result = act_iteration(
        critical_action_ids = plan_result["critical_action_ids"],
        alignment_result    = alignment_result,
    )

    # Iteration 3 — REFLECT
    reflect_result = reflect_iteration(act_result, alignment_result)

    # Build a plain-English summary
    n_critical = len(plan_result["critical_action_ids"])
    n_improved = len(act_result)
    avg_score  = (
        sum(r["score"] for r in reflect_result.values()) / len(reflect_result)
        if reflect_result else 0.0
    )
    needs_more = sum(1 for r in reflect_result.values() if r.get("needs_more_work"))

    summary = (
        f"Agent reasoning complete. "
        f"Identified {n_critical} critical action(s), "
        f"generated {n_improved} improvement suggestion(s). "
        f"Average suggestion quality score: {avg_score:.1f}/10. "
        f"{needs_more} suggestion(s) flagged as needing more work."
    )

    print("\n" + "="*60)
    print("  AGENT REASONING COMPLETE")
    print("="*60)
    print(f"  {summary}")

    return {
        "plan":    plan_result,
        "act":     act_result,
        "reflect": reflect_result,
        "summary": summary,
    }
