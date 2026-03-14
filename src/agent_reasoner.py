"""
agent_reasoner.py — Agentic 3-iteration Plan → Act → Reflect loop using GPT-4o-mini.
"""

import json

from src.config import get_openai_client, OPENAI_MODEL, THRESHOLD_GOOD, OBJECTIVES_COLLECTION
from src import rag_engine, vector_store


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text


def plan_iteration(alignment_result: dict) -> dict:
    """PLAN: Ask the LLM which poor/orphan actions are most critical to fix."""
    print("\n" + "="*60)
    print("  === PLAN ===")
    print("="*60)

    classifications = alignment_result["classifications"]
    actions         = alignment_result["actions"]

    best_scores = {}
    for c in classifications:
        aid = c["action_id"]
        if aid not in best_scores or c["score"] > best_scores[aid]:
            best_scores[aid] = c["score"]

    poor_actions = [a for a in actions if best_scores.get(a["id"], 0.0) < THRESHOLD_GOOD]
    orphan_ids   = {a["id"] for a in alignment_result.get("orphan_actions", [])}

    if not poor_actions:
        return {"critical_action_ids": [], "reasoning": "All actions are well-aligned."}

    action_summary = "\n".join(
        f"- [{a['id']}]{' [ORPHAN]' if a['id'] in orphan_ids else ''} "
        f"\"{a['title']}\" (score: {best_scores.get(a['id'], 0):.3f})"
        for a in poor_actions
    )

    prompt = f"""You are a hospital strategy consultant.

Poorly-aligned actions:
{action_summary}

Pick the 3 MOST CRITICAL to improve. Return valid JSON only:
{{
  "critical_action_ids": ["A1", "A5", "A12"],
  "reasoning": "Brief explanation."
}}"""

    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        plan_result = json.loads(_strip_fences(response.choices[0].message.content.strip()))
    except (json.JSONDecodeError, Exception) as e:
        sorted_poor = sorted(poor_actions, key=lambda a: best_scores.get(a["id"], 0.0))
        plan_result = {
            "critical_action_ids": [a["id"] for a in sorted_poor[:3]],
            "reasoning": f"Fallback (top 3 lowest scores): {str(e)[:80]}",
        }

    print(f"   Critical: {plan_result['critical_action_ids']}")
    print(f"   Reason:   {plan_result['reasoning']}")
    return plan_result


def act_iteration(critical_action_ids: list, alignment_result: dict) -> dict:
    """ACT: Generate RAG improvement suggestions for each critical action."""
    print("\n" + "="*60)
    print("  === ACT ===")
    print("="*60)

    action_lookup = {a["id"]: a for a in alignment_result["actions"]}
    suggestions   = {}

    for act_id in critical_action_ids:
        act = action_lookup.get(act_id)
        if act is None:
            continue

        print(f"\n   → [{act_id}]: {act['title'][:55]}...")
        query_text  = act["title"] + " " + act.get("description", "")
        top_results = vector_store.query(query_text, OBJECTIVES_COLLECTION, top_k=3)

        try:
            suggestion = rag_engine.get_improvement_suggestion(act, top_results)
            suggestions[act_id] = suggestion
            print(f"   ✅ [{act_id}] done. Preview: {suggestion[:120].replace(chr(10), ' ')}...")
        except RuntimeError as e:
            suggestions[act_id] = f"Error: {e}"

    return suggestions


def reflect_iteration(suggestions: dict, alignment_result: dict) -> dict:
    """REFLECT: LLM scores each suggestion 1–10 and flags if more work is needed."""
    print("\n" + "="*60)
    print("  === REFLECT ===")
    print("="*60)

    if not suggestions:
        return {}

    act_ids = list(suggestions.keys())
    suggestions_block = "\n\n---\n\n".join(
        f"Action [{aid}]:\n{text[:400]}"
        for aid, text in suggestions.items()
    )

    prompt = f"""You are a senior hospital strategy consultant reviewing AI-generated
improvement recommendations.

Score each on specificity, strategic relevance, and feasibility (1–10).

{suggestions_block}

Return valid JSON only:
{{
  "reflections": [
    {{
      "action_id": "A1",
      "score": 8,
      "note": "Brief evaluation",
      "needs_more_work": false
    }}
  ]
}}"""

    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        parsed = json.loads(_strip_fences(response.choices[0].message.content.strip()))
        reflections_list = parsed.get("reflections", [])
    except Exception as e:
        reflections_list = [
            {"action_id": aid, "score": 5, "note": f"Auto-scored: {str(e)[:60]}", "needs_more_work": False}
            for aid in act_ids
        ]

    reflection_result = {}
    for r in reflections_list:
        aid = r.get("action_id", "unknown")
        reflection_result[aid] = {
            "score":           r.get("score", 5),
            "note":            r.get("note", ""),
            "needs_more_work": r.get("needs_more_work", False),
        }
        flag = " ⚠️ needs work" if reflection_result[aid]["needs_more_work"] else ""
        print(f"   [{aid}] {reflection_result[aid]['score']}/10{flag} — {reflection_result[aid]['note'][:80]}")

    return reflection_result


def run_agent_reasoning(alignment_result: dict) -> dict:
    """
    Run the full Plan → Act → Reflect agentic loop.

    Returns:
        {plan, act, reflect, summary}
    """
    print("\n" + "="*60)
    print("  RUNNING AGENT REASONER (Plan → Act → Reflect)")
    print("="*60)

    plan_result    = plan_iteration(alignment_result)
    act_result     = act_iteration(plan_result["critical_action_ids"], alignment_result)
    reflect_result = reflect_iteration(act_result, alignment_result)

    avg_score  = sum(r["score"] for r in reflect_result.values()) / len(reflect_result) if reflect_result else 0.0
    needs_more = sum(1 for r in reflect_result.values() if r.get("needs_more_work"))

    summary = (
        f"Identified {len(plan_result['critical_action_ids'])} critical action(s), "
        f"generated {len(act_result)} suggestion(s). "
        f"Avg quality: {avg_score:.1f}/10. {needs_more} flagged for more work."
    )

    print(f"\n✅ Agent complete. {summary}")

    return {"plan": plan_result, "act": act_result, "reflect": reflect_result, "summary": summary}
