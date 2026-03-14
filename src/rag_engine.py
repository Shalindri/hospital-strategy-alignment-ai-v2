"""
rag_engine.py — RAG pipeline: retrieve relevant objectives from ChromaDB, then generate
improvement suggestions and new action proposals via GPT-4o-mini.
"""

from src.config import get_openai_client, OPENAI_MODEL, THRESHOLD_GOOD, OBJECTIVES_COLLECTION
from src import vector_store


def get_improvement_suggestion(action: dict, top_k_results: list) -> str:
    """
    Ask GPT-4o-mini for 3 improvements to a poorly-aligned action.

    Args:
        action: Action dict (id, title, description).
        top_k_results: Retrieved objective chunks from vector_store.query().

    Returns:
        LLM suggestion text.
    """
    context = "\n".join(f"[{r['id']}] {r['document']}" for r in top_k_results)

    prompt = f"""You are a hospital strategy consultant helping align operational actions
with strategic objectives.

Strategic objective context:
{context}

Poorly-aligned action:
Title: {action['title']}
Description: {action.get('description', '')}

Suggest exactly 3 specific, actionable improvements:
1. [Title]: [Description]
2. [Title]: [Description]
3. [Title]: [Description]"""

    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"API call failed for [{action['id']}]: {e}")


def generate_new_action_proposal(objective: dict) -> str:
    """
    Propose one new action for an objective with no Excellent-tier action.

    Args:
        objective: Objective dict (id, title, description).

    Returns:
        Proposed action text.
    """
    prompt = f"""You are a hospital strategy consultant reviewing an action plan.

This objective has NO action that strongly supports it:
[{objective['id']}] {objective['title']}
{objective.get('description', '')}

Propose ONE specific new action that would directly operationalise this objective.
Format:
Proposed Action Title: [title]
Description: [2-3 sentences]"""

    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"API call failed for [{objective['id']}]: {e}")


def run_rag_suggestions(alignment_result: dict) -> dict:
    """
    Generate improvement suggestions for poorly-aligned actions and new action
    proposals for objectives with no Excellent-tier action.

    Returns:
        {
            "improvement_suggestions": {action_id: suggestion_text},
            "new_action_proposals":    {objective_id: proposal_text}
        }
    """
    print("\n" + "="*60)
    print("  RUNNING RAG ENGINE")
    print("="*60)

    classifications = alignment_result["classifications"]
    objectives      = alignment_result["objectives"]
    actions         = alignment_result["actions"]

    # Part A — improvements for poorly-aligned actions
    best_scores = {}
    for c in classifications:
        aid = c["action_id"]
        if aid not in best_scores or c["score"] > best_scores[aid]:
            best_scores[aid] = c["score"]

    poor_actions = [a for a in actions if best_scores.get(a["id"], 0.0) < THRESHOLD_GOOD]
    print(f"\n📌 Part A: {len(poor_actions)} action(s) need improvement...")

    improvement_suggestions = {}
    for act in poor_actions:
        print(f"   Generating for [{act['id']}]: {act['title'][:50]}...")
        query_text  = act["title"] + " " + act.get("description", "")
        top_results = vector_store.query(query_text, OBJECTIVES_COLLECTION, top_k=3)

        if not top_results:
            improvement_suggestions[act["id"]] = "No objectives in ChromaDB. Run embed_and_store first."
            continue

        try:
            improvement_suggestions[act["id"]] = get_improvement_suggestion(act, top_results)
            print(f"   ✅ Done [{act['id']}].")
        except RuntimeError as e:
            improvement_suggestions[act["id"]] = f"Error: {e}"

    # Part B — new proposals for uncovered objectives
    excellent_obj_ids = {c["objective_id"] for c in classifications if c["tier"] == "Excellent"}
    uncovered = [o for o in objectives if o["id"] not in excellent_obj_ids]
    print(f"\n📌 Part B: {len(uncovered)} objective(s) lack an Excellent action...")

    new_action_proposals = {}
    for obj in uncovered:
        print(f"   Generating for [{obj['id']}]: {obj['title'][:50]}...")
        try:
            new_action_proposals[obj["id"]] = generate_new_action_proposal(obj)
            print(f"   ✅ Done [{obj['id']}].")
        except RuntimeError as e:
            new_action_proposals[obj["id"]] = f"Error: {e}"

    print(f"\n✅ RAG complete. {len(improvement_suggestions)} suggestions, {len(new_action_proposals)} proposals.")

    return {
        "improvement_suggestions": improvement_suggestions,
        "new_action_proposals":    new_action_proposals,
    }
