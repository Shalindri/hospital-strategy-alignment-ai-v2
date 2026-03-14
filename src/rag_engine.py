"""
rag_engine.py
-------------
Retrieval-Augmented Generation (RAG) pipeline for alignment improvement suggestions.

How it fits in the pipeline:
  alignment_scorer → [this module] → agent_reasoner / dashboard

Beginner note:
  RAG = Retrieve then Generate.

  Step 1 — RETRIEVE: query ChromaDB for the most relevant strategic objective
           chunks for a given poorly-aligned action.

  Step 2 — GENERATE: send those retrieved chunks + the action text to GPT-4o-mini
           and ask it to suggest concrete improvements.

  Why RAG instead of just prompting the LLM?
  Because the LLM's training data does NOT include YOUR hospital's specific
  strategic objectives. RAG gives the LLM exactly the right context from YOUR
  documents, making suggestions much more relevant.
"""

# --- Standard library ---
import json

# --- Local ---
from src.config import (
    get_openai_client,
    OPENAI_MODEL,
    THRESHOLD_GOOD,
    THRESHOLD_EXCELLENT,
    OBJECTIVES_COLLECTION,
)
from src import vector_store


# =============================================================================
# HELPER: IMPROVEMENT SUGGESTION VIA RAG
# =============================================================================

def get_improvement_suggestion(action: dict, top_k_results: list) -> str:
    """
    Ask GPT-4o-mini to suggest 3 improvements for a poorly-aligned action.

    The retrieved objective chunks are provided as context so the LLM
    grounds its suggestions in the actual strategic plan.

    Args:
        action (dict): The poorly-aligned action (id, title, description).
        top_k_results (list): Top-K results from vector_store.query() —
                              each has keys: id, document, distance, metadata.

    Returns:
        str: The LLM's 3 improvement suggestions as plain text.

    Raises:
        RuntimeError: If the OpenAI API call fails.
    """
    # Step 1 — Build the context string from retrieved objective chunks
    # Join the top-K retrieved documents into one block of text
    context_parts = []
    for result in top_k_results:
        obj_id  = result["id"]
        obj_text = result["document"]
        context_parts.append(f"[{obj_id}] {obj_text}")

    context = "\n".join(context_parts)

    # Step 2 — Build the prompt
    prompt = f"""You are a hospital strategy consultant helping align operational actions
with strategic objectives.

Strategic objective context (retrieved from the strategic plan):
{context}

This annual action plan item has POOR alignment with the strategic objectives:
Title: {action['title']}
Description: {action.get('description', 'No description provided.')}

Please suggest exactly 3 specific, actionable improvements to better align
this action with the strategic objectives above. Format your response as:

1. [Improvement title]: [Specific description of what to change and why it helps]
2. [Improvement title]: [Specific description of what to change and why it helps]
3. [Improvement title]: [Specific description of what to change and why it helps]"""

    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model       = OPENAI_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.3,   # slightly creative but still focused
        )
        suggestion = response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(
            f"OpenAI API call failed for action [{action['id']}]: {e}\n"
            "Check your OPENAI_API_KEY and internet connection."
        )

    return suggestion


# =============================================================================
# HELPER: NEW ACTION PROPOSAL
# =============================================================================

def generate_new_action_proposal(objective: dict) -> str:
    """
    For an objective with no Excellent-tier action, propose one new action.

    This helps decision-makers identify gaps in the action plan.

    Args:
        objective (dict): Objective dict (id, title, description).

    Returns:
        str: A proposed new action as plain text.
    """
    prompt = f"""You are a hospital strategy consultant reviewing a hospital's annual action plan.

The following strategic objective currently has NO action plan item that directly
and strongly supports it (no "Excellent" alignment):

Strategic Objective [{objective['id']}]:
Title: {objective['title']}
Description: {objective.get('description', 'No description provided.')}

Propose ONE specific, concrete new action plan item that would directly and
strongly operationalise this strategic objective. Format your response as:

Proposed Action Title: [short, clear title]
Description: [2-3 sentences describing what the action involves, who is responsible,
and how it directly advances the objective above]"""

    try:
        client   = get_openai_client()
        response = client.chat.completions.create(
            model       = OPENAI_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.4,
        )
        proposal = response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(
            f"OpenAI API call failed for objective [{objective['id']}]: {e}\n"
            "Check your OPENAI_API_KEY and internet connection."
        )

    return proposal


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_rag_suggestions(alignment_result: dict) -> dict:
    """
    Generate RAG improvement suggestions for poor actions and new action proposals
    for uncovered objectives.

    Poor action = best alignment score < THRESHOLD_GOOD across all objectives.
    Uncovered objective = no action has "Excellent" tier for that objective.

    Args:
        alignment_result (dict): Output from alignment_scorer.run_alignment().
                                 Must have keys: classifications, objectives, actions.

    Returns:
        dict: {
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

    # -------------------------------------------------------------------------
    # PART A: Improvement suggestions for poorly-aligned actions
    # -------------------------------------------------------------------------
    print("\n📌 Part A: Generating improvement suggestions for poor actions...")

    # Find each action's best score across all objectives
    # best_scores: {action_id: float}
    best_scores: dict = {}
    for c in classifications:
        act_id = c["action_id"]
        if act_id not in best_scores or c["score"] > best_scores[act_id]:
            best_scores[act_id] = c["score"]

    # Actions whose best score is below THRESHOLD_GOOD need improvement
    poor_actions = [
        act for act in actions
        if best_scores.get(act["id"], 0.0) < THRESHOLD_GOOD
    ]

    print(f"   Found {len(poor_actions)} action(s) with best score < {THRESHOLD_GOOD}")

    improvement_suggestions: dict = {}

    for act in poor_actions:
        print(f"\n   Generating suggestion for action [{act['id']}]: {act['title'][:50]}...")

        # Retrieve top-3 most relevant objectives from ChromaDB
        query_text = act["title"] + " " + act.get("description", "")
        top_results = vector_store.query(
            text            = query_text,
            collection_name = OBJECTIVES_COLLECTION,
            top_k           = 3,
        )

        if not top_results:
            print(f"   ⚠️  No objectives in ChromaDB — skipping RAG for [{act['id']}]")
            improvement_suggestions[act["id"]] = (
                "Could not generate suggestion: no objectives indexed in ChromaDB. "
                "Run vector_store.embed_and_store() on objectives first."
            )
            continue

        # Call GPT-4o-mini with the retrieved context
        try:
            suggestion = get_improvement_suggestion(act, top_results)
            improvement_suggestions[act["id"]] = suggestion
            print(f"   ✅ Suggestion generated for [{act['id']}].")
        except RuntimeError as e:
            print(f"   ❌ Error: {e}")
            improvement_suggestions[act["id"]] = f"Error generating suggestion: {e}"

    # -------------------------------------------------------------------------
    # PART B: New action proposals for uncovered objectives
    # -------------------------------------------------------------------------
    print("\n📌 Part B: Generating new action proposals for uncovered objectives...")

    # Find objectives with no Excellent-tier action
    # Build a set of objective IDs that have at least one "Excellent" action
    excellent_objective_ids = {
        c["objective_id"]
        for c in classifications
        if c["tier"] == "Excellent"
    }

    uncovered_objectives = [
        obj for obj in objectives
        if obj["id"] not in excellent_objective_ids
    ]

    print(f"   Found {len(uncovered_objectives)} objective(s) with no Excellent-tier action.")

    new_action_proposals: dict = {}

    for obj in uncovered_objectives:
        print(f"\n   Generating proposal for objective [{obj['id']}]: {obj['title'][:50]}...")

        try:
            proposal = generate_new_action_proposal(obj)
            new_action_proposals[obj["id"]] = proposal
            print(f"   ✅ Proposal generated for [{obj['id']}].")
        except RuntimeError as e:
            print(f"   ❌ Error: {e}")
            new_action_proposals[obj["id"]] = f"Error generating proposal: {e}"

    print("\n✅ RAG engine complete.")
    print(f"   Generated {len(improvement_suggestions)} improvement suggestions.")
    print(f"   Generated {len(new_action_proposals)} new action proposals.")

    return {
        "improvement_suggestions": improvement_suggestions,
        "new_action_proposals":    new_action_proposals,
    }
