"""
config.py — Central configuration: thresholds, model names, paths, and helpers.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Alignment score thresholds (cosine similarity 0.0–1.0)
THRESHOLD_EXCELLENT = 0.70
THRESHOLD_GOOD      = 0.60
THRESHOLD_FAIR      = 0.47
ORPHAN_THRESHOLD    = 0.47

# Models
EMBEDDING_MODEL = "all-mpnet-base-v2"
OPENAI_MODEL    = "gpt-4o-mini"

# Paths
DATA_DIR    = "data"
CHROMA_DIR  = "chroma_db"
OUTPUTS_DIR = "outputs"

STRATEGIC_PLAN_FILE = os.path.join(DATA_DIR, "strategic_plan.json")
ACTION_PLAN_FILE    = os.path.join(DATA_DIR, "action_plan.json")

OBJECTIVES_COLLECTION = "strategic_objectives"
ACTIONS_COLLECTION    = "action_plan_items"


def get_openai_client() -> OpenAI:
    """Return a configured OpenAI client. Raises ValueError if API key is missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set.\n"
            "Add OPENAI_API_KEY=sk-your-key-here to a .env file in the project root."
        )
    return OpenAI(api_key=api_key)


def classify_score(score: float) -> str:
    """Return tier label for a cosine similarity score."""
    if score >= THRESHOLD_EXCELLENT:
        return "Excellent"
    elif score >= THRESHOLD_GOOD:
        return "Good"
    elif score >= THRESHOLD_FAIR:
        return "Fair"
    return "Poor"
