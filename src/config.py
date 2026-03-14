"""
config.py
---------
Central configuration: all constants, thresholds, and model names for ISPS.

How it fits in the pipeline:
  [this module] → every other src/ module imports from here

Beginner note: Never scatter "magic numbers" (like 0.75) across your code.
Put them all in one place so you only need to change one file to tune the system.
"""

# --- Standard library ---
import os

# --- Third-party ---
from dotenv import load_dotenv   # reads values from the .env file
from openai import OpenAI        # official OpenAI Python client

# Load .env file so OPENAI_API_KEY is available via os.getenv()
load_dotenv()

# =============================================================================
# ALIGNMENT SCORE THRESHOLDS
# Cosine similarity ranges from 0.0 (no relation) to 1.0 (identical meaning).
# =============================================================================
THRESHOLD_EXCELLENT = 0.70   # ≥ 0.70 → action directly operationalises strategy
THRESHOLD_GOOD      = 0.50   # ≥ 0.50 → clear strategic support
THRESHOLD_FAIR      = 0.53   # ≥ 0.39 → partial / indirect alignment (tuned via threshold sweep, F1: 0.667 → 0.833)
ORPHAN_THRESHOLD    = 0.53   # action scores < 0.39 against EVERY objective → orphan (kept equal to THRESHOLD_FAIR)

# =============================================================================
# MODEL NAMES
# =============================================================================
#EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model (384-dim)
EMBEDDING_MODEL = "all-mpnet-base-v2"
OPENAI_MODEL    = "gpt-4o-mini"         # OpenAI chat model for all LLM calls

# =============================================================================
# FILE & FOLDER PATHS  (relative to the project root)
# =============================================================================
DATA_DIR    = "data"       # JSON data files
CHROMA_DIR  = "chroma_db"  # ChromaDB persistent storage
OUTPUTS_DIR = "outputs"    # Generated outputs (HTML, TTL, JSON, ...)

STRATEGIC_PLAN_FILE = os.path.join(DATA_DIR, "strategic_plan.json")
ACTION_PLAN_FILE    = os.path.join(DATA_DIR, "action_plan.json")

# ChromaDB collection names (think of these like table names)
OBJECTIVES_COLLECTION = "strategic_objectives"
ACTIONS_COLLECTION    = "action_plan_items"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_openai_client() -> OpenAI:
    """
    Create and return a configured OpenAI client.

    Reads OPENAI_API_KEY from the environment (set by load_dotenv above).

    Returns:
        openai.OpenAI: Ready-to-use OpenAI client.

    Raises:
        ValueError: With a clear message if the API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set.\n"
            "Create a file called .env in the project root and add:\n"
            "  OPENAI_API_KEY=sk-your-key-here"
        )

    return OpenAI(api_key=api_key)


def classify_score(score: float) -> str:
    """
    Convert a numeric cosine-similarity score into a human-readable tier label.

    Args:
        score (float): A similarity score between 0.0 and 1.0.

    Returns:
        str: "Excellent", "Good", "Fair", or "Poor".
    """
    if score >= THRESHOLD_EXCELLENT:
        return "Excellent"
    elif score >= THRESHOLD_GOOD:
        return "Good"
    elif score >= THRESHOLD_FAIR:
        return "Fair"
    else:
        return "Poor"
