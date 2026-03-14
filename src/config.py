"""
Shared configuration constants for the ISPS alignment system.

Centralises thresholds and defaults that are used across multiple modules
so they are defined in exactly one place.

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (ignored if not found, e.g. on Streamlit Cloud)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# On Streamlit Cloud, secrets are set via the dashboard UI and exposed as
# environment variables.  We also try st.secrets as a fallback.
def _get_secret(key: str, default: str = "") -> str:
    """Read a config value from env vars, falling back to st.secrets."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default

logger = logging.getLogger("config")

# ---------------------------------------------------------------------------
# Alignment classification thresholds
# ---------------------------------------------------------------------------
# Calibrated against the all-MiniLM-L6-v2 model's typical similarity range
# for hospital-domain documents (unrelated pairs: 0.05–0.20, strongly
# related pairs: 0.45–0.65).

THRESHOLD_EXCELLENT = 0.75   # Near-direct operationalisation of strategy
THRESHOLD_GOOD = 0.60        # Clear strategic support
THRESHOLD_FAIR = 0.42        # Partial or indirect alignment
ORPHAN_THRESHOLD = 0.42      # Below this for ALL objectives → orphan

# ---------------------------------------------------------------------------
# Default hospital name (used when no name is provided in data)
# ---------------------------------------------------------------------------
DEFAULT_HOSPITAL_NAME = "Hospital"

# ---------------------------------------------------------------------------
# OpenAI LLM Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY", "")
OPENAI_MODEL = _get_secret("OPENAI_MODEL", "gpt-4o-mini")

# Shared LLM parameters
LLM_TEMPERATURE = 0.2

# Agent configuration
MAX_ITERATIONS = 3
MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

class _OpenAIWrapper:
    """Wraps ChatOpenAI so .invoke(str) returns str (not AIMessage)."""

    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        msg = self._llm.invoke(prompt)
        return msg.content


def get_llm(temperature: float = LLM_TEMPERATURE):
    """Return a LangChain ChatOpenAI wrapper.

    The returned object exposes ``.invoke(prompt_str) -> str``.
    """
    from langchain_openai import ChatOpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or Streamlit Cloud secrets."
        )
    logger.info("Using OpenAI model: %s", OPENAI_MODEL)
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
    )
    return _OpenAIWrapper(llm)
