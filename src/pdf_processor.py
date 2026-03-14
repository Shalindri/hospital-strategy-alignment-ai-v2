"""
PDF Processor for the ISPS system.

Extracts text from uploaded PDF files and uses an LLM (OpenAI GPT-4o-mini)
to parse the content into the structured JSON schema expected by the
downstream embedding and alignment pipeline.

Two LLM prompts are used:
  1. Strategic plan extraction -> objectives with goals, KPIs, etc.
  2. Action plan extraction   -> action items with titles, owners, budgets, etc.

Typical usage (from the Streamlit dashboard)::

    from src.pdf_processor import extract_strategic_plan_from_pdf, extract_action_plan_from_pdf

    strategic_data = extract_strategic_plan_from_pdf(uploaded_bytes)
    action_data    = extract_action_plan_from_pdf(uploaded_bytes)

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber

from src.config import get_llm

logger = logging.getLogger("pdf_processor")

# Maximum characters of PDF text to send to the LLM.
# GPT-4o-mini has a 128K context window — plenty for full documents.
_MAX_TEXT_CHARS = 100_000


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF file given as bytes.

    Uses pdfplumber which handles multi-column layouts, tables, and
    embedded fonts better than simpler libraries.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        Concatenated text from all pages.
    """
    import io
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    full_text = "\n\n".join(text_parts)
    logger.info("Extracted %d characters from PDF (%d pages).",
                len(full_text), len(text_parts))
    return full_text


# ---------------------------------------------------------------------------
# LLM helper (provider-agnostic)
# ---------------------------------------------------------------------------

def _query_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM and return the response."""
    llm = get_llm(temperature=0.1)
    logger.info("Calling LLM ...")
    response = llm.invoke(prompt)
    return response


def _clean_json_string(raw: str) -> str:
    """Fix common LLM JSON mistakes: trailing commas, unescaped chars."""
    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return raw


def _extract_json_from_response(text: str) -> dict | list:
    """Find and parse the first JSON object or array in LLM output."""
    logger.debug("LLM response length: %d chars", len(text))

    # Try to find JSON block in markdown code fences
    match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if match:
        try:
            return json.loads(_clean_json_string(match.group(1)))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching closing brace/bracket
        depth = 0
        last_valid = -1
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
            if depth == 0:
                last_valid = i
                break

        if last_valid > start:
            candidate = text[start:last_valid + 1]
            try:
                return json.loads(_clean_json_string(candidate))
            except json.JSONDecodeError:
                pass

    # Last resort: try to repair truncated JSON by closing open brackets
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        candidate = text[start:]
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if open_braces > 0 or open_brackets > 0:
            # Remove any trailing partial entry (incomplete key-value)
            candidate = re.sub(r',\s*"[^"]*"?\s*:?\s*[^}\]]*$', "", candidate)
            candidate += "]" * open_brackets + "}" * open_braces
            candidate = _clean_json_string(candidate)
            try:
                result = json.loads(candidate)
                logger.warning("Repaired truncated JSON from LLM output.")
                return result
            except json.JSONDecodeError:
                pass

    raise ValueError("No valid JSON found in LLM response")


# ---------------------------------------------------------------------------
# Strategic plan extraction
# ---------------------------------------------------------------------------

_STRATEGIC_PROMPT = """\
You are an expert at analyzing strategic plans. Read the following document \
text and extract the strategic objectives in JSON format.

Return a JSON object with this EXACT structure:
{{
  "metadata": {{
    "title": "<document title>",
    "period": "<planning period, e.g. 2026-2030>",
    "version": "1.0"
  }},
  "vision": "<vision statement if present, else empty string>",
  "mission": "<mission statement if present, else empty string>",
  "objectives": [
    {{
      "code": "<letter A, B, C, etc.>",
      "name": "<objective name>",
      "goal_statement": "<main goal description>",
      "strategic_goals": [
        {{"id": "<e.g. A1>", "description": "<goal description>"}}
      ],
      "kpis": [
        {{"KPI": "<kpi name>", "Baseline": "<current value>", "Target": "<target value>"}}
      ],
      "keywords": ["<keyword1>", "<keyword2>"]
    }}
  ]
}}

Rules:
- Extract ALL strategic objectives found in the document
- Assign letter codes (A, B, C, ...) if not explicitly labeled
- Extract as many strategic goals and KPIs as mentioned
- Keywords should be 5-10 domain-relevant terms per objective
- Return ONLY valid JSON, no other text

DOCUMENT TEXT:
{text}
"""

_ACTION_PROMPT = """\
You are an expert at analyzing action plans. Read the following document \
text and extract the action items in JSON format.

Return a JSON object with this EXACT structure:
{{
  "metadata": {{
    "title": "<document title>",
    "period": "<action plan period, e.g. 2025>",
    "version": "1.0"
  }},
  "actions": [
    {{
      "action_number": <integer>,
      "title": "<action title>",
      "strategic_objective_code": "<letter matching the strategic objective>",
      "strategic_objective_name": "<objective name>",
      "description": "<full description of the action>",
      "action_owner": "<responsible person/department>",
      "timeline": "<e.g. Q1-Q4 2025>",
      "quarters": ["Q1", "Q2"],
      "budget_lkr_millions": <number or 0>,
      "budget_raw": "<original budget text>",
      "expected_outcome": "<expected result>",
      "kpis": ["<kpi1>", "<kpi2>"],
      "keywords": ["<keyword1>", "<keyword2>"]
    }}
  ]
}}

Rules:
- Extract ALL action items found in the document
- Number them sequentially starting from 1 if not explicitly numbered
- Map each action to its strategic objective using the letter code
- Extract budget as a number in millions if possible (set 0 if not mentioned)
- Extract timeline quarters (Q1, Q2, Q3, Q4) when mentioned
- Keywords should be 5-8 domain-relevant terms per action
- Return ONLY valid JSON, no other text

DOCUMENT TEXT:
{text}
"""


def extract_strategic_plan_from_pdf(
    pdf_bytes: bytes,
) -> dict[str, Any]:
    """Extract strategic plan structure from a PDF using LLM.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        Structured strategic plan dict.

    Raises:
        ValueError: If LLM output cannot be parsed as JSON.
    """
    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise ValueError("PDF appears to be empty or image-only.")

    truncated = text[:_MAX_TEXT_CHARS]
    prompt = _STRATEGIC_PROMPT.format(text=truncated)

    logger.info("Sending strategic plan to LLM for parsing (%d / %d chars)...",
                len(truncated), len(text))
    response = _query_llm(prompt)
    logger.info("LLM strategic plan response length: %d chars", len(response))
    result = _extract_json_from_response(response)

    # Flexible key detection - LLM may use different key names
    objectives_list = None
    if isinstance(result, dict):
        for key in ("objectives", "strategic_objectives"):
            if key in result and isinstance(result[key], list):
                objectives_list = result[key]
                break
        if objectives_list is None:
            for key, val in result.items():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    logger.warning("Using unexpected key '%s' as objectives list.", key)
                    objectives_list = val
                    break

    if objectives_list is not None:
        result["objectives"] = objectives_list
        n_obj = len(objectives_list)
        logger.info("LLM extracted %d strategic objectives.", n_obj)
        for obj in objectives_list:
            obj.setdefault("strategic_goals", [])
            obj.setdefault("kpis", [])
            obj.setdefault("keywords", [])
            obj.setdefault("goal_statement", obj.get("name", ""))
        return result

    logger.error("LLM response keys: %s",
                 list(result.keys()) if isinstance(result, dict) else type(result).__name__)
    raise ValueError(
        "LLM response does not contain expected 'objectives' key. "
        f"Got keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}"
    )


def extract_action_plan_from_pdf(
    pdf_bytes: bytes,
) -> dict[str, Any]:
    """Extract action plan structure from a PDF using LLM.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        Structured action plan dict.

    Raises:
        ValueError: If LLM output cannot be parsed as JSON.
    """
    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise ValueError("PDF appears to be empty or image-only.")

    truncated = text[:_MAX_TEXT_CHARS]
    prompt = _ACTION_PROMPT.format(text=truncated)

    logger.info("Sending action plan to LLM for parsing (%d / %d chars)...",
                len(truncated), len(text))
    response = _query_llm(prompt)
    logger.info("LLM action plan response length: %d chars", len(response))
    result = _extract_json_from_response(response)

    # Flexible key detection - LLM may use different key names
    actions_list = None
    if isinstance(result, dict):
        for key in ("actions", "action_items", "action_plan", "items"):
            if key in result and isinstance(result[key], list):
                actions_list = result[key]
                break
        if actions_list is None:
            for key, val in result.items():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    logger.warning("Using unexpected key '%s' as actions list.", key)
                    actions_list = val
                    break
    elif isinstance(result, list) and len(result) > 0:
        actions_list = result
        result = {"actions": actions_list}

    if actions_list is not None:
        result["actions"] = actions_list
        n_act = len(actions_list)
        logger.info("LLM extracted %d action items.", n_act)
        for idx, act in enumerate(actions_list):
            act.setdefault("action_number", idx + 1)
            act.setdefault("description", act.get("title", ""))
            act.setdefault("expected_outcome", "")
            act.setdefault("kpis", [])
            act.setdefault("keywords", [])
            act.setdefault("action_owner", "")
            act.setdefault("timeline", "")
            act.setdefault("quarters", [])
            act.setdefault("budget_lkr_millions", 0.0)
            act.setdefault("budget_raw", "")
        return result

    logger.error("LLM response keys: %s",
                 list(result.keys()) if isinstance(result, dict) else type(result).__name__)
    raise ValueError(
        "LLM response does not contain expected 'actions' key. "
        f"Got keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}"
    )


def check_llm_available() -> bool:
    """Check if the OpenAI API key is configured."""
    from src.config import OPENAI_API_KEY
    return bool(OPENAI_API_KEY)
