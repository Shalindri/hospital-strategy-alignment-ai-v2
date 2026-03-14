"""
pdf_processor.py — Extract PDF text and parse into structured JSON via GPT-4o-mini.
"""

import json
import os

import pdfplumber

from src.config import get_openai_client, OPENAI_MODEL, DATA_DIR


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file, joined by newlines."""
    print(f"📌 Extracting text: {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"   {len(pdf.pages)} pages found.")
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    page_texts.append(text)
                print(f"   Page {i}/{len(pdf.pages)}...", end="\r")
    except Exception as e:
        raise RuntimeError(f"Could not read PDF: {e}")

    full_text = "\n".join(page_texts)
    print(f"\n✅ Extracted {len(full_text):,} characters.")
    return full_text


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output if present."""
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text


def _call_llm(prompt: str) -> str:
    """Send a prompt to GPT-4o-mini and return the response text."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return _strip_fences(response.choices[0].message.content.strip())


def parse_strategic_plan(raw_text: str) -> dict:
    """Parse strategic plan text into {objectives: [{id, title, description}]}."""
    print("📌 Parsing strategic plan with GPT-4o-mini...")

    prompt = f"""You are a healthcare strategy analyst. Extract ONLY the strategic objectives
from this hospital strategic plan document.

IGNORE: forewords, table of contents, vision/mission statements, KPIs, timelines,
risk registers, sign-off pages, contact details, headers, footers.

Return ONLY valid JSON (no markdown):
{{
  "objectives": [
    {{
      "id": "O1",
      "title": "Short clear title (5-10 words)",
      "description": "4-6 sentences with domain-specific clinical/operational/financial terminology unique to this objective."
    }}
  ]
}}

Rules:
- Sequential IDs: O1, O2, O3, ...
- Use vocabulary UNIQUE to each domain (patient safety: HAI, infection control, adverse events; technology: EHR, cybersecurity; finance: deficit, procurement, EBITDA)
- No KPIs, timelines, or risk text in descriptions
- Include each objective only once

Document text:
{raw_text[:12000]}"""

    try:
        parsed = json.loads(_call_llm(prompt))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}. Try again.")
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    print(f"✅ Found {len(parsed.get('objectives', []))} objectives.")
    return parsed


def parse_action_plan(raw_text: str) -> dict:
    """Parse action plan text into {actions: [{id, title, description}]}."""
    print("📌 Parsing action plan with GPT-4o-mini...")

    prompt = f"""You are a healthcare operations analyst. Extract ONLY the operational actions
from this hospital action plan document.

IGNORE: cover page, "how to read this plan", objective section headers, owner names,
timelines, KPI lines, sign-off tables, page numbers, headers, footers.

Return ONLY valid JSON (no markdown):
{{
  "actions": [
    {{
      "id": "A1",
      "title": "Short clear title (5-10 words)",
      "description": "4-6 sentences with domain-specific terminology linking this action to its strategic area."
    }}
  ]
}}

Rules:
- Sequential IDs: A1, A2, A3, ...
- Use domain-specific vocabulary (infection control: HAI, sterilisation, PPE; technology: EHR, cybersecurity; workforce: turnover, burnout, mentorship; finance: procurement, deficit, grants)
- No owner names or KPIs in descriptions
- Include each action only once

Document text:
{raw_text[:12000]}"""

    try:
        parsed = json.loads(_call_llm(prompt))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}. Try again.")
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    print(f"✅ Found {len(parsed.get('actions', []))} actions.")
    return parsed


def save_to_json(data: dict, output_filename: str) -> str:
    """Save a dict as JSON to the data/ folder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to {output_path}")
    return output_path


def process_pdf(pdf_path: str, doc_type: str, output_filename: str) -> dict:
    """
    Full pipeline: extract PDF text → parse with LLM → save JSON.

    Args:
        pdf_path: Path to the PDF file.
        doc_type: "strategic_plan" or "action_plan".
        output_filename: Filename to save in data/.
    """
    print(f"\n{'='*60}\n  Processing: {pdf_path} ({doc_type})\n{'='*60}\n")

    raw_text = extract_text_from_pdf(pdf_path)

    if doc_type == "strategic_plan":
        parsed_data = parse_strategic_plan(raw_text)
    elif doc_type == "action_plan":
        parsed_data = parse_action_plan(raw_text)
    else:
        raise ValueError(f"Unknown doc_type: '{doc_type}'. Use 'strategic_plan' or 'action_plan'.")

    save_to_json(parsed_data, output_filename)
    print(f"\n✅ Done: {doc_type}.\n")
    return parsed_data


if __name__ == "__main__":
    sample_strategic = os.path.join(DATA_DIR, "strategic_plan.pdf")
    sample_action    = os.path.join(DATA_DIR, "action_plan.pdf")

    if os.path.exists(sample_strategic):
        process_pdf(sample_strategic, "strategic_plan", "strategic_plan.json")
    else:
        print(f"⚠️  Not found: {sample_strategic}")

    if os.path.exists(sample_action):
        process_pdf(sample_action, "action_plan", "action_plan.json")
    else:
        print(f"⚠️  Not found: {sample_action}")
