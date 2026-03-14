"""
pdf_processor.py
----------------
Extract text from a PDF file and parse it into structured JSON using an LLM.

How it fits in the pipeline:
  [this module] → vector_store → alignment_scorer → dashboard

Beginner note: PDFs store content in a layout format (positions, fonts, boxes),
not as plain text. pdfplumber reads those boxes and gives us the text per page.
We then send all that raw text to GPT-4o-mini and ask it to return clean JSON
with a list of objectives or actions.
"""

# --- Standard library ---
import json
import os

# --- Third-party ---
import pdfplumber   # PDF text extraction

# --- Local ---
from src.config import get_openai_client, OPENAI_MODEL, DATA_DIR


# =============================================================================
# STEP 1: EXTRACT RAW TEXT FROM PDF
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all its text as one big string.

    Each page's text is joined with a newline. pdfplumber handles complex
    PDF layouts better than simple tools like PyPDF2.

    Args:
        pdf_path (str): Path to the PDF file (absolute or relative to project root).

    Returns:
        str: All text from every page, joined with newlines.

    Raises:
        FileNotFoundError: If the file does not exist (with a helpful message).
        RuntimeError: If pdfplumber cannot read the file.
    """
    print(f"📌 Step 1: Extracting text from PDF: {pdf_path}")

    # Check the file actually exists before trying to open it
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Please check the file path and try again."
        )

    page_texts = []  # We'll collect text from each page here

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"   Found {total_pages} pages.")

            for i, page in enumerate(pdf.pages, start=1):
                # extract_text() returns a string or None if the page is empty
                text = page.extract_text()
                if text:
                    page_texts.append(text)
                print(f"   Processed page {i}/{total_pages}...", end="\r")

    except Exception as e:
        raise RuntimeError(
            f"Could not read the PDF file: {e}\n"
            "Make sure the file is not password-protected and is a valid PDF."
        )

    # Join all pages into one string, separated by newlines
    full_text = "\n".join(page_texts)
    print(f"\n✅ Step 1 complete. Extracted {len(full_text):,} characters.")
    return full_text


# =============================================================================
# STEP 2: PARSE TEXT INTO STRUCTURED JSON USING LLM
# =============================================================================

def parse_strategic_plan(raw_text: str) -> dict:
    """
    Send strategic-plan text to GPT-4o-mini and parse it into a structured dict.

    The LLM identifies each strategic objective and returns JSON with fields:
      objectives: list of {id, title, description}

    Args:
        raw_text (str): Full text from the strategic plan PDF.

    Returns:
        dict: {"objectives": [{"id": "O1", "title": "...", "description": "..."}, ...]}

    Raises:
        RuntimeError: If the LLM call fails or returns invalid JSON.
    """
    print("📌 Step 2a: Parsing strategic plan with GPT-4o-mini...")

    # We only send the first ~12,000 characters to keep API costs low.
    # Most plans fit in this limit; adjust if needed.
    truncated_text = raw_text[:12000]

    prompt = f"""You are a healthcare strategy analyst. Extract all strategic objectives
from this hospital strategic plan document.

Return ONLY valid JSON in this exact format (no extra text):
{{
  "objectives": [
    {{
      "id": "O1",
      "title": "Short title of the objective",
      "description": "Full description of what this objective aims to achieve"
    }}
  ]
}}

Rules:
- Use sequential IDs: O1, O2, O3, ...
- Each objective must have a meaningful title (5-10 words) and a description (1-3 sentences)
- Extract ALL distinct objectives from the document
- Do not include sub-objectives or KPIs as separate objectives

Document text:
{truncated_text}"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # Low temperature = more consistent, less creative output
        )

        # The LLM response is a string; we parse it as JSON
        response_text = response.choices[0].message.content.strip()

        # Sometimes the LLM wraps JSON in markdown code fences — remove them
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        parsed = json.loads(response_text)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned invalid JSON: {e}\n"
            "Try running again — LLMs occasionally produce malformed responses."
        )
    except Exception as e:
        raise RuntimeError(
            f"OpenAI API call failed: {e}\n"
            "Check your OPENAI_API_KEY in .env and your internet connection."
        )

    print(f"✅ Step 2a complete. Found {len(parsed.get('objectives', []))} objectives.")
    return parsed


def parse_action_plan(raw_text: str) -> dict:
    """
    Send action-plan text to GPT-4o-mini and parse it into a structured dict.

    The LLM identifies each action item and returns JSON with fields:
      actions: list of {id, title, description}

    Args:
        raw_text (str): Full text from the action plan PDF.

    Returns:
        dict: {"actions": [{"id": "A1", "title": "...", "description": "..."}, ...]}

    Raises:
        RuntimeError: If the LLM call fails or returns invalid JSON.
    """
    print("📌 Step 2b: Parsing action plan with GPT-4o-mini...")

    truncated_text = raw_text[:12000]

    prompt = f"""You are a healthcare operations analyst. Extract all action items
from this hospital annual action plan document.

Return ONLY valid JSON in this exact format (no extra text):
{{
  "actions": [
    {{
      "id": "A1",
      "title": "Short title of the action",
      "description": "What this action involves and what it aims to achieve"
    }}
  ]
}}

Rules:
- Use sequential IDs: A1, A2, A3, ...
- Each action must have a clear title (5-10 words) and a description (1-3 sentences)
- Extract ALL distinct actions from the document
- Include operational tasks, projects, and initiatives

Document text:
{truncated_text}"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        response_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        parsed = json.loads(response_text)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned invalid JSON: {e}\n"
            "Try running again — LLMs occasionally produce malformed responses."
        )
    except Exception as e:
        raise RuntimeError(
            f"OpenAI API call failed: {e}\n"
            "Check your OPENAI_API_KEY in .env and your internet connection."
        )

    print(f"✅ Step 2b complete. Found {len(parsed.get('actions', []))} actions.")
    return parsed


# =============================================================================
# STEP 3: SAVE RESULT TO JSON FILE
# =============================================================================

def save_to_json(data: dict, output_filename: str) -> str:
    """
    Save a Python dict as a JSON file inside the data/ folder.

    Args:
        data (dict): The data to save.
        output_filename (str): Filename only (e.g. "strategic_plan.json").

    Returns:
        str: The full path to the saved file.
    """
    # Create the data/ folder if it doesn't exist yet
    os.makedirs(DATA_DIR, exist_ok=True)

    output_path = os.path.join(DATA_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to {output_path}")
    return output_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_pdf(pdf_path: str, doc_type: str, output_filename: str) -> dict:
    """
    Full pipeline: extract PDF text → parse with LLM → save JSON → return data.

    Args:
        pdf_path (str): Path to the input PDF file.
        doc_type (str): "strategic_plan" or "action_plan".
        output_filename (str): Filename to save in data/ (e.g. "strategic_plan.json").

    Returns:
        dict: The parsed structured data.

    Raises:
        ValueError: If doc_type is not "strategic_plan" or "action_plan".
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {pdf_path} (type: {doc_type})")
    print(f"{'='*60}\n")

    # Step 1: Extract text from PDF
    raw_text = extract_text_from_pdf(pdf_path)

    # Step 2: Parse text into structured JSON using the LLM
    if doc_type == "strategic_plan":
        parsed_data = parse_strategic_plan(raw_text)
    elif doc_type == "action_plan":
        parsed_data = parse_action_plan(raw_text)
    else:
        raise ValueError(
            f"Unknown doc_type: '{doc_type}'. "
            "Use 'strategic_plan' or 'action_plan'."
        )

    # Step 3: Save to JSON file in data/
    save_to_json(parsed_data, output_filename)

    print(f"\n✅ Processing complete for {doc_type}.\n")
    return parsed_data


# =============================================================================
# CONVENIENCE: run this file directly to process both PDFs
# =============================================================================
if __name__ == "__main__":
    # Quick test: process sample PDFs if they exist in data/
    import sys

    sample_strategic = os.path.join(DATA_DIR, "StrategicPlan20262030_test1.pdf")
    sample_action    = os.path.join(DATA_DIR, "AnnualActionPlan2026_test1.pdf")

    if os.path.exists(sample_strategic):
        process_pdf(sample_strategic, "strategic_plan", "strategic_plan.json")
    else:
        print(f"⚠️  No strategic plan PDF found at {sample_strategic}")

    if os.path.exists(sample_action):
        process_pdf(sample_action, "action_plan", "action_plan.json")
    else:
        print(f"⚠️  No action plan PDF found at {sample_action}")
