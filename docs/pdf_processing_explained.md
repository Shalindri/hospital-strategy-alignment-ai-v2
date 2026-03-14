# What Happens When You Click "Process PDFs"

A complete step-by-step technical walkthrough of the PDF ingestion pipeline in ISPS.

---

## Overview

When you click the **⚙️ Process PDFs** button in the sidebar, the system runs a 6-stage pipeline
that transforms a raw hospital PDF document into structured JSON data, which then drives
the entire alignment scoring, knowledge graph, and recommendations engine.

```
User clicks "Process PDFs"
         │
         ▼
 Stage 1 — Save uploaded file to disk
         │
         ▼
 Stage 2 — Extract raw text from PDF (pdfplumber)
         │
         ▼
 Stage 3 — Send text to GPT-4o-mini (OpenAI API)
         │
         ▼
 Stage 4 — Parse LLM response into structured JSON
         │
         ▼
 Stage 5 — Save JSON to data/ folder
         │
         ▼
 Stage 6 — Clear cache and reload the dashboard
```

---

## Stage 1 — Save the Uploaded File to Disk

**Where in the code:** `dashboard/app.py` → `sidebar_pdf_upload()` function

Streamlit's file uploader gives you the file as **bytes in memory** (not a file path).
But `pdfplumber` — the library that reads PDFs — needs a **file path on disk**.

So the first thing the code does is write those bytes to a temporary file:

```python
import tempfile

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(sp_file.read())   # write uploaded bytes to disk
    tmp_path = tmp.name         # remember the temporary path
```

The file goes into your operating system's temp folder (e.g. `/tmp/tmpXXXXXX.pdf`
on macOS/Linux). It is automatically cleaned up later.

The progress bar in the sidebar advances to 10% at this point.

---

## Stage 2 — Extract Raw Text from the PDF

**Where in the code:** `src/pdf_processor.py` → `extract_text_from_pdf()`

The temporary PDF path is passed to `pdfplumber`, which opens the file and reads
every page one by one:

```python
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()   # returns the text content of that page
        page_texts.append(text)
```

`pdfplumber` handles the complex internal structure of PDFs (fonts, layout boxes,
coordinates) and gives back plain text strings per page.

All pages are joined into one large string:

```python
full_text = "\n".join(page_texts)
```

**What comes out of this stage:** A single long string containing everything printed
in the PDF — including cover pages, forewords, table of contents, objective descriptions,
KPI tables, timelines, and sign-off pages.

The progress bar advances to 30%.

---

## Stage 3 — Send the Text to GPT-4o-mini

**Where in the code:** `src/pdf_processor.py` → `parse_strategic_plan()` or `parse_action_plan()`

The full extracted text is truncated to the first **12,000 characters** (to keep API
costs low and stay within the token limit) and sent to GPT-4o-mini via the OpenAI API:

```python
client = get_openai_client()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,    # low temperature = consistent, factual output
)
```

### The Prompt (for a Strategic Plan)

The prompt gives the LLM four explicit instructions:

**1. What to IGNORE (boilerplate filtering):**
```
IGNORE all of the following:
- Foreword, CEO message, or introductory paragraphs
- Table of contents entries
- Vision, Mission, and Values statements
- Performance targets, KPIs, or metrics tables
- Risk registers or risk descriptions
- Delivery timelines or milestone lists
- Sign-off pages, approvals, and signatures
- Page numbers, footers, and headers
```

This is the most important part. Without this, GPT-4o-mini would include "Committed
to Excellence in Healthcare" (a strapline) or "Q3 2025 — Complete gap analysis"
(a timeline entry) as objectives.

**2. What to EXTRACT:**
```
Extract ONLY the core strategic objectives — the high-level goals the organisation
is committing to achieve over the plan period.
```

**3. Output format:**
```json
{
  "objectives": [
    {
      "id": "O1",
      "title": "Short clear title (5-10 words)",
      "description": "Rich description with domain-specific vocabulary (4-6 sentences)"
    }
  ]
}
```

**4. Vocabulary rules:**
```
Write rich, detailed descriptions using specific domain vocabulary:
- Patient safety: HAI, infection control, adverse events, near-miss, PPE, sterilisation
- Technology: EHR, cybersecurity, digital infrastructure, interoperability
- Finance: deficit, procurement, EBITDA, overhead expenditure, grants
Each objective must use vocabulary UNIQUE to its domain.
```

This vocabulary rule is critical for getting **high alignment scores** — if all
descriptions use the same generic words ("patient", "hospital", "improve"), the
embedding model cannot distinguish between domains and scores will be low.

The progress bar advances to 50% (for a strategic plan) or 75% (for an action plan).

---

## Stage 4 — Parse the LLM Response into Structured JSON

**Where in the code:** `src/pdf_processor.py` → `parse_strategic_plan()` (continued)

The LLM returns a text response. Usually it is clean JSON, but sometimes it wraps
the JSON in markdown code fences like:

````
```json
{ "objectives": [...] }
```
````

The code strips these fences if present:

```python
response_text = response.choices[0].message.content.strip()

if response_text.startswith("```"):
    response_text = response_text.split("```")[1]
    if response_text.startswith("json"):
        response_text = response_text[4:]
```

Then it parses the cleaned string as JSON:

```python
parsed = json.loads(response_text)
```

If `json.loads()` raises an error (the LLM produced malformed JSON), the code
raises a `RuntimeError` with a plain-English message asking the user to try again.
This occasionally happens — GPT-4o-mini is not 100% reliable at producing valid JSON
on every single call. Clicking Process PDFs a second time almost always works.

**What comes out of this stage:** A Python dictionary like:

```python
{
  "objectives": [
    {
      "id": "O1",
      "title": "Improve Patient Safety",
      "description": "Eliminate hospital-acquired infections and adverse clinical events..."
    },
    ...
  ]
}
```

---

## Stage 5 — Save JSON to the data/ Folder

**Where in the code:** `src/pdf_processor.py` → `save_to_json()`

The structured dictionary is written to disk as a JSON file:

```python
output_path = os.path.join("data", "strategic_plan.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

This **overwrites** the existing `data/strategic_plan.json` or `data/action_plan.json`.

The two output files are:

| File | Contents |
|---|---|
| `data/strategic_plan.json` | `{"objectives": [{id, title, description}, ...]}` |
| `data/action_plan.json` | `{"actions": [{id, title, description}, ...]}` |

These files are the single source of truth for everything else in the pipeline —
the alignment scorer, RAG engine, ontology mapper, knowledge graph, and evaluation
all read from these two JSON files.

The progress bar advances to 100%.

---

## Stage 6 — Clear Cache and Reload the Dashboard

**Where in the code:** `dashboard/app.py` → `sidebar_pdf_upload()` (end of function)

Streamlit caches heavy computations to avoid re-running them on every page
interaction. After saving new JSON files, the cache must be cleared so the
dashboard picks up the new data instead of the old cached values:

```python
st.cache_data.clear()                       # clear all cached function results

if "alignment_result" in st.session_state:
    del st.session_state["alignment_result"]  # clear cached alignment scores

st.rerun()                                  # reload the entire Streamlit app
```

`st.rerun()` triggers a full page reload. On reload, the dashboard:

1. Calls `load_data()` → reads the new `strategic_plan.json` and `action_plan.json`
2. Calls `get_alignment_result()` → recomputes the full cosine similarity matrix
3. Calls `vector_store.embed_and_store()` → re-embeds all text and rebuilds ChromaDB
4. Renders all 5 tabs with the new data

This re-computation takes approximately **20–40 seconds** depending on your machine.
A spinner is shown while it runs.

---

## What Triggers a Low Overall Score After PDF Upload

If your alignment score drops after uploading a PDF, the most common cause is that
GPT-4o-mini extracted **shorter or more generic descriptions** than the hand-crafted
JSON files.

The ISPS alignment score is calculated as:

```
overall_score = mean( max_cosine_similarity_per_action )
```

Cosine similarity is higher when the text vectors are close in meaning. Vectors
are close when descriptions share **specific, domain-unique vocabulary**.

If descriptions are short or use generic hospital vocabulary (e.g. "improve patient
care", "enhance services"), the embedding model sees all objectives as similar to
all actions and scores remain low (≈ 0.50–0.58).

The PDF processor prompt now explicitly instructs GPT-4o-mini to:
- Write 4–6 sentence descriptions (not 2–3)
- Use domain-specific terminology (HAI, EHR, EBITDA, turnover, etc.)
- Use vocabulary UNIQUE to each domain

This should produce scores of **0.65–0.75** after PDF upload.

If scores are still low after uploading, you can directly edit
`data/strategic_plan.json` and `data/action_plan.json` and add richer descriptions
before restarting the dashboard — no PDF upload required.

---

## Summary Table

| Stage | What happens | Code location | Time |
|---|---|---|---|
| 1 | Upload bytes saved to temp file | `app.py: sidebar_pdf_upload()` | < 1 sec |
| 2 | pdfplumber reads all PDF pages | `pdf_processor.py: extract_text_from_pdf()` | 1–3 sec |
| 3 | Text sent to GPT-4o-mini API | `pdf_processor.py: parse_strategic_plan()` | 5–15 sec |
| 4 | LLM JSON response parsed | `pdf_processor.py: parse_strategic_plan()` | < 1 sec |
| 5 | Structured JSON saved to data/ | `pdf_processor.py: save_to_json()` | < 1 sec |
| 6 | Streamlit cache cleared, app reloads | `app.py: sidebar_pdf_upload()` | 20–40 sec |
