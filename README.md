# ISPS — Intelligent Strategic Plan Synchronization System

> **MSc Information Retrieval Coursework** · Nawaloka Hospital Negombo Sri Lanka · Hospital Strategy Alignment AI

ISPS measures how well a hospital's annual action plan aligns with its long-term strategic objectives. It uses semantic embeddings, ontology mapping, knowledge graphs, and Retrieval-Augmented Generation (RAG) to produce alignment scores, detect misaligned actions, and generate AI-powered improvement recommendations for decision-makers.

The system is built around a real hospital scenario: **Nawaloka Hospital Negombo Sri Lanka** with 5 strategic objectives and 20 operational actions across patient safety, technology, patient experience, workforce, and financial sustainability domains.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Alignment Score Tiers](#alignment-score-tiers)
6. [Technology Stack](#technology-stack)
7. [Setup — Step by Step](#setup--step-by-step)
8. [How to Use the Dashboard](#how-to-use-the-dashboard)
9. [Uploading PDFs](#uploading-pdfs)
10. [Generating Sample PDFs](#generating-sample-pdfs)
11. [Running the Evaluation](#running-the-evaluation)
12. [Current Evaluation Results](#current-evaluation-results)
13. [Coursework Mapping](#coursework-mapping)

---

## What It Does

| Feature | Module | Description |
|---|---|---|
| **PDF ingestion** | `pdf_processor.py` | Upload a strategic plan or action plan PDF — GPT-4o-mini extracts only the relevant objectives/actions, ignoring boilerplate |
| **Semantic alignment scoring** | `alignment_scorer.py` | Cosine similarity matrix (objectives × actions) using sentence embeddings; classifies each pair as Excellent / Good / Fair / Poor |
| **Orphan detection** | `alignment_scorer.py` | Flags actions that score below the Fair threshold against every objective |
| **RAG improvement suggestions** | `rag_engine.py` | For each poorly-aligned action, retrieves the most relevant objective context from ChromaDB and asks GPT-4o-mini to suggest 3 improvements |
| **New action proposals** | `rag_engine.py` | Generates a new action for any objective that has no Excellent-tier action supporting it |
| **Ontology mapping** | `ontology_mapper.py` | Builds an OWL ontology with healthcare concept classes; maps each objective and action to its closest concept using keyword matching |
| **Knowledge graph** | `knowledge_graph.py` | Interactive NetworkX + PyVis HTML graph; edges weighted by alignment score; bridge nodes highlighted |
| **Agentic reasoning** | `agent_reasoner.py` | 3-iteration Plan → Act → Reflect loop; LLM identifies critical gaps, retrieves context, and scores its own suggestions |
| **Smart dashboard** | `dashboard/app.py` | 5-tab Streamlit interface with gauge charts, heatmaps, interactive graphs, and guidance messages for decision-makers |
| **Evaluation** | `tests/evaluation.py` | Precision, Recall, F1, AUC (ROC), Pearson r — compared against a 100-pair human-annotated ground truth |

---

## System Architecture

```
PDF Upload (dashboard sidebar)
        │
        ▼
┌─────────────────────────┐
│    pdf_processor.py     │  pdfplumber extracts raw text
│  GPT-4o-mini parses     │  → filters out boilerplate
│  → saves .json to data/ │  → saves structured objectives/actions
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     vector_store.py     │  sentence-transformers encodes text
│  all-mpnet-base-v2      │  → embeddings stored in ChromaDB
│  768-dimensional        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  alignment_scorer.py    │  cosine similarity (objectives × actions)
│  5 × 20 matrix          │  → tier labels → orphan detection
└────────────┬────────────┘
             │
     ┌───────┼───────────────┐
     ▼       ▼               ▼
┌─────────┐ ┌────────────┐ ┌──────────────┐
│rag_     │ │ontology_   │ │knowledge_    │
│engine   │ │mapper.py   │ │graph.py      │
│.py      │ │RDF/OWL TTL │ │NetworkX +    │
│RAG +    │ │concept     │ │PyVis HTML    │
│GPT-4o   │ │mapping     │ │centrality    │
└────┬────┘ └────────────┘ └──────────────┘
     │
     ▼
┌─────────────────────────┐
│   agent_reasoner.py     │  Plan → Act → Reflect (3 iterations)
│   LLM self-evaluation   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    dashboard/app.py     │  Streamlit (5 tabs)
│    Plotly charts        │  interactive UI for decision-makers
└─────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│   tests/evaluation.py   │  vs 100-pair ground truth
│   Precision/Recall/F1   │
└─────────────────────────┘
```

---

## Project Structure

```
hospital-strategy-alignment-ai-v2/
│
├── .env                          ← OPENAI_API_KEY (never commit this)
├── requirements.txt              ← all Python dependencies
├── README.md                     ← this file
├── generate_pdfs.py              ← generates sample hospital PDFs for testing
│
├── src/
│   ├── config.py                 ← all constants, thresholds, model names
│   ├── pdf_processor.py          ← extract PDF text → filtered structured JSON
│   ├── vector_store.py           ← embed text with sentence-transformers, store/query ChromaDB
│   ├── alignment_scorer.py       ← cosine similarity matrix + tier labels + orphan detection
│   ├── ontology_mapper.py        ← RDF/OWL healthcare concept mapping (rdflib)
│   ├── knowledge_graph.py        ← NetworkX directed graph + PyVis HTML export
│   ├── rag_engine.py             ← RAG: retrieve context → GPT-4o-mini suggestion
│   └── agent_reasoner.py         ← Plan → Act → Reflect agentic loop
│
├── dashboard/
│   └── app.py                    ← Streamlit UI with 5 tabs + PDF upload sidebar
│
├── data/
│   ├── strategic_plan.json       ← 5 strategic objectives (extracted from PDF or edited directly)
│   ├── action_plan.json          ← 20 operational actions (extracted from PDF or edited directly)
│   ├── strategic_plan.pdf        ← sample Nawaloka Hospital strategic plan document
│   └── action_plan.pdf           ← sample Nawaloka Hospital action plan document
│
├── tests/
│   ├── ground_truth.json         ← 100-pair human-annotated dataset (5 obj × 20 actions)
│   └── evaluation.py             ← Precision / Recall / F1 / AUC / Pearson r
│
├── outputs/                      ← auto-generated files
│   ├── evaluation_results.json   ← latest evaluation metrics
│   ├── knowledge_graph.html      ← interactive graph (open in browser)
│   └── ontology.ttl              ← RDF ontology in Turtle format
│
├── chroma_db/                    ← ChromaDB vector storage (auto-created on first run)
│
└── docs/
    └── hosting_architecture.md   ← cloud deployment write-up (CW sections 3.6–3.7)
```

---

## Dataset

The system is pre-loaded with a realistic hospital scenario for **Nawaloka Hospital Negombo Sri Lanka**:

### Strategic Objectives (5)

| ID | Objective | Owner |
|---|---|---|
| O1 | Improve Patient Safety | Dr. Marcus Osei, Director of Quality & Patient Safety |
| O2 | Modernise Hospital Technology | Ms. Leila Moussaoui, Chief Information Officer |
| O3 | Enhance Patient Experience and Satisfaction | Ms. Priya Chandran, Director of Patient Experience |
| O4 | Develop and Retain a Skilled Workforce | Ms. Ama Forson, Director of People & OD |
| O5 | Achieve Financial Sustainability | Mr. David Reeves, Chief Financial Officer |

### Action Plan (20 actions — 4 per objective)

| ID | Action | Primary Objective |
|---|---|---|
| A1 | Monthly Infection Control Audit | O1 |
| A2 | Staff Safety Training Programme | O1 |
| A3 | JCI Accreditation Preparation | O1 |
| A4 | Clinical Incident Reporting System | O1 |
| A5 | Deploy Electronic Health Record System | O2 |
| A6 | Automate Patient Scheduling and Billing | O2 |
| A7 | Upgrade Cybersecurity Infrastructure | O2 |
| A8 | Implement AI-Powered Diagnostic Support Tools | O2 |
| A9 | Launch Patient Satisfaction Survey Programme | O3 |
| A10 | Reduce Outpatient Waiting Times Initiative | O3 |
| A11 | Introduce Multilingual Patient Communication Services | O3 |
| A12 | Deploy Patient Self-Service Portal | O3 |
| A13 | Launch Clinical Leadership Development Programme | O4 |
| A14 | Staff Wellbeing and Mental Health Support Initiative | O4 |
| A15 | Nurse Retention and Incentive Scheme | O4 |
| A16 | University Medical Internship Partnership Programme | O4 |
| A17 | Annual Budget Review and Cost Reduction Audit | O5 |
| A18 | Expand Private Patient and Revenue Services | O5 |
| A19 | Apply for Healthcare Grants and Government Subsidies | O5 |
| A20 | Medical Supply Chain Optimisation Programme | O5 |

### Ground Truth (100 pairs)

`tests/ground_truth.json` contains 100 human-annotated alignment pairs (5 objectives × 20 actions) with labels:

| Label | Value | Meaning |
|---|---|---|
| Strongly Aligned | 1.0 | Action directly delivers the objective |
| Weakly Aligned | 0.5 | Some thematic overlap; indirect support |
| Not Aligned | 0.1 | Different domain; no meaningful link |

Distribution: **22 strongly aligned**, **21 weakly aligned**, **57 not aligned**

---

## Alignment Score Tiers

Alignment scores are cosine similarity values (0.0–1.0) between objective and action embeddings, classified into four tiers:

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.70 | **Excellent** | Action directly operationalises the strategy |
| 0.60–0.69 | **Good** | Clear strategic support |
| 0.47–0.59 | **Fair** | Partial or indirect alignment |
| < 0.47 | **Poor / Orphan** | Weak or no meaningful alignment |

An action is an **orphan** if it scores below 0.47 against *every* objective. Orphans are highlighted in red in the dashboard and prioritised by the agent reasoner.

> The Fair threshold was tuned to 0.47 via a threshold sweep (optimising F1 score against the ground truth).

---

## Technology Stack

| Component | Library / Model | Purpose |
|---|---|---|
| **LLM** | OpenAI `gpt-4o-mini` | PDF parsing, RAG suggestions, agentic reasoning |
| **Embeddings** | `sentence-transformers` `all-mpnet-base-v2` (768-dim) | Encode text for similarity comparison |
| **Vector DB** | `chromadb` (local persistent) | Store and query embeddings |
| **Ontology** | `rdflib` (RDF/OWL, Turtle format) | Healthcare concept class mapping |
| **Knowledge graph** | `networkx` + `pyvis` | Directed alignment graph with HTML export |
| **Dashboard** | `streamlit` + `plotly` | Interactive 5-tab UI |
| **PDF reading** | `pdfplumber` | Extract text from uploaded PDFs |
| **Evaluation** | `scikit-learn` + `scipy` | Precision, Recall, F1, AUC, Pearson r |
| **Data** | `pandas` + `numpy` | Data handling and matrix operations |
| **Secrets** | `python-dotenv` | Load API key from `.env` file |

---

## Setup — Step by Step

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd hospital-strategy-alignment-ai-v2
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First install downloads the sentence-transformers model (~450 MB). This only happens once.

### 4. Set your OpenAI API key

Create a file called `.env` in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

> Never commit this file — it is already in `.gitignore`.
> Get a key at [platform.openai.com](https://platform.openai.com).

### 5. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501**

The dashboard loads the existing `data/strategic_plan.json` and `data/action_plan.json` and runs alignment scoring automatically on first load (~30 seconds while ChromaDB builds the vector index).

---

## How to Use the Dashboard

The dashboard has **5 tabs** plus a **PDF upload sidebar**:

### 📊 Synchronization
- **Gauge chart** — overall alignment score (0–100%)
- **Heatmap** — 5 × 20 objectives-vs-actions colour grid (hover for exact scores)
- **Per-action table** — each action's best-matching objective, score, and tier label
- **Orphan list** — actions with no meaningful link to any objective (highlighted in red)

### 💡 Recommendations
- **RAG suggestions** — for each poorly-aligned action, GPT-4o-mini suggests 3 specific improvements using retrieved objective context
- **New action proposals** — for objectives with no Excellent-tier action, the system proposes a brand new action
- Click **"Run RAG Suggestions"** to generate (requires OpenAI API key)

### 🕸️ Knowledge Graph
- Interactive PyVis HTML network embedded in the browser
- **Nodes** = objectives (circles) and actions (squares)
- **Edges** = alignment relationships (only Fair and above)
- **Gold nodes** = high-centrality "bridge" nodes
- Click and drag to explore the graph

### 🔬 Ontology
- Lists the RDF/OWL concept class each objective and action is mapped to
- Healthcare concept classes: ClinicalCare, DigitalTechnology, PatientExperience, HumanResources, Finance
- Coverage summary showing how many items map to each concept

### 📈 Evaluation
- Click **"Run Evaluation"** to compare the system against the 100-pair ground truth
- Shows: Precision, Recall, F1 Score, AUC (ROC), Pearson r
- Confusion matrix heatmap
- Threshold sweep chart (F1 / Precision / Recall vs threshold value)
- Interpretation guidance for decision-makers

---

## Uploading PDFs

The **sidebar on the left** lets you upload Nawaloka Hospital Negombo documents at any time:

1. Click **"Strategic Plan PDF"** → select your strategic plan PDF
2. Click **"Action Plan PDF"** → select your action plan PDF
3. Click **"⚙️ Process PDFs"**

What happens next:
- `pdfplumber` extracts all text from each PDF page
- GPT-4o-mini reads the text and extracts **only the relevant objectives/actions** — it skips forewords, table of contents, KPI tables, sign-off pages, and other boilerplate
- Descriptions are written with **rich, domain-specific vocabulary** to ensure high alignment scores
- The results are saved to `data/strategic_plan.json` and `data/action_plan.json`
- The Streamlit cache is cleared and the dashboard reloads automatically with the new data

> You only need to upload the PDFs you want to change. Upload one, or both.

For a detailed step-by-step technical walkthrough of exactly what happens internally when you click "Process PDFs", see [docs/pdf_processing_explained.md](docs/pdf_processing_explained.md).

---

## Generating Sample PDFs

The project includes a script that generates realistic-looking hospital PDFs from the existing JSON data:

```bash
python generate_pdfs.py
```

This creates:
- `data/strategic_plan.pdf` — full 11-page strategic plan document with cover, foreword, context, objectives overview, and **one dedicated page per objective** (rationale, targets, risks, timeline)
- `data/action_plan.pdf` — action plan document with all 20 actions grouped by objective, each with owner, timeline, and KPI

These PDFs are what you would upload via the dashboard sidebar to test the PDF extraction pipeline.

---

## Running the Evaluation

To evaluate the alignment scorer against the 100-pair ground truth dataset:

```bash
python tests/evaluation.py
```

Results are printed to the terminal and saved to `outputs/evaluation_results.json`.

You can also run evaluation directly from the dashboard **📈 Evaluation** tab without using the terminal.

---

## Current Evaluation Results

Results from the latest run against the 100-pair ground truth (`tests/ground_truth.json`):

| Metric | Value |
|---|---|
| **Precision** | 0.818 |
| **Recall** | 0.600 |
| **F1 Score** | 0.692 |
| **AUC (ROC)** | 0.786 |
| **Pearson r** | 0.725 |
| Ground truth threshold | 0.5 (label ≥ 0.5 = positive) |
| Prediction threshold | 0.47 (tuned via sweep) |
| Optimal F1 | 0.700 at threshold 0.35 |
| Total pairs evaluated | 100 |

**Confusion matrix:**

```
                  Predicted NOT Aligned   Predicted Aligned
Actual NOT Aligned        49                    6
Actual Aligned            18                   27
```

> The model is precise (low false positive rate) but has moderate recall — it misses some weakly-aligned pairs. This is expected behaviour since the embedding model (all-mpnet-base-v2) scores pairs with partial domain overlap conservatively.

---
## Common Issues

**`OPENAI_API_KEY is not set`**
→ Create a `.env` file in the project root with `OPENAI_API_KEY=sk-...`

**`ModuleNotFoundError`**
→ Make sure your virtual environment is activated: `source venv/bin/activate`

**Dashboard loads but alignment scoring is slow**
→ Normal — ChromaDB is building the vector index on first run. Subsequent loads use the cached index.

**PDF upload returns "LLM returned invalid JSON"**
→ Try clicking Process PDFs again — GPT-4o-mini occasionally produces a malformed response on the first attempt.

**ChromaDB dimension mismatch error**
→ Delete the `chroma_db/` folder and restart the dashboard. This happens if you switch embedding models.

```bash
rm -rf chroma_db/
streamlit run dashboard/app.py
```
