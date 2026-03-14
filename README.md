# ISPS — Intelligent Strategic Plan Synchronization System

An MSc Information Retrieval coursework project that measures how well a hospital's
annual action plan aligns with its long-term strategic plan.

The system uses semantic embeddings, ontology mapping, knowledge graphs, and
Retrieval-Augmented Generation (RAG) to produce alignment scores, detect misaligned
actions, and generate improvement recommendations for decision-makers.

---

## What It Does

| Feature | Description |
|---|---|
| Alignment scoring | Cosine similarity matrix (objectives × actions) with tier labels |
| Orphan detection | Actions with no meaningful link to any strategic objective |
| RAG suggestions | GPT-4o-mini improves poorly-aligned actions using retrieved context |
| Ontology mapping | Maps each item to a healthcare concept (ClinicalCare, Finance, etc.) |
| Knowledge graph | Interactive network showing objective→action connections |
| Agentic reasoning | Plan→Act→Reflect loop for automated analysis |
| Evaluation | Precision / Recall / F1 / AUC vs 58-pair human-annotated ground truth |

---

## Project Structure

```
hospital-strategy-alignment-ai-v2/
├── .env                          ← your OPENAI_API_KEY (never commit this)
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py                 ← all constants, thresholds, model names
│   ├── pdf_processor.py          ← extract PDF text → structured JSON
│   ├── vector_store.py           ← embed text, store/query ChromaDB
│   ├── alignment_scorer.py       ← cosine similarity matrix + tier labels
│   ├── ontology_mapper.py        ← RDF/OWL concept mapping
│   ├── knowledge_graph.py        ← NetworkX graph + pyvis HTML export
│   ├── rag_engine.py             ← RAG: retrieve context → LLM suggestion
│   └── agent_reasoner.py         ← Plan → Act → Reflect agentic loop
│
├── dashboard/
│   └── app.py                    ← Streamlit UI (5 tabs)
│
├── data/
│   ├── strategic_plan.json       ← extracted objectives
│   └── action_plan.json          ← extracted action items
│
├── tests/
│   ├── ground_truth.json         ← 58-pair annotated dataset
│   └── evaluation.py             ← Precision / Recall / F1 / AUC
│
├── outputs/                      ← generated files (HTML, TTL, JSON)
├── chroma_db/                    ← ChromaDB vector storage (auto-created)
└── docs/
    └── hosting_architecture.md   ← deployment write-up (CW sections 3.6–3.7)
```

---

## Setup (Step by Step)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd hospital-strategy-alignment-ai-v2
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # on macOS/Linux
# or on Windows:
# venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything: OpenAI SDK, sentence-transformers, ChromaDB, rdflib,
networkx, pyvis, Streamlit, Plotly, scikit-learn, and more.

### 4. Set your OpenAI API key

Create a file called `.env` in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

Never commit this file — it is already in `.gitignore`.

### 5. (Optional) Process your own PDFs

If you want to extract objectives/actions from new PDFs:

```bash
python -c "
from src.pdf_processor import process_pdf
process_pdf('data/StrategicPlan20262030_test1.pdf', 'strategic_plan', 'strategic_plan.json')
process_pdf('data/AnnualActionPlan2026_test1.pdf',  'action_plan',    'action_plan.json')
"
```

The existing `data/strategic_plan.json` and `data/action_plan.json` are already
populated, so you can skip this step for testing.

### 6. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

On first load the alignment scoring runs automatically (~30 seconds).

---

## Running the Evaluation

To compare the system against the 58-pair ground truth dataset:

```bash
python tests/evaluation.py
```

Results are printed to the terminal and saved to `outputs/evaluation_results.json`.

---

## Alignment Score Tiers

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.75 | Excellent | Action directly operationalises the strategy |
| 0.60–0.74 | Good | Clear strategic support |
| 0.42–0.59 | Fair | Partial or indirect alignment |
| < 0.42 | Poor / Orphan | Weak or no meaningful alignment |

An action is an **orphan** if it scores below 0.42 against *every* objective.

---

## Technology Stack

| Component | Library |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) |
| Vector DB | `chromadb` (local persistent) |
| Ontology | `rdflib` (RDF/OWL, Turtle format) |
| Knowledge graph | `networkx` + `pyvis` |
| Dashboard | `streamlit` + `plotly` |
| PDF reading | `pdfplumber` |
| Evaluation | `scikit-learn` + `scipy` |
| Secrets | `python-dotenv` |

---

## Coursework Mapping

| CW Section | Marks | File |
|---|---|---|
| 3.1 & 3.2 Overall + strategy-wise sync | 20 | `src/alignment_scorer.py` |
| 3.3 RAG improvement suggestions | 10 | `src/rag_engine.py` |
| 3.4 Smart dashboard | 10 | `dashboard/app.py` |
| 3.5 Agentic AI reasoning | 10 | `src/agent_reasoner.py` |
| 3.5 Ontology mapping | 10 | `src/ontology_mapper.py` |
| 3.5 Knowledge graph | 10 | `src/knowledge_graph.py` |
| 3.6 & 3.7 Hosting architecture | 10 | `docs/hosting_architecture.md` |
| 3.8 Testing & evaluation | 10 | `tests/evaluation.py` |
