# Intelligent Strategic Plan Synchronization System (ISPS)

An AI-powered system that measures how well a hospital's annual action plan aligns with its long-term strategic plan. Uses semantic embeddings, ontology mapping, knowledge graphs, and Retrieval-Augmented Generation (RAG) to produce quantitative alignment scores, detect misaligned actions, and generate improvement recommendations.

## Problem Statement

Hospitals produce multi-year strategic plans and annual action plans, but manually checking whether 25+ operational actions genuinely support 5+ strategic objectives is slow, subjective, and error-prone. Misalignment means wasted budgets, missed goals, and strategic drift. ISPS automates this assessment with a data-driven, reproducible approach — replacing subjective review with objective, evidence-based analysis.

## Case Study

**Nawaloka Hospital Negombo, Sri Lanka**
- Strategic Plan: 2026-2030 (5 strategic objectives, capacity expansion from 75 to 150 beds)
- Action Plan: 2026 (25 operational actions across the five objectives)
- 58-pair human-annotated ground truth dataset for system evaluation
- Validated: AUC = 0.94 | F1 = 0.87 | Pearson r = 0.76

The system is **domain-agnostic** — any hospital or organisation can upload their strategic and action plan PDFs through the dashboard for analysis.

## Architecture

```
┌──────────────┐    ┌──────────────┐
│  Strategic   │    │   Action     │
│  Plan (PDF)  │    │  Plan (PDF)  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 ▼
       ┌─────────────────┐
       │  PDF Processor   │  GPT-4o-mini structured extraction
       │  → JSON          │
       └────────┬─────────┘
                │
    ┌───────────┼───────────────────────────────┐
    │           │                               │
    ▼           ▼                               ▼
┌────────┐ ┌──────────┐                  ┌────────────┐
│ChromaDB│ │Alignment │                  │  Ontology  │
│Vector  │ │Scoring   │                  │  Mapper    │
│Store   │ │(Cosine)  │                  │  (RDF/OWL) │
└───┬────┘ └────┬─────┘                  └─────┬──────┘
    │           │                               │
    │     ┌─────┴──────┐               ┌────────┴────────┐
    │     │ Knowledge  │               │  Gap Detection  │
    │     │ Graph      │               │                 │
    │     │ (NetworkX) │               └─────────────────┘
    │     └─────┬──────┘
    │           │
    ├───────────┤
    ▼           ▼
┌────────┐ ┌──────────┐
│  RAG   │ │  Agent   │
│Engine  │ │ Reasoner │
│(LLM)  │ │ (LLM)   │
└───┬────┘ └────┬─────┘
    │           │
    └─────┬─────┘
          ▼
  ┌───────────────┐
  │   Streamlit   │
  │   Dashboard   │
  └───────────────┘
```

## Pipeline Stages

| Stage | Module | Description | LLM |
|-------|--------|-------------|:---:|
| 1. PDF Extraction | `pdf_processor.py` | PDF text → structured JSON via GPT-4o-mini | Yes |
| 2. Vector Embedding | `vector_store.py` | Text → 384-dim embeddings, stored in ChromaDB | No |
| 3. Alignment Scoring | `synchronization_analyzer.py` | 5x25 cosine similarity matrix + classification | No |
| 4. Ontology Mapping | `ontology_mapper.py` | RDF/OWL concept mapping + gap detection | No |
| 5. Knowledge Graph | `knowledge_graph.py` | NetworkX graph with centrality & community analysis | No |
| 6. RAG Recommendations | `rag_engine.py` | Context-aware improvement suggestions via GPT-4o-mini | Yes |
| 7. Agent Reasoning | `agent_reasoner.py` | Plan-Act-Reflect diagnostic reasoning | Yes |

## Project Structure

```
hospital-strategy-alignment-ai/
├── src/                              # Core pipeline modules
│   ├── config.py                     # Centralised config, thresholds, LLM factory
│   ├── pdf_processor.py              # PDF extraction via OpenAI GPT-4o-mini
│   ├── vector_store.py               # ChromaDB embeddings (all-MiniLM-L6-v2)
│   ├── synchronization_analyzer.py   # Alignment scoring engine
│   ├── dynamic_analyzer.py           # Dynamic analysis for uploaded PDFs
│   ├── ontology_mapper.py            # RDF/OWL ontology mapping
│   ├── knowledge_graph.py            # NetworkX knowledge graph
│   ├── rag_engine.py                 # RAG recommendation engine
│   └── agent_reasoner.py             # Agentic AI reasoning (Plan-Act-Reflect)
├── dashboard/                        # Streamlit UI
│   ├── app.py                        # Main dashboard application
│   ├── pipeline_runner.py            # Dynamic pipeline orchestration
│   ├── data_adapter.py               # Data format conversion
│   └── utils.py                      # PDF report, charts, exports
├── data/                             # Processed data (JSON)
│   ├── strategic_plan.json
│   ├── action_plan.json
│   └── alignment_report.json
├── tests/                            # Evaluation & ground truth
│   ├── evaluation.py                 # P/R/F1/AUC against baselines
│   ├── evaluate_suggestions.py       # Suggestion quality metrics
│   ├── create_ground_truth.py        # Ground truth labelling tool
│   └── ground_truth.json             # 58-pair human-annotated dataset
├── experiments/                      # Parameter tuning notebooks
│   └── parameter_tuning.ipynb
├── outputs/                          # Generated artefacts (ontology, KG, etc.)
├── models/                           # Model artefacts and caches
├── chroma_db/                        # ChromaDB persistent storage
├── .env                              # API keys (gitignored)
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | OpenAI GPT-4o-mini | PDF extraction, RAG recommendations, agent reasoning |
| LLM Framework | LangChain + langchain-openai | Prompt templating, chain composition |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | Semantic text encoding (14K sent/sec on CPU) |
| Vector DB | ChromaDB | Persistent local storage, HNSW indexing, cosine search |
| Ontology | RDFLib (RDF/OWL) | Healthcare concept hierarchy, Turtle export |
| Knowledge Graph | NetworkX | Centrality analysis, community detection |
| Dashboard | Streamlit + Plotly | Interactive UI with heatmaps, gauges, network graphs |
| PDF Processing | pdfplumber + ReportLab | Multi-column PDF extraction + report generation |
| Evaluation | scikit-learn, SciPy | Precision/Recall/F1, ROC/AUC, statistical tests |
| Language | Python 3.10+ | ML/NLP ecosystem |

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone the Repository

```bash
git clone <repo-url>
cd hospital-strategy-alignment-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the OpenAI API Key

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

The system reads this file automatically via `python-dotenv`. Never commit this file — it is already in `.gitignore`.

### 5. Download the spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 6. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

## Usage

1. Open the dashboard in your browser
2. Upload a **Strategic Plan PDF** (e.g., a 5-year hospital strategic plan)
3. Upload an **Action Plan PDF** (e.g., an annual operational action plan)
4. Click **Run Analysis** — the 6-step pipeline processes both documents
5. Explore the five dashboard tabs:

| Tab | What You See |
|-----|-------------|
| **Synchronization** | Overall alignment score gauge, 5x25 heatmap, score distribution, action detail table |
| **Recommendations** | AI-generated improvement suggestions, new action proposals, strategic gap alerts |
| **Knowledge Graph** | Interactive network visualization, bridge nodes, suggested connections |
| **Ontology Browser** | Healthcare concept hierarchy, coverage badges, sunburst chart |
| **Evaluation** | Cross-method agreement, precision/recall metrics, declaration mismatch table |

6. Export results as a **PDF report** or **CSV data** from the dashboard

## Alignment Scoring

The system classifies each action-objective pair into four tiers (configurable in `src/config.py`):

| Score | Classification | Interpretation |
|-------|---------------|----------------|
| >= 0.75 | Excellent | Near-direct operationalisation of strategy |
| 0.60 - 0.74 | Good | Clear strategic support |
| 0.42 - 0.59 | Fair | Partial or indirect alignment |
| < 0.42 | Poor / Orphan | Weak or no meaningful alignment |

Actions scoring below 0.42 for **all** objectives are flagged as **orphan actions** — operational activities with no strategic anchor.

## Evaluation Results

Validated against a **58-pair human-annotated ground truth** dataset:

| Metric | Score |
|--------|:-----:|
| AUC (ROC) | 0.94 |
| F1 Score | 0.87 |
| Precision | 0.83 |
| Recall | 0.91 |
| Pearson r | 0.76 |
| MRR (RAG retrieval) | 0.85 |

## Deploying to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file path to `dashboard/app.py`
4. Add your secrets in the Streamlit Cloud dashboard under **Settings > Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-your-api-key-here"
   OPENAI_MODEL = "gpt-4o-mini"
   ```
5. Deploy — the app will install dependencies from `requirements.txt` automatically

## Running Experiments

### Evaluation Against Ground Truth

```bash
python -m tests.evaluation
```

### Parameter Tuning Notebooks

```bash
jupyter notebook experiments/parameter_tuning.ipynb
```

Four experiments are included:
1. **Embedding Model Comparison** — 4 sentence-transformer models compared on AUC, F1, Pearson r
2. **Threshold Sweep** — 46 thresholds tested to find F1-optimal classification boundary
3. **Ontology Weight Calibration** — embedding vs keyword balance for hybrid scoring
4. **RAG top_k Tuning** — retrieval depth optimisation for recommendation quality

## Configuration

All key parameters are centralised in `src/config.py`:

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `THRESHOLD_EXCELLENT` | 0.75 | Cosine similarity for "Excellent" alignment |
| `THRESHOLD_GOOD` | 0.60 | Cosine similarity for "Good" alignment |
| `THRESHOLD_FAIR` | 0.42 | Cosine similarity for "Fair" alignment |
| `ORPHAN_THRESHOLD` | 0.42 | Below this for all objectives = orphan action |
| `OPENAI_MODEL` | gpt-4o-mini | OpenAI model for extraction and recommendations |
| `LLM_TEMPERATURE` | 0.2 | LLM temperature for generation |
| `MAX_ITERATIONS` | 3 | Agent reasoning iterations |

## License

This project is developed as part of an MSc in Computer Science (Information Retrieval) programme. For academic use only.
