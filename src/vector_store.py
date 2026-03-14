"""
Vector Store for Hospital Strategy–Action Plan Alignment System (ISPS).

This module creates semantic embeddings from the parsed strategic-plan
objectives and action-plan items, stores them in a persistent ChromaDB
vector database, and exposes search and retrieval functions used by the
downstream alignment-scoring engine.

Embedding model
---------------
We use ``all-MiniLM-L6-v2`` from the Sentence-Transformers library.

*  **Dimensions** : 384-dimensional dense vectors.
*  **Max tokens**  : 256 word-pieces (≈ 200 English words).  Inputs that
   exceed this length are truncated.  The composite text fields we
   construct (goal statement + goal descriptions + KPIs) typically stay
   well within this limit.
*  **Training**    : Distilled from ``microsoft/MiniLM-L12-H384``, then
   fine-tuned on > 1 billion sentence pairs for semantic-similarity tasks.
*  **Speed**       : ~14 000 sentences/sec on a single CPU, making it
   practical for laptop-scale use without a GPU.

Similarity metric
-----------------
ChromaDB computes **cosine similarity** by default when the collection
is created with ``metadata={"hnsw:space": "cosine"}``.  Cosine similarity
measures the angle between two vectors in the 384-dimensional space and
is robust to differences in document length (because vectors are
L2-normalised before comparison).

*  **1.0**  → identical direction  (maximum similarity)
*  **0.0**  → orthogonal           (no similarity)
*  **−1.0** → opposite direction   (anti-similarity)

ChromaDB's ``query`` method returns a *distance* (``1 − cosine_similarity``),
so lower distance = higher similarity.  The helper functions in this
module convert distances back to similarity scores for easier
interpretation.

Collections
-----------
Two ChromaDB collections are maintained:

``strategic_objectives``
    One document per strategic objective (A–E).  The embedded text is a
    composite of the goal statement, individual strategic-goal
    descriptions, and KPI names.  Metadata includes the objective code,
    name, and keyword list.

``action_items``
    One document per action item (1–25).  The embedded text is a
    composite of the action title, description, expected outcome, and
    KPI text.  Metadata includes objective code, owner, budget, and
    timeline quarters.

Typical usage::

    from src.vector_store import VectorStore

    vs = VectorStore()
    vs.build_from_json()          # Loads JSON, embeds, persists

    results = vs.search_similar(
        "cardiac catheterisation equipment",
        collection_name="action_items",
        top_k=3,
    )

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vector_store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

STRATEGIC_JSON = DATA_DIR / "strategic_plan.json"
ACTION_JSON = DATA_DIR / "action_plan.json"

# Embedding model configuration
# ─────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
# It balances quality and speed for semantic-similarity tasks.
# See https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
# ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # fixed for this model

# Collection names
STRATEGIC_COLLECTION = "strategic_objectives"
ACTION_COLLECTION = "action_items"

# ChromaDB similarity metric
# ─────────────────────────────────────────────────────────────────────
# "cosine" normalises vectors before comparison, so document length
# does not skew similarity.  Alternatives supported by ChromaDB:
#   "l2"  — Euclidean distance (sensitive to magnitude)
#   "ip"  — inner product (requires pre-normalised vectors)
# We choose cosine because our documents vary in length and we want
# pure directional similarity.
# ─────────────────────────────────────────────────────────────────────
SIMILARITY_SPACE = "cosine"


# ---------------------------------------------------------------------------
# Text composition helpers
# ---------------------------------------------------------------------------

def _compose_objective_text(obj: dict[str, Any]) -> str:
    """Build a single embedding-ready text from a strategic-objective dict.

    The composite includes the goal statement, each strategic goal's
    description, and KPI names — the three most semantically meaningful
    components for alignment matching.

    Args:
        obj: A parsed strategic-objective dictionary (from
             ``strategic_plan.json``).

    Returns:
        A plain-text string suitable for sentence-transformer encoding.
    """
    parts: list[str] = []

    # Goal statement
    if obj.get("goal_statement"):
        parts.append(obj["goal_statement"])

    # Individual strategic goals
    for goal in obj.get("strategic_goals", []):
        parts.append(f"{goal['id']}: {goal['description']}")

    # KPI names (not baselines/targets — just the indicator label)
    for kpi in obj.get("kpis", []):
        kpi_name = kpi.get("KPI", "")
        if kpi_name:
            parts.append(kpi_name)

    return " ".join(parts)


def _compose_action_text(action: dict[str, Any]) -> str:
    """Build a single embedding-ready text from an action-item dict.

    The composite includes the title, full description, expected outcome,
    and KPI text — capturing the operational semantics of the action.

    Args:
        action: A parsed action-item dictionary (from
                ``action_plan.json``).

    Returns:
        A plain-text string suitable for sentence-transformer encoding.
    """
    parts: list[str] = [
        action.get("title", ""),
        action.get("description", ""),
        action.get("expected_outcome", ""),
    ]

    for kpi in action.get("kpis", []):
        parts.append(kpi)

    return " ".join(p for p in parts if p)


def _compose_objective_metadata(obj: dict[str, Any]) -> dict[str, str]:
    """Build ChromaDB metadata for a strategic-objective document.

    ChromaDB metadata values must be ``str``, ``int``, ``float``, or
    ``bool``.  We serialise list fields as comma-separated strings.

    Args:
        obj: A parsed strategic-objective dictionary.

    Returns:
        A flat metadata dict for ChromaDB storage.
    """
    return {
        "code": obj.get("code", ""),
        "name": obj.get("name", ""),
        "goal_statement": obj.get("goal_statement", "")[:500],
        "num_goals": len(obj.get("strategic_goals", [])),
        "num_kpis": len(obj.get("kpis", [])),
        "keywords": ", ".join(obj.get("keywords", [])),
    }


def _compose_action_metadata(action: dict[str, Any]) -> dict[str, str]:
    """Build ChromaDB metadata for an action-item document.

    Args:
        action: A parsed action-item dictionary.

    Returns:
        A flat metadata dict for ChromaDB storage.
    """
    return {
        "action_number": action.get("action_number", 0),
        "title": action.get("title", ""),
        "strategic_objective_code": action.get("strategic_objective_code", ""),
        "strategic_objective_name": action.get("strategic_objective_name", ""),
        "action_owner": action.get("action_owner", ""),
        "budget_lkr_millions": action.get("budget_lkr_millions", 0.0),
        "timeline": action.get("timeline", ""),
        "quarters": ", ".join(action.get("quarters", [])),
        "keywords": ", ".join(action.get("keywords", [])),
    }


# ---------------------------------------------------------------------------
# VectorStore class
# ---------------------------------------------------------------------------

class VectorStore:
    """Manages ChromaDB collections and sentence-transformer embeddings.

    This class encapsulates the full lifecycle of the vector store:
    loading the embedding model, initialising persistent ChromaDB
    collections, inserting documents, and querying by semantic
    similarity.

    Attributes:
        chroma_dir:  Path to the persistent ChromaDB storage directory.
        model:       The loaded ``SentenceTransformer`` instance.
        client:      The ``chromadb.PersistentClient`` instance.

    Example::

        vs = VectorStore()
        vs.build_from_json()

        results = vs.search_similar("elderly care", "action_items", top_k=3)
        for doc_id, score, meta in zip(
            results["ids"], results["scores"], results["metadatas"]
        ):
            print(f"{doc_id}  sim={score:.3f}  {meta['title']}")
    """

    def __init__(
        self,
        chroma_dir: Path | str = CHROMA_DIR,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        """Initialise the vector store with embedding model and ChromaDB.

        Args:
            chroma_dir:  Directory for ChromaDB persistent storage.
                         Created automatically if it does not exist.
            model_name:  Sentence-transformer model identifier.

        The constructor loads the embedding model into memory and opens
        (or creates) the ChromaDB persistent client.  No documents are
        embedded at this stage — call :meth:`build_from_json` or
        :meth:`embed_documents` to populate the collections.
        """
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        # ── Load sentence-transformer model ──────────────────────────
        # The model is downloaded on first use and cached locally by the
        # sentence-transformers library (~80 MB for all-MiniLM-L6-v2).
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded. Embedding dimension: %d (expected %d).",
            embedding_dim,
            EMBEDDING_DIMENSION,
        )

        # ── Initialise ChromaDB persistent client ────────────────────
        # PersistentClient writes to disk automatically after every
        # add / update / delete, so no explicit .persist() call is
        # needed (that API was removed in ChromaDB >= 0.4).
        logger.info("Opening ChromaDB at: %s", self.chroma_dir)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB initialised.")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _get_or_create_collection(
        self, name: str
    ) -> chromadb.Collection:
        """Return an existing collection or create a new one.

        Collections are configured with the HNSW index using cosine
        distance.  HNSW (Hierarchical Navigable Small World) is an
        approximate-nearest-neighbour algorithm that provides sub-linear
        query time — ideal for interactive dashboard queries.

        Args:
            name: The collection name.

        Returns:
            A ``chromadb.Collection`` object.
        """
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={
                # ── Similarity metric ────────────────────────────
                # "cosine" means ChromaDB will L2-normalise vectors
                # and use (1 - cosine_similarity) as the distance.
                # Lower distance → higher similarity.
                "hnsw:space": SIMILARITY_SPACE,
            },
        )
        logger.info(
            "Collection '%s': %d existing documents.", name, collection.count()
        )
        return collection

    def reset_collection(self, name: str) -> chromadb.Collection:
        """Delete and recreate a collection, removing all documents.

        Args:
            name: The collection name to reset.

        Returns:
            A fresh, empty ``chromadb.Collection``.
        """
        try:
            self.client.delete_collection(name)
            logger.info("Deleted collection '%s'.", name)
        except (ValueError, Exception) as exc:
            # ChromaDB raises NotFoundError (subclass of Exception) in
            # newer versions, ValueError in older ones.
            if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                pass  # collection did not exist — nothing to delete
            else:
                raise
        return self._get_or_create_collection(name)

    # ------------------------------------------------------------------
    # Embedding & insertion
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
        collection_name: str,
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> chromadb.Collection:
        """Embed a list of texts and insert them into a ChromaDB collection.

        This is the low-level insertion method.  For the standard
        pipeline, prefer :meth:`build_from_json` which handles text
        composition and metadata automatically.

        Args:
            texts:           Plain-text documents to embed.
            collection_name: Target ChromaDB collection name.
            ids:             Unique document IDs (auto-generated as
                             ``"doc_0"``, ``"doc_1"``, … if omitted).
            metadatas:       Optional list of metadata dicts, one per
                             document.

        Returns:
            The ``chromadb.Collection`` containing the new documents.

        Raises:
            ValueError: If *ids* or *metadatas* length doesn't match
                        *texts*.
        """
        if not texts:
            logger.warning("embed_documents called with empty text list.")
            return self._get_or_create_collection(collection_name)

        if ids and len(ids) != len(texts):
            raise ValueError(
                f"ids length ({len(ids)}) != texts length ({len(texts)})"
            )
        if metadatas and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) != texts length ({len(texts)})"
            )

        # ── Generate embeddings ──────────────────────────────────────
        # SentenceTransformer.encode() returns a numpy array of shape
        # (n_texts, 384).  ChromaDB accepts list-of-lists, so we
        # convert via .tolist().
        logger.info(
            "Encoding %d documents for collection '%s' …",
            len(texts),
            collection_name,
        )
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalise for cosine
        ).tolist()

        # ── Auto-generate IDs if not supplied ────────────────────────
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        # ── Upsert into collection ───────────────────────────────────
        # upsert = insert-or-update, safe to call repeatedly.
        collection = self._get_or_create_collection(collection_name)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "Upserted %d documents into '%s'. Total: %d.",
            len(texts),
            collection_name,
            collection.count(),
        )
        return collection

    # ------------------------------------------------------------------
    # Build from JSON
    # ------------------------------------------------------------------

    def build_from_json(
        self,
        strategic_json: Path | str = STRATEGIC_JSON,
        action_json: Path | str = ACTION_JSON,
        reset: bool = True,
    ) -> dict[str, int]:
        """Load parsed JSON files, embed documents, and populate ChromaDB.

        This is the primary pipeline method that reads the output of
        ``DocumentProcessor``, composes embedding-ready text for each
        objective and action, generates embeddings, and upserts them
        into the two collections.

        Args:
            strategic_json: Path to ``strategic_plan.json``.
            action_json:    Path to ``action_plan.json``.
            reset:          If ``True`` (default), delete existing
                            documents before inserting.  Set to
                            ``False`` to incrementally add.

        Returns:
            A dict with keys ``"strategic_objectives"`` and
            ``"action_items"``, each holding the document count.

        Raises:
            FileNotFoundError: If either JSON file is missing.
        """
        strategic_json = Path(strategic_json)
        action_json = Path(action_json)

        for path in (strategic_json, action_json):
            if not path.exists():
                raise FileNotFoundError(
                    f"JSON file not found: {path}. "
                    "Run DocumentProcessor.save_all() first."
                )

        # ── Load JSON ────────────────────────────────────────────────
        with open(strategic_json, encoding="utf-8") as fh:
            strategic_data = json.load(fh)
        with open(action_json, encoding="utf-8") as fh:
            action_data = json.load(fh)

        # ── Reset collections if requested ───────────────────────────
        if reset:
            self.reset_collection(STRATEGIC_COLLECTION)
            self.reset_collection(ACTION_COLLECTION)

        # ── Embed strategic objectives ───────────────────────────────
        objectives = strategic_data.get("objectives", [])
        obj_texts: list[str] = []
        obj_ids: list[str] = []
        obj_metas: list[dict[str, Any]] = []

        for obj in objectives:
            text = _compose_objective_text(obj)
            obj_texts.append(text)
            obj_ids.append(f"obj_{obj['code']}")
            obj_metas.append(_compose_objective_metadata(obj))

        self.embed_documents(
            texts=obj_texts,
            collection_name=STRATEGIC_COLLECTION,
            ids=obj_ids,
            metadatas=obj_metas,
        )

        # ── Embed action items ───────────────────────────────────────
        actions = action_data.get("actions", [])
        act_texts: list[str] = []
        act_ids: list[str] = []
        act_metas: list[dict[str, Any]] = []

        for action in actions:
            text = _compose_action_text(action)
            act_texts.append(text)
            act_ids.append(f"action_{action['action_number']}")
            act_metas.append(_compose_action_metadata(action))

        self.embed_documents(
            texts=act_texts,
            collection_name=ACTION_COLLECTION,
            ids=act_ids,
            metadatas=act_metas,
        )

        counts = {
            STRATEGIC_COLLECTION: len(obj_texts),
            ACTION_COLLECTION: len(act_texts),
        }
        logger.info("Build complete: %s", counts)
        return counts

    # ------------------------------------------------------------------
    # Query / search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
    ) -> dict[str, list]:
        """Find the most semantically similar documents to a query string.

        The query is encoded with the same sentence-transformer model
        and compared against all documents in the specified collection
        using cosine similarity (via ChromaDB's HNSW index).

        Args:
            query:           Free-text search query.
            collection_name: Which collection to search
                             (``"strategic_objectives"`` or
                             ``"action_items"``).
            top_k:           Number of results to return (default 5).

        Returns:
            A dictionary with four parallel lists::

                {
                    "ids":       ["action_1", "action_3", …],
                    "scores":    [0.87, 0.72, …],       # cosine similarity
                    "documents": ["Adjacent Land …", …], # original text
                    "metadatas": [{…}, {…}, …],          # stored metadata
                }

            Results are sorted by descending similarity score.

        Raises:
            ValueError: If the collection does not exist.
        """
        # ── Encode query ─────────────────────────────────────────────
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # ── Query ChromaDB ───────────────────────────────────────────
        collection = self._get_or_create_collection(collection_name)
        if collection.count() == 0:
            logger.warning("Collection '%s' is empty.", collection_name)
            return {"ids": [], "scores": [], "documents": [], "metadatas": []}

        actual_k = min(top_k, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        # ── Convert distances to similarity scores ───────────────────
        # ChromaDB returns cosine *distance* = 1 − cosine_similarity.
        # We invert to get a 0–1 similarity score where 1 = identical.
        distances = results["distances"][0]
        similarities = [round(1.0 - d, 4) for d in distances]

        return {
            "ids": results["ids"][0],
            "scores": similarities,
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
        }

    # ------------------------------------------------------------------
    # Bulk retrieval
    # ------------------------------------------------------------------

    def get_all_embeddings(
        self, collection_name: str
    ) -> dict[str, Any]:
        """Retrieve all documents, embeddings, and metadata from a collection.

        This is used by the alignment engine to compute pairwise
        similarity matrices between objectives and actions without
        repeated model inference.

        Args:
            collection_name: The collection to retrieve from.

        Returns:
            A dictionary::

                {
                    "ids":        ["obj_A", "obj_B", …],
                    "embeddings": [[0.012, −0.034, …], …],  # 384-d each
                    "documents":  ["Deliver consistently …", …],
                    "metadatas":  [{…}, {…}, …],
                }

            Returns empty lists if the collection has no documents.
        """
        collection = self._get_or_create_collection(collection_name)
        count = collection.count()
        if count == 0:
            logger.warning("Collection '%s' is empty.", collection_name)
            return {
                "ids": [],
                "embeddings": [],
                "documents": [],
                "metadatas": [],
            }

        results = collection.get(
            include=["embeddings", "documents", "metadatas"],
        )
        logger.info(
            "Retrieved %d documents + embeddings from '%s'.",
            len(results["ids"]),
            collection_name,
        )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def collection_stats(self) -> dict[str, dict[str, Any]]:
        """Return basic statistics for all managed collections.

        Returns:
            A dict keyed by collection name, each containing
            ``"count"`` and ``"metadata"`` (collection-level config).
        """
        stats = {}
        for name in [STRATEGIC_COLLECTION, ACTION_COLLECTION]:
            try:
                col = self.client.get_collection(name)
                stats[name] = {
                    "count": col.count(),
                    "metadata": col.metadata,
                }
            except ValueError:
                stats[name] = {"count": 0, "metadata": {}}
        return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Build the vector store from JSON and run demonstration queries.

    This entry point is intended for development and verification.
    It performs the full build pipeline, prints collection statistics,
    and runs a set of sample queries to demonstrate semantic search.
    """
    logger.info("=" * 60)
    logger.info("ISPS Vector Store — Starting")
    logger.info("=" * 60)

    # ── Build ─────────────────────────────────────────────────────────
    vs = VectorStore()
    counts = vs.build_from_json()

    print("\n" + "=" * 60)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 60)
    for name, count in counts.items():
        print(f"  Collection '{name}': {count} documents")

    # ── Stats ─────────────────────────────────────────────────────────
    stats = vs.collection_stats()
    print(f"\nCollection config:")
    for name, info in stats.items():
        print(f"  {name}: {info['count']} docs, space={info['metadata']}")

    # ── Sample queries ────────────────────────────────────────────────
    sample_queries = [
        ("cardiac catheterisation and heart procedures", ACTION_COLLECTION),
        ("telemedicine and Maldives outreach", ACTION_COLLECTION),
        ("elderly care for senior citizens", ACTION_COLLECTION),
        ("staff recruitment and workforce", ACTION_COLLECTION),
        ("digital health and cybersecurity", STRATEGIC_COLLECTION),
        ("community health screening", STRATEGIC_COLLECTION),
    ]

    print("\n" + "=" * 60)
    print("SAMPLE SIMILARITY SEARCHES")
    print("=" * 60)

    for query, collection in sample_queries:
        results = vs.search_similar(query, collection, top_k=3)
        print(f"\nQuery: \"{query}\" → [{collection}]")
        for doc_id, score, meta in zip(
            results["ids"], results["scores"], results["metadatas"]
        ):
            if collection == ACTION_COLLECTION:
                label = f"Action {meta.get('action_number', '?')}: {meta.get('title', '')[:45]}"
            else:
                label = f"Obj {meta.get('code', '?')}: {meta.get('name', '')}"
            print(f"  {score:.3f}  {doc_id:<12}  {label}")

    # ── Embedding shape verification ─────────────────────────────────
    all_obj = vs.get_all_embeddings(STRATEGIC_COLLECTION)
    if all_obj["embeddings"] is not None and len(all_obj["embeddings"]) > 0:
        dim = len(all_obj["embeddings"][0])
        print(f"\nEmbedding dimension verification: {dim}")
        assert dim == EMBEDDING_DIMENSION, (
            f"Expected {EMBEDDING_DIMENSION}, got {dim}"
        )

    print(f"\nChromaDB persisted at: {vs.chroma_dir}")


if __name__ == "__main__":
    main()
