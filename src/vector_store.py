"""
vector_store.py — Embed text with sentence-transformers and store/query in ChromaDB.
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, CHROMA_DIR


_embedding_model = None
_chroma_client   = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"📌 Loading embedding model: {EMBEDDING_MODEL}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Embedding model loaded.")
    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    """Connect to (or create) the local ChromaDB persistent store."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def embed_and_store(items: list, collection_name: str) -> None:
    """
    Embed items and store their vectors in a named ChromaDB collection.

    Each item must have: id, title, description.
    Existing items in the collection are cleared before adding new ones.
    """
    print(f"\n📌 Embedding {len(items)} items → '{collection_name}'...")

    client = get_chroma_client()
    model  = get_embedding_model()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids       = [str(item["id"]) for item in items]
    texts     = [item["title"] + " " + item.get("description", "") for item in items]
    metadatas = [{"title": item["title"], "description": item.get("description", "")} for item in items]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"✅ Stored {len(items)} items in '{collection_name}'.")


def query(text: str, collection_name: str, top_k: int = 3) -> list:
    """
    Return top_k most similar items to the query text from a ChromaDB collection.

    Returns a list of dicts: {id, document, distance, metadata}.
    """
    model        = get_embedding_model()
    query_vector = model.encode([text]).tolist()

    client     = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    n_results = min(top_k, collection.count())
    if n_results == 0:
        print(f"⚠️  Collection '{collection_name}' is empty.")
        return []

    results = collection.query(query_embeddings=query_vector, n_results=n_results)

    return [
        {
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def load_collection_items(collection_name: str) -> list:
    """Return all items stored in a ChromaDB collection."""
    collection = get_chroma_client().get_or_create_collection(name=collection_name)
    data = collection.get()

    return [
        {
            "id":       data["ids"][i],
            "document": data["documents"][i] if data["documents"] else "",
            "metadata": data["metadatas"][i] if data["metadatas"] else {},
        }
        for i in range(len(data["ids"]))
    ]
