"""
vector_store.py
---------------
Embed text items and store/query them in a local ChromaDB vector database.

How it fits in the pipeline:
  pdf_processor → [this module] → alignment_scorer / rag_engine

Beginner note:
  "Embedding" = converting text into a list of numbers (a vector) that captures
  the *meaning* of the text. Similar meanings → similar vectors.

  "ChromaDB" = a lightweight vector database that stores these vectors on disk.
  You can query it with any text, and it returns the most similar stored items.
  Think of it like a Google Search that works on meaning, not just keywords.

The embedding model used is all-MiniLM-L6-v2 (384 numbers per text).
"""

# --- Standard library ---
import os

# --- Third-party ---
import chromadb                                        # local vector database
from sentence_transformers import SentenceTransformer  # text → vector model

# --- Local ---
from src.config import EMBEDDING_MODEL, CHROMA_DIR


# =============================================================================
# SINGLETON HELPERS
# These functions load heavy objects once and reuse them (no re-loading per call).
# =============================================================================

_embedding_model = None   # module-level cache for the embedding model
_chroma_client   = None   # module-level cache for the ChromaDB client


def get_embedding_model() -> SentenceTransformer:
    """
    Load and return the sentence-transformers embedding model.

    Uses a module-level cache so the model is only loaded once (loading
    takes ~5 seconds the first time; subsequent calls are instant).

    Returns:
        SentenceTransformer: Model that converts text → 384-dim float vectors.
    """
    global _embedding_model

    if _embedding_model is None:
        print(f"📌 Loading embedding model: {EMBEDDING_MODEL} (first load may take a moment)...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Embedding model loaded.")

    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create and return a persistent ChromaDB client.

    ChromaDB stores its data in the chroma_db/ folder. Persistent storage means
    embeddings survive between program runs — we don't re-embed every time.

    Returns:
        chromadb.PersistentClient: Connected to the local chroma_db/ folder.
    """
    global _chroma_client

    if _chroma_client is None:
        # Create the storage folder if it doesn't exist
        os.makedirs(CHROMA_DIR, exist_ok=True)
        print(f"📌 Connecting to ChromaDB at: {CHROMA_DIR}")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        print("✅ ChromaDB connected.")

    return _chroma_client


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def embed_and_store(items: list, collection_name: str) -> None:
    """
    Embed a list of items and store their vectors in a named ChromaDB collection.

    What gets embedded is: item["title"] + " " + item["description"]
    This combined text gives the model enough context to capture the meaning.

    If the collection already has items, they are cleared and re-added fresh.

    Args:
        items (list): List of dicts, each with keys: id, title, description.
        collection_name (str): Name of the ChromaDB collection (like a table name).
    """
    print(f"\n📌 Embedding and storing {len(items)} items into '{collection_name}'...")

    # Step 1 — Get our tools
    client = get_chroma_client()
    model  = get_embedding_model()

    # Step 2 — Get or create the collection
    # get_or_create_collection: creates it if it doesn't exist, reuses if it does
    collection = client.get_or_create_collection(
        name=collection_name,
        # cosine distance is better than Euclidean for text similarity
        metadata={"hnsw:space": "cosine"}
    )

    # Step 3 — Clear existing items so we start fresh on each run
    # This avoids duplicate embeddings when re-running the pipeline
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"   Cleared {len(existing['ids'])} existing items from '{collection_name}'.")

    # Step 4 — Build the text strings to embed
    ids       = []
    texts     = []
    metadatas = []

    for item in items:
        ids.append(str(item["id"]))
        # Combine title and description for richer embeddings
        texts.append(item["title"] + " " + item.get("description", ""))
        metadatas.append({
            "title":       item["title"],
            "description": item.get("description", ""),
        })

    # Step 5 — Encode all texts into vectors at once (batch is faster than one by one)
    print(f"   Encoding {len(texts)} texts with {EMBEDDING_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=False)

    # ChromaDB needs embeddings as a plain Python list of lists, not numpy arrays
    embeddings_list = embeddings.tolist()

    # Step 6 — Store everything in ChromaDB
    collection.add(
        ids        = ids,
        embeddings = embeddings_list,
        documents  = texts,      # raw text stored alongside vector for retrieval
        metadatas  = metadatas,
    )

    print(f"✅ Stored {len(items)} items in ChromaDB collection '{collection_name}'.")


def query(text: str, collection_name: str, top_k: int = 3) -> list:
    """
    Find the top_k most similar items in a collection to the given query text.

    How it works:
      1. Embed the query text into a vector.
      2. ChromaDB compares that vector to all stored vectors using cosine similarity.
      3. Return the closest matches.

    Args:
        text (str): The query text to search for.
        collection_name (str): Which ChromaDB collection to search.
        top_k (int): How many results to return (default 3).

    Returns:
        list: List of dicts, each with keys:
              id (str), document (str), distance (float), metadata (dict).
              Lower distance = more similar (cosine distance, not similarity).
    """
    print(f"📌 Querying '{collection_name}' for top-{top_k} matches...")

    # Step 1 — Embed the query text
    model = get_embedding_model()
    query_vector = model.encode([text]).tolist()

    # Step 2 — Run the query against ChromaDB
    client     = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # n_results must not exceed the number of stored items
    n_items = collection.count()
    n_results = min(top_k, n_items)

    if n_results == 0:
        print(f"⚠️  Collection '{collection_name}' is empty. Run embed_and_store first.")
        return []

    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results,
    )

    # Step 3 — Reformat results into a clean list of dicts
    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    print(f"✅ Query complete. Returned {len(output)} results.")
    return output


def load_collection_items(collection_name: str) -> list:
    """
    Retrieve all items stored in a ChromaDB collection as a list of dicts.

    Useful for inspecting what's stored without running a search query.

    Args:
        collection_name (str): Name of the ChromaDB collection.

    Returns:
        list: List of dicts with keys: id, document, metadata.
    """
    client     = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    data       = collection.get()

    output = []
    for i in range(len(data["ids"])):
        output.append({
            "id":       data["ids"][i],
            "document": data["documents"][i] if data["documents"] else "",
            "metadata": data["metadatas"][i] if data["metadatas"] else {},
        })

    return output
