"""
qdrant_search.py
────────────────
Search backend for CUAD contracts using Qdrant vector database.

Supports:
  - Semantic similarity search (vector-based)
  - Hybrid search combining metadata filtering and vector similarity
  - Document filtering by title
  - Configurable result count and scoring thresholds

Qdrant Collection Structure:
  - collection_name: "cuad_contracts"
  - vector_size: 384 (all-MiniLM-L6-v2)
  - distance: COSINE
  - payload fields: doc_id, title, text, page_start, page_end, pdf_path, char_start, char_end
"""

import os
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cached model and client
_model: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None


def init_qdrant(url: str = QDRANT_URL, api_key: Optional[str] = QDRANT_API_KEY):
    """Initialize Qdrant client (called at app startup)."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=30,
        )
        print(f"[INFO] Qdrant client initialized: {url}")
    return _client


def init_model(device: str = "cpu"):
    """Initialize embedding model (called at app startup)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"[INFO] Embedding model loaded: {EMBEDDING_MODEL}")
    return _model


def get_client() -> QdrantClient:
    """Get or initialize Qdrant client."""
    if _client is None:
        init_qdrant()
    return _client


def get_model() -> SentenceTransformer:
    """Get or initialize embedding model."""
    if _model is None:
        init_model()
    return _model


def semantic_search(
    query: str,
    top_k: int = 10,
    document_name: Optional[str] = None,
    min_score: float = 0.0,
) -> Tuple[list[dict], dict]:
    """
    Semantic search using vector similarity in Qdrant.

    Args:
        query: Search text to embed and find similar results
        top_k: Number of results to return (1-100)
        document_name: Optional filter by contract title
        min_score: Minimum cosine similarity score (0.0-1.0)

    Returns:
        (results, metadata) where:
          results: List of search result dicts with scores, text, source info
          metadata: Dict with query, top_k, strategy used, filter info
    """
    client = get_client()
    model = get_model()

    # Embed query
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # Build filter if document_name specified
    search_filter = None
    if document_name:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="title",
                    match=MatchValue(value=document_name),
                )
            ]
        )

    # Search in Qdrant
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )

    # Format results
    results = []
    for point in search_results:
        if point.score >= min_score:
            payload = point.payload
            results.append(
                {
                    "id": payload.get("doc_id"),
                    "score": point.score,
                    "title": payload.get("title"),
                    "text": payload.get("text"),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "char_start": payload.get("char_start"),
                    "char_end": payload.get("char_end"),
                    "pdf_path": payload.get("pdf_path"),
                    "source": ["embeddings"],  # Qdrant is vector/semantic search
                }
            )

    metadata = {
        "query": query,
        "top_k": top_k,
        "strategy": "semantic_search",
        "document_filter": document_name,
        "min_score": min_score,
        "results_count": len(results),
    }

    return results, metadata


def hybrid_search(
    query: str,
    top_k: int = 10,
    document_name: Optional[str] = None,
    min_score: float = 0.0,
) -> Tuple[list[dict], dict]:
    """
    Hybrid search combining semantic similarity with metadata filtering.

    For now, this is equivalent to semantic_search since Qdrant stores
    pre-computed embeddings. In the future, could implement:
    - BM25 via full-text search (requires additional indexing)
    - RRF fusion of BM25 + semantic scores

    Args:
        query: Search text
        top_k: Number of results
        document_name: Optional document title filter
        min_score: Minimum score threshold

    Returns:
        (results, metadata) tuple
    """
    return semantic_search(
        query=query,
        top_k=top_k,
        document_name=document_name,
        min_score=min_score,
    )


def search(
    query: str,
    top_k: int = 10,
    document_name: Optional[str] = None,
    strategy: str = "semantic_search",
    min_score: float = 0.0,
) -> Tuple[list[dict], dict]:
    """
    Main search router supporting multiple strategies.

    Args:
        query: Search text
        top_k: Number of results (1-100)
        document_name: Optional filter by contract title
        strategy: One of "semantic_search", "hybrid_search"
        min_score: Minimum score threshold

    Returns:
        (results, metadata) tuple
    """
    top_k = max(1, min(top_k, 100))  # Clamp to valid range

    if strategy == "hybrid_search":
        return hybrid_search(
            query=query,
            top_k=top_k,
            document_name=document_name,
            min_score=min_score,
        )
    else:  # default: "semantic_search"
        return semantic_search(
            query=query,
            top_k=top_k,
            document_name=document_name,
            min_score=min_score,
        )


def get_collection_stats() -> dict:
    """Get collection statistics for health checks."""
    try:
        client = get_client()
        collection_info = client.get_collection(COLLECTION_NAME)
        return {
            "collection": COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": str(collection_info.config.params.vectors.distance),
            "status": "ready",
        }
    except Exception as e:
        return {
            "collection": COLLECTION_NAME,
            "status": "error",
            "error": str(e),
        }


# %%
