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
from pathlib import Path as PathLib
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import embedding client
from embeddings.embedding_client import get_embedding_client


# Configuration
QDANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")

# Cached client
_client: Optional[QdrantClient] = None


def init_qdrant(url: str = QDANT_URL, api_key: Optional[str] = QDANT_API_KEY):
    """Initialize Qdrant client (called at app startup)."""
    global _client
    if _client is None:
        try:
            _client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=30,
            )
            # Test connection
            _client.get_collections()
            print(f"[INFO] Qdrant client initialized and connected: {url}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Qdrant client: {e}")
            raise
    return _client


def init_embedding_service(url: str = EMBEDDING_SERVICE_URL):
    """Initialize embedding service client (called at app startup)."""
    try:
        embedding_client = get_embedding_client()
        # Test connection
        health = embedding_client.health()
        print(f"[INFO] Embedding service connected: {url} - {health}")
        return embedding_client
    except Exception as e:
        print(f"[ERROR] Failed to connect to embedding service at {url}: {e}")
        raise


def get_client() -> QdrantClient:
    """Get or initialize Qdrant client."""
    if _client is None:
        init_qdrant()
    return _client


def embed_query(query: str) -> list[float]:
    """Embed query using remote embedding service."""
    try:
        embedding_client = get_embedding_client()
        print(f"[DEBUG] Encoding query via embedding service: {query[:50]}...")
        embeddings = embedding_client.embed([query])
        embedding_vector = embeddings[0] if embeddings else []
        print(f"[DEBUG] Query embedding shape: {len(embedding_vector)}")
        return embedding_vector
    except Exception as e:
        print(f"[ERROR] Failed to embed query: {e}")
        raise


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
    try:
        client = get_client()

        # Embed query using embedding service
        query_embedding = embed_query(query)

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
        print(f"[DEBUG] Searching Qdrant collection '{COLLECTION_NAME}'...")
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )
        print(f"[DEBUG] Found {len(search_results)} results")

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

    except AttributeError as e:
        print(f"[ERROR] Qdrant client method error: {e}")
        raise ValueError(f"Qdrant client error - method not found: {e}")
    except Exception as e:
        print(f"[ERROR] Search error: {type(e).__name__}: {e}")
        raise


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
