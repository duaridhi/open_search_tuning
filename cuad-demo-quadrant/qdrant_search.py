"""
qdrant_search.py
────────────────
Search backend for CUAD contracts using Qdrant vector database.

Supports:
    - Semantic similarity search (vector-based)
    - Hybrid search combining metadata filtering and vector similarity
    - Document filtering by title
    - Configurable result count and scoring thresholds
highlight_qdrant_data_point
Qdrant Collection Structure:
  - collection_name: "cuad_contracts"
  - vector_size: 384 (all-MiniLM-L6-v2)
  - distance: COSINE
    - payload fields: doc_id, title, text, page_start, page_end, pdf_path, char_start, char_end,
        page_offset_start, page_offset_end
"""

import logging
import os
from typing import Optional, Tuple
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import cluster connection and embedding client
from qdrant_cluster_connect import get_qdrant_client, get_cluster_info
from embeddings.embedding_client import get_embedding_client

logger = logging.getLogger(__name__)


# Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")


def init_qdrant():
    """Initialize Qdrant client via cluster connection module."""
    return get_qdrant_client()


def get_client():
    """Get Qdrant client from cluster connection module."""
    return get_qdrant_client()


def embed_query(query: str) -> list[float]:
    """Embed query using remote embedding service."""
    try:
        embedding_client = get_embedding_client()
        embeddings = embedding_client.embed([query])
        embedding_vector = embeddings[0] if embeddings else []
        return embedding_vector
    except Exception as e:
        logger.error("Failed to embed query: %s", e)
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

        # Search in Qdrant using query_points (for qdrant-client 1.7.x)
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        embedding_client = get_embedding_client()
        
        for point in search_results.points:
            if point.score >= min_score:
                payload = point.payload
                text = payload.get("text", "")
                title = payload.get("title", "Unknown")
                chunk_page_offset_start = payload.get("page_offset_start", 0)
                
                # Get highlights from embedding service
                highlighted_sentences = []
                highlight_sentence_indexes = []
                highlight_page_offset_starts = []
                highlight_page_offset_ends = []
                try:
                    highlights_response = embedding_client.highlight(
                        query=query,
                        document=text
                    )
                    highlighted_sentences = highlights_response.get("highlighted_sentences", [])
                    highlight_sentence_indexes = highlights_response.get("highlight_sentence_indexes", [])
                    highlight_offsets = highlights_response.get("highlight_offsets", [])
                    
                    # Convert chunk-relative offsets to page-relative offsets
                    for chunk_start, chunk_end in highlight_offsets:
                        page_start = chunk_page_offset_start + chunk_start if chunk_page_offset_start else chunk_start
                        page_end = chunk_page_offset_start + chunk_end if chunk_page_offset_start else chunk_end
                        highlight_page_offset_starts.append(page_start)
                        highlight_page_offset_ends.append(page_end)
                except Exception as e:
                    logger.warning("Failed to highlight result for '%s': %s", title, e)
                
                results.append(
                    {
                        "id": payload.get("doc_id"),
                        "score": point.score,
                        "title": title,
                        "text": text,
                        "page_start": payload.get("page_start"),
                        "page_end": payload.get("page_end"),
                        "char_start": payload.get("char_start"),
                        "char_end": payload.get("char_end"),
                        "page_offset_start": highlight_page_offset_starts if highlight_page_offset_starts else payload.get("page_offset_start"),
                        "page_offset_end": highlight_page_offset_ends if highlight_page_offset_ends else payload.get("page_offset_end"),
                        "pdf_path": payload.get("pdf_path"),
                        "source": ["embeddings"],  # Qdrant is vector/semantic search
                        "highlighted_sentences": highlighted_sentences,
                        "highlight_sentence_indexes": highlight_sentence_indexes,
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
        logger.error("Qdrant client method error: %s", e)
        raise ValueError(f"Qdrant client error - using query_points method: {e}")
    except Exception as e:
        logger.error("Search error: %s: %s", type(e).__name__, e)
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
    -  fusion of BM25 + semantic scores

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

        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)

        # Extract points count
        points_count = getattr(collection_info, "points_count", 0)
        
        # Extract vector config - handle different nested structures
        vector_size = None
        distance = None
        
        try:
            # Try new structure first
            if hasattr(collection_info, "config"):
                config = collection_info.config
                if hasattr(config, "params"):
                    params = config.params
                    if hasattr(params, "vectors"):
                        vectors_config = params.vectors
                        if isinstance(vectors_config, dict):
                            vector_size = vectors_config.get("size")
                            distance = vectors_config.get("distance")
                        else:
                            vector_size = getattr(vectors_config, "size", None)
                            distance = getattr(vectors_config, "distance", None)
        except Exception as e:
            logger.debug("Could not extract vector config: %s", e)
        
        return {
            "collection": COLLECTION_NAME,
            "points_count": points_count,
            "vector_size": vector_size,
            "distance": str(distance) if distance else "unknown",
            "status": "ready" if points_count is not None else "ready",
        }
    except Exception as e:
        logger.error("Failed to get collection stats: %s: %s", type(e).__name__, e)
        return {
            "collection": COLLECTION_NAME,
            "status": "error",
            "error": str(e),
            "points_count": None,
            "vector_size": None,
        }


# %%
