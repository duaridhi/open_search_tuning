"""
CUAD Hybrid Search API
Exposes BM25 + k-NN hybrid search over the cuad_dataset OpenSearch index.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from opensearchpy import OpenSearch, NotFoundError
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration — all values overridable via environment variables
# ---------------------------------------------------------------------------
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
INDEX_NAME      = os.getenv("INDEX_NAME", "cuad_dataset")
MODEL_NAME      = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# App startup / shutdown — load heavy resources once
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["client"] = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )
    _state["model"] = SentenceTransformer(MODEL_NAME, device="cpu")
    yield
    _state["client"].close()


app = FastAPI(
    title="CUAD Hybrid Search API",
    description="BM25 + k-NN semantic search over CUAD contract chunks.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    id: str
    score: float
    title: str
    text: str
    char_start: int
    char_end: int


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResponse | SearchResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bm25_search(client: OpenSearch, query: str, top_k: int) -> dict[str, float]:
    """Run BM25 match query; return {doc_id: bm25_score}."""
    resp = client.search(
        index=INDEX_NAME,
        body={
            "query": {"match": {"text": query}},
            "size": top_k,
        },
    )
    return {
        hit["_id"]: hit["_score"]
        for hit in resp["hits"]["hits"]
    }


def _knn_search(client: OpenSearch, vector: list[float], top_k: int) -> dict[str, float]:
    """Run k-NN search; return {doc_id: knn_score}."""
    resp = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": top_k,
                    }
                }
            },
            "size": top_k,
        },
    )
    return {
        hit["_id"]: hit["_score"]
        for hit in resp["hits"]["hits"]
    }


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-based)."""
    return 1.0 / (k + rank)


def _fuse_rrf(
    bm25_scores: dict[str, float],
    knn_scores: dict[str, float],
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion (RRF).
    Returns [(doc_id, rrf_score)] sorted descending, capped at top_k.
    """
    bm25_ranked = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
    knn_ranked  = sorted(knn_scores,  key=knn_scores.get,  reverse=True)

    fused: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25_ranked, start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + _rrf_score(rank)
    for rank, doc_id in enumerate(knn_ranked, start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + _rrf_score(rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/search", response_model=dict)
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
):
    """
    Hybrid BM25 + k-NN search with Reciprocal Rank Fusion.

    Returns up to `top_k` ranked contract chunks.
    """
    client: OpenSearch = _state["client"]
    model: SentenceTransformer = _state["model"]

    try:
        query_vector = model.encode(q, show_progress_bar=False).tolist()
        bm25_scores  = _bm25_search(client, q, top_k)
        knn_scores   = _knn_search(client, query_vector, top_k)
        fused        = _fuse_rrf(bm25_scores, knn_scores, top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    results = []
    for doc_id, rrf_score in fused:
        try:
            doc = client.get(index=INDEX_NAME, id=doc_id)
            src = doc["_source"]
            results.append({
                "id":         doc_id,
                "score":      round(rrf_score, 6),
                "title":      src.get("title", ""),
                "text":       src.get("text", ""),
                "char_start": src.get("char_start", 0),
                "char_end":   src.get("char_end", 0),
            })
        except NotFoundError:
            continue

    return {"query": q, "top_k": top_k, "results": results}
