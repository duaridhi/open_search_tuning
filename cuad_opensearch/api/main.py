"""
CUAD Hybrid Search API
Exposes BM25 + k-NN hybrid search over the cuad_dataset OpenSearch index.
"""

import os
from contextlib import asynccontextmanager

import boto3
from botocore.config import Config

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch, NotFoundError
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration — all values overridable via environment variables
# ---------------------------------------------------------------------------
OPENSEARCH_HOST   = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT   = int(os.getenv("OPENSEARCH_PORT", "9200"))
INDEX_NAME        = os.getenv("INDEX_NAME", "cuad_dataset")
MODEL_NAME        = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MINIO_ENDPOINT    = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")     # internal
MINIO_PUBLIC_ENDPOINT = os.getenv("MINIO_PUBLIC_ENDPOINT", MINIO_ENDPOINT)   # rewritten in URLs
MINIO_ACCESS_KEY  = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY  = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME       = os.getenv("MINIO_BUCKET", "cuad-contracts")
PRESIGNED_EXPIRY  = int(os.getenv("PRESIGNED_EXPIRY_SECONDS", "3600"))  # 1 hour

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
    _state["s3"] = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )
    yield
    _state["client"].close()


app = FastAPI(
    title="CUAD Hybrid Search API",
    description="BM25 + k-NN semantic search over CUAD contract chunks.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    pdf_url: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]


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
    s3 = _state["s3"]

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
            title = src.get("title", "")
            s3_key = f"raw/{title}.PDF"
            pdf_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": s3_key},
                ExpiresIn=PRESIGNED_EXPIRY,
            )
            # Rewrite internal Docker hostname → public endpoint so the
            # browser can actually reach MinIO on localhost:9000
            if MINIO_PUBLIC_ENDPOINT != MINIO_ENDPOINT:
                pdf_url = pdf_url.replace(MINIO_ENDPOINT, MINIO_PUBLIC_ENDPOINT, 1)
            results.append({
                "id":         doc_id,
                "score":      round(rrf_score, 6),
                "title":      title,
                "text":       src.get("text", ""),
                "char_start": src.get("char_start", 0),
                "char_end":   src.get("char_end", 0),
                "pdf_url":    pdf_url,
            })
        except NotFoundError:
            continue

    return {"query": q, "top_k": top_k, "results": results}
