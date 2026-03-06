"""
CUAD Hybrid Search API
Exposes BM25 + k-NN hybrid search over the cuad_dataset OpenSearch index.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the cuad_opensearch/ directory, regardless of working directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import boto3
from botocore.config import Config

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch, NotFoundError

# Add notebooks dir to sys.path so hybrid_search can be imported directly.
_notebooks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
if _notebooks_dir not in sys.path:
    sys.path.insert(0, _notebooks_dir)

from hybrid_search import _bm25_search, _knn_search, _rrf_score, _fuse_rrf  # noqa: E402
from open_search_connect import connect  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — all values overridable via environment variables
# ---------------------------------------------------------------------------
# OPENSEARCH_HOST / OPENSEARCH_PORT are consumed inside open_search_connect.py
INDEX_NAME            = os.getenv("INDEX_NAME")
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
    _state["client"] = connect()
    _state["s3"] = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )
    # Separate client for presigned URLs — must be signed with the public
    # endpoint so the signature is valid when the browser hits localhost:9000.
    _state["s3_public"] = boto3.client(
        "s3",
        endpoint_url=MINIO_PUBLIC_ENDPOINT,
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
    # Full-document char offsets (position within the entire concatenated PDF text)
    char_start: int
    char_end: int
    # PDF page range this chunk spans
    page_start: int = 1
    page_end: int = 1
    # Char offset of the chunk start within page_start's text,
    # and char offset of the chunk end within page_end's text.
    # Use these in the PDF viewer: scroll to page_start, then highlight
    # from page_char_start to page_char_end within the page text layer.
    page_char_start: int = 0
    page_char_end: int = 0
    pdf_url: str | None = None
    highlight: str | None = None
    source: list[str] = []  # ["bm25"], ["embeddings"], or ["bm25", "embeddings"]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]


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
    document_name: str = Query(None, description="Optional: Filter search to a specific document name (exact match)"),
):
    """
    Hybrid BM25 + k-NN search with Reciprocal Rank Fusion.

    Returns up to `top_k` ranked contract chunks with highlights and source information.
    If `document_name` is provided, only searches within that specific document.
    """
    client: OpenSearch = _state["client"]
    s3 = _state["s3"]
    s3_public = _state["s3_public"]

    try:
        print(f"[DEBUG] Starting search for query: '{q}'")
        if document_name:
            print(f"[DEBUG] Filtering to document: '{document_name}'")

        print(f"[DEBUG] Running BM25 search...")
        bm25_scores, bm25_highlights = _bm25_search(client, q, top_k, document_name)
        print(f"[DEBUG] BM25 search returned {len(bm25_scores)} results")

        print(f"[DEBUG] Running neural search...")
        knn_scores, knn_highlights = _knn_search(client, q, top_k, document_name)
        print(f"[DEBUG] Neural search returned {len(knn_scores)} results")
        
        # Track which docs came from which search method
        bm25_docs = set(bm25_scores.keys())
        knn_docs = set(knn_scores.keys())
        
        print(f"[DEBUG] Running RRF fusion...")
        fused        = _fuse_rrf(bm25_scores, knn_scores, top_k)
        print(f"[DEBUG] RRF fusion produced {len(fused)} results")
    except Exception as exc:
        print(f"[ERROR] Search failed: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(exc)}")

    results = []
    for doc_id, rrf_score in fused:
        try:
            doc = client.get(index=INDEX_NAME, id=doc_id)
            src = doc["_source"]
            title = src.get("title", "")
            s3_key = f"raw/{title}.pdf"
            
            # Try to generate presigned URL, but don't fail if S3 is unavailable
            pdf_url = None
            try:
                pdf_url = s3_public.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": BUCKET_NAME, "Key": s3_key},
                    ExpiresIn=PRESIGNED_EXPIRY,
                )
            except Exception as s3_err:
                print(f"[WARNING] Could not generate presigned URL for {s3_key}: {s3_err}")
                # Continue without PDF URL instead of failing entire search
                pdf_url = None
            
            # Determine which search method(s) returned this doc
            sources = []
            if doc_id in bm25_docs:
                sources.append("bm25")
            if doc_id in knn_docs:
                sources.append("embeddings")
            
            result = {
                "id":              doc_id,
                "score":           round(rrf_score, 6),
                "title":           title,
                "text":            src.get("text", ""),
                "char_start":      src.get("char_start", 0),
                "char_end":        src.get("char_end", 0),
                "page_start":      src.get("page_start", 1),
                "page_end":        src.get("page_end", 1),
                "page_char_start": src.get("page_char_start", 0),
                "page_char_end":   src.get("page_char_end", 0),
                "pdf_url":         pdf_url,
                "source":          sources if sources else ["rrf"],
            }
            # Add highlight — prefer BM25 highlight; fall back to kNN semantic highlight
            if doc_id in bm25_highlights:
                result["highlight"] = bm25_highlights[doc_id]
            elif doc_id in knn_highlights:
                result["highlight"] = knn_highlights[doc_id]
            results.append(result)
        except NotFoundError:
            continue
        except Exception as doc_err:
            print(f"[ERROR] Failed to process doc {doc_id}: {doc_err}")
            continue

    if document_name:
        print(f"[DEBUG] Returned {len(results)} results for document: '{document_name}'")
    
    return {"query": q, "top_k": top_k, "results": results}
