"""
app.py
──────
FastAPI application for CUAD contract search using Qdrant.

Provides REST API endpoints for:
  - /health          - Service health check
  - /search          - Semantic search over contracts
  - /documents       - List indexed documents
  - /documents/{name} - Get specific document info
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import Qdrant search and utilities
import sys
from pathlib import Path as PathLib

qdrant_dir = PathLib(__file__).parent / "cuad-demo-quadrant"
sys.path.insert(0, str(qdrant_dir))

from qdrant_search import (
    init_qdrant,
    init_embedding_service,
    init_highlighting,
    search,
    get_collection_stats,
)
from document_utils import get_unique_documents, get_document_info
from qdrant_cluster_connect import get_cluster_info
from s3_utils import init_s3_clients, generate_presigned_url, list_s3_documents


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Response Models
# ─────────────────────────────────────────────────────────────────────────────


class Highlight(BaseModel):
    """Highlight position from Qdrant."""

    start: Optional[int] = Field(None, description="Start position (if available)")
    end: Optional[int] = Field(None, description="End position (if available)")
    field: Optional[str] = Field(None, description="Field name containing highlight")


class SearchResult(BaseModel):
    """Single search result."""

    id: str = Field(..., description="Unique chunk ID")
    score: float = Field(..., description="Similarity score (0-1)")
    title: str = Field(..., description="Contract title")
    text: str = Field(..., description="Chunk text")
    highlights: list[Highlight] = Field(
        default=[],
        description="Query term highlights from Qdrant",
    )
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")
    char_start: int = Field(..., description="Character offset (page start)")
    char_end: int = Field(..., description="Character offset (page end)")
    pdf_path: str = Field(..., description="Relative path to PDF")
    pdf_url: Optional[str] = Field(
        None,
        description="Presigned URL for PDF download",
    )
    source: list[str] = Field(
        default=["embeddings"],
        description="Search method source",
    )


class SearchResponse(BaseModel):
    """Response for search endpoint."""

    query: str = Field(..., description="Original search query")
    top_k: int = Field(..., description="Number of results requested")
    strategy: str = Field(..., description="Search strategy used")
    results_count: int = Field(..., description="Actual number of results")
    results: list[SearchResult] = Field(..., description="Search results")


class DocumentMetadata(BaseModel):
    """Document metadata."""

    title: str = Field(..., description="Contract title/name")
    pdf_path: str = Field(..., description="Path to PDF file")
    s3_key: Optional[str] = Field(
        None,
        description="S3 object key for PDF",
    )
    pdf_url: Optional[str] = Field(
        None,
        description="Presigned URL for PDF download",
    )
    chunk_count: int = Field(..., description="Number of chunks")
    total_chars: int = Field(..., description="Total characters")


class DocumentListResponse(BaseModel):
    """Response for documents list endpoint."""

    documents: list[DocumentMetadata] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")


class DocumentDetailResponse(BaseModel):
    """Response for document detail endpoint."""

    title: str
    pdf_path: str
    chunks: list[dict] = Field(..., description="Chunk details")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    collection: Optional[str] = None
    points_count: Optional[int] = None
    vector_size: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan & Initialization
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    print("[INFO] Starting up CUAD Qdrant API...")

    # Initialize Qdrant client
    try:
        qdrant_client = init_qdrant()
        print("[INFO] Qdrant client initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Qdrant: {e}")

    # Initialize embedding service client
    try:
        embedding_service = init_embedding_service()
        print(f"[INFO] Embedding service client initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize embedding service: {e}")

    # Initialize S3 clients for presigned URLs
    try:
        init_s3_clients()
        print("[INFO] S3 clients initialized for presigned URL generation")
    except Exception as e:
        print(f"[WARN] S3 initialization failed (PDFs may not have presigned URLs): {e}")

    # Initialize highlighting (create text indices)
    try:
        init_highlighting()
        print("[INFO] Highlighting initialized with text indices")
    except Exception as e:
        print(f"[WARN] Could not initialize highlighting: {e}")

    # Check collection stats
    try:
        stats = get_collection_stats()
        print(f"[INFO] Collection stats: {stats}")
    except Exception as e:
        print(f"[WARN] Could not fetch collection stats: {e}")

    yield  # App runs here

    print("[INFO] Shutting down CUAD Qdrant API...")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CUAD Qdrant Search API",
    description="Search CUAD contracts using Qdrant vector database",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    try:
        stats = get_collection_stats()
        print(f"[DEBUG] Health check - stats: {stats}")
        
        if stats.get("status") == "error":
            print(f"[WARN] Collection error: {stats.get('error')}")
            return HealthResponse(
                status="degraded",
                collection=stats.get("collection"),
            )
        
        return HealthResponse(
            status="ok",
            collection=stats.get("collection"),
            points_count=stats.get("points_count"),
            vector_size=stats.get("vector_size"),
        )
    except Exception as e:
        print(f"[ERROR] Health check exception: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}",
        )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_contracts(
    q: str = Query(..., description="Search query", min_length=1),
    top_k: int = Query(
        10,
        description="Number of results",
        ge=1,
        le=100,
    ),
    document_name: Optional[str] = Query(
        None,
        description="Filter by contract name",
    ),
    strategy: str = Query(
        "semantic_search",
        description="Search strategy: semantic_search or hybrid_search",
    ),
) -> SearchResponse:
    """
    Search contracts using semantic similarity (vector search).

    Parameters:
      - q: Search query (required)
      - top_k: Number of results (1-100, default: 10)
      - document_name: Optional filter by contract title
      - strategy: Search strategy (semantic_search or hybrid_search)

    Returns:
      List of matching contract chunks with scores, metadata, and presigned PDF URLs.
    """
    try:
        results, metadata = search(
            query=q,
            top_k=top_k,
            document_name=document_name,
            strategy=strategy,
        )

        # Convert to SearchResult objects with presigned URLs and highlights
        search_results = []
        for r in results:
            # Generate presigned URL: format is "raw/{title}.pdf"
            s3_key = f"raw/{r['title']}.pdf"
            pdf_url = generate_presigned_url(s3_key)
            
            # Convert highlight positions to Highlight objects
            hl_list = []
            for hl in r.get("highlights", []):
                hl_list.append(
                    Highlight(
                        start=hl.get("start"),
                        end=hl.get("end"),
                        field=hl.get("field"),
                    )
                )
            
            search_results.append(
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    title=r["title"],
                    text=r["text"],
                    highlights=hl_list,
                    page_start=r["page_start"],
                    page_end=r["page_end"],
                    char_start=r["char_start"],
                    char_end=r["char_end"],
                    pdf_path=r["pdf_path"],
                    pdf_url=pdf_url,
                    source=r.get("source", ["embeddings"]),
                )
            )

        return SearchResponse(
            query=q,
            top_k=top_k,
            strategy=strategy,
            results_count=len(search_results),
            results=search_results,
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Search failed: {str(e)}",
        )


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents() -> DocumentListResponse:
    """
    List all indexed documents (contracts) in the Qdrant collection.
    Each document includes a presigned URL for direct PDF download.

    Returns:
      List of document metadata including title, path, chunk count, and presigned URLs.
    """
    try:
        qdrant_client = init_qdrant()
        qdrant_documents = get_unique_documents(qdrant_client, "cuad_contracts")

        # Try to enrich with S3 presigned URLs
        try:
            s3_documents = list_s3_documents()
            
            # Create mapping of title -> S3 doc info
            s3_map = {doc["title"]: doc for doc in s3_documents}
            
            # Combine Qdrant metadata with S3 URLs
            documents = []
            for d in qdrant_documents:
                s3_info = s3_map.get(d["title"], {})
                documents.append(DocumentMetadata(
                    title=d["title"],
                    pdf_path=d["pdf_path"],
                    s3_key=s3_info.get("s3_key"),
                    pdf_url=s3_info.get("pdf_url"),
                    chunk_count=d["chunk_count"],
                    total_chars=d["total_chars"],
                ))
        except Exception as s3_error:
            print(f"[WARN] Could not fetch S3 documents, returning Qdrant-only: {s3_error}")
            documents = [
                DocumentMetadata(
                    title=d["title"],
                    pdf_path=d["pdf_path"],
                    chunk_count=d["chunk_count"],
                    total_chars=d["total_chars"],
                )
                for d in qdrant_documents
            ]

        return DocumentListResponse(
            documents=documents,
            total=len(documents),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}",
        )


@app.get("/documents/{document_name}", response_model=DocumentDetailResponse, tags=["Documents"])
async def get_document_detail(
    document_name: str = Path(..., description="Document/contract title"),
) -> DocumentDetailResponse:
    """
    Get detailed information about a specific document.

    Returns:
      Document metadata, chunk information, and presigned PDF URL.
    """
    try:
        qdrant_client = init_qdrant()
        doc_info = get_document_info(
            qdrant_client,
            "cuad_contracts",
            document_name,
        )

        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_name}' not found",
            )

        return DocumentDetailResponse(
            title=doc_info["title"],
            pdf_path=doc_info["pdf_path"],
            chunks=doc_info["chunks"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document details: {str(e)}",
        )


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "CUAD Qdrant Search API",
        "version": "1.0.0",
        "description": "Search CUAD contracts using Qdrant vector database",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "documents": "/documents",
            "documents_detail": "/documents/{document_name}",
            "docs": "/docs",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
# ──────────────────────────────────────────────────────────────────────────────


