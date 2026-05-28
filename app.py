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
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Path, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import Qdrant search and utilities
import sys
from pathlib import Path as PathLib

qdrant_dir = PathLib(__file__).parent / "cuad-demo-quadrant"
sys.path.insert(0, str(qdrant_dir))

from qdrant_search_hf import (
    init_qdrant,
    search,
    get_collection_stats,
)
from document_utils import get_unique_documents, get_document_info
from chat_hf import chat
from perf_trace import start_trace, span, spans_header_value, record_span
from qdrant_cluster_connect import get_cluster_info
from hf_utils import init_hf_client, generate_hf_url, list_hf_documents


# ─────────────────────────────────────────────────────────────────────────────
# Logger Configuration
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Timeout Configurations (in seconds)
# ─────────────────────────────────────────────────────────────────────────────
INIT_QDRANT_TIMEOUT = int(os.getenv("INIT_QDRANT_TIMEOUT", "30"))
INIT_HF_TIMEOUT = int(os.getenv("INIT_HF_TIMEOUT", "10"))
COLLECTION_STATS_TIMEOUT = int(os.getenv("COLLECTION_STATS_TIMEOUT", "10"))
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "60"))
DOCS_LIST_TIMEOUT = int(os.getenv("DOCS_LIST_TIMEOUT", "20"))
DOCS_DETAIL_TIMEOUT = int(os.getenv("DOCS_DETAIL_TIMEOUT", "15"))


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Response Models
# ─────────────────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    """Single search result."""

    id: str = Field(..., description="Unique chunk ID")
    score: float = Field(..., description="Similarity score (0-1)")
    title: str = Field(..., description="Contract title")
    text: str = Field(..., description="Chunk text")
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")
    char_start: int = Field(..., description="Character offset (page start)")
    char_end: int = Field(..., description="Character offset (page end)")
    page_offset_start: Optional[int | list[int]] = Field(
        None,
        description="Chunk start offset local to page_start, or array of highlight offsets if highlights found",
    )
    page_offset_end: Optional[int | list[int]] = Field(
        None,
        description="Chunk end offset local to page_end, or array of highlight offsets if highlights found",
    )
    pdf_path: str = Field(..., description="Relative path to PDF")
    pdf_url: Optional[str] = Field(
        None,
        description="HuggingFace Hub URL for PDF download",
    )
    source: list[str] = Field(
        default=["embeddings"],
        description="Search method source",
    )
    highlighted_sentences: list[str] = Field(
        default=[],
        description="Sentences highlighted by semantic highlighter",
    )
    highlight_sentence_indexes: list[int] = Field(
        default=[],
        description="Indexes of highlighted sentences",
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
    hf_path: Optional[str] = Field(
        None,
        description="HuggingFace Hub path for PDF",
    )
    pdf_url: Optional[str] = Field(
        None,
        description="HuggingFace Hub URL for PDF download",
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
    logger.info("Starting up CUAD Qdrant API...")

    # Initialize Qdrant client
    try:
        qdrant_client = await asyncio.wait_for(
            asyncio.to_thread(init_qdrant),
            timeout=INIT_QDRANT_TIMEOUT
        )
        logger.info("Qdrant client initialized successfully")
    except asyncio.TimeoutError:
        logger.error(f"Qdrant initialization timed out after {INIT_QDRANT_TIMEOUT}s")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}", exc_info=True)

    # Initialize HuggingFace Hub client for document URLs
    try:
        await asyncio.wait_for(
            asyncio.to_thread(init_hf_client),
            timeout=INIT_HF_TIMEOUT
        )
        logger.info("HuggingFace Hub client initialized for document URL generation")
    except asyncio.TimeoutError:
        logger.warning(f"HuggingFace Hub initialization timed out after {INIT_HF_TIMEOUT}s")
    except Exception as e:
        logger.warning(f"HuggingFace Hub initialization failed (PDFs may not have URLs): {e}", exc_info=True)

    # Check collection stats
    try:
        stats = await asyncio.wait_for(
            asyncio.to_thread(get_collection_stats),
            timeout=COLLECTION_STATS_TIMEOUT
        )
        logger.info(f"Collection stats: {stats}")
    except asyncio.TimeoutError:
        logger.warning(f"Collection stats timed out after {COLLECTION_STATS_TIMEOUT}s")
    except Exception as e:
        logger.warning(f"Could not fetch collection stats: {e}", exc_info=True)

    yield  # App runs here

    logger.info("Shutting down CUAD Qdrant API...")


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
    allow_origins=["http://localhost:5173", "http://localhost:3000","https://ginntonicfun-cuad-ai-demo.hf.space", "*"],
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
    logger.debug("Health check endpoint called")
    try:
        try:
            stats = await asyncio.wait_for(
                asyncio.to_thread(get_collection_stats),
                timeout=COLLECTION_STATS_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Health check timed out after {COLLECTION_STATS_TIMEOUT}s")
            raise HTTPException(
                status_code=504,
                detail=f"Health check timed out after {COLLECTION_STATS_TIMEOUT}s",
            )
        
        logger.debug(f"Health check - stats: {stats}")
        
        if stats.get("status") == "error":
            logger.warning(f"Collection error: {stats.get('error')}")
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check exception: {type(e).__name__}: {e}", exc_info=True)
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
    highlight: bool = Query(
        True,
        description="If false, skip per-sentence reranking and return chunks without highlights. Faster.",
    ),
    response: Response = None,
) -> SearchResponse:
    """
    Search contracts using semantic similarity (vector search).

    Parameters:
      - q: Search query (required)
      - top_k: Number of results (1-100, default: 10)
      - document_name: Optional filter by contract title
      - strategy: Search strategy (semantic_search or hybrid_search)
      - highlight: Whether to compute per-sentence highlights (default: true)

    Returns:
      List of matching contract chunks with scores, metadata, and HuggingFace Hub PDF URLs.
    """
    logger.info(
        f"Search request: query='{q}', top_k={top_k}, document_name={document_name}, "
        f"strategy={strategy}, highlight={highlight}"
    )
    start_trace()
    try:
        try:
            with span("total"):
                results, metadata = await asyncio.wait_for(
                    asyncio.to_thread(
                        search,
                        q,
                        top_k,
                        document_name,
                        strategy,
                        0.0,
                        highlight,
                    ),
                    timeout=SEARCH_TIMEOUT
                )
        except asyncio.TimeoutError:
            logger.error(f"Search timed out after {SEARCH_TIMEOUT}s for query: {q}")
            raise HTTPException(
                status_code=504,
                detail=f"Search operation timed out after {SEARCH_TIMEOUT}s",
            )

        # Convert to SearchResult objects with HuggingFace Hub URLs
        search_results = []
        for r in results:
            # Generate HF Hub URL: format is "raw/{title}.pdf"
            hf_path = f"raw/{r['title']}.pdf"
            pdf_url = generate_hf_url(hf_path)
            
            search_results.append(
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    title=r["title"],
                    text=r["text"],
                    page_start=r["page_start"],
                    page_end=r["page_end"],
                    char_start=r["char_start"],
                    char_end=r["char_end"],
                    page_offset_start=r.get("page_offset_start"),
                    page_offset_end=r.get("page_offset_end"),
                    pdf_path=r["pdf_path"],
                    pdf_url=pdf_url,
                    source=r.get("source", ["embeddings"]),
                    highlighted_sentences=r.get("highlighted_sentences", []),
                    highlight_sentence_indexes=r.get("highlight_sentence_indexes", []),
                )
            )
        
        logger.info(f"Search query '{q}' returned {len(search_results)} results with strategy '{strategy}'")
        header = spans_header_value()
        if header and response is not None:
            response.headers["X-Perf-Spans"] = header
        return SearchResponse(
            query=q,
            top_k=top_k,
            strategy=strategy,
            results_count=len(search_results),
            results=search_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed for query '{q}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents() -> DocumentListResponse:
    """
    List all indexed documents (contracts) in the Qdrant collection.
    Each document includes a HuggingFace Hub URL for direct PDF download.

    Returns:
      List of document metadata including title, path, chunk count, and HF Hub URLs.
    """
    logger.info("Document list request received")
    try:
        try:
            qdrant_client = await asyncio.wait_for(
                asyncio.to_thread(init_qdrant),
                timeout=INIT_QDRANT_TIMEOUT
            )
            qdrant_documents = await asyncio.wait_for(
                asyncio.to_thread(get_unique_documents, qdrant_client, "cuad_contracts"),
                timeout=DOCS_LIST_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Document list operation timed out after {DOCS_LIST_TIMEOUT}s")
            raise HTTPException(
                status_code=504,
                detail=f"Document list operation timed out after {DOCS_LIST_TIMEOUT}s",
            )

        # Try to enrich with HuggingFace Hub URLs
        try:
            hf_documents = await asyncio.wait_for(
                asyncio.to_thread(list_hf_documents),
                timeout=5
            )
            
            # Create mapping of title -> HF doc info
            hf_map = {doc["title"]: doc for doc in hf_documents}
            
            # Combine Qdrant metadata with HF URLs
            documents = []
            for d in qdrant_documents:
                hf_info = hf_map.get(d["title"], {})
                documents.append(DocumentMetadata(
                    title=d["title"],
                    pdf_path=d["pdf_path"],
                    hf_path=hf_info.get("hf_path"),
                    pdf_url=hf_info.get("pdf_url"),
                    chunk_count=d["chunk_count"],
                    total_chars=d["total_chars"],
                ))
            logger.info(f"Retrieved {len(documents)} documents from Qdrant with HuggingFace Hub URLs")
        except (asyncio.TimeoutError, Exception) as hf_error:
            logger.warning(f"Could not fetch HuggingFace Hub documents, returning Qdrant-only: {hf_error}", exc_info=True)
            documents = [
                DocumentMetadata(
                    title=d["title"],
                    pdf_path=d["pdf_path"],
                    chunk_count=d["chunk_count"],
                    total_chars=d["total_chars"],
                )
                for d in qdrant_documents
            ]
            logger.info(f"Retrieved {len(documents)} documents from Qdrant without HuggingFace Hub URLs")

        return DocumentListResponse(
            documents=documents,
            total=len(documents),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}", exc_info=True)
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
      Document metadata, chunk information, and HuggingFace Hub PDF URL.
    """
    logger.info(f"Document detail request for: '{document_name}'")
    try:
        try:
            qdrant_client = await asyncio.wait_for(
                asyncio.to_thread(init_qdrant),
                timeout=INIT_QDRANT_TIMEOUT
            )
            doc_info = await asyncio.wait_for(
                asyncio.to_thread(
                    get_document_info,
                    qdrant_client,
                    "cuad_contracts",
                    document_name,
                ),
                timeout=DOCS_DETAIL_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Document detail operation timed out after {DOCS_DETAIL_TIMEOUT}s for document: {document_name}")
            raise HTTPException(
                status_code=504,
                detail=f"Document detail operation timed out after {DOCS_DETAIL_TIMEOUT}s",
            )

        if not doc_info:
            logger.warning(f"Document not found: '{document_name}'")
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_name}' not found",
            )

        logger.info(f"Retrieved details for document: '{document_name}' with {len(doc_info['chunks'])} chunks")
        return DocumentDetailResponse(
            title=doc_info["title"],
            pdf_path=doc_info["pdf_path"],
            chunks=doc_info["chunks"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document details for '{document_name}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document details: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chat Endpoint
# ─────────────────────────────────────────────────────────────────────────────

CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "120"))


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question", min_length=1)
    top_k: int = Field(5, description="Number of search results to use as context", ge=1, le=20)
    document_name: Optional[str] = Field(None, description="Filter search to a specific contract")
    strategy: str = Field("semantic_search", description="Search strategy")
    system_prompt: Optional[str] = Field(None, description="Override default system instructions")


class ChatResponse(BaseModel):
    query: str
    answer: str
    document_name: Optional[str] = Field(None, description="Document name used to scope the search, if provided")
    sources: list[SearchResult] = Field(default=[], description="Chunks used as context, with full metadata and highlights")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest, response: Response = None) -> ChatResponse:
    """
    Answer a question using RAG: retrieves top-k contract passages from Qdrant
    then generates a grounded answer via HuggingFace Inference API.
    """
    logger.info(f"Chat request: query='{request.query}', top_k={request.top_k}")
    start_trace()
    _t_chat_start = time.perf_counter()
    try:
        try:
            with span("retrieve"):
                results, _ = await asyncio.wait_for(
                    asyncio.to_thread(
                        search,
                        request.query,
                        request.top_k,
                        request.document_name,
                        request.strategy,
                    ),
                    timeout=SEARCH_TIMEOUT,
                )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Search timed out after {SEARCH_TIMEOUT}s",
            )

        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(chat, request.query, results, request.system_prompt),
                timeout=CHAT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Chat generation timed out after {CHAT_TIMEOUT}s",
            )

        sources = []
        for r in results:
            hf_path = f"raw/{r['title']}.pdf"
            pdf_url = generate_hf_url(hf_path)
            sources.append(
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    title=r["title"],
                    text=r["text"],
                    page_start=r["page_start"],
                    page_end=r["page_end"],
                    char_start=r["char_start"],
                    char_end=r["char_end"],
                    page_offset_start=r.get("page_offset_start"),
                    page_offset_end=r.get("page_offset_end"),
                    pdf_path=r["pdf_path"],
                    pdf_url=pdf_url,
                    source=r.get("source", ["embeddings"]),
                    highlighted_sentences=r.get("highlighted_sentences", []),
                    highlight_sentence_indexes=r.get("highlight_sentence_indexes", []),
                )
            )
        logger.info(f"Chat answered query '{request.query}' using {len(results)} passages")
        record_span("total", (time.perf_counter() - _t_chat_start) * 1000.0)
        header = spans_header_value()
        if header and response is not None:
            response.headers["X-Perf-Spans"] = header
        return ChatResponse(
            query=request.query,
            answer=answer,
            document_name=request.document_name,
            sources=sources,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed for query '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    logger.debug("Root endpoint called")
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
