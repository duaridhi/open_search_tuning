# OpenSearch → Qdrant Migration Guide

## Overview

This document outlines the migration from OpenSearch to Qdrant for the CUAD contract search backend.

### Key Differences

| Aspect | OpenSearch | Qdrant |
|--------|-----------|--------|
| **Architecture** | Cluster-based, distributed full-text search | Vector database (specialized for semantic search) |
| **Search Types** | BM25 (lexical), Neural (semantic), Hybrid RRF | Semantic/Vector (cosine similarity) |
| **Infrastructure** | Complex setup, separate embedding service | Simpler: vectors pre-computed during ingestion |
| **Scaling** | Scales to huge datasets | Optimized for focused collections (millions not billions) |
| **Full-text** | Native BM25 | Requires external indexing (planned future work) |

---

## New Architecture

### Data Flow

```
Contract PDFs
    ↓
[upload_to_qdrant.py] — Extract text, create chunks
    ↓
[Embedding Model] — Generate 384-D vectors (all-MiniLM-L6-v2)
    ↓
Qdrant Collection — Store vectors + metadata
    ↓
[FastAPI App] — Query API
    ↓
[qdrant_search.py] — Vector search + filtering
    ↓
Search Results → JSON Response
```

### Collection Schema

**Collection Name:** `cuad_contracts`

**Vector Config:**
- Size: 384 dimensions (all-MiniLM-L6-v2)
- Distance: Cosine similarity
- Storage: Dense vectors

**Payload (Metadata per chunk):**
```
{
  "doc_id": "contract_name-chunk-0",
  "title": "Contract Name",
  "text": "Chunk text content...",
  "page_start": 1,
  "page_end": 3,
  "char_start": 0,
  "char_end": 500,
  "pdf_path": "path/to/contract.pdf"
}
```

---

## API Endpoints

### 1. Health Check
```
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "collection": "cuad_contracts",
  "points_count": 5000,
  "vector_size": 384
}
```

### 2. Search (Semantic)
```
GET /search?q=liability%20clause&top_k=10&document_name=optional_contract_name&strategy=semantic_search
```

**Query Parameters:**
- `q` (required): Search query
- `top_k` (optional, 1-100, default: 10): Number of results
- `document_name` (optional): Filter by contract title
- `strategy` (optional, default: "semantic_search"): "semantic_search" or "hybrid_search"

**Response:**
```json
{
  "query": "liability clause",
  "top_k": 10,
  "strategy": "semantic_search",
  "results_count": 10,
  "results": [
    {
      "id": "contract-chunk-0",
      "score": 0.892,
      "title": "Contract Name",
      "text": "The liability clause states...",
      "page_start": 5,
      "page_end": 5,
      "char_start": 100,
      "char_end": 200,
      "pdf_path": "path/to/contract.pdf",
      "source": ["embeddings"]
    }
  ]
}
```

### 3. List Documents
```
GET /documents
```

**Response:**
```json
{
  "total": 50,
  "documents": [
    {
      "title": "Contract A",
      "pdf_path": "path/to/contractA.pdf",
      "chunk_count": 120,
      "total_chars": 45000
    }
  ]
}
```

### 4. Document Details
```
GET /documents/Contract%20Name
```

**Response:**
```json
{
  "title": "Contract Name",
  "pdf_path": "path/to/contract.pdf",
  "chunks": [
    {
      "doc_id": "Contract Name-chunk-0",
      "page_start": 1,
      "page_end": 1,
      "char_count": 512
    }
  ]
}
```

---

## Setup & Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key-if-using-cloud
QDRANT_COLLECTION=cuad_contracts

# Optional: Customize embedding
ENCODING_MODEL=all-MiniLM-L6-v2
```

### 2. Start Qdrant (Local)

**Using Docker:**
```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

**Or Qdrant Cloud:**
- Sign up at https://cloud.qdrant.io
- Get your cluster URL and API key
- Update `.env` with `QDRANT_URL` and `QDRANT_API_KEY`

### 3. Upload Contracts

```bash
cd cuad-demo-quadrant
python upload_to_qdrant.py
```

This will:
- Find all PDFs in configured path
- Extract text and create chunks
- Generate embeddings
- Upload to Qdrant collection

**Configuration (via environment variables):**
```bash
MAX_DOCS=1000          # Max chunks to upload
CHUNK_SIZE=500         # Characters per chunk
CHUNK_OVERLAP=50       # Overlap between chunks
ENCODE_BATCH_SIZE=32   # Batch size for embedding
UPLOAD_BATCH_SIZE=100  # Batch size for upload
```

### 4. Start API Server

```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Or production:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

---

## Code Structure

```
cuad-ai-demo/
├── app.py                          # FastAPI application
├── requirements.txt                # Dependencies
├── cuad-demo-quadrant/
│   ├── upload_to_qdrant.py        # Data ingestion pipeline
│   ├── qdrant_search.py           # Search functions (replaces hybrid_search.py)
│   ├── document_utils.py          # Document metadata utilities
│   ├── .env                       # Configuration
│   ├── CHUNKING_STRATEGY.md       # Chunking documentation
│   └── tests/
│       └── verify_collection.py   # Collection validation
└── MIGRATION.md                   # This file
```

---

## Search Strategies

### Semantic Search (Recommended)
- **How it works:** Embeds query using same model, finds nearest vectors
- **Speed:** Fast (sub-second)
- **Quality:** Good for semantic/conceptual matching
- **Use case:** General contract questions, clause finding

```python
results, meta = search(
    query="What are liability limitations?",
    top_k=10,
    strategy="semantic_search"
)
```

### Hybrid Search (Future)
- **Current implementation:** Same as semantic search
- **Future roadmap:** Combine BM25 (keyword exact match) + semantic scores using RRF
- **Benefit:** Would handle both keyword-exact and semantic matches

---

## Differences from OpenSearch API

### What's Changed:
1. **No BM25 option** — Qdrant does vector search only. Future: can add BM25 via hybrid strategy.
2. **No highlighting** — Qdrant doesn't compute semantic highlights. Can be added via post-processing.
3. **No presigned S3 URLs** — PDF URLs not handled yet (can integrate S3 client).
4. **Simpler deployment** — No cluster coordination, single instance or cloud service.

### What's Kept:
1. **Same endpoint structure** — `/search`, `/documents`, `/health`
2. **Same response format** — SearchResult objects with score, title, text, page info
3. **Same filtering** — Can filter by document name
4. **Same configuration pattern** — Environment variables

---

## Migration Checklist

- [ ] Install Qdrant (cloud or local)
- [ ] Set up `.env` with Qdrant credentials
- [ ] Run `upload_to_qdrant.py` to ingest contracts
- [ ] Start FastAPI: `uvicorn app:app --reload`
- [ ] Test `/health` endpoint
- [ ] Test `/search` endpoint
- [ ] Verify `/documents` lists contracts
- [ ] Update frontend to use new endpoints (same interfaces)
- [ ] Performance test and tune (CHUNK_SIZE, top_k, etc.)

---

## Troubleshooting

### Collection Not Found
```
Error: Collection 'cuad_contracts' not found
```
**Fix:** Run `upload_to_qdrant.py` to create and populate collection.

### Embedding Model Download Slow
```
[INFO] Loading embedding model ... (downloading ~90 MB)
```
**Normal:** First run downloads model from HuggingFace. Subsequent runs use cache (~5 seconds).

### Qdrant Connection Failed
```
Error: failed to connect to Qdrant at http://localhost:6333
```
**Fix:** 
1. Check Qdrant is running: `curl http://localhost:6333/health`
2. Or update `QDRANT_URL` in `.env`

### Low Search Results Quality
1. Check `CHUNK_SIZE` in `upload_to_qdrant.py` (try 500-800 chars)
2. Verify embeddings are normalized (they are by default)
3. Try lowering `min_score` threshold in search call

---

## Performance Notes

- **Ingestion:** ~100 chunks/second (depends on model, batch size)
- **Search:** <50ms per query (sub-second)
- **Memory:** ~2GB for ~10k contracts with embeddings
- **Scaling:** Qdrant handles millions of vectors efficiently

---

## Next Steps

1. **Hybrid Search:** Add BM25 via external indexer or Qdrant sparse vectors
2. **Highlighting:** Implement post-processing to mark relevant phrases
3. **S3 Integration:** Add presigned URLs for PDF downloads
4. **UI Integration:** Update frontend to use new API
5. **Analytics:** Track search queries and results quality

---
