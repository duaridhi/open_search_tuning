# CUAD AI Demo - Qdrant Backend

A FastAPI-based search application for CUAD (Contract Understanding Atticus Dataset) contracts using Qdrant vector database for semantic search.

## Features

- **Semantic Search:** Find relevant contract clauses using natural language queries
- **Vector Database:** Fast similarity search using Qdrant with all-MiniLM-L6-v2 embeddings
- **Contract Filtering:** Search within specific documents
- **Document Discovery:** Browse indexed contracts and their metadata
- **REST API:** Complete OpenAPI/Swagger documentation
- **Production Ready:** Async FastAPI with proper lifecycle management

## Quick Start

### 1. Setup Qdrant

**Option A: Local Docker**
```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

**Option B: Qdrant Cloud**
- Sign up at https://cloud.qdrant.io
- Create cluster and get API key

### 2. Configure Environment

Create `.env` in project root:
```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional-cloud-api-key
QDRANT_COLLECTION=cuad_contracts
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Ingest Contracts (Optional)

```bash
cd cuad-demo-quadrant
python upload_to_qdrant.py
```

Configure ingestion via environment variables:
- `MAX_DOCS` - Number of contracts to upload (default: 1000)
- `CHUNK_SIZE` - Characters per chunk (default: 500)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 50)

### 5. Start API Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Visit: http://localhost:8000/docs for interactive API documentation

## API Endpoints

### Search
```bash
curl "http://localhost:8000/search?q=liability%20clause&top_k=10"
```

### List Documents
```bash
curl http://localhost:8000/documents
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Project Structure

```
├── app.py                          # FastAPI application
├── requirements.txt                # Python dependencies
├── cuad-demo-quadrant/
│   ├── upload_to_qdrant.py        # Data ingestion pipeline
│   ├── qdrant_search.py           # Core search functions
│   ├── document_utils.py          # Document utilities
│   ├── CHUNKING_STRATEGY.md       # Chunking documentation
│   └── tests/
│       └── verify_collection.py   # Collection validation
├── MIGRATION.md                    # OpenSearch → Qdrant migration guide
└── Dockerfile                      # Docker configuration
```

## Architecture

```
Contract PDFs → Extract & Chunk → Generate Embeddings → Qdrant Collection
                                                              ↓
                                                       FastAPI /search
                                                              ↓
                                                         JSON Response
```

## Search Strategies

- **semantic_search** (default): Vector similarity search
- **hybrid_search**: Future - combines keyword + semantic matching

## Configuration

All configuration via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant cluster URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Cloud API key (optional) | None |
| `QDRANT_COLLECTION` | Collection name | `cuad_contracts` |
| `ENCODING_MODEL` | Embedding model | `all-MiniLM-L6-v2` |

## Migration from OpenSearch

See [MIGRATION.md](MIGRATION.md) for detailed migration guide including:
- Architecture differences
- API endpoint comparison
- Code structure overview
- Troubleshooting guide

## Performance

- **Search:** <50ms per query
- **Ingestion:** ~100 chunks/second
- **Memory:** ~2GB for ~10k contracts
- **Embedding Model:** 384-dimensional vectors

## API Documentation

Once running, interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technology Stack

- **FastAPI** - Web framework
- **Qdrant** - Vector database
- **Sentence-Transformers** - Embeddings (all-MiniLM-L6-v2)
- **Pydantic** - Data validation
- **uvicorn** - ASGI server

## Contact

For issues or questions, refer to MIGRATION.md or project documentation.
