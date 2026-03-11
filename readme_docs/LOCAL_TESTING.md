# Local Testing with Docker Compose (Development)

Before deploying to HF Spaces, test locally with Docker Compose.

## Prerequisites

- Docker & Docker Compose installed
- `.env.dev` file configured (copy from `.env.example` if needed)
- Contract PDFs indexed in Qdrant (run `upload_to_qdrant.py` first)

## Quick Start

### 1. Setup Environment

```bash
# Copy dev environment template
cp .env.example .env.dev

# (Optional) Edit if you have a different local Qdrant setup
# nano .env.dev
```

### 2. Build and Run (Development)

```bash
# Using default docker-compose.yml (dev)
docker-compose up --build

# Or explicitly use dev configuration
docker-compose -f docker-compose.dev.yml up --build
```

### 3. Verify Services are Running

In a separate terminal:

```bash
# Health check - Main API
curl http://localhost:7860/health

# Health check - Embedding Service
curl http://localhost:8001/health

# Health check - Qdrant
curl http://localhost:6333/health
```

### 4. Test Endpoints

```bash
# Search
curl "http://localhost:7860/search?q=liability%20clause&top_k=5"

# List documents
curl http://localhost:7860/documents

# Get document detail
curl "http://localhost:7860/documents/ContractName"

# API Documentation
# Open: http://localhost:7860/docs
```

## Populating Qdrant with Contract Data

Before testing search, you need to populate Qdrant:

### Option A: Local Python (Recommended for first time)

```bash
# From your Python environment (not in Docker)
python cuad-demo-quadrant/upload_to_qdrant.py \
  --pdf-dir /path/to/contract/pdfs \
  --qdrant-url http://localhost:6333 \
  --chunk-size 500 \
  --overlap 100
```

### Option B: Inside Docker Container

```bash
# Start services first
docker-compose up

# In another terminal, run upload script in container
docker-compose exec cuad-api python cuad-demo-quadrant/upload_to_qdrant.py \
  --pdf-dir /app/data/pdfs \
  --chunk-size 500
```

## Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f cuad-api
docker-compose logs -f qdrant

# Main API only
docker-compose logs -f cuad-api | grep "embedding\|search"
```

## Stopping Services

```bash
# Stop running services (keeps volumes)
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Troubleshooting Local Setup

### Port already in use

Edit `docker-compose.yml` and change port mapping:

```yaml
services:
  cuad-api:
    ports:
      - "7861:7860"  # Map to 7861 instead
      - "8002:8001"  # Map to 8002 instead
```

### Out of memory

Docker Desktop → Preferences → Resources

Increase memory limit to at least 8GB.

### Models loading timeout

Increase timeout in `.env.dev`:

```bash
MODEL_LOAD_TIMEOUT=300  # 5 minutes instead of 2
```

### "No module named" errors

Ensure you're using relative imports in highlights module:

```python
from ..embeddings.embedding_service import _state  # Correct
from embeddings.embedding_service import _state    # Wrong
```

### Qdrant connection refused

```bash
# Verify Qdrant is running
docker-compose logs qdrant

# Check if port 6333 is open
curl http://localhost:6333/health
```

## Performance Tips

1. **Use CPU-only** (default, faster startup than GPU)
2. **Keep embedding model small**: `all-MiniLM-L6-v2` is ideal
3. **Cache models**: They're downloaded on first run, then cached
4. **Live reload**: Volume mounts allow code changes without rebuild

## Development Workflow

```bash
# Terminal 1: Start services
docker-compose up

# Terminal 2: Watch logs
docker-compose logs -f

# Terminal 3: Test/develop
curl http://localhost:7860/health
# Edit code
# Changes reflected due to volume mounts
```

## Next: Deploy to HF Spaces

Once local testing passes:

1. Review `DEPLOYMENT_ROADMAP.md`
2. Create HF Space
3. Configure `.env.prod` secrets
4. Push code

See [HF_SPACES_DEPLOYMENT.md](./HF_SPACES_DEPLOYMENT.md) for production deployment.
