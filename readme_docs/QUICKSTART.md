# Quick Start Guide

Fast reference for getting the project running locally or deploying to production.

## Prerequisites

1. **Cloud Qdrant Account** - [Create free account](https://qdrant.tech/cloud/)
2. **Qdrant Cluster URL and API Key** - From your Qdrant Cloud dashboard
3. **Docker** installed locally

## Local Development

### First-Time Setup
```bash
# 1. Clone and navigate
git clone <repo-url>
cd cuad-ai-demo

# 2. Copy dev environment template
cp .env.example .env.dev

# 3. Update .env.dev with cloud Qdrant credentials
# Edit .env.dev and set:
#   QDRANT_URL=https://<your-cluster>.qdrant.io:6333
#   QDRANT_API_KEY=<your-api-key>

# 4. Start services
docker-compose up --build
```

Services will start:
- **API**: http://localhost:7860
- **Embedding Service**: http://localhost:8001

### Index Documents
```bash
# In another terminal, index contracts to cloud Qdrant
python cuad-demo-quadrant/upload_to_qdrant.py
```

### API Testing
```bash
# Health check
curl http://localhost:7860/health

# Search contracts
curl -X POST http://localhost:7860/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the term of the agreement?", "top_k": 3}'

# Get info
curl http://localhost:7860/info
```

### Stop Services
```bash
docker-compose down
```

---

## Production (HuggingFace Spaces)

### Prerequisites
1. Create free account at [huggingface.co](https://huggingface.co)
2. Create a new **Space** repository
3. Select `Docker` as the SDK

### Deploy
```bash
# 1. Clone HF Space to your machine
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>

# 2. Copy this repo into the space
# Copy all files from cuad-ai-demo/ into the space directory

# 3. Configure secrets in HF Spaces web UI
# Go to Space settings → Secrets and add:
#   - QDRANT_URL: <your-qdrant-cloud-url>
#   - QDRANT_API_KEY: <your-qdrant-api-key>

# 4. Commit and push
git add .
git commit -m "Deploy CUAD search API"
git push
```

HF Spaces will:
- Auto-detect Dockerfile
- Build using `FROM ... AS production` stage
- Deploy on URL: `https://huggingface.co/spaces/<username>/<space-name>`

### Post-Deployment
1. Index documents once:
   ```bash
   # SSH into space or use HF Spaces Terminal
   python cuad-demo-quadrant/upload_to_qdrant.py
   ```

2. Test production:
   ```bash
   curl -X POST https://<space-url>/search \
     -H "Content-Type: application/json" \
     -d '{"query": "payment terms", "top_k": 5}'
   ```

---

## Testing Production Build Locally

Before deploying to HF Spaces, test the production build locally:

```bash
# 1. Build production image
docker-compose -f docker-compose.prod.yml build

# 2. Set cloud credentials in .env.prod (same as .env.dev)
# Update with your Qdrant Cloud credentials

# 3. Run production container
docker-compose -f docker-compose.prod.yml up
```

This uses the `production` Dockerfile stage and the same cloud Qdrant, exactly like HF Spaces.

---

## Environment Files

Both dev and production use cloud Qdrant. Set your credentials in the config files:

### `.env.dev` (Local Development)
```
QDRANT_URL=https://<your-cluster>.qdrant.io:6333
QDRANT_API_KEY=<your-api-key>
EMBEDDING_SERVICE_URL=http://localhost:8001
DEBUG=true
LOG_LEVEL=INFO
```

### `.env.prod` (Production/HF Spaces)
```
QDRANT_URL=https://<your-cluster>.qdrant.io:6333
QDRANT_API_KEY=<your-api-key>
EMBEDDING_SERVICE_URL=http://localhost:8001
DEBUG=false
LOG_LEVEL=WARNING
```

---

## Troubleshooting

### Models take too long to load?
- First request loads SentenceTransformer (~1-2 min)
- Subsequent requests are instant
- Use `/info` endpoint to check status

### Can't access API on localhost:7860?
```bash
# Check if container is running
docker ps | grep cuad

# View logs
docker-compose logs cuad-api

# Check port is available
lsof -i :7860
```

### Can't connect to cloud Qdrant?
- Verify QDRANT_URL format: `https://cluster-name.qdrant.io:6333`
- Check QDRANT_API_KEY is correct
- Verify the cluster is running in Qdrant Cloud dashboard
- Check network firewall allows HTTPS outbound

### Production build fails?
- Ensure Docker supports multi-stage builds (Docker 17.05+)
- Check `docker --version`
- Try: `docker-compose -f docker-compose.prod.yml build --no-cache`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `app.py` | FastAPI main application |
| `Dockerfile` | Multi-stage build (dev/prod) |
| `docker-compose.yml` | Local dev compose (default) |
| `docker-compose.dev.yml` | Explicit dev config |
| `docker-compose.prod.yml` | Production config |
| `.env.dev` | Dev environment variables |
| `.env.prod` | Production environment variables |
| `ENVIRONMENTS.md` | Detailed environment documentation |
| `LOCAL_TESTING.md` | Local testing procedures |

---

## Next Steps

1. **Create Qdrant Cloud account** at https://qdrant.tech/cloud/
2. **Get cluster credentials** - QDRANT_URL and QDRANT_API_KEY
3. **Local dev**: Update `.env.dev` with credentials and run `docker-compose up`
4. **Index documents**: `python cuad-demo-quadrant/upload_to_qdrant.py`
5. **Production**: Create HF Space and push code → configure secrets → deploy

For detailed information, see [ENVIRONMENTS.md](ENVIRONMENTS.md) and [LOCAL_TESTING.md](LOCAL_TESTING.md).
