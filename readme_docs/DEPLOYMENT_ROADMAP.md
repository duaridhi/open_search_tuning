# Deployment Roadmap for HF Spaces

## What Was Updated ✅

### 1. **Dockerfile** (`Dockerfile`)
   - Multi-service setup (Embedding Service + Main API in one container)
   - Optimized for HF Spaces (`python:3.9-slim`, 7860 port)
   - Pre-downloads NLTK data at build time
   - Sets up entrypoint script for managing both services

### 2. **Startup Script** (`docker-entrypoint.sh`)
   - Starts Embedding Service on port 8001 (background)
   - Waits for warm-up (10 seconds)
   - Starts Main API on port 7860 (foreground)
   - Manages both services lifecycle

### 3. **Requirements** (`requirements.txt`)
   - Added missing dependencies: `transformers`, `datasets`, `httpx`, `numpy`
   - All packages pinned to stable versions
   - Ready for production deployment

### 4. **Environment Template** (`.env.example`)
   - Reference for all configurable variables
   - Safe to commit (no secrets)
   - Copy to `.env` before running

### 5. **Documentation**
   - `HF_SPACES_DEPLOYMENT.md` - Complete HF Spaces deployment guide
   - `LOCAL_TESTING.md` - Local Docker Compose testing guide

## Next Steps (In Order)

### Phase 1️⃣: Local Testing (15 minutes)

```bash
# 1. Create .env from template
cp .env.example .env

# 2. Edit .env with your Qdrant URL
# QDRANT_URL=http://qdrant:6333  # or cloud URL

# 3. Populate Qdrant with contract data
python cuad-demo-quadrant/upload_to_qdrant.py \
  --pdf-dir /path/to/pdfs \
  --qdrant-url http://localhost:6333

# 4. Test with Docker Compose
docker-compose up --build

# 5. In another terminal, test endpoints
curl http://localhost:7860/health
curl "http://localhost:7860/search?q=test&top_k=5"
```

**Expected result**: API responds with search results ✓

---

### Phase 2️⃣: Prepare for HF Spaces (10 minutes)

```bash
# 1. Clean up Docker
docker-compose down -v

# 2. Make entrypoint executable (if not already)
chmod +x docker-entrypoint.sh

# 3. Verify critical files exist
ls -la Dockerfile docker-entrypoint.sh requirements.txt
ls -la app.py cuad-demo-quadrant/

# 4. Commit all changes
git add -A
git commit -m "Prepare for HF Spaces deployment"
```

---

### Phase 3️⃣: Create HF Spaces Repository (5 minutes)

1. Go to **https://huggingface.co/spaces**
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `cuad-ai-demo`
   - **Space SDK**: Docker
   - **Visibility**: Public
4. Click **"Create Space"**
5. Get your Space URL: `https://huggingface.co/spaces/{username}/cuad-ai-demo`

---

### Phase 4️⃣: Connect & Deploy (5 minutes)

```bash
# 1. Clone HF Space repository
git clone https://huggingface.co/spaces/{username}/cuad-ai-demo
cd cuad-ai-demo

# 2. Add your code
cp -r /path/to/local/repo/* .

# 3. Add HF secrets (via web UI)
# HF Space → Settings → Repository secrets
# Add:
#   QDRANT_URL=<your-qdrant-url>
#   QDRANT_API_KEY=<if-needed>
#   (Optional S3 credentials)

# 4. Push to HF
git add .
git commit -m "Deploy CUAD AI Demo to HF Spaces"
git push

# 5. Monitor deployment
# Watch HF Space → App tab for build logs
# Wait 2-3 minutes for models to load
```

---

### Phase 5️⃣: Test on HF Spaces (5 minutes)

Once deployed, test:

```bash
# Replace {username} with your HF username
APP_URL="https://huggingface.co/spaces/{username}/cuad-ai-demo"

# Health check
curl $APP_URL/health

# Search (if you have contracts indexed)
curl "$APP_URL/search?q=liability&top_k=5"

# View API docs
# Open: $APP_URL/docs
```

---

## Important Considerations

### 🔑 Qdrant Setup (REQUIRED)

You MUST have a running Qdrant instance before deploying. **Choose one:**

**Option A: Cloud Qdrant (Recommended)**
- Sign up: https://cloud.qdrant.io/
- Create cluster (free tier available)
- Get URL + API key
- Update `.env` and HF secrets

**Option B: Self-hosted**
- Run Qdrant Docker container
- Ensure it's accessible from HF Spaces
- Update `QDRANT_URL` to external IP

**Option C: Docker Compose (Local testing only)**
- Use included `docker-compose.yml`
- Qdrant only accessible locally

### 📊 Contract Data

**Must populate Qdrant BEFORE deploying:**

```bash
python cuad-demo-quadrant/upload_to_qdrant.py \
  --pdf-dir /path/to/contracts \
  --qdrant-url http://your-qdrant-url:6333
```

Without contracts, search returns empty results.

### 💾 Resource Limits (HF Spaces)

| Tier | CPU | Memory | Cost | Uptime |
|------|-----|--------|------|--------|
| Free | 2 cores | 16GB | $0 | Suspends after inactivity |
| Pro | 8 cores | 32GB | $9-15/mo | Always-on |
| GPU | 1x T4 GPU | 32GB | $0.60/hr | Always-on |

**Recommendation**: Start with free tier, upgrade if needed

### ⏱️ Cold Start Time

- First request: ~2-3 minutes (models loading)
- Subsequent requests: 1-5 seconds
- Cached after first use

---

## Quick Reference Checklist

```
LOCAL TESTING
☐ .env created from .env.example
☐ Qdrant instance running
☐ Contracts indexed in Qdrant
☐ docker-compose up builds successfully
☐ /health endpoint responds
☐ /search returns results

HF SPACES SETUP
☐ Created Space on HF
☐ Qdrant instance prepared (Cloud or self-hosted)
☐ docker-entrypoint.sh is executable
☐ All files committed and pushed
☐ Secrets configured in HF Space settings
☐ Dockerfile builds successfully

DEPLOYMENT VERIFICATION
☐ App loads (2-3 min wait)
☐ /health endpoint returns 200
☐ /docs shows API documentation
☐ /search?q=test returns results (if contracts indexed)
☐ /documents lists indexed contracts
```

---

## Troubleshooting Guide

See detailed troubleshooting in:
- **Local issues**: `LOCAL_TESTING.md` → Troubleshooting
- **HF Spaces issues**: `HF_SPACES_DEPLOYMENT.md` → Troubleshooting

Common issues:
- **Module not found**: Check relative imports (use `../` for sibling modules)
- **Qdrant connection refused**: Verify URL and network access
- **Memory errors**: Use smaller model or GPU tier
- **Timeout on models**: Increase `MODEL_LOAD_TIMEOUT`

---

## Support

If you encounter issues:

1. Check the relevant troubleshooting guide above
2. Review Docker logs: `docker-compose logs -f`
3. Check HF Space logs (web UI)
4. Verify `QDRANT_URL` is correct and accessible
5. Ensure contracts are indexed in Qdrant

---

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│        Frontend Application (React/Vue)     │
│              User Interface                 │
└────────────────────┬────────────────────────┘
                     │
                     │ HTTPS
                     ▼
┌─────────────────────────────────────────────┐
│        HF Spaces Docker Container           │
│            Port 7860 (Public)               │
├─────────────────────────────────────────────┤
│   FastAPI Main App (app.py)                 │
│   ├─ /search                                │
│   ├─ /documents                             │
│   ├─ /health                                │
│   └─ /docs (Swagger UI)                     │
│          │                                  │
│          │ HTTP localhost:8001              │
│          ▼                                  │
│   Embedding Service (embedding_service.py) │
│   ├─ /embed      (SentenceTransformer)      │
│   ├─ /highlight  (BERT Highlighter)        │
│   └─ Models (pre-loaded at startup)         │
└──────────┬──────────────────────────────────┘
           │
           │ HTTP/HTTPS
           ▼
┌─────────────────────────────────────────────┐
│     Qdrant Vector Database (External)       │
│     cloud.qdrant.io or self-hosted          │
│                                             │
│  Collection: cuad_contracts                 │
│  ├─ Vectors (384-D embeddings)              │
│  ├─ Text chunks                             │
│  └─ Metadata (page, offset, etc)            │
└─────────────────────────────────────────────┘
```

---

## Success Criteria ✅

Your deployment is successful when:

1. ✅ App loads without errors
2. ✅ `/health` responds with `{"status": "ok"}`
3. ✅ `/docs` displays Swagger UI
4. ✅ `/search?q=test` returns results (if contracts indexed)
5. ✅ Highlights show in results with offsets
6. ✅ Response time < 5 seconds (after cold start)

---

**Ready to deploy? Start with Phase 1️⃣ Local Testing above!**
