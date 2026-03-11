# Deployment Guide: Hugging Face Spaces

This guide walks you through deploying the CUAD AI Demo to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: Create one at https://huggingface.co
2. **Git & GitHub**: Push your code to GitHub
3. **Docker**: Already configured in the Dockerfile

## Architecture Overview

The application has 3 components:

```
┌─────────────────────────────────────────────┐
│      HF Spaces Docker Container (7860)      │
├─────────────────────────────────────────────┤
│  Main API (app.py): FastAPI on port 7860    │
│  ↓ (calls)                                  │
│  Embedding Service: FastAPI on port 8001    │
│  ↓ (uses)                                   │
│  Highlight Models (loaded at startup)       │
└─────────────────────────────────────────────┘
         ↓ (needs external)
┌─────────────────────────────────────────────┐
│   Qdrant Vector Database (external)         │
│   http://qdrant:6333                        │
└─────────────────────────────────────────────┘
```

## Step-by-Step Deployment

### Step 1: Prepare Your Code

1. **Update `.env` file** for HF Spaces:
   ```bash
   cp .env.example .env
   ```

2. **Generate a Qdrant instance** (Choose one option):

   **Option A: Cloud Qdrant (Recommended)**
   - Go to https://cloud.qdrant.io/
   - Create a free account
   - Create a new cluster
   - Get your API key and URL
   - Update `.env`:
     ```
     QDRANT_URL=https://your-cluster-url:6333
     QDRANT_API_KEY=your-api-key
     ```

   **Option B: Using a Docker Compose setup locally (test only)**
   ```yaml
   version: '3.8'
   services:
     qdrant:
       image: qdrant/qdrant:latest
       ports:
         - "6333:6333"
       volumes:
         - ./qdrant_storage:/qdrant/storage
   ```

3. **Initialize Qdrant with data** (before deploying):
   ```bash
   # Run the upload script locally to populate Qdrant
   python cuad-demo-quadrant/upload_to_qdrant.py --pdf-dir path/to/pdfs
   ```

### Step 2: Create a Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `cuad-ai-demo`
   - **License**: Apache 2.0
   - **Space SDK**: Docker
   - **Visibility**: Public or Private
4. Click **"Create Space"**

### Step 3: Set Up GitHub Integration

1. Clone your HF Space repo:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/cuad-ai-demo
   cd cuad-ai-demo
   ```

2. Copy your code:
   ```bash
   cp -r /path/to/cuad-ai-demo/* .
   ```

3. Ensure these files exist:
   - `Dockerfile`
   - `docker-entrypoint.sh`
   - `requirements.txt`
   - `app.py`
   - `cuad-demo-quadrant/` directory
   - `.env` (with secrets configured)

### Step 4: Configure Environment Variables

1. In HF Spaces, go to **Settings** → **Repository secrets**
2. Add these secrets:
   ```
   QDRANT_URL=<your-qdrant-url>
   QDRANT_API_KEY=<your-api-key>
   S3_ENDPOINT_URL=<optional>
   AWS_ACCESS_KEY_ID=<optional>
   AWS_SECRET_ACCESS_KEY=<optional>
   ```

3. Update `.env` in your repo to reference these via environment variables:
   ```bash
   # .env
   QDRANT_URL=${QDRANT_URL:-http://qdrant:6333}
   QDRANT_API_KEY=${QDRANT_API_KEY}
   ```

### Step 5: Push Code to HF Spaces

```bash
git add .
git commit -m "Deploy CUAD AI Demo"
git push
```

HF Spaces will automatically:
1. Detect the Dockerfile
2. Build the container
3. Start the services
4. Serve on the Space URL

### Step 6: Monitor Deployment

1. Watch the **App** tab for startup logs
2. Check for errors in the logs
3. Wait 2-3 minutes for models to load
4. Once ready, the app will be accessible at: `https://huggingface.co/spaces/YOUR_USERNAME/cuad-ai-demo`

## Testing the Deployment

Once deployed, test the endpoints:

```bash
# Health check
curl https://{HF_SPACE_URL}/health

# Search
curl "https://{HF_SPACE_URL}/search?q=liability%20clause&top_k=5"

# List documents
curl "https://{HF_SPACE_URL}/documents"
```

## Troubleshooting

### Issue: "Module not found" errors
**Solution**: Ensure all imports use relative paths in highlights module
```python
from ..embeddings.embedding_service import _state  # Correct
from embeddings.embedding_service import _state    # Wrong
```

### Issue: "Qdrant connection refused"
**Solution**: 
1. Verify `QDRANT_URL` is correct
2. Ensure Qdrant instance is running (if using Cloud Qdrant, it should be)
3. Check firewall/network access

### Issue: "Model loading timeout"
**Solution**: 
1. Increase `MODEL_LOAD_TIMEOUT` in `.env`
2. Use CPU (default) instead of GPU for HF Spaces
3. Consider using smaller models

### Issue: "Out of memory"
**Solution**:
1. HF Spaces has limited memory (~16GB for paid)
2. Use smaller embedding model: `all-MiniLM-L6-v2` (recommended)
3. Reduce batch size in inference

## Production Checklist

- [ ] Qdrant database is initialized with contract data
- [ ] Environment variables are set in HF Spaces secrets
- [ ] `.env.example` is included in repo (no secrets)
- [ ] `docker-entrypoint.sh` is executable
- [ ] All imports use relative paths
- [ ] Dockerfile uses `python:3.9-slim` (lightweight)
- [ ] Test /health endpoint after deployment
- [ ] Test /search endpoint with sample query
- [ ] Monitor Space resource usage

## Cost Considerations (as of 2026)

**HF Spaces Free Tier:**
- CPU-only machines
- Limited memory (~16GB)
- Suspends after inactivity
- Good for demos/development

**HF Spaces Paid Tier (Pro Spaces):**
- $9-15/month
- Always-on
- GPU options available (additional cost)
- Better for production

## Next Steps

1. **Custom Domain**: Add your own domain to the Space
2. **Add UI**: Create a frontend (React/Vue/Streamlit) on top of the API
3. **Analytics**: Track API usage and search patterns
4. **Auto-indexing**: Set up pipeline to auto-index new PDFs
5. **Scaling**: If needed, migrate to Kubernetes or cloud providers

## Support & Resources

- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- Qdrant Docs: https://qdrant.tech/documentation/
- FastAPI Docs: https://fastapi.tiangolo.com/
- Docker Docs: https://docs.docker.com/

---

**Questions?** Open an issue on the repository.
