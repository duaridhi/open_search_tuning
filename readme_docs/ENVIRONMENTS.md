# Environment Configuration Guide

This project supports separate configurations for **development** (local) and **production** (HF Spaces/cloud).

Both environments use **cloud Qdrant** for the vector database. The difference is in service organization and logging/debug settings.

## Overview

| Environment | Purpose | Qdrant | Services | Network | Port |
|-------------|---------|--------|----------|---------|------|
| **Dev** (`.env.dev`) | Local development & testing | Cloud Qdrant | Both separate | `localhost:*` | 7860, 8001 |
| **Prod** (`.env.prod`) | HF Spaces deployment | Cloud Qdrant | Both in container | Private | 7860 |

---

## Development Environment (Local)

### Use When:
- ✅ Developing locally on your machine
- ✅ Testing new features
- ✅ Debugging issues
- ✅ Running locally for demos

### Configuration Files:
- **Docker Compose**: `docker-compose.yml` (or `docker-compose.dev.yml`)
- **Environment**: `.env.dev`
- **Dockerfile Stage**: `development`

### Setup:

```bash
# 1. Get cloud Qdrant credentials
# - Create account at https://qdrant.tech/cloud/
# - Copy cluster URL and API key

# 2. Configure dev environment
cp .env.example .env.dev
# Edit .env.dev with your Qdrant credentials

# 3. Start services
docker-compose up --build

# 4. Both services accessible locally:
#    - Main API: http://localhost:7860
#    - Embedding Service: http://localhost:8001
```

### Characteristics:
- ✅ Uses cloud Qdrant (same cluster as production)
- ✅ Both services separated (easier debugging)
- ✅ Volume mounts for live code reload
- ✅ Port 8001 exposed (Embedding Service visible)
- ✅ Debug mode enabled
- ✅ Full logging active

---

## Production Environment (HF Spaces)

### Use When:
- ✅ Deploying to HF Spaces
- ✅ Production deployment
- ✅ Cloud staging environment
- ✅ Public-facing application

### Configuration Files:
- **Docker Compose**: `docker-compose.prod.yml` (for local staging)
- **Environment**: `.env.prod`
- **Dockerfile Stage**: `production`
- **Entrypoint**: `docker-entrypoint.sh` (same for both)

### Setup:

```bash
# 1. Configure production environment (same Qdrant cluster)
cp .env.example .env.prod
# Edit .env.prod with your Qdrant credentials

# 2. For local staging:
docker-compose -f docker-compose.prod.yml up --build

# 3. For HF Spaces (automatic):
# - Set QDRANT_URL and QDRANT_API_KEY as secrets in HF Spaces
# - Push code to HF Spaces repo
# - HF automatically detects Dockerfile and uses production stage
# API accessible at: https://huggingface.co/spaces/{username}/app
```

### Characteristics:
- ✅ Both services run in single container
- ✅ Uses cloud Qdrant (external)
- ✅ Port 8001 NOT exposed (internal only)
- ✅ Only port 7860 exposed
- ✅ Health checks enabled
- ✅ Auto-restart on failure
- ✅ Minimal logging (production)

---

## Environment Variables

### Key Differences

Both development and production use the same cloud Qdrant cluster. The configuration is nearly identical:

**Development (.env.dev):**
```bash
QDRANT_URL=https://your-cluster.qdrant.io:6333  # Cloud Qdrant
QDRANT_API_KEY=your-api-key-here                # API credentials
EMBEDDING_SERVICE_URL=http://localhost:8001      # Local
DEBUG=true
LOG_LEVEL=INFO
```

**Production (.env.prod):**
```bash
QDRANT_URL=https://your-cluster.qdrant.io:6333  # Same cloud Qdrant
QDRANT_API_KEY=your-api-key-here                # Same credentials
EMBEDDING_SERVICE_URL=http://localhost:8001     # Internal in container
DEBUG=false
LOG_LEVEL=WARNING
```

---

## Quick Commands

### Development

```bash
# Start (uses docker-compose.yml = docker-compose.dev.yml)
docker-compose up --build

# View logs
docker-compose logs -f

# Execute command in container
docker-compose exec cuad-api bash

# Stop
docker-compose down
```

### Production (Local Staging)

```bash
# Start (explicitly use production config)
docker-compose -f docker-compose.prod.yml up --build

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop
docker-compose -f docker-compose.prod.yml down
```

### HF Spaces (Automatic)

```bash
# Just push code - HF handles the rest
git push

# Monitor in HF Spaces → App tab
# See real-time build and startup logs
```

---

## Docker Build Stages

The Dockerfile uses **multi-stage build** for efficiency:

### Base Stage
```dockerfile
FROM python:3.9-slim AS base
# Install all dependencies once
# Download NLTK data
```

### Development Stage
```dockerfile
FROM base AS development
# Copy full source code
# Mount volumes for live reload
# Expose both ports (7860, 8001)
# Enable debugging
```

### Production Stage
```dockerfile
FROM base AS production
# Copy production code only
# Expose only port 7860
# Add health checks
# Disable debugging
```

### Usage

```bash
# Development (default)
docker build -t cuad:dev .
docker build --target development -t cuad:dev .

# Production
docker build --target production -t cuad:prod .
```

---

## When to Use Each

### Use Development When:

1. **Local Testing**
   ```bash
   docker-compose up
   # Make changes to code
   # Volume mounts reload automatically
   # Test at http://localhost:7860
   ```

2. **Debugging Issues**
   ```bash
   docker-compose logs -f
   # See detailed logs and stack traces
   ```

3. **Development Loop**
   ```bash
   # Edit code
   docker-compose up
   # Test
   # Repeat
   ```

### Use Production When:

1. **Cloud Deployment**
   ```bash
   # Configure .env.prod with cloud Qdrant
   # Push to HF Spaces
   # HF builds production stage
   ```

2. **Local Staging/Testing**
   ```bash
   # Test production build locally before deploying
   docker-compose -f docker-compose.prod.yml up
   ```

3. **Final Verification**
   ```bash
   # Verify prod setup works before HF deployment
   curl http://localhost:7860/health
   ```

---

## Troubleshooting

### "docker-compose command not found"
Use: `docker-compose` (with hyphen) or `docker compose` (v2+)

### Wrong stage building
```bash
# Explicitly specify stage
docker build --target production -f Dockerfile -t app:prod .
```

### Environment not loading
```bash
# Check which env file is loaded
cat .env.dev   # For dev
cat .env.prod  # For prod

# Verify in container
docker-compose exec cuad-api env | grep QDRANT
```

### Services not communicating
- **Dev**: Check `localhost:6333` is accessible
- **Prod**: Check `QDRANT_URL` is correct and reachable from HF Spaces

---

## Migration Path

```
Development (localhost)
        ↓
    [Test locally]
        ↓
Production Staging (docker-compose.prod.yml)
        ↓
    [Final testing]
        ↓
HF Spaces Production
        ↓
    [Public facing]
```

---

## Summary

| Task | Environment | Command |
|------|-------------|---------|
| Start local dev | Dev | `docker-compose up --build` |
| View dev logs | Dev | `docker-compose logs -f` |
| Debug issue | Dev | `docker-compose exec cuad-api bash` |
| Test prod build | Prod | `docker-compose -f docker-compose.prod.yml up` |
| Deploy to HF | Prod | `git push` |
| Monitor HF | Prod | Check HF Spaces → App tab |

---

For more details:
- Local testing guide: [LOCAL_TESTING.md](./LOCAL_TESTING.md)
- HF Spaces deployment: [HF_SPACES_DEPLOYMENT.md](./HF_SPACES_DEPLOYMENT.md)
- Deployment roadmap: [DEPLOYMENT_ROADMAP.md](./DEPLOYMENT_ROADMAP.md)
