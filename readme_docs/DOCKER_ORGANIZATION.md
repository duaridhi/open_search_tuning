# Docker and Environment File Organization

## File Structure

```
cuad-ai-demo/
├── .env.dev                      # DEV environment - cloud Qdrant
├── .env.prod                     # PROD environment - cloud Qdrant
├── .env.example                  # Template
│
├── docker-compose.yml            # DEV (default) - runs with .env.dev
├── docker-compose.dev.yml        # DEV (explicit) - runs with .env.dev
├── docker-compose.prod.yml       # PROD - runs with .env.prod
│
├── docker/
│   ├── Dockerfile               # Single multi-stage Dockerfile (base → dev/prod)
│   ├── docker-compose.dev.yml   # DEV config (KEPT FOR REFERENCE, don't use directly)
│   ├── docker-compose.prod.yml  # PROD config (KEPT FOR REFERENCE, don't use directly)
│   └── docker-entrypoint.sh     # Entrypoint for both dev and prod
│
├── app.py
├── requirements.txt
├── cuad-demo-quadrant/
│   └── ...
```

## Key Points

✅ **Root-level docker-compose files** - Easy commands from repository root
✅ **/docker/Dockerfile** - Single source of truth for image definition
✅ **docker/ reference configs** - Kept for documentation (not used in workflow)
✅ **.env files at root** - Easy path references in volume mounts
✅ **Automatic env file loading** - Docker Compose auto-loads `.env` by default

## Usage

### Development (Local)
```bash
# Default: Uses .env.dev automatically
docker-compose up --build

# Or be explicit:
docker-compose -f docker-compose.dev.yml --env-file .env.dev up --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production (Local Test or HF Spaces)
```bash
# Test production build locally
docker-compose -f docker-compose.prod.yml --env-file .env.prod up --build

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop
docker-compose -f docker-compose.prod.yml down
```

### HF Spaces Deployment
1. Push code to HF Spaces repository
2. HF Spaces auto-detects `docker/Dockerfile`
3. Auto-builds using `production` stage
4. Set secrets: QDRANT_URL, QDRANT_API_KEY in HF Spaces settings

## Environment Variables

### .env.dev (Development)
- QDRANT_URL: cloud Qdrant URL
- QDRANT_API_KEY: cloud Qdrant API key
- DEBUG: true
- LOG_LEVEL: INFO

### .env.prod (Production)
- QDRANT_URL: same cloud Qdrant URL
- QDRANT_API_KEY: same cloud Qdrant API key
- DEBUG: false
- LOG_LEVEL: WARNING

## Docker Build Targets

The Dockerfile has two targets:

1. **development** (docker-compose.dev.yml)
   - Exposes port 8001 (Embedding Service)
   - Debug mode enabled
   - Live reload with volume mounts

2. **production** (docker-compose.prod.yml)
   - Port 8001 NOT exposed (internal only)
   - Debug mode disabled
   - Minimal image size
   - Health checks enabled

## Why This Structure?

| Aspect | Benefit |
|--------|---------|
| Root compose files | Easy `docker-compose up` from any directory |
| Dockerfile in /docker/ | Clean, organized, single source of truth |
| .env at root | Path references work naturally (./cuad-demo-quadrant) |
| Multi-stage Dockerfile | Single image definition for both dev & prod |
| Reference configs in /docker/ | Documentation for advanced users |
