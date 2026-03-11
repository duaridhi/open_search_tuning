# Follow HF Spaces guide: https://huggingface.co/docs/hub/spaces-sdks-docker
# Multi-service setup: Embedding Service + Main API
# Multi-stage build for development and production
# Uses BuildKit for efficient layer caching

# syntax=docker/dockerfile:1.4

# ==================== SHARED BASE ====================
FROM python:3.12-slim AS base

# Create app user (minimal layer)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching (this layer only rebuilds when requirements.txt changes)
COPY --chown=user --chmod=644 requirements.txt requirements.txt

# Install dependencies with caching
# Use --mount=type=cache for pip cache in BuildKit
RUN --mount=type=cache,target=/home/user/.cache/pip,uid=1000,gid=1000 \
    pip install --no-cache-dir --upgrade pip && \
    pip install --upgrade -r requirements.txt

# Pre-download NLTK data (cached separately)
RUN python -m nltk.downloader punkt -d /home/user/nltk_data 2>/dev/null || true


# ==================== DEVELOPMENT STAGE ====================
FROM base AS development

# Copy application code (with volumes for live reload)
COPY --chown=user . /app

# Copy and make entrypoint script executable
COPY --chown=user docker/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose ports:
# 7860 - Main API
# 8001 - Embedding Service (visible locally)
EXPOSE 7860 8001

# Development environment variables
ENV EMBEDDING_SERVICE_URL=http://localhost:8001
ENV QDRANT_URL=http://qdrant:6333
ENV DEBUG=true

# Use the entrypoint script to start both services
ENTRYPOINT ["/app/docker-entrypoint.sh"]


# ==================== PRODUCTION STAGE ====================
FROM base AS production

# Copy application code (minimal, for production)
COPY --chown=user . /app

# Copy and make entrypoint script executable
COPY --chown=user docker/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose only port 7860 (API)
# 8001 is internal/not exposed in production
EXPOSE 7860

# Production environment variables
ENV EMBEDDING_SERVICE_URL=http://localhost:8001
ENV QDRANT_URL=http://qdrant:6333
ENV DEBUG=false

# Health check for production
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Use the entrypoint script to start both services
ENTRYPOINT ["/app/docker-entrypoint.sh"]