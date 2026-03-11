# Follow HF Spaces guide: https://huggingface.co/docs/hub/spaces-sdks-docker
# Multi-service setup: Embedding Service + Main API
# Multi-stage build for development and production

# ==================== SHARED BASE ====================
FROM python:3.9-slim AS base

# Create app user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements
COPY --chown=user requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Pre-download NLTK data
RUN python -m nltk.downloader punkt -d /home/user/nltk_data 2>/dev/null || true


# ==================== DEVELOPMENT STAGE ====================
FROM base AS development

# Copy application code (with volumes for live reload)
COPY --chown=user . /app

# Copy and make entrypoint script executable
COPY --chown=user docker-entrypoint.sh /app/docker-entrypoint.sh
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
COPY --chown=user docker-entrypoint.sh /app/docker-entrypoint.sh
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