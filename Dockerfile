# Hugging Face Spaces Docker deployment
# FastAPI application for CUAD contract search using Qdrant
# Follow HF Spaces guide: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.12-slim

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
    pip install --upgrade -r requirements.txt

# Pre-download NLTK data for sentence tokenization
RUN python -m nltk.downloader punkt -d /home/user/nltk_data 2>/dev/null || true

# Copy application code
COPY --chown=user . /app

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Set environment variables for HF Spaces
ENV EMBEDDING_SERVICE_DEVICE=cpu
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health', timeout=5)" || exit 1

# Run the application with uvicorn on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
