"""
Embedding Inference Service
Standalone API for computing embeddings using SentenceTransformer.
Can be deployed separately and reused by multiple search services.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_SERVICE_DEVICE = os.getenv("EMBEDDING_SERVICE_DEVICE", "cpu")

# Global state
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[STARTUP] Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
    _state["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_SERVICE_DEVICE)
    print("[STARTUP] Embedding service ready")
    yield
    print("[SHUTDOWN] Embedding service closing")


app = FastAPI(
    title="Embedding Inference Service",
    description="Standalone service for computing text embeddings using SentenceTransformer",
    version="1.0.0",
    lifespan=lifespan,
)


# --------- Schemas ---------
class EmbedRequest(BaseModel):
    texts: list[str]
    convert_to_numpy: bool = False


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimension: int


# --------- Routes ---------
@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "service": "embedding"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """
    Compute embeddings for a batch of texts.
    
    Args:
        texts: List of text strings to embed
        convert_to_numpy: Whether to return as numpy arrays (not recommended for JSON)
    
    Returns:
        Embeddings as list of float lists
    """
    try:
        model = _state["model"]
        embeddings = model.encode(
            request.texts,
            convert_to_numpy=request.convert_to_numpy
        )
        
        # Convert to list if numpy array
        if hasattr(embeddings, 'tolist'):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
        
        return EmbedResponse(
            embeddings=embeddings_list,
            model=EMBEDDING_MODEL_NAME,
            dimension=len(embeddings_list[0]) if embeddings_list else 0
        )
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


@app.get("/info")
def info():
    """Get service information."""
    model = _state.get("model")
    if model:
        # Get model info
        config = model.get_sentence_embedding_dimension()
        return {
            "service": "embedding",
            "model": EMBEDDING_MODEL_NAME,
            "dimension": config,
            "device": EMBEDDING_SERVICE_DEVICE,
        }
    return {"status": "model not loaded"}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("EMBEDDING_SERVICE_PORT", "8001"))
    host = os.getenv("EMBEDDING_SERVICE_HOST", "0.0.0.0")
    
    print(f"Starting Embedding Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
