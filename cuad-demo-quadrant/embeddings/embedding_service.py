"""
Embedding Inference Service
Standalone API for computing embeddings using SentenceTransformer.
Can be deployed separately and reused by multiple search services.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from functools import wraps

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Import highlighter components
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Authenticate with Hugging Face Hub if token is available
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("[STARTUP] Authenticated with Hugging Face Hub")
    except Exception as e:
        print(f"[WARN] Failed to authenticate with Hugging Face Hub: {e}")

# Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_SERVICE_DEVICE = os.getenv("EMBEDDING_SERVICE_DEVICE", "cpu")

# Highlights model configuration
HIGHLIGHTER_MODEL_ID = "opensearch-project/opensearch-semantic-highlighter-v1"
HIGHLIGHTER_BASE_MODEL_ID = "bert-base-uncased"

# Timeout configurations (in seconds)
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "30"))
MODEL_LOAD_TIMEOUT = int(os.getenv("MODEL_LOAD_TIMEOUT", "120"))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "30"))

# Global state
_state: dict = {}


# --------- Highlights Model Definition ---------
class BertTaggerForSentenceExtractionWithBackoff(BertPreTrainedModel):
    """Sentence-level BERT classifier with a confidence-backoff rule."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        import torch.nn as nn
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    @property
    def all_tied_weights_keys(self):
        return {}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_ids=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = self.dropout(outputs[0])

        def _get_agg_output(ids, seq_out):
            max_sentences = torch.max(ids) + 1
            d_model = seq_out.size(-1)

            agg_out, global_offsets, num_sents = [], [], []
            for i, sen_ids in enumerate(ids):
                out, local_ids = [], sen_ids.clone()
                mask = local_ids != -100
                offset = local_ids[mask].min()
                global_offsets.append(offset)
                local_ids[mask] -= offset
                n_sent = local_ids.max() + 1
                num_sents.append(n_sent)

                for j in range(int(n_sent)):
                    out.append(seq_out[i, local_ids == j].mean(dim=-2, keepdim=True))

                if max_sentences - n_sent:
                    padding = torch.zeros(
                        (int(max_sentences - n_sent), d_model), device=seq_out.device
                    )
                    out.append(padding)
                agg_out.append(torch.cat(out, dim=0))
            return torch.stack(agg_out), global_offsets, num_sents

        agg_output, offsets, num_sents_item = _get_agg_output(sentence_ids, sequence_output)
        logits = self.classifier(agg_output)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]

        def _get_preds(pp, offs, num_s, threshold=0.5, alpha=0.05):
            preds = []
            for p, off, ns in zip(pp, offs, num_s):
                rel_probs = p[:ns]
                hits = (rel_probs >= threshold).int()
                if hits.sum() == 0 and rel_probs.max().item() >= alpha:
                    hits[rel_probs.argmax()] = 1
                preds.append(torch.where(hits == 1)[0] + off)
            return preds

        return tuple(_get_preds(probs, offsets, num_sents_item))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f"[STARTUP] Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
        model_loaded = False
        for retry in range(3):
            try:
                print(f"[STARTUP] SentenceTransformer load attempt {retry + 1}/3...")
                _state["model"] = await asyncio.wait_for(
                    asyncio.to_thread(
                        SentenceTransformer,
                        EMBEDDING_MODEL_NAME,
                        device=EMBEDDING_SERVICE_DEVICE
                    ),
                    timeout=MODEL_LOAD_TIMEOUT
                )
                model_loaded = True
                print("[STARTUP] SentenceTransformer model loaded successfully")
                break
            except asyncio.TimeoutError:
                print(f"[WARN] SentenceTransformer loading attempt {retry + 1} timed out after {MODEL_LOAD_TIMEOUT}s")
                if retry == 2:
                    raise RuntimeError(f"SentenceTransformer loading timeout after 3 attempts")
            except Exception as e:
                print(f"[WARN] SentenceTransformer loading attempt {retry + 1} failed: {e}")
                if retry == 2:
                    raise RuntimeError(f"Failed to load SentenceTransformer: {e}")
                await asyncio.sleep(2)  # Wait before retry
        
        if not model_loaded:
            raise RuntimeError("SentenceTransformer model failed to load after all retry attempts")
        
        print(f"[STARTUP] Loading Highlighter model: {HIGHLIGHTER_MODEL_ID}")
        model_loaded = False
        for retry in range(3):
            try:
                print(f"[STARTUP] Highlighter model load attempt {retry + 1}/3...")
                _state["highlighter_model"] = await asyncio.wait_for(
                    asyncio.to_thread(
                        BertTaggerForSentenceExtractionWithBackoff.from_pretrained,
                        HIGHLIGHTER_MODEL_ID,
                        trust_remote_code=True
                    ),
                    timeout=MODEL_LOAD_TIMEOUT
                )
                _state["highlighter_model"].eval()
                model_loaded = True
                print("[STARTUP] Highlighter model loaded successfully")
                break
            except asyncio.TimeoutError:
                print(f"[WARN] Highlighter model loading attempt {retry + 1} timed out after {MODEL_LOAD_TIMEOUT}s")
                if retry == 2:
                    raise RuntimeError(f"Highlighter model loading timeout after 3 attempts")
            except Exception as e:
                print(f"[WARN] Highlighter model loading attempt {retry + 1} failed: {e}")
                if retry == 2:
                    raise RuntimeError(f"Failed to load highlighter model: {e}")
                await asyncio.sleep(2)  # Wait before retry
        
        if not model_loaded:
            raise RuntimeError("Highlighter model failed to load after all retry attempts")
        
        print(f"[STARTUP] Loading Highlighter tokenizer: {HIGHLIGHTER_BASE_MODEL_ID}")
        tokenizer_loaded = False
        for retry in range(3):
            try:
                print(f"[STARTUP] Tokenizer load attempt {retry + 1}/3...")
                _state["highlighter_tokenizer"] = await asyncio.wait_for(
                    asyncio.to_thread(
                        AutoTokenizer.from_pretrained,
                        HIGHLIGHTER_BASE_MODEL_ID,
                        trust_remote_code=True
                    ),
                    timeout=MODEL_LOAD_TIMEOUT
                )
                tokenizer_loaded = True
                print("[STARTUP] Highlighter tokenizer loaded successfully")
                break
            except asyncio.TimeoutError:
                print(f"[WARN] Highlighter tokenizer loading attempt {retry + 1} timed out after {MODEL_LOAD_TIMEOUT}s")
                if retry == 2:
                    raise RuntimeError(f"Highlighter tokenizer loading timeout after 3 attempts")
            except Exception as e:
                print(f"[WARN] Highlighter tokenizer loading attempt {retry + 1} failed: {e}")
                if retry == 2:
                    raise RuntimeError(f"Failed to load highlighter tokenizer: {e}")
                await asyncio.sleep(2)  # Wait before retry
        
        if not tokenizer_loaded:
            raise RuntimeError("Highlighter tokenizer failed to load after all retry attempts")
        
        print("[STARTUP] Embedding service ready with both embedding and highlighting models")
        yield
        
    except Exception as e:
        print(f"[STARTUP FAILED] {e}")
        raise
    finally:
        print("[SHUTDOWN] Embedding service closing")
        _state.clear()


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
async def embed(request: EmbedRequest):
    """
    Compute embeddings for a batch of texts with timeout protection.
    
    Args:
        texts: List of text strings to embed
        convert_to_numpy: Whether to return as numpy arrays (not recommended for JSON)
    
    Returns:
        Embeddings as list of float lists
    """
    try:
        if "model" not in _state:
            raise HTTPException(
                status_code=503,
                detail="Embedding model not loaded"
            )
        
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="texts list cannot be empty"
            )
        
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=413,
                detail="Maximum 100 texts allowed per request"
            )
        
        model = _state["model"]
        
        try:
            embeddings = await asyncio.wait_for(
                asyncio.to_thread(
                    model.encode,
                    request.texts,
                    convert_to_numpy=request.convert_to_numpy
                ),
                timeout=INFERENCE_TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"[ERROR] Embedding inference exceeded {INFERENCE_TIMEOUT}s")
            raise HTTPException(
                status_code=504,
                detail=f"Embedding computation timed out after {INFERENCE_TIMEOUT}s"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[ERROR] Out of memory during embedding: {e}")
                raise HTTPException(
                    status_code=507,
                    detail="Insufficient memory for embedding computation"
                )
            raise
        
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
        
    except HTTPException:
        raise
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
            "highlighter_model": HIGHLIGHTER_MODEL_ID,
            "highlighter_base_model": HIGHLIGHTER_BASE_MODEL_ID,
        }
    return {"status": "model not loaded"}


@app.get("/highlight-ready")
def highlight_ready():
    """Check if highlighter models are ready."""
    has_model = "highlighter_model" in _state
    has_tokenizer = "highlighter_tokenizer" in _state
    return {
        "ready": has_model and has_tokenizer,
        "highlighter_model_loaded": has_model,
        "highlighter_tokenizer_loaded": has_tokenizer,
    }


class HighlightRequest(BaseModel):
    query: str
    document: str


class HighlightResponse(BaseModel):
    highlighted_sentences: list[str]
    highlight_sentence_indexes: list[int]
    highlight_offsets: list[tuple[int, int]] = Field(
        ..., 
        description="Character offsets (start, end) for each highlighted sentence relative to document start"
    )


@app.post("/highlight", response_model=HighlightResponse)
def highlight(request: HighlightRequest):
    """
    Highlight relevant sentences in a document using semantic highlighter.
    
    Args:
        query: Search query
        document: Document text to highlight
    
    Returns:
        Highlighted sentences and their indexes
    """
    try:
        if "highlighter_model" not in _state or "highlighter_tokenizer" not in _state:
            raise HTTPException(
                status_code=503,
                detail="Highlighter models not loaded"
            )
        
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="query cannot be empty"
            )
        
        if not request.document or not request.document.strip():
            raise HTTPException(
                status_code=400,
                detail="document cannot be empty"
            )
        
        # Use the qdrant_opensearch_highlights module to run highlighting
        from highlights.qdrant_opensearch_highlights import highlight_document
        
        highlights = highlight_document(
            query=request.query,
            document=request.document
        )
        
        return HighlightResponse(
            highlighted_sentences=highlights.get("highlighted_sentences", []),
            highlight_sentence_indexes=highlights.get("highlight_sentence_indexes", []),
            highlight_offsets=highlights.get("highlight_offsets", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Highlighting failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Highlighting error: {str(e)}")


# --------- Helper Functions ---------
def get_highlighter_model():
    """Get the loaded highlighter model from global state."""
    return _state.get("highlighter_model")


def get_highlighter_tokenizer():
    """Get the loaded highlighter tokenizer from global state."""
    return _state.get("highlighter_tokenizer")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("EMBEDDING_SERVICE_PORT", "8001"))
    host = os.getenv("EMBEDDING_SERVICE_HOST", "0.0.0.0")
    
    print(f"Starting Embedding Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
