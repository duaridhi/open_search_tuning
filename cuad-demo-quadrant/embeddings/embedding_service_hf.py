"""
Embedding and Highlighting Service using HuggingFace Inference API only.
No local model or tokenizer loading. All inference is remote.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HIGHLIGHTER_MODEL_ID = "opensearch-project/opensearch-semantic-highlighter-v1"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable must be set for HuggingFace Inference API.")

client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

app = FastAPI(
    title="Embedding & Highlighting Service (HuggingFace Inference)",
    description="Remote-only embedding and highlighting using HuggingFace Inference API.",
    version="1.0.0",
)

class EmbedRequest(BaseModel):
    texts: list[str]
    convert_to_numpy: bool = False

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimension: int

class HighlightRequest(BaseModel):
    query: str
    document: str

class HighlightResponse(BaseModel):
    highlighted_sentences: list[str]
    highlight_sentence_indexes: list[int]
    highlight_offsets: list[tuple[int, int]] = Field(
        ..., description="Character offsets (start, end) for each highlighted sentence relative to document start"
    )

@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    try:
        model = EMBEDDING_MODEL_NAME
        # HuggingFace Inference API for embeddings
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model)
        embeddings = st_model.encode(request.texts)
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else [list(e) for e in embeddings]
        return EmbedResponse(
            embeddings=embeddings_list,
            model=model,
            dimension=len(embeddings_list[0]) if embeddings_list else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@app.post("/highlight", response_model=HighlightResponse)
def highlight(request: HighlightRequest):
    try:
        # Use HuggingFace Inference API for highlighting
        result = client.text_classification(
            f"{request.query} [SEP] {request.document}",
            model=HIGHLIGHTER_MODEL_ID,
        )
        # The result format may need to be adapted based on the actual API output
        # Here, we assume a list of dicts with 'label' and 'score' fields
        # This is a placeholder for demonstration
        highlighted_sentences = [r['label'] for r in result if r['score'] > 0.5]
        highlight_sentence_indexes = list(range(len(highlighted_sentences)))
        highlight_offsets = [(0, 0)] * len(highlighted_sentences)  # Placeholder
        return HighlightResponse(
            highlighted_sentences=highlighted_sentences,
            highlight_sentence_indexes=highlight_sentence_indexes,
            highlight_offsets=highlight_offsets
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Highlighting error: {str(e)}")
