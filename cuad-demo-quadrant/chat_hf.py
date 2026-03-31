"""
chat_hf.py
----------
RAG chat using HuggingFace Inference API (OpenAI-compatible endpoint).
Takes top_k search results from qdrant_search_hf.search() and generates
a grounded answer from the retrieved contract passages.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen3-235B-A22B:novita")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable must be set for HuggingFace Inference API.")

_inference_client: Optional[InferenceClient] = None


def _get_inference_client() -> InferenceClient:
    global _inference_client
    if _inference_client is None:
        _inference_client = InferenceClient(api_key=HF_TOKEN)
    return _inference_client


def _build_context(documents: list[dict]) -> str:
    """Format retrieved documents into a context block for the prompt."""
    parts = []
    for i, doc in enumerate(documents, start=1):
        title = doc.get("title", "Unknown")
        text = doc.get("text", "").strip()
        score = doc.get("score", 0.0)
        parts.append(f"[Document {i}] Title: {title} (score: {score:.3f})\n{text}")
    return "\n\n---\n\n".join(parts)


def chat(
    query: str,
    documents: list[dict],
    system_prompt: Optional[str] = None,
) -> str:
    """
    Generate an answer grounded in the provided search results.

    Parameters
    ----------
    query : str
        The user's original question.
    documents : list[dict]
        Top-K results returned by qdrant_search_hf.search().
        Each dict must contain at least 'title' and 'text'.
    system_prompt : str, optional
        Override the default system instructions.

    Returns
    -------
    str
        The model's generated answer.
    """
    if not documents:
        return "No relevant contract passages were found for your query."

    context = _build_context(documents)

    if system_prompt is None:
        system_prompt = (
            "You are a legal contract analysis assistant. "
            "Answer the user's question using ONLY the contract passages provided below. "
            "Cite the document title and sections when referencing specific clauses. "
            "If the answer cannot be determined from the passages, say so explicitly."
        )

    user_message = (
        f"Contract passages:\n\n{context}\n\n"
        f"Question: {query}"
    )

    logger.info("Calling HuggingFace chat model: %s", CHAT_MODEL)

    client = _get_inference_client()
    _t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    _elapsed = time.perf_counter() - _t0

    answer = completion.choices[0].message.content
    logger.info("HuggingFace chat API responded in %.2fs (%d chars)", _elapsed, len(answer))
    return answer
