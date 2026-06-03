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
from typing import Generator, Optional

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from perf_trace import span

logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent.parent / ".env.dev"
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
# Default to a free, fast 8B model on HF Inference serverless. Override with CHAT_MODEL.
CHAT_MODEL = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "512"))
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.1"))

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable must be set for HuggingFace Inference API.")

# Load system prompt from prompts/chat_system.txt (two levels up from this file).
# Falls back to a minimal inline prompt if the file is missing.
_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "chat_system.txt"
try:
    DEFAULT_SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8").strip()
    logger.info("Loaded system prompt from %s", _PROMPT_FILE)
except FileNotFoundError:
    DEFAULT_SYSTEM_PROMPT = (
        "You are a legal contract analysis assistant. "
        "Synthesize an answer from the contract passages below. "
        "Cite documents inline as [Source N]. Do not copy passages verbatim."
    )
    logger.warning("prompts/chat_system.txt not found; using fallback system prompt")

_inference_client: Optional[InferenceClient] = None


def _get_inference_client() -> InferenceClient:
    global _inference_client
    if _inference_client is None:
        _inference_client = InferenceClient(api_key=HF_TOKEN)
    return _inference_client


def _build_context(documents: list[dict]) -> str:
    """
    Format retrieved documents into a context block for the prompt.

    When a doc has `highlighted_sentences`, use ONLY those sentences instead of
    the full chunk — typically 60–80% fewer input tokens with no quality loss,
    since the reranker already picked the relevant spans.
    """
    parts = []
    for i, doc in enumerate(documents, start=1):
        title = doc.get("title", "Unknown")
        score = doc.get("score", 0.0)
        highlights = doc.get("highlighted_sentences") or []
        if highlights:
            body = " ".join(s.strip() for s in highlights if s.strip())
        else:
            body = (doc.get("text", "") or "").strip()
        parts.append(f"[Document {i}] Title: {title} (score: {score:.3f})\n{body}")
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
        system_prompt = DEFAULT_SYSTEM_PROMPT

    user_message = (
        f"Contract passages:\n\n{context}\n\n"
        f"Question: {query}"
    )

    logger.info("Calling HuggingFace chat model: %s", CHAT_MODEL)

    client = _get_inference_client()
    _t0 = time.perf_counter()
    with span("chat_completion"):
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=CHAT_MAX_TOKENS,
            temperature=CHAT_TEMPERATURE,
        )
    _elapsed = time.perf_counter() - _t0

    answer = completion.choices[0].message.content
    logger.info("HuggingFace chat API responded in %.2fs (%d chars)", _elapsed, len(answer))
    return answer


def chat_stream(
    query: str,
    documents: list[dict],
    system_prompt: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Like chat() but yields token strings as they arrive from the HF model.
    Must be called from a background thread (blocks on network I/O per token).
    """
    if not documents:
        yield "No relevant contract passages were found for your query."
        return

    context = _build_context(documents)

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    user_message = (
        f"Contract passages:\n\n{context}\n\n"
        f"Question: {query}"
    )

    client = _get_inference_client()
    logger.info("Streaming HuggingFace chat model: %s", CHAT_MODEL)

    with span("chat_completion_stream"):
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=CHAT_MAX_TOKENS,
            temperature=CHAT_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
