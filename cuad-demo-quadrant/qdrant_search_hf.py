"""
qdrant_search_hf.py
-------------------
Qdrant search backend using HuggingFace Inference API for embeddings and highlighting.
No local embedding service required.
"""

import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests as _requests
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Fusion, SparseVector
from qdrant_cluster_connect import get_qdrant_client, get_cluster_info

from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

from perf_trace import span

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
# Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
EMBEDDING_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "0").lower() not in ("0", "false", "no")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
# Set ENABLE_RERANKER=0 to skip the CrossEncoder highlighting step (useful for ablation evals).
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "1").lower() not in ("0", "false", "no")

# Result-list reranking (distinct from highlighting). When RERANK_RESULTS=1, the
# search functions fetch a larger candidate pool (top_k * RERANK_POOL), score each
# (query, chunk_text) pair with a CrossEncoder, and re-sort the *result list* by that
# score before truncating to top_k. This is the precision lever: it reorders which
# chunks rank first, unlike ENABLE_RERANKER which only highlights sentences within an
# already-ranked chunk. Default model is small (CPU-friendly); override via RERANK_MODEL.
RERANK_RESULTS = os.getenv("RERANK_RESULTS", "0").lower() not in ("0", "false", "no")
RERANK_MODEL_ID = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_POOL = max(1, int(os.getenv("RERANK_POOL", "5")))

# When HF_PROVIDER is set, embed queries via the HF Inference API instead of
# loading the model locally — needed for large models (>2 GB) that won't fit in RAM.
HF_PROVIDER = os.getenv("HF_PROVIDER", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "")   # "voyageai" to use VoyageAI API
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

_embedder = None
_hf_embed_url: str | None = None
_hf_embed_headers: dict | None = None
_hf_embed_url_v1: str | None = None
_voyage_embed_url: str | None = None
_voyage_embed_headers: dict | None = None

if EMBED_PROVIDER == "voyageai":
    _voyage_embed_url = "https://api.voyageai.com/v1/embeddings"
    _voyage_embed_headers = {"Authorization": f"Bearer {VOYAGE_API_KEY}", "Content-Type": "application/json"}
    logger.info("Query embedding via VoyageAI API: %s", EMBEDDING_MODEL_NAME)
elif HF_PROVIDER:
    _hf_embed_url = f"https://router.huggingface.co/{HF_PROVIDER}/models/{EMBEDDING_MODEL_NAME}/pipeline/feature-extraction"
    _hf_embed_url_v1 = f"https://router.huggingface.co/{HF_PROVIDER}/v1/embeddings"
    _hf_embed_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    logger.info("Query embedding via HF API (%s): %s", HF_PROVIDER, EMBEDDING_MODEL_NAME)
else:
    logger.info("Loading local embedding model: %s", EMBEDDING_MODEL_NAME)
    _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

if ENABLE_RERANKER:
    logger.info("Loading local CrossEncoder reranker: %s", RERANKER_MODEL_ID)
    _reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=256)
else:
    logger.info("CrossEncoder reranker disabled (ENABLE_RERANKER=0) — highlights will be empty")
    _reranker = None

_result_reranker = None
if RERANK_RESULTS:
    logger.info("Loading result-reranker CrossEncoder: %s (candidate pool = top_k x %d)", RERANK_MODEL_ID, RERANK_POOL)
    _result_reranker = CrossEncoder(RERANK_MODEL_ID, max_length=512)

_sparse_encoder = None
if ENABLE_HYBRID:
    try:
        from fastembed import SparseTextEmbedding
        logger.info("Loading sparse embedding model: %s", SPARSE_MODEL_NAME)
        _sparse_encoder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        logger.info("Sparse model loaded.")
    except ImportError:
        logger.warning("fastembed not installed; ENABLE_HYBRID disabled. pip install fastembed")
        ENABLE_HYBRID = False

def init_qdrant():
	return get_qdrant_client()

def get_client():
	return get_qdrant_client()

def _voyage_embed_query(query: str) -> np.ndarray:
	"""Call VoyageAI embeddings with retry on 429 (3 req/min free-tier limit)."""
	for attempt in range(5):
		try:
			resp = _requests.post(
				_voyage_embed_url, headers=_voyage_embed_headers,
				json={"input": [query], "model": EMBEDDING_MODEL_NAME, "input_type": "query"},
				timeout=30,
			)
			resp.raise_for_status()
			data = resp.json()
			arr = np.array(data["data"][0]["embedding"], dtype=np.float32)
			norm = np.linalg.norm(arr)
			if norm > 0:
				arr /= norm
			return arr
		except _requests.exceptions.HTTPError as exc:
			status = exc.response.status_code if exc.response is not None else 0
			if status == 429:
				if attempt == 4:
					raise
				wait = 20 * (attempt + 1)
				logger.warning("VoyageAI rate limited (429), attempt %d/5 — retrying in %ds", attempt + 1, wait)
				time.sleep(wait)
			else:
				raise
	raise RuntimeError("unreachable")


@lru_cache(maxsize=1024)
def _embed_query_cached(query: str) -> tuple[float, ...]:
	_t0 = time.perf_counter()
	if _voyage_embed_url:
		vec = _voyage_embed_query(query)
	elif _hf_embed_url:
		resp = _requests.post(
			_hf_embed_url, headers=_hf_embed_headers,
			json={"inputs": [query]}, timeout=30,
		)
		if resp.status_code == 400 and _hf_embed_url_v1:
			resp2 = _requests.post(
				_hf_embed_url_v1, headers=_hf_embed_headers,
				json={"model": EMBEDDING_MODEL_NAME, "input": [query]}, timeout=30,
			)
			if resp2.ok:
				data = resp2.json()
				arr = np.array(data["data"][0]["embedding"], dtype=np.float32)
				norm = np.linalg.norm(arr)
				if norm > 0:
					arr /= norm
				return tuple(float(x) for x in arr)
		resp.raise_for_status()
		raw = resp.json()
		arr = np.array(raw[0], dtype=np.float32)
		if arr.ndim == 2:   # token-level → mean pool
			arr = arr.mean(axis=0)
		norm = np.linalg.norm(arr)
		if norm > 0:
			arr /= norm
		vec = arr
	else:
		vec = _embedder.encode(query, normalize_embeddings=True)
	logger.info("Query embedding: %d-d in %.3fs", len(vec), time.perf_counter() - _t0)
	return tuple(float(x) for x in vec)


def embed_query(query: str) -> list[float]:
	try:
		return list(_embed_query_cached(query))
	except Exception as e:
		logger.error("Failed to embed query: %s", e)
		raise

@lru_cache(maxsize=256)
def highlight_text(query: str, document: str):
	"""
	Identify the most relevant sentences in `document` for `query` using a local
	CrossEncoder reranker. Scores are sigmoid-normalized so the 0.5 threshold
	remains semantically "more relevant than not."
	"""
	try:
		# Split document into sentences (handle common delimiters)
		with span("highlight_assemble"):
			sentences = re.split(r'(?<=[.!?])\s+', document.strip())
			sentences = [s.strip() for s in sentences if s.strip()]

		if not sentences:
			return {"highlighted_sentences": [], "highlight_sentence_indexes": [], "highlight_offsets": []}

		# Score all sentences in a single batched local forward pass
		_reranker_t0 = time.perf_counter()
		if _reranker is not None:
			try:
				with span("rerank"):
					raw_scores = _reranker.predict([(query, s) for s in sentences], batch_size=32)
				sentence_scores = (1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=np.float32)))).tolist()
			except Exception as e:
				logger.warning("Local CrossEncoder failed: %s", e)
				sentence_scores = [0.0] * len(sentences)
			logger.info(
				"Local CrossEncoder scored %d sentences in %.3fs",
				len(sentences),
				time.perf_counter() - _reranker_t0,
			)
		else:
			sentence_scores = [0.0] * len(sentences)
		
		with span("highlight_assemble"):
			# Find top-scoring sentences (above threshold)
			threshold = 0.5
			highlighted_pairs = [(i, sentence, score) for i, (sentence, score) in enumerate(zip(sentences, sentence_scores)) if score >= threshold]

			# Sort by score descending
			highlighted_pairs.sort(key=lambda x: x[2], reverse=True)

			# Limit to top 5 most relevant sentences
			top_k = 5
			highlighted_pairs = highlighted_pairs[:top_k]

			# Calculate character offsets in original document
			highlight_offsets = []
			current_pos = 0
			sentence_positions = {}

			for i, sentence in enumerate(sentences):
				pos = document.find(sentence, current_pos)
				if pos != -1:
					sentence_positions[i] = (pos, pos + len(sentence))
					current_pos = pos + len(sentence)

			highlighted_sentences = []
			highlight_sentence_indexes = []

			for sent_idx, sentence, score in highlighted_pairs:
				highlighted_sentences.append(sentence)
				highlight_sentence_indexes.append(sent_idx)
				if sent_idx in sentence_positions:
					start, end = sentence_positions[sent_idx]
					highlight_offsets.append((start, end))

			avg_score = np.mean([s[2] for s in highlighted_pairs]) if highlighted_pairs else 0.0

		return {
			"highlighted_sentences": highlighted_sentences,
			"highlight_sentence_indexes": highlight_sentence_indexes,
			"highlight_offsets": highlight_offsets
		}
		
	except Exception as e:
		logger.warning("Failed to highlight using reranker: %s", e)
		return {"highlighted_sentences": [], "highlight_sentence_indexes": [], "highlight_offsets": []}

def _build_candidate(payload: dict, score: float, source: list[str]) -> dict:
	"""Assemble a result dict from a Qdrant point payload (highlights added later)."""
	return {
		"id": payload.get("doc_id"),
		"score": score,
		"title": payload.get("title", "Unknown"),
		"text": payload.get("text", ""),
		"page_start": payload.get("page_start"),
		"page_end": payload.get("page_end"),
		"char_start": payload.get("char_start"),
		"char_end": payload.get("char_end"),
		"page_offset_start": payload.get("page_offset_start"),
		"page_offset_end": payload.get("page_offset_end"),
		"pdf_path": payload.get("pdf_path"),
		"source": source,
		"highlighted_sentences": [],
		"highlight_sentence_indexes": [],
		"_raw_page_offset_start": payload.get("page_offset_start", 0),
	}


def _apply_highlight(query: str, cand: dict) -> None:
	"""Compute sentence highlights for a finalized candidate and fold the highlight
	offsets into page_offset_start/page_offset_end (mirrors the original inline logic)."""
	chunk_page_offset_start = cand.get("_raw_page_offset_start", 0)
	hl = highlight_text(query, cand["text"])
	offsets = hl.get("highlight_offsets", [])
	starts, ends = [], []
	for chunk_start, chunk_end in offsets:
		starts.append(chunk_page_offset_start + chunk_start if chunk_page_offset_start else chunk_start)
		ends.append(chunk_page_offset_start + chunk_end if chunk_page_offset_start else chunk_end)
	cand["highlighted_sentences"] = hl.get("highlighted_sentences", [])
	cand["highlight_sentence_indexes"] = hl.get("highlight_sentence_indexes", [])
	if starts:
		cand["page_offset_start"] = starts
		cand["page_offset_end"] = ends


@lru_cache(maxsize=512)
def _rerank_scores_cached(query: str, texts: tuple) -> tuple:
	"""Score (query, chunk_text) pairs with the result-reranker CrossEncoder.
	Cached so the eval's repeated latency probes for one query reuse one forward pass."""
	raw = _result_reranker.predict([(query, t) for t in texts], batch_size=32)
	return tuple(float(x) for x in np.asarray(raw, dtype=np.float32).ravel())


def _rerank_candidates(query: str, candidates: list[dict], top_k: int) -> list[dict]:
	"""Re-sort the candidate pool by CrossEncoder (query, chunk) score, return top_k.
	Keeps the original vector/RRF score as `vector_score`; `score` becomes the
	sigmoid-normalized rerank score so it stays in a comparable 0–1 range for the UI."""
	if not _result_reranker or not candidates:
		return candidates[:top_k]
	texts = tuple(c["text"] for c in candidates)
	try:
		_t0 = time.perf_counter()
		with span("result_rerank"):
			scores = _rerank_scores_cached(query, texts)
		logger.info("Result reranker scored %d candidates in %.3fs", len(texts), time.perf_counter() - _t0)
	except Exception as e:
		logger.warning("Result reranker failed: %s — keeping retrieval order", e)
		return candidates[:top_k]
	for cand, raw in zip(candidates, scores):
		cand["vector_score"] = cand.get("score")
		cand["rerank_score"] = raw
		cand["score"] = float(1.0 / (1.0 + np.exp(-raw)))
	candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
	return candidates[:top_k]


def semantic_search(
	query: str,
	top_k: int = 10,
	document_name: Optional[str] = None,
	min_score: float = 0.0,
	highlight: bool = True,
) -> Tuple[list[dict], dict]:
	try:
		client_qdrant = get_client()
		with span("embed"):
			query_embedding = embed_query(query)
		search_filter = None
		if document_name:
			search_filter = Filter(
				must=[FieldCondition(key="title", match=MatchValue(value=document_name))]
			)
		rerank_on = RERANK_RESULTS and _result_reranker is not None
		fetch_limit = top_k * RERANK_POOL if rerank_on else top_k
		with span("qdrant_query"):
			search_results = client_qdrant.query_points(
				collection_name=COLLECTION_NAME,
				query=query_embedding,
				query_filter=search_filter,
				limit=fetch_limit,
				with_payload=True,
			)
		candidates = [
			_build_candidate(point.payload, point.score, ["embeddings"])
			for point in search_results.points
			if point.score >= min_score
		]
		if rerank_on:
			candidates = _rerank_candidates(query, candidates, top_k)
		else:
			candidates = candidates[:top_k]
		for cand in candidates:
			if highlight:
				_apply_highlight(query, cand)
			cand.pop("_raw_page_offset_start", None)
		results = candidates
		metadata = {
			"query": query,
			"top_k": top_k,
			"strategy": "semantic_search",
			"document_filter": document_name,
			"min_score": min_score,
			"reranked": rerank_on,
			"results_count": len(results),
		}
		return results, metadata
	except Exception as e:
		logger.error("Search error: %s: %s", type(e).__name__, e)
		raise

def hybrid_search(
	query: str,
	top_k: int = 10,
	document_name: Optional[str] = None,
	min_score: float = 0.0,
	highlight: bool = True,
) -> Tuple[list[dict], dict]:
	if not ENABLE_HYBRID or _sparse_encoder is None:
		logger.warning("hybrid_search called but ENABLE_HYBRID=0 or sparse model not loaded; falling back to semantic_search")
		return semantic_search(query=query, top_k=top_k, document_name=document_name, min_score=min_score, highlight=highlight)
	try:
		client_qdrant = get_client()
		with span("embed"):
			query_embedding = embed_query(query)
		with span("sparse_embed"):
			sp = list(_sparse_encoder.embed([query]))[0]
			sparse_vec = SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
		search_filter = None
		if document_name:
			search_filter = Filter(must=[FieldCondition(key="title", match=MatchValue(value=document_name))])
		rerank_on = RERANK_RESULTS and _result_reranker is not None
		fetch_limit = top_k * RERANK_POOL if rerank_on else top_k
		with span("qdrant_query"):
			search_results = client_qdrant.query_points(
				collection_name=COLLECTION_NAME,
				prefetch=[
					Prefetch(query=query_embedding, using="", limit=fetch_limit * 2, filter=search_filter),
					Prefetch(query=sparse_vec, using="sparse", limit=fetch_limit * 2, filter=search_filter),
				],
				query=FusionQuery(fusion=Fusion.RRF),
				limit=fetch_limit,
				with_payload=True,
			)
		candidates = [
			_build_candidate(point.payload, point.score, ["embeddings", "sparse"])
			for point in search_results.points
		]
		if rerank_on:
			candidates = _rerank_candidates(query, candidates, top_k)
		else:
			candidates = candidates[:top_k]
		for cand in candidates:
			if highlight:
				_apply_highlight(query, cand)
			cand.pop("_raw_page_offset_start", None)
		results = candidates
		metadata = {
			"query": query,
			"top_k": top_k,
			"strategy": "hybrid_search",
			"document_filter": document_name,
			"min_score": min_score,
			"reranked": rerank_on,
			"results_count": len(results),
		}
		return results, metadata
	except Exception as e:
		logger.error("Hybrid search error: %s: %s", type(e).__name__, e)
		raise

def search(
	query: str,
	top_k: int = 10,
	document_name: Optional[str] = None,
	strategy: str = "semantic_search",
	min_score: float = 0.0,
	highlight: bool = True,
) -> Tuple[list[dict], dict]:
	top_k = max(1, min(top_k, 100))
	if strategy == "hybrid_search":
		return hybrid_search(
			query=query, top_k=top_k, document_name=document_name,
			min_score=min_score, highlight=highlight,
		)
	else:
		return semantic_search(
			query=query, top_k=top_k, document_name=document_name,
			min_score=min_score, highlight=highlight,
		)

_STATS_CACHE: dict = {"value": None, "ts": 0.0}
_STATS_TTL_SECONDS = float(os.getenv("STATS_CACHE_TTL", "30"))


def get_collection_stats() -> dict:
	now = time.monotonic()
	cached = _STATS_CACHE["value"]
	if cached is not None and (now - _STATS_CACHE["ts"]) < _STATS_TTL_SECONDS:
		return cached
	try:
		client = get_client()
		collection_info = client.get_collection(COLLECTION_NAME)
		points_count = getattr(collection_info, "points_count", 0)
		vector_size = None
		distance = None
		try:
			if hasattr(collection_info, "config"):
				config = collection_info.config
				if hasattr(config, "params"):
					params = config.params
					if hasattr(params, "vectors"):
						vectors_config = params.vectors
						if isinstance(vectors_config, dict):
							vector_size = vectors_config.get("size")
							distance = vectors_config.get("distance")
						else:
							vector_size = getattr(vectors_config, "size", None)
							distance = getattr(vectors_config, "distance", None)
		except Exception as e:
			logger.debug("Could not extract vector config: %s", e)
		result = {
			"collection": COLLECTION_NAME,
			"points_count": points_count,
			"vector_size": vector_size,
			"distance": str(distance) if distance else "unknown",
			"status": "ready" if points_count is not None else "ready",
		}
		_STATS_CACHE["value"] = result
		_STATS_CACHE["ts"] = now
		return result
	except Exception as e:
		logger.error("Failed to get collection stats: %s: %s", type(e).__name__, e)
		# Don't cache errors — let the next call retry.
		return {
			"collection": COLLECTION_NAME,
			"status": "error",
			"error": str(e),
			"points_count": None,
			"vector_size": None,
		}
