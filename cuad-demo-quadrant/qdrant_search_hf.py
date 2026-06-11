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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests as _requests
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, RrfQuery, Rrf, SparseVector
from qdrant_cluster_connect import get_qdrant_client, get_cluster_info

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from perf_trace import span

logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent.parent / ".env.dev"
load_dotenv(env_path)
# Configuration
# ----------------------------------------------------------------------------
# Production default: bge-large query embeddings + bge-reranker-v2-m3 result
# reranking, BOTH served via the HuggingFace serverless Inference API
# (huggingface_hub.InferenceClient). No model weights load locally and no GPU
# code runs in this process; HF hosts the compute (ZeroGPU/serverless).
#
# EMBED_BACKEND / RERANK_BACKEND let the offline eval harness fall back to local
# sentence-transformers models ("local"); the deployed default is "hf".
# ----------------------------------------------------------------------------
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cuad_bgelarge_hybrid")
EMBEDDING_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
# Canonical search-strategy vocabulary, shared by the request-time `strategy` arg
# AND the startup LOAD_MODEL_STRATEGY env var so the two read identically.
STRATEGY_SEMANTIC = "semantic_search"   # dense only
STRATEGY_SPARSE = "sparse_search"       # BM42 sparse only
STRATEGY_HYBRID = "hybrid_search"       # weighted-RRF fusion of both
SEARCH_STRATEGIES = (STRATEGY_SEMANTIC, STRATEGY_SPARSE, STRATEGY_HYBRID)

# LOAD_MODEL_STRATEGY selects which query-time embedding models load at startup,
# using the same vocabulary as the request strategy:
#   semantic_search → dense embedder only
#   sparse_search   → BM42 sparse encoder only
#   hybrid_search   → both (required by hybrid_search requests)
# Model loading happens once at import, so this is a deploy-time capability switch.
# The per-request `strategy` then routes to a search needing those models and fails
# loudly if they were not loaded, rather than silently degrading. Back-compat:
# legacy ENABLE_HYBRID=1 maps to hybrid_search when LOAD_MODEL_STRATEGY is unset.
_legacy_hybrid = os.getenv("ENABLE_HYBRID", "").lower() in ("1", "true", "yes")
LOAD_MODEL_STRATEGY = os.getenv(
    "LOAD_MODEL_STRATEGY", STRATEGY_HYBRID if _legacy_hybrid else STRATEGY_SEMANTIC
).lower()
if LOAD_MODEL_STRATEGY not in SEARCH_STRATEGIES:
    raise ValueError(
        f"LOAD_MODEL_STRATEGY must be one of {SEARCH_STRATEGIES}; got {LOAD_MODEL_STRATEGY!r}"
    )
LOAD_DENSE = LOAD_MODEL_STRATEGY in (STRATEGY_SEMANTIC, STRATEGY_HYBRID)
LOAD_SPARSE = LOAD_MODEL_STRATEGY in (STRATEGY_SPARSE, STRATEGY_HYBRID)
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")

# Weighted RRF fusion for hybrid search. Qdrant fuses the dense + sparse prefetch
# branches with reciprocal-rank fusion; these weights bias the blend. Tuned on
# cuad_bgelarge_hybrid_50 (see tests/eval/RRF_WEIGHT_TUNING.md): dense:sparse =
# 0.7:0.3 beats the balanced 0.5:0.5 default on both recall and ranking. NOTE: that
# sweep ran with the result reranker OFF, so the win is established for the fused
# retrieval order (the rerank pool feed), not yet re-confirmed reranker-on end to
# end. Only the ratio matters. `... or "0.7"` keeps a blank env var from crashing.
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT") or "0.7")
RRF_SPARSE_WEIGHT = float(os.getenv("RRF_SPARSE_WEIGHT") or "0.3")

# Backend selection: "hf" (serverless Inference API, production default) | "local"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hf").lower()
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "hf").lower()

# Result-list reranking (distinct from per-sentence highlighting). Fetch a larger
# candidate pool from Qdrant (min(top_k * RERANK_POOL, RERANK_FETCH_LIMIT)), score
# each (query, chunk) pair with the cross-encoder, re-sort, and truncate to top_k.
# Operating point: top_k=20, pool=5, fetch_limit=60. Default ON in production.
RERANK_RESULTS = os.getenv("RERANK_RESULTS", "1").lower() not in ("0", "false", "no")
RERANK_MODEL_ID = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_POOL = max(1, int(os.getenv("RERANK_POOL", "5")))
# Pool cap. With batched reranking (one HF call for the whole pool), this controls
# batch size / recall rather than round-trip count. 100 is the natural uncapped value
# (top_k=20 × RERANK_POOL=5); raise further via env if you want deeper recall.
RERANK_FETCH_LIMIT = max(1, int(os.getenv("RERANK_FETCH_LIMIT", "100")))
# Bounded concurrency for the HF per-pair rerank calls (huggingface_hub has no
# native batch route; we parallelize instead of running calls strictly serially).
RERANK_HF_WORKERS = max(1, int(os.getenv("RERANK_HF_WORKERS", "16")))

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Legacy HF_PROVIDER raw-router embedding path (kept for backward compat / eval).
HF_PROVIDER = os.getenv("HF_PROVIDER", "")
_hf_embed_url: str | None = None
_hf_embed_url_v1: str | None = None
_hf_embed_headers: dict | None = None

# Shared serverless Inference client (embeddings + reranking).
_inference_client: InferenceClient | None = None
_embedder = None   # local SentenceTransformer, only when EMBED_BACKEND="local"
_dense_available = False   # True once a dense query-embedding path is initialized

if LOAD_DENSE:
    if EMBED_BACKEND == "hf":
        _inference_client = InferenceClient(api_key=HF_TOKEN or None, provider="hf-inference")
        logger.info("Query embedding via HF Inference API: %s", EMBEDDING_MODEL_NAME)
    elif HF_PROVIDER:
        _hf_embed_url = f"https://router.huggingface.co/{HF_PROVIDER}/models/{EMBEDDING_MODEL_NAME}/pipeline/feature-extraction"
        _hf_embed_url_v1 = f"https://router.huggingface.co/{HF_PROVIDER}/v1/embeddings"
        _hf_embed_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        logger.info("Query embedding via HF router (%s): %s", HF_PROVIDER, EMBEDDING_MODEL_NAME)
    else:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    _dense_available = True
else:
    logger.info("LOAD_MODEL_STRATEGY=%s: dense query embedder not loaded", LOAD_MODEL_STRATEGY)

# Highlight + result-list reranker. In "hf" mode both use the serverless Inference
# API (text-classification cross-encoder). In "local" mode a CrossEncoder loads.
_reranker = None   # local CrossEncoder for highlights, only when RERANK_BACKEND="local"
if RERANK_BACKEND == "hf":
    if _inference_client is None:
        _inference_client = InferenceClient(api_key=HF_TOKEN or None, provider="hf-inference")
    logger.info(
        "Reranking via HF Inference API: %s (highlight) / %s (result-list)",
        RERANKER_MODEL_ID, RERANK_MODEL_ID,
    )
else:
    from sentence_transformers import CrossEncoder
    logger.info("Loading local CrossEncoder reranker: %s", RERANKER_MODEL_ID)
    _reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=256)

_sparse_encoder = None
if LOAD_SPARSE:
    try:
        from fastembed import SparseTextEmbedding
        logger.info("Loading sparse embedding model: %s", SPARSE_MODEL_NAME)
        _sparse_encoder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        logger.info("Sparse model loaded.")
    except ImportError:
        logger.warning("fastembed not installed; sparse model unavailable. pip install fastembed")
_sparse_available = _sparse_encoder is not None

def init_qdrant():
	return get_qdrant_client()

def get_client():
	return get_qdrant_client()


def _ce_pair_input(query: str, passage: str) -> str:
	"""Encode a (query, passage) pair as a single text for the HF text-classification
	cross-encoder. bge-reranker-v2-m3 expects the pair joined with [SEP]."""
	return f"{query} [SEP] {passage}"


def _hf_ce_score(query: str, passage: str) -> float:
	"""Score one (query, passage) pair via the HF Inference API cross-encoder.
	Returns a sigmoid-normalized 0–1 relevance score. One HTTPS round-trip."""
	out = _inference_client.text_classification(
		_ce_pair_input(query, passage), model=RERANK_MODEL_ID,
	)
	# bge-reranker emits a single logit/score label; take the top element's score.
	raw = float(out[0].score) if out else 0.0
	return raw


def _hf_ce_scores_parallel(query: str, passages: tuple) -> list[float]:
	"""Score many (query, passage) pairs via the HF cross-encoder using a bounded
	thread pool. huggingface_hub exposes no batch rerank route, so we fan out
	per-pair calls with at most RERANK_HF_WORKERS concurrent HTTPS requests rather
	than running them strictly sequentially."""
	if not passages:
		return []
	workers = min(RERANK_HF_WORKERS, len(passages))
	with ThreadPoolExecutor(max_workers=workers) as pool:
		return list(pool.map(lambda p: _hf_ce_score(query, p), passages))


def _hf_ce_scores_batch(query: str, passages: tuple) -> list[float]:
	"""Score all (query, passage) pairs in a single batched HF Inference API call.
	Reduces N round-trips to 1 regardless of passage count — preferred over
	_hf_ce_scores_parallel for SageMaker Serverless where per-invocation overhead
	dominates latency.

	TextClassificationOutputElement is a dict subclass; access score via .score /
	["score"], NOT [0] (which raises KeyError because 0 is not a dict key)."""
	if not passages:
		return []
	inputs = [_ce_pair_input(query, p) for p in passages]
	results = _inference_client.text_classification(inputs, model=RERANK_MODEL_ID)
	# Flat batch response: one TextClassificationOutputElement per input.
	if len(results) != len(inputs):
		raise ValueError(
			f"Batch reranker returned {len(results)} scores for {len(inputs)} inputs; "
			"falling back to retrieval order"
		)
	return [float(r.score) if r else 0.0 for r in results]


@lru_cache(maxsize=1024)
def _embed_query_cached(query: str) -> tuple[float, ...]:
	_t0 = time.perf_counter()
	if _inference_client is not None and EMBED_BACKEND == "hf":
		arr = np.asarray(
			_inference_client.feature_extraction(query, model=EMBEDDING_MODEL_NAME),
			dtype=np.float32,
		)
		if arr.ndim == 2:   # token-level → mean pool
			arr = arr.mean(axis=0)
		norm = np.linalg.norm(arr)
		if norm > 0:
			arr /= norm
		vec = arr
		logger.info("Query embedding (HF API): %d-d in %.3fs", len(vec), time.perf_counter() - _t0)
		return tuple(float(x) for x in vec)
	if _hf_embed_url:
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
	if not _dense_available:
		raise RuntimeError(
			f"Dense embedding model not loaded (LOAD_MODEL_STRATEGY={LOAD_MODEL_STRATEGY}); "
			"set LOAD_MODEL_STRATEGY=semantic_search or hybrid_search to use semantic/hybrid search."
		)
	try:
		return list(_embed_query_cached(query))
	except Exception as e:
		logger.error("Failed to embed query: %s", e)
		raise


def warmup_inference() -> None:
	"""Fire one cheap embedding + one cheap reranker call to warm the HF serverless
	endpoints, cutting first-query cold-start. Only meaningful on the HF backend;
	no-op otherwise. Non-fatal: logs and returns on any failure so it can never
	block startup. Intended to run in a background thread, gated by WARMUP_ON_START."""
	if _inference_client is None:
		logger.info("Warm-up skipped: not on HF Inference backend")
		return
	_t0 = time.perf_counter()
	if EMBED_BACKEND == "hf":
		try:
			_inference_client.feature_extraction("warm up", model=EMBEDDING_MODEL_NAME)
			logger.info("Warm-up: embed endpoint (%s) warmed", EMBEDDING_MODEL_NAME)
		except Exception as e:
			logger.warning("Warm-up embed call failed (non-fatal): %s", e)
	if RERANK_BACKEND == "hf":
		try:
			_hf_ce_score("warm up", "warm up passage")
			logger.info("Warm-up: reranker endpoint (%s) warmed", RERANK_MODEL_ID)
		except Exception as e:
			logger.warning("Warm-up reranker call failed (non-fatal): %s", e)
	logger.info("Warm-up finished in %.2fs", time.perf_counter() - _t0)


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

		# Score all sentences against the query with the cross-encoder.
		# HF backend: bounded-parallel per-pair Inference API calls.
		# local backend: single batched forward pass.
		_reranker_t0 = time.perf_counter()
		try:
			with span("rerank"):
				if RERANK_BACKEND == "hf":
					raw_scores = _hf_ce_scores_batch(query, tuple(sentences))
				else:
					raw_scores = _reranker.predict([(query, s) for s in sentences], batch_size=32)
			sentence_scores = (1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=np.float32)))).tolist()
		except Exception as e:
			logger.warning("Highlight reranker (%s) failed: %s", RERANK_BACKEND, e)
			sentence_scores = [0.0] * len(sentences)
		logger.info(
			"Highlight reranker (%s) scored %d sentences in %.3fs",
			RERANK_BACKEND, len(sentences),
			time.perf_counter() - _reranker_t0,
		)
		
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


# ----------------------------------------------------------------------------
# Lexical keyword highlighting for sparse / hybrid search.
#
# BM42 retrieval matches a query token to a chunk token when their Snowball
# *stems* collide (see fastembed/sparse/bm42.py: tokenize → reconstruct BPE →
# drop stopwords/punctuation → stem → hash). We reproduce that same pipeline on
# the already-loaded sparse encoder to surface which query keywords matched each
# chunk, with char offsets — faithful to retrieval, no model call, no HF round
# trip. Empty for semantic_search (no lexical signal) and when sparse isn't loaded.
# ----------------------------------------------------------------------------
def _bm42_model():
	"""The underlying fastembed `Bm42` instance (exposes tokenizer/stemmer/
	stopwords/punctuation), or None when the sparse encoder isn't loaded."""
	return getattr(_sparse_encoder, "model", None) if _sparse_available else None


def _bm42_words_with_offsets(model, text: str) -> list[tuple[str, str, int, int]]:
	"""Run `text` through BM42's word pipeline, keeping char offsets.

	Returns (surface, stem, start, end) per content word. Mirrors the Bm42
	reconstruct-BPE → filter → stem steps, but merges subword char offsets
	(which `_reconstruct_bpe` discards) so callers can locate each word in `text`.
	"""
	encoded = model.tokenizer.encode(text)
	prefix = model.tokenizer.model.continuing_subword_prefix
	plen = len(prefix)
	# Merge WordPiece subwords back into whole words, accumulating offsets.
	words: list[tuple[str, int, int]] = []
	acc = ""
	acc_start = acc_end = 0
	for token, (start, end) in zip(encoded.tokens, encoded.offsets):
		if token in model.special_tokens:
			continue
		if token.startswith(prefix):
			acc += token[plen:]
			acc_end = end
		else:
			if acc:
				words.append((acc, acc_start, acc_end))
			acc, acc_start, acc_end = token, start, end
	if acc:
		words.append((acc, acc_start, acc_end))
	# Drop stopwords/punctuation, then Snowball-stem each surviving word.
	result: list[tuple[str, str, int, int]] = []
	for surface, start, end in words:
		if surface in model.stopwords or surface in model.punctuation:
			continue
		result.append((surface, model.stemmer.stem_word(surface), start, end))
	return result


@lru_cache(maxsize=1024)
def _query_stem_set(query: str) -> frozenset:
	"""Distinct Snowball stems of the query's content tokens — the set BM42 would
	hash and match against. Cached; empty frozenset if sparse isn't loaded."""
	model = _bm42_model()
	if model is None:
		return frozenset()
	return frozenset(stem for _, stem, _, _ in _bm42_words_with_offsets(model, query))


def extract_matched_keywords(query: str, document: str) -> dict:
	"""Identify which query keywords appear in `document` under BM42 stemming.

	Returns `matched_keywords` (distinct surface terms, first-seen order, original
	casing) and `keyword_offsets` ([start, end] char ranges in `document` for every
	occurrence). Never raises — returns empty lists on any failure."""
	empty = {"matched_keywords": [], "keyword_offsets": []}
	model = _bm42_model()
	if model is None:
		return empty
	try:
		with span("keyword_match"):
			query_stems = _query_stem_set(query)
			if not query_stems:
				return empty
			matched_keywords: list[str] = []
			keyword_offsets: list[list[int]] = []
			seen: set[str] = set()
			for surface, stem, start, end in _bm42_words_with_offsets(model, document):
				if stem not in query_stems:
					continue
				keyword_offsets.append([start, end])
				term = document[start:end]   # original casing
				if term.lower() not in seen:
					seen.add(term.lower())
					matched_keywords.append(term)
			return {"matched_keywords": matched_keywords, "keyword_offsets": keyword_offsets}
	except Exception as e:
		logger.warning("Keyword match failed: %s", e)
		return empty


def _result_rerank_enabled() -> bool:
	return RERANK_RESULTS and (RERANK_BACKEND == "hf" or _reranker is not None)


def _rerank_fetch_limit(top_k: int) -> int:
	"""How many candidates to pull from Qdrant before reranking down to top_k."""
	if not _result_rerank_enabled():
		return top_k
	return min(top_k * RERANK_POOL, RERANK_FETCH_LIMIT)


def _rerank_points(query: str, points: list, top_k: int) -> list:
	"""Re-sort Qdrant result points by cross-encoder (query, chunk_text) score and
	truncate to top_k. HF backend uses bounded-parallel Inference API calls; local
	backend uses a single batched CrossEncoder forward pass. On failure, keeps the
	original retrieval order (truncated to top_k)."""
	if not _result_rerank_enabled() or not points:
		return points[:top_k]
	texts = tuple((p.payload or {}).get("text", "") for p in points)
	try:
		_t0 = time.perf_counter()
		with span("result_rerank"):
			if RERANK_BACKEND == "hf":
				scores = _hf_ce_scores_batch(query, texts)
			else:
				scores = _reranker.predict([(query, t) for t in texts], batch_size=32)
		logger.info(
			"Result reranker (%s) scored %d candidates in %.3fs",
			RERANK_BACKEND, len(texts), time.perf_counter() - _t0,
		)
	except Exception as e:
		logger.warning("Result reranker failed: %s — keeping retrieval order", e)
		return points[:top_k]
	scores = np.asarray(scores, dtype=np.float32).ravel()
	order = np.argsort(-scores)
	return [points[i] for i in order[:top_k]]


def _assemble_results(
	query: str,
	raw_points: list,
	top_k: int,
	min_score: float,
	highlight: bool,
	source: list[str],
	strategy: str,
	document_name: Optional[str],
) -> Tuple[list[dict], dict]:
	"""Shared post-retrieval stage for all three search strategies: optional
	cross-encoder result reranking, min_score filtering, per-chunk highlight +
	page-offset mapping, result-dict construction, and metadata. `source` labels
	which retrieval signals produced the chunk; `strategy` is echoed into metadata."""
	rerank_on = _result_rerank_enabled()
	points = _rerank_points(query, raw_points, top_k) if rerank_on else raw_points[:top_k]
	# Lexical keyword highlighting: only for strategies that use the sparse signal
	# (sparse_search / hybrid_search both carry "sparse" in `source`). Always on for
	# those, independent of the `highlight` flag, since it makes no HF call.
	keyword_match_on = "sparse" in source and _sparse_available
	results = []
	for point in points:
		if point.score < min_score:
			continue
		payload = point.payload or {}
		text = payload.get("text", "")
		title = payload.get("title", "Unknown")
		chunk_page_offset_start = payload.get("page_offset_start", 0)
		if highlight:
			highlights_response = highlight_text(query, text)
			highlighted_sentences = highlights_response.get("highlighted_sentences", [])
			highlight_sentence_indexes = highlights_response.get("highlight_sentence_indexes", [])
			highlight_offsets = highlights_response.get("highlight_offsets", [])
		else:
			highlighted_sentences = []
			highlight_sentence_indexes = []
			highlight_offsets = []
		highlight_page_offset_starts = []
		highlight_page_offset_ends = []
		for chunk_start, chunk_end in highlight_offsets:
			page_start = chunk_page_offset_start + chunk_start if chunk_page_offset_start else chunk_start
			page_end = chunk_page_offset_start + chunk_end if chunk_page_offset_start else chunk_end
			highlight_page_offset_starts.append(page_start)
			highlight_page_offset_ends.append(page_end)
		if keyword_match_on:
			kw = extract_matched_keywords(query, text)
			matched_keywords = kw["matched_keywords"]
			keyword_offsets = kw["keyword_offsets"]
		else:
			matched_keywords = []
			keyword_offsets = []
		results.append(
			{
				"id": payload.get("doc_id"),
				"score": point.score,
				"title": title,
				"text": text,
				"page_start": payload.get("page_start"),
				"page_end": payload.get("page_end"),
				"char_start": payload.get("char_start"),
				"char_end": payload.get("char_end"),
				"page_offset_start": highlight_page_offset_starts if highlight_page_offset_starts else payload.get("page_offset_start"),
				"page_offset_end": highlight_page_offset_ends if highlight_page_offset_ends else payload.get("page_offset_end"),
				"pdf_path": payload.get("pdf_path"),
				"source": source,
				"highlighted_sentences": highlighted_sentences,
				"highlight_sentence_indexes": highlight_sentence_indexes,
				"matched_keywords": matched_keywords,
				"keyword_offsets": keyword_offsets,
			}
		)
	metadata = {
		"query": query,
		"top_k": top_k,
		"strategy": strategy,
		"document_filter": document_name,
		"min_score": min_score,
		"reranked": rerank_on,
		"results_count": len(results),
	}
	return results, metadata


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
		with span("qdrant_query"):
			search_results = client_qdrant.query_points(
				collection_name=COLLECTION_NAME,
				query=query_embedding,
				query_filter=search_filter,
				limit=_rerank_fetch_limit(top_k),
				with_payload=True,
			)
		return _assemble_results(
			query, search_results.points, top_k, min_score, highlight,
			source=["embeddings"], strategy=STRATEGY_SEMANTIC, document_name=document_name,
		)
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
	# Hybrid needs both query embedders loaded. Fail loudly rather than silently
	# degrading to semantic — the caller asked for HYBRID explicitly.
	if not _sparse_available:
		raise RuntimeError(
			f"hybrid_search requires the sparse model, but LOAD_MODEL_STRATEGY={LOAD_MODEL_STRATEGY} "
			"did not load it. Set LOAD_MODEL_STRATEGY=hybrid_search."
		)
	if not _dense_available:
		raise RuntimeError(
			f"hybrid_search requires the dense model, but LOAD_MODEL_STRATEGY={LOAD_MODEL_STRATEGY} "
			"did not load it. Set LOAD_MODEL_STRATEGY=hybrid_search."
		)
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
		fetch_limit = _rerank_fetch_limit(top_k)
		with span("qdrant_query"):
			search_results = client_qdrant.query_points(
				collection_name=COLLECTION_NAME,
				prefetch=[
					Prefetch(query=query_embedding, using="", limit=fetch_limit * 2, filter=search_filter),
					Prefetch(query=sparse_vec, using="sparse", limit=fetch_limit * 2, filter=search_filter),
				],
				query=RrfQuery(rrf=Rrf(weights=[RRF_DENSE_WEIGHT, RRF_SPARSE_WEIGHT])),
				limit=fetch_limit,
				with_payload=True,
			)
		return _assemble_results(
			query, search_results.points, top_k, min_score, highlight,
			source=["embeddings", "sparse"], strategy=STRATEGY_HYBRID, document_name=document_name,
		)
	except Exception as e:
		logger.error("Hybrid search error: %s: %s", type(e).__name__, e)
		raise


def sparse_search(
	query: str,
	top_k: int = 10,
	document_name: Optional[str] = None,
	min_score: float = 0.0,
	highlight: bool = True,
) -> Tuple[list[dict], dict]:
	"""Lexical (BM42 sparse) retrieval only — no dense vector. Mirrors the SPARSE
	value of LOAD_MODEL_STRATEGY. Fails loudly if the sparse model isn't loaded."""
	if not _sparse_available:
		raise RuntimeError(
			f"sparse_search requires the sparse model, but LOAD_MODEL_STRATEGY={LOAD_MODEL_STRATEGY} "
			"did not load it. Set LOAD_MODEL_STRATEGY=sparse_search or hybrid_search."
		)
	try:
		client_qdrant = get_client()
		with span("sparse_embed"):
			sp = list(_sparse_encoder.embed([query]))[0]
			sparse_vec = SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
		search_filter = None
		if document_name:
			search_filter = Filter(must=[FieldCondition(key="title", match=MatchValue(value=document_name))])
		with span("qdrant_query"):
			search_results = client_qdrant.query_points(
				collection_name=COLLECTION_NAME,
				query=sparse_vec,
				using="sparse",
				query_filter=search_filter,
				limit=_rerank_fetch_limit(top_k),
				with_payload=True,
			)
		return _assemble_results(
			query, search_results.points, top_k, min_score, highlight,
			source=["sparse"], strategy=STRATEGY_SPARSE, document_name=document_name,
		)
	except Exception as e:
		logger.error("Sparse search error: %s: %s", type(e).__name__, e)
		raise

def normalize_strategy(strategy: str) -> str:
	"""Validate/canonicalize a request strategy (case-insensitive) against
	SEARCH_STRATEGIES. Raises ValueError on anything else."""
	s = (strategy or "").strip().lower()
	if s not in SEARCH_STRATEGIES:
		raise ValueError(f"strategy must be one of {SEARCH_STRATEGIES}; got {strategy!r}")
	return s


def search(
	query: str,
	top_k: int = 10,
	document_name: Optional[str] = None,
	strategy: str = STRATEGY_SEMANTIC,
	min_score: float = 0.0,
	highlight: bool = True,
) -> Tuple[list[dict], dict]:
	top_k = max(1, min(top_k, 100))
	strat = normalize_strategy(strategy)
	kwargs = dict(
		query=query, top_k=top_k, document_name=document_name,
		min_score=min_score, highlight=highlight,
	)
	if strat == STRATEGY_HYBRID:
		return hybrid_search(**kwargs)
	if strat == STRATEGY_SPARSE:
		return sparse_search(**kwargs)
	return semantic_search(**kwargs)

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
