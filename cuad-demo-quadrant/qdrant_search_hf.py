"""
qdrant_search_hf.py
-------------------
Qdrant search backend using HuggingFace Inference API for embeddings and highlighting.
No local embedding service required.
"""

import os
from typing import Optional, Tuple
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_cluster_connect import get_qdrant_client, get_cluster_info

from huggingface_hub import InferenceClient
import requests

# Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

if not HF_TOKEN:
	raise RuntimeError("HF_TOKEN environment variable must be set for HuggingFace Inference API.")

# Initialize inference client
print(f"[INFO] Initializing HuggingFace Inference client")
client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

def init_qdrant():
	return get_qdrant_client()

def get_client():
	return get_qdrant_client()

def embed_query(query: str) -> list[float]:
	try:
		# Use HuggingFace Inference API for embeddings
		response = client.feature_extraction(
			query,
			model=EMBEDDING_MODEL_NAME,
		)
		print(f"[DEBUG] Raw embedding response from HF Inference API: {response}")
		# Accept both a flat list (single embedding) and a list of lists (batch), handle numpy types
		import numpy as np
		def to_float_list(arr):
			if isinstance(arr, np.ndarray):
				return arr.astype(float).tolist()
			elif isinstance(arr, list):
				return [float(x) for x in arr]
			else:
				raise TypeError(f"Unexpected embedding type: {type(arr)}")

		if isinstance(response, (list, np.ndarray)) and len(response) > 0:
			# Flat list or numpy array: single embedding
			if all(isinstance(x, (float, int, np.floating, np.integer)) for x in response):
				return to_float_list(response)
			elif all(isinstance(x, (list, np.ndarray)) for x in response):
				# List of lists: batch
				return to_float_list(response[0])
		raise RuntimeError(f"Empty or invalid embedding response from HF Inference API: {response}")
	except Exception as e:
		print(f"[ERROR] Failed to embed query via HF Inference API: {e}")
		raise

def highlight_text(query: str, document: str):
	"""
	Use HuggingFace Inference API reranker model to identify the most relevant sentences.
	The BAAI/bge-reranker-v2-m3 model scores query-sentence pairs for semantic relevance.
	"""
	try:
		import re
		import numpy as np
		
		# Split document into sentences (handle common delimiters)
		sentences = re.split(r'(?<=[.!?])\s+', document.strip())
		sentences = [s.strip() for s in sentences if s.strip()]
		
		if not sentences:
			return {"highlighted_sentences": [], "highlight_sentence_indexes": [], "highlight_offsets": []}
		
		print(f"[DEBUG] Split document into {len(sentences)} sentences for reranking")
		
		# Score each sentence using the reranker
		sentence_scores = []
		
		for i, sentence in enumerate(sentences):
			try:
				# Use HF Inference API text_classification with the reranker model
				print(f"[DEBUG] Scoring sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
				
				result = client.text_classification(
					f"{query} [SEP] {sentence}",
					model=RERANKER_MODEL_ID,
				)
				
				print(f"[DEBUG] Reranker result for sentence {i+1}: {result}")
				
				# Extract the relevance score
				# The model returns a list with label and score
				if result and isinstance(result, list) and len(result) > 0:
					score = result[0].get("score", 0.0)
				else:
					score = 0.0
				
				sentence_scores.append(score)
				
			except Exception as e:
				print(f"[WARN] Failed to score sentence {i+1}: {e}")
				sentence_scores.append(0.0)
		
		print(f"[DEBUG] Sentence scores: {sentence_scores}")
		
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
		print(f"[DEBUG] Highlighted {len(highlighted_sentences)} sentences with avg score: {avg_score:.3f}")
		
		return {
			"highlighted_sentences": highlighted_sentences,
			"highlight_sentence_indexes": highlight_sentence_indexes,
			"highlight_offsets": highlight_offsets
		}
		
	except Exception as e:
		print(f"[WARN] Failed to highlight using reranker: {e}")
		import traceback
		traceback.print_exc()
		return {"highlighted_sentences": [], "highlight_sentence_indexes": [], "highlight_offsets": []}

def semantic_search(
	query: str,
	top_k: int = 10,
	document_name: Optional[str] = None,
	min_score: float = 0.0,
) -> Tuple[list[dict], dict]:
	try:
		client_qdrant = get_client()
		query_embedding = embed_query(query)
		search_filter = None
		if document_name:
			search_filter = Filter(
				must=[FieldCondition(key="title", match=MatchValue(value=document_name))]
			)
		print(f"[DEBUG] Searching Qdrant collection '{COLLECTION_NAME}'...")
		search_results = client_qdrant.query_points(
			collection_name=COLLECTION_NAME,
			query=query_embedding,
			query_filter=search_filter,
			limit=top_k,
			with_payload=True,
		)
		print(f"[DEBUG] Found {len(search_results.points)} results")
		results = []
		for point in search_results.points:
			if point.score >= min_score:
				payload = point.payload
				text = payload.get("text", "")
				title = payload.get("title", "Unknown")
				chunk_page_offset_start = payload.get("page_offset_start", 0)
				highlights_response = highlight_text(query, text)
				highlighted_sentences = highlights_response.get("highlighted_sentences", [])
				highlight_sentence_indexes = highlights_response.get("highlight_sentence_indexes", [])
				highlight_offsets = highlights_response.get("highlight_offsets", [])
				highlight_page_offset_starts = []
				highlight_page_offset_ends = []
				for chunk_start, chunk_end in highlight_offsets:
					page_start = chunk_page_offset_start + chunk_start if chunk_page_offset_start else chunk_start
					page_end = chunk_page_offset_start + chunk_end if chunk_page_offset_start else chunk_end
					highlight_page_offset_starts.append(page_start)
					highlight_page_offset_ends.append(page_end)
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
						"source": ["embeddings"],
						"highlighted_sentences": highlighted_sentences,
						"highlight_sentence_indexes": highlight_sentence_indexes,
					}
				)
		metadata = {
			"query": query,
			"top_k": top_k,
			"strategy": "semantic_search",
			"document_filter": document_name,
			"min_score": min_score,
			"results_count": len(results),
		}
		return results, metadata
	except Exception as e:
		print(f"[ERROR] Search error: {type(e).__name__}: {e}")
		raise

def hybrid_search(
	query: str,
	top_k: int = 10,
	document_name: Optional[str] = None,
	min_score: float = 0.0,
) -> Tuple[list[dict], dict]:
	return semantic_search(query=query, top_k=top_k, document_name=document_name, min_score=min_score)

def search(
	query: str,
	top_k: int = 10,
	document_name: Optional[str] = None,
	strategy: str = "semantic_search",
	min_score: float = 0.0,
) -> Tuple[list[dict], dict]:
	top_k = max(1, min(top_k, 100))
	if strategy == "hybrid_search":
		return hybrid_search(query=query, top_k=top_k, document_name=document_name, min_score=min_score)
	else:
		return semantic_search(query=query, top_k=top_k, document_name=document_name, min_score=min_score)

def get_collection_stats() -> dict:
	try:
		client = get_client()
		print(f"[DEBUG] Fetching stats for collection: {COLLECTION_NAME}")
		collection_info = client.get_collection(COLLECTION_NAME)
		print(f"[DEBUG] Collection info retrieved: {collection_info}")
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
			print(f"[DEBUG] Could not extract vector config: {e}")
		return {
			"collection": COLLECTION_NAME,
			"points_count": points_count,
			"vector_size": vector_size,
			"distance": str(distance) if distance else "unknown",
			"status": "ready" if points_count is not None else "ready",
		}
	except Exception as e:
		print(f"[ERROR] Failed to get collection stats: {type(e).__name__}: {e}")
		import traceback
		traceback.print_exc()
		return {
			"collection": COLLECTION_NAME,
			"status": "error",
			"error": str(e),
			"points_count": None,
			"vector_size": None,
		}
