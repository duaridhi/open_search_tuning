
# %%
import os
import sys

from open_search_connect import connect
from sentence_transformers import SentenceTransformer
from opensearchpy import Search
from ranx import Run, fuse
import json
from dotenv import load_dotenv

# %%

# Load environment variables
load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME")
NEURAL_MODEL_ID    = os.getenv("NEURAL_MODEL_ID", "YiiCrpwBjTNT0beiSYlI")   # text embedding (bi-encoder) — used for neural search
HIGHLIGHT_MODEL_ID = os.getenv("HIGHLIGHT_MODEL_ID", "tXOuuZwBQxryLfg1Ym_a")                     # cross-encoder — used for semantic highlighting

# Heavy resources — only initialized when running as a script/notebook,
# not when imported as a module (e.g. by the API).
embedding_model = None
client = None

if __name__ == "__main__":
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    client = connect()

# ---------------------------------------------------------------------------
# Helper search functions (shared with the API)
# ---------------------------------------------------------------------------

def _bm25_search(client, query: str, top_k: int, document_name: str = None) -> tuple[dict, dict]:
    """Run BM25 match query; return ({doc_id: bm25_score}, {doc_id: highlight})."""
    if document_name:
        print("Searching in document", document_name)
        query_body = {
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
                    "filter": [{"term": {"title": document_name}}],
                }
            },
            "size": top_k,
            "highlight": {
                "fields": {
                    "text": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                    }
                }
            },
        }
    else:
        print("Searching all documents")
        query_body = {
            "query": {"match": {"text": query}},
            "size": top_k,
            "highlight": {
                "fields": {
                    "text": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                    }
                }
            },
        }

    resp = client.search(index=INDEX_NAME, body=query_body, request_timeout=60)

    scores: dict[str, float] = {}
    highlights: dict[str, str] = {}
    for hit in resp["hits"]["hits"]:
        scores[hit["_id"]] = hit["_score"]
        if "highlight" in hit and "text" in hit["highlight"]:
            highlights[hit["_id"]] = " ... ".join(hit["highlight"]["text"])
    return scores, highlights


def _knn_search(client, query: str, top_k: int, document_name: str = None) -> tuple[dict, dict]:
    """Run neural search; return ({doc_id: score}, {doc_id: highlight}).

    Semantic highlight (type: semantic) is only applied when the neural clause
    is the top-level query.  When a document_name filter wraps it inside a
    bool, OpenSearch cannot resolve clauseText from the nested neural clause
    and raises a NullPointerException.  In that case we fall back to a
    standard fragment highlight so the API never returns a 500.
    """
    neural_clause = {
        "neural": {
            "embedding": {
                "query_text": query,
                "model_id": NEURAL_MODEL_ID,
                "k": top_k,
            }
        }
    }
    if document_name:
        print("Searching in document", document_name)
        query_body = {
            "query": {
                "bool": {
                    "must": [neural_clause],
                    "filter": [{"term": {"title": document_name}}],
                }
            },
            "size": top_k,
            # Fallback: plain fragment highlight — works inside bool
            "highlight": {
                "fields": {
                    "text": {
                        "pre_tags": ["<em>"],
                        "post_tags": ["</em>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                    }
                }
            },
        }
    else:
        print("Searching all documents")
        # Semantic highlight is safe here — neural is the top-level query
        highlight_block = {
            "fields": {"text": {"type": "semantic"}},
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"],
            "options": {
                "model_id": HIGHLIGHT_MODEL_ID,
                "query_text": query,
            },
        } if HIGHLIGHT_MODEL_ID else None
        query_body = {
            "query": neural_clause,
            "size": top_k,
        }
        if highlight_block:
            query_body["highlight"] = highlight_block

    resp = client.search(index=INDEX_NAME, body=query_body, request_timeout=60)
    scores: dict[str, float] = {}
    highlights: dict[str, str] = {}
    for hit in resp["hits"]["hits"]:
        scores[hit["_id"]] = hit["_score"]
        if "highlight" in hit and "text" in hit["highlight"]:
            highlights[hit["_id"]] = " ... ".join(hit["highlight"]["text"])
    return scores, highlights


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-based)."""
    return 1.0 / (k + rank)


def _fuse_rrf(
    bm25_scores: dict,
    knn_scores: dict,
    top_k: int,
) -> list:
    """
    Merge two ranked lists via Reciprocal Rank Fusion (RRF).
    Returns [(doc_id, rrf_score)] sorted descending, capped at top_k.
    """
    bm25_ranked = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
    knn_ranked = sorted(knn_scores, key=knn_scores.get, reverse=True)

    fused: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25_ranked, start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + _rrf_score(rank)
    for rank, doc_id in enumerate(knn_ranked, start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + _rrf_score(rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


# %%

def rrf_pipeline(query, top_k=10):
    """
    Hybrid BM25 + neural search using OpenSearch's built-in rrf-pipeline.
    Delegates fusion to the server-side search pipeline instead of doing
    RRF in Python.
    """
    response = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                "title": {
                                    "query": query
                                }
                            }
                        },
                        {
                            "neural": {
                                "embedding": {
                                    "query_text": query,
                                    "model_id": NEURAL_MODEL_ID,
                                    "k": top_k
                                }
                            }
                        }
                    ]
                }
            },
            "size": top_k
        },
        params={"search_pipeline": "rrf-pipeline"},
        request_timeout=60,
    )

    return [
        {
            "id":    hit["_id"],
            "score": hit["_score"],
            **hit["_source"],
        }
        for hit in response["hits"]["hits"]
    ]

# %%

def hybrid_search(query, top_k=10, rrf_k=60, document_name=None):
    """
    Client-side Reciprocal Rank Fusion over BM25 + kNN results using ranx.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        rrf_k: RRF parameter (default 60)
        document_name: Optional document name to filter results to a specific document
    """
    # BM25 search with highlights
    if document_name:
        bm25_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": query}}
                    ],
                    "filter": [
                        {"term": {"title": document_name}}
                    ]
                }
            },
            "size": top_k,
            "highlight": {
                "fields": {
                    "text": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        }
    else:
        bm25_body = {
            "query": {"match": {"text": query}},
            "size": top_k,
            "highlight": {
                "fields": {
                    "text": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        }
    
    bm25_resp = client.search(index=INDEX_NAME, body=bm25_body, request_timeout=60)
    
    # Store highlights from BM25 response
    highlights_map = {}
    for hit in bm25_resp["hits"]["hits"]:
        if "highlight" in hit and "text" in hit["highlight"]:
            highlights_map[hit["_id"]] = " ... ".join(hit["highlight"]["text"])
    
    # kNN/neural search — use helper so highlights are captured
    knn_scores, knn_highlights = _knn_search(client, query, top_k, document_name)
    # Merge neural highlights (BM25 takes precedence where both exist)
    for doc_id, hl in knn_highlights.items():
        if doc_id not in highlights_map:
            highlights_map[doc_id] = hl

    # Build ranx Run objects — {query_id: {doc_id: score}}
    Q = "q"
    bm25_run = Run({Q: {hit["_id"]: hit["_score"] for hit in bm25_resp["hits"]["hits"]}})
    knn_run  = Run({Q: knn_scores})
    
    # Track which docs came from which search method
    bm25_docs = set(bm25_run.run[Q].keys())
    knn_docs = set(knn_run.run[Q].keys())
    
    # Fuse with RRF via ranx
    fused_run = fuse(runs=[bm25_run, knn_run], method="rrf", params={"k": rrf_k})
    ("finished fusion")
    ranked = list(fused_run.run[Q].items())[:top_k]  # already sorted by ranx
    ("finished re-ranking")
    results = []
    for doc_id, score in ranked:
        doc = client.get(index=INDEX_NAME, id=doc_id)
        result = {"id": doc_id, "score": round(score, 6), **doc["_source"]}
        
        # Determine which search method(s) returned this doc
        sources = []
        if doc_id in bm25_docs:
            sources.append("bm25")
        if doc_id in knn_docs:
            sources.append("embeddings")
        result["source"] = sources if sources else ["rrf"]
        
        # Add highlight if available
        if doc_id in highlights_map:
            result["highlight"] = highlights_map[doc_id]
        results.append(result)
    print("finished get docs")

    return results

# %%

# Example usage
if __name__ == "__main__":
    query = "end of contract"

 #   print("=== rrf_pipeline (server-side RRF) ===")
 #   results = rrf_pipeline(query, top_k=5)
 #   print(json.dumps(results, indent=2))

    print("\n=== hybrid_search (client-side RRF) ===")
    results = hybrid_search(query, 5)
    print(json.dumps(results, indent=2))
    
    # Search within a specific document
    print("\n=== hybrid_search with document filter ===")
    results = hybrid_search(query, 5, document_name="Affiliate_Agreements")
    print(json.dumps(results, indent=2))

# %%
