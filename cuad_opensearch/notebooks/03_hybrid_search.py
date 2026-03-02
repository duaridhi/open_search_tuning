
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
INDEX_NAME = "cuad_dataset"
NEURAL_MODEL_ID = os.getenv("NEURAL_MODEL_ID", "YiiCrpwBjTNT0beiSYlI")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Connect to OpenSearch
client = connect()

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
        params={"search_pipeline": "rrf-pipeline"}
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

def hybrid_search(query, top_k=10, rrf_k=60):
    """
    Client-side Reciprocal Rank Fusion over BM25 + kNN results using ranx.
    """
    # BM25 search
    bm25_resp = client.search(
        index=INDEX_NAME,
        body={
            "query": {"match": {"text": query}},
            "size": top_k,
        },
    )
    # kNN search using local embedding model
    query_embedding = embedding_model.encode(query, show_progress_bar=False)
    knn_resp = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding.tolist(),
                        "k": top_k,
                    }
                }
            },
            "size": top_k,
        },
    )

    # Build ranx Run objects — {query_id: {doc_id: score}}
    Q = "q"
    bm25_run = Run({Q: {hit["_id"]: hit["_score"] for hit in bm25_resp["hits"]["hits"]}})
    print("finished bm25")
    knn_run  = Run({Q: {hit["_id"]: hit["_score"] for hit in knn_resp["hits"]["hits"]}})
    print("finished knn")
    # Fuse with RRF via ranx
    fused_run = fuse(runs=[bm25_run, knn_run], method="rrf", params={"k": rrf_k})
    ("finished fusion")
    ranked = list(fused_run.run[Q].items())[:top_k]  # already sorted by ranx
    ("finished re-ranking")
    results = []
    for doc_id, score in ranked:
        doc = client.get(index=INDEX_NAME, id=doc_id)
        results.append({"id": doc_id, "score": round(score, 6), **doc["_source"]})
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

# %%
