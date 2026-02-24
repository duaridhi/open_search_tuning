# %%
print("Start")

# %%
# Load environment variables from .env file (e.g. HF_TOKEN)
from dotenv import load_dotenv
import os

load_dotenv()
token = os.environ["HF_TOKEN"]

# %%
# Connect to OpenSearch and set the target index for IR dataset
import open_search_connect
client = open_search_connect.connect()
INDEX_NAME = "ir-dataset-train-v2"

# %%
# Disable tokenizer parallelism and limit CPU threads to avoid
# deadlocks and resource contention when running under WSL
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Authenticate with Hugging Face Hub using the token from environment
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# Load the bi-encoder model for generating query/document embeddings
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
def hybrid_opensearch_reranked(query, k):
    """
    Run a hybrid search (BM25 + k-NN vector) on OpenSearch.

    Combines a BM25 keyword match on 'text_field' with a k-NN
    vector search on 'text_vector' (boosted) so that both lexical
    and semantic relevance contribute to the ranking.

    Args:
        query (str): The user search query.
        k (int): Number of top results to return.

    Returns:
        list[tuple[str, str]]: List of (doc_id, text) pairs ranked
        by OpenSearch's hybrid scoring.
    """
    # Encode the query into a dense vector for k-NN search
    query_vector = embedding_model.encode(query).tolist()

    body = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    # BM25 lexical match on the text field
                    {
                        "match": {
                            "text_field": {
                                "query": query
                            }
                        }
                    },
                    # k-NN semantic vector search with boost to weight it higher
                    {
                        "knn": {
                            "text_vector": {
                                "k": 10,
                                "vector": query_vector,
                                "boost": 4.5
                            }
                        }
                    }
                ]
            }
        }
    }

    INDEX_NAME = "ir-dataset-train-v2"
    res = client.search(index=INDEX_NAME, body=body)

    # Extract doc_id and text from each hit
    hits = [(hit["_source"]["doc_id"], hit["_source"]["text"]) for hit in res["hits"]["hits"]]
    return hits


# %%
def rerank_with_cross_encoder(query, docs, rerank_count):
    """
    Re-rank candidate documents using a cross-encoder model.

    The cross-encoder scores each (query, document) pair jointly,
    producing more accurate relevance scores than the bi-encoder
    used during retrieval.

    Args:
        query (str): The original search query.
        docs (list[tuple[str, str]]): Candidate (doc_id, text) pairs
            from the initial retrieval step.
        rerank_count (int): How many of the top candidates to re-rank.

    Returns:
        list[tuple[str, str, float]]: Re-ranked list of
        (doc_id, text, score) sorted by descending relevance score.
    """
    # Build (query, document_text) pairs for the cross-encoder
    pairs = [(query, text) for _, text in docs[:rerank_count]]

    # Score all pairs in a single batch
    scores = model.predict(pairs)

    # Sort by score descending to get the most relevant docs first
    reranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [(doc_id, text, score) for ((doc_id, text), score) in reranked]


# %%
# Run a quick single-query test to verify the hybrid search pipeline
query = "communication amid scientific minds"
single_docs = hybrid_opensearch_reranked(query, 10)
print(single_docs)

# %%

# %%
# Load the cross-encoder model for re-ranking (MS MARCO fine-tuned)
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# %%
# Re-rank the hybrid search results using the cross-encoder
reranked_docs = rerank_with_cross_encoder(query, single_docs, 10)
print(reranked_docs)

# %%
