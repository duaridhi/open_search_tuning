import sys

from open_search_connect import connect
from sentence_transformers import SentenceTransformer
from opensearchpy import Search
import json
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
INDEX_NAME = "cuad_dataset"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Connect to OpenSearch
client = connect()

def hybrid_search(query, top_k=10):
    # Perform keyword search
    keyword_search = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k
        }
    )

    # Get the document IDs from the keyword search
    keyword_doc_ids = [hit["_id"] for hit in keyword_search["hits"]["hits"]]

    # Perform semantic search using embeddings
    query_embedding = embedding_model.encode(query, show_progress_bar=False)
    semantic_search = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding.tolist(),
                        "k": top_k
                    }
                }
            },
            "size": top_k
        }
    )

    # Get the document IDs from the semantic search
    semantic_doc_ids = [hit["_id"] for hit in semantic_search["hits"]["hits"]]

    # Combine results and remove duplicates
    combined_doc_ids = list(set(keyword_doc_ids + semantic_doc_ids))

    # Fetch the combined results
    combined_results = []
    for doc_id in combined_doc_ids[:top_k]:
        doc = client.get(index=INDEX_NAME, id=doc_id)
        combined_results.append(doc["_source"])

    return combined_results

# Example usage
if __name__ == "__main__":
    query = "What is the purpose of the CUAD dataset?"
    results = hybrid_search(query, 20)
    print(json.dumps(results, indent=2))