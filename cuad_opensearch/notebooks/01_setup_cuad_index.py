# filepath: /cuad_opensearch/cuad_opensearch/notebooks/01_setup_cuad_index.py

from opensearchpy import OpenSearch

# Load environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# OpenSearch connection settings
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,   # gzip-compress request/response bodies
        use_ssl=False,        # set True for HTTPS clusters
        verify_certs=False    # set True when using a valid TLS cert
    )

client.info()

# Define the index name
INDEX_NAME = "cuad_dataset"

# Define index settings and mappings
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True
    },
    "mappings": {
        "properties": {
            # Chunk text — the primary search field
            "text": {
                "type": "text",
                "analyzer": "english"
            },
            # Contract filename — stored for display / filtering
            "title": {
                "type": "keyword"
            },
            # Character offsets of this chunk within the original context
            "char_start": {
                "type": "integer"
            },
            "char_end": {
                "type": "integer"
            },
            # all-MiniLM-L6-v2 produces 384-dim vectors
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene"
                }
            }
        }
    }
}
#client.indices.create(index=INDEX_NAME, body=index_settings)

try:
    response = client.indices.create(
        index=INDEX_NAME,
        body=index_settings
    )
    print(f"Index '{INDEX_NAME}' created: {response}")
except Exception as e:
    print(f"Error creating index: {e}")