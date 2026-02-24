OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
INDEX_NAME = "ir-dataset-train-v2"

from opensearchpy import OpenSearch

# ----------------------------
# OpenSearch client
# ----------------------------
def connect():
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False)
    client.info()
    return client
