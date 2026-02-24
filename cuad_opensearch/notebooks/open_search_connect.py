# ---------------------------------------------------------------------------
# open_search_connect.py
# Shared utility for creating an OpenSearch client.
# Import this module from any notebook or script to avoid duplicating
# connection logic across the project.
# ---------------------------------------------------------------------------

# Default connection settings — override via environment variables if needed
OPENSEARCH_HOST = "localhost"   # hostname where OpenSearch is running
OPENSEARCH_PORT = 9200          # default OpenSearch REST API port
INDEX_NAME = "cuad_dataset"  # default index used by this project

from opensearchpy import OpenSearch


# ----------------------------
# OpenSearch client
# ----------------------------
def connect() -> OpenSearch:
    """
    Create and return an authenticated OpenSearch client.

    Connects to the host/port defined by OPENSEARCH_HOST and
    OPENSEARCH_PORT.  SSL is disabled for local/Docker deployments;
    enable it (and verify_certs) when connecting to a secured cluster.

    Returns:
        OpenSearch: A connected client instance ready for indexing
        and search operations.

    Raises:
        opensearchpy.ConnectionError: If the cluster is unreachable.
    """
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,   # gzip-compress request/response bodies
        use_ssl=False,        # set True for HTTPS clusters
        verify_certs=False    # set True when using a valid TLS cert
    )
    # Verify the cluster is reachable; raises an exception if not
    client.info()
    return client
