"""
qdrant_cluster_connect.py
─────────────────────────
Centralized Qdrant cluster connection manager.
Reusable by upload_to_qdrant.py, qdrant_search.py, and other modules.

Handles:
  - Environment variable loading
  - Client initialization and connection testing
  - Error handling with helpful diagnostics
  - Connection pooling (singleton pattern)
"""

import os
import logging
from typing import Optional
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pathlib import Path


# Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

logger.info("File path is: %s", Path(__file__).resolve())
# Load environment variables from .env
env_path = Path(__file__).resolve().parent / ".env"
logger.info(f"Loading environment variables from: {env_path}");
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Path found: {env_path}");




# Configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_CLUSTER_URL = os.getenv("CLUSTER_URL", None)  # For Qdrant Cloud
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
logger.info(f"Configuration - QDRANT_URL: {QDRANT_URL}, CLUSTER_URL: {QDRANT_CLUSTER_URL}, QDRANT_PORT: {QDRANT_PORT}")

# Cached client (singleton)
_qdrant_client: Optional[QdrantClient] = None


def _get_connection_url() -> str:
    """
    Determine the correct Qdrant connection URL.
    
    Priority:
    1. CLUSTER_URL (Qdrant Cloud) - format: host.region.aws.qdrant.io
    2. QDRANT_URL (local/custom) - format: http://localhost:6333
    
    Returns:
        Full connection URL
    """
    if QDRANT_CLUSTER_URL:
        # Qdrant Cloud format: append port and protocol if needed
        url = QDRANT_CLUSTER_URL.strip()
        if not url.startswith("http"):
            url = f"https://{url}"
        if not url.endswith(f":{QDRANT_PORT}"):
            url = f"{url}:{QDRANT_PORT}"
        return url
    else:
        # Local or custom deployment
        return QDRANT_URL


def init_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> QdrantClient:
    """
    Initialize and connect to Qdrant cluster.

    Args:
        url: Override connection URL (uses env var if None)
        api_key: Override API key (uses env var if None)
        timeout: Connection timeout in seconds

    Returns:
        Initialized QdrantClient connected to cluster

    Raises:
        ConnectionError: If connection fails
    """
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    # Use provided values or environment defaults
    connection_url = url or _get_connection_url()
    api_token = api_key or QDRANT_API_KEY

    logger.info(f"Connecting to Qdrant: {connection_url}")

    try:
        _qdrant_client = QdrantClient(
            url=connection_url,
            api_key=api_token,
            timeout=timeout,
            prefer_grpc=False,
        )

        # Test connection
        collections = _qdrant_client.get_collections()
        collection_count = len(collections.collections) if collections else 0
        logger.info(f"Qdrant connected successfully ({collection_count} collections found)")

        return _qdrant_client

    except Exception as e:
        logger.error(f"Failed to connect to Qdrant at {connection_url}")
        logger.error(f"Make sure Qdrant is running:")
        logger.error(f"")
        logger.error(f"For local deployment:")
        logger.error(f"  docker run -p 6333:6333 qdrant/qdrant:latest")
        logger.error(f"")
        logger.error(f"For Qdrant Cloud:")
        logger.error(f"  Set CLUSTER_URL and QDRANT_API_KEY in .env")
        logger.error(f"")
        logger.error(f"Error details: {type(e).__name__}: {e}", exc_info=True)
        raise ConnectionError(
            f"Cannot connect to Qdrant at {connection_url}: {e}"
        ) from e


def get_qdrant_client() -> QdrantClient:
    """
    Get cached Qdrant client, initializing if needed.

    Returns:
        QdrantClient instance connected to cluster
    """
    global _qdrant_client
    if _qdrant_client is None:
        logger.debug("Initializing new Qdrant client")
        init_qdrant_client()
    return _qdrant_client


def reset_client():
    """Reset cached client (useful for testing)."""
    global _qdrant_client
    logger.debug("Resetting Qdrant client cache")
    _qdrant_client = None


def get_cluster_info() -> dict:
    """
    Get information about connected Qdrant cluster.

    Returns:
        Dict with cluster and connection info
    """
    logger.debug("Fetching cluster information")
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        
        info = {
            "status": "ok",
            "url": _get_connection_url(),
            "collections_count": len(collections.collections) if collections else 0,
            "collections": (
                [c.name for c in collections.collections] if collections else []
            ),
        }
        logger.info(f"Cluster info retrieved: {info['collections_count']} collections found")
        return info
    except Exception as e:
        logger.error(f"Failed to fetch cluster information: {e}", exc_info=True)
        return {
            "status": "error",
            "url": _get_connection_url(),
            "error": str(e),
        }
