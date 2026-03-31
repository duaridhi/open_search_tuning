#!/usr/bin/env python
"""
verify_collection.py
────────────────────
Test script to verify that documents are properly uploaded to the Qdrant collection.

Connects to the Qdrant cluster and retrieves collection stats and sample documents.

Usage
─────
    python tests/verify_collection.py
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables from parent directory's .env
env_path = Path(__file__).resolve().parent.parent / "cuad-demo-quadrant" / ".env"
load_dotenv(env_path)

# Configuration
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
CLUSTER_URL = os.environ.get("CLUSTER_URL", "").strip()
COLLECTION_NAME = "cuad_contracts"

if not QDRANT_API_KEY or not CLUSTER_URL:
    logger.error("Missing QDRANT_API_KEY or CLUSTER_URL in .env")
    sys.exit(1)

logger.info("Connecting to Qdrant cluster: %s", CLUSTER_URL)

try:
    qdrant = QdrantClient(
        url=f"{CLUSTER_URL}:6333",
        api_key=QDRANT_API_KEY,
        timeout=30,
        prefer_grpc=False,
    )
    logger.info("Connected to Qdrant cluster successfully.")
except Exception as e:
    logger.error("Failed to connect: %s", e)
    sys.exit(1)

# Get collection info
logger.info("Retrieving collection '%s' info...", COLLECTION_NAME)
try:
    info = qdrant.get_collection(COLLECTION_NAME)
    logger.info("Collection found. Points count: %s", info.points_count)
except Exception as e:
    logger.error("Failed to get collection info: %s", e)
    sys.exit(1)

# List all collections
logger.info("Available collections:")
try:
    collections = qdrant.get_collections().collections
    for coll in collections:
        logger.info("  - %s", coll.name)
except Exception as e:
    logger.error("Failed to list collections: %s", e)

# Retrieve and display sample documents
logger.info("Retrieving sample documents...")
try:
    # Scroll through the collection to get sample points
    points, next_page_offset = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    
    if not points:
        logger.warning("No points found in collection!")
    else:
        logger.info("Retrieved %d sample documents:", len(points))
        for i, point in enumerate(points, 1):
            payload = point.payload
            text_preview = payload.get('text', '')[:100].replace('\n', ' ')
            logger.info(
                "  [%d] id=%s  title=%s  pages=%s-%s  text=%s...",
                i, point.id,
                payload.get('title', 'N/A'),
                payload.get('page_start', 'N/A'),
                payload.get('page_end', 'N/A'),
                text_preview,
            )
except Exception as e:
    logger.error("Failed to retrieve documents: %s", e)
    sys.exit(1)

logger.info("Verification complete! Collection: %s  Total vectors: %s", COLLECTION_NAME, info.points_count)
