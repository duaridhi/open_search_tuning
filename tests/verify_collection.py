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

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from parent directory's .env
env_path = Path(__file__).resolve().parent.parent / "cuad-demo-quadrant" / ".env"
load_dotenv(env_path)

# Configuration
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
CLUSTER_URL = os.environ.get("CLUSTER_URL", "").strip()
COLLECTION_NAME = "cuad_contracts"

if not QDRANT_API_KEY or not CLUSTER_URL:
    print("[ERROR] Missing QDRANT_API_KEY or CLUSTER_URL in .env")
    sys.exit(1)

print(f"[INFO] Connecting to Qdrant cluster: {CLUSTER_URL}")

try:
    qdrant = QdrantClient(
        url=f"{CLUSTER_URL}:6333",
        api_key=QDRANT_API_KEY,
        timeout=30,
        prefer_grpc=False,
    )
    print("[✓] Connected to Qdrant cluster successfully.")
except Exception as e:
    print(f"[ERROR] Failed to connect: {e}")
    sys.exit(1)

# Get collection info
print(f"\n[INFO] Retrieving collection '{COLLECTION_NAME}' info...")
try:
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"[✓] Collection found.")
    print(f"    Points count    : {info.points_count}")
    print(f"    Vector size     : {info.config.vectors_config.size if hasattr(info.config, 'vectors_config') else 'N/A'}")
except Exception as e:
    print(f"[ERROR] Failed to get collection info: {e}")
    sys.exit(1)

# List all collections
print(f"\n[INFO] Available collections:")
try:
    collections = qdrant.get_collections().collections
    for coll in collections:
        print(f"    - {coll.name}")
except Exception as e:
    print(f"[ERROR] Failed to list collections: {e}")

# Retrieve and display sample documents
print(f"\n[INFO] Retrieving sample documents...")
try:
    # Scroll through the collection to get sample points
    points, next_page_offset = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    
    if not points:
        print("[WARN] No points found in collection!")
    else:
        print(f"[✓] Retrieved {len(points)} sample documents:\n")
        for i, point in enumerate(points, 1):
            payload = point.payload
            print(f"  [{i}] ID: {point.id}")
            print(f"      Doc ID    : {payload.get('doc_id', 'N/A')}")
            print(f"      Title     : {payload.get('title', 'N/A')}")
            print(f"      PDF Path  : {payload.get('pdf_path', 'N/A')}")
            print(f"      Pages     : {payload.get('page_start', 'N/A')}-{payload.get('page_end', 'N/A')}")
            text_preview = payload.get('text', '')[:100].replace('\n', ' ')
            print(f"      Text      : {text_preview}...")
            print()
except Exception as e:
    print(f"[ERROR] Failed to retrieve documents: {e}")
    sys.exit(1)

print("[✓] Verification complete!")
print(f"\n[INFO] Summary:")
print(f"       Collection: {COLLECTION_NAME}")
print(f"       Total uploaded documents: {info.points_count}")
