"""
document_utils.py
─────────────────
Utilities for document discovery and metadata from Qdrant collection.
"""

import os
from typing import Optional, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


def get_unique_documents(client: QdrantClient, collection_name: str) -> List[Dict]:
    """
    Get list of unique documents (contracts) from Qdrant collection.

    Returns:
        List of dicts with document metadata: title, word_count, chunk_count, pdf_path
    """
    try:
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name)
            print(f"[DEBUG] Collection '{collection_name}' has {collection_info.points_count} points")
        except Exception as e:
            print(f"[ERROR] Collection '{collection_name}' not found: {e}")
            return []

        # Retrieve all points (with pagination if collection is large)
        scroll_results, _ = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        print(f"[DEBUG] Retrieved {len(scroll_results)} points from collection")

        # Aggregate by title to get unique documents
        doc_map: Dict[str, Dict] = {}
        for point in scroll_results:
            payload = point.payload
            title = payload.get("title")
            if title and title not in doc_map:
                doc_map[title] = {
                    "title": title,
                    "pdf_path": payload.get("pdf_path"),
                    "chunk_count": 0,
                    "total_chars": 0,
                }
            if title:
                doc_map[title]["chunk_count"] += 1
                text = payload.get("text", "")
                doc_map[title]["total_chars"] += len(text)

        # Convert to list and sort
        documents = sorted(
            doc_map.values(),
            key=lambda x: x["title"],
        )

        return documents

    except Exception as e:
        print(f"[ERROR] Failed to get documents: {e}")
        return []


def get_document_info(
    client: QdrantClient,
    collection_name: str,
    document_title: str,
) -> Optional[Dict]:
    """
    Get detailed info about a specific document.

    Args:
        client: Qdrant client
        collection_name: Collection name
        document_title: Document title/contract name

    Returns:
        Dict with document stats or None if not found
    """
    scroll_results, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    doc_info = None
    for point in scroll_results:
        payload = point.payload
        if payload.get("title") == document_title:
            if doc_info is None:
                doc_info = {
                    "title": document_title,
                    "pdf_path": payload.get("pdf_path"),
                    "chunks": [],
                }
            doc_info["chunks"].append(
                {
                    "doc_id": payload.get("doc_id"),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "char_count": len(payload.get("text", "")),
                }
            )

    return doc_info
