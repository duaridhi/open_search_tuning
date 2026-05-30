"""
document_utils.py
─────────────────
Utilities for document discovery and metadata from Qdrant collection.
"""

import logging
import os
from typing import Optional, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


_SCROLL_PAGE_SIZE = 512         # per-doc detail (filtered, small result set)
_FULL_SCAN_PAGE_SIZE = 5000     # full-collection scan — minimises Cloud round-trips


def get_unique_documents(client: QdrantClient, collection_name: str) -> List[Dict]:
    """
    Get list of unique documents (contracts) from Qdrant collection.

    Streams points in pages of `_SCROLL_PAGE_SIZE`, aggregating by `title` as it
    goes so the full payload list never has to live in memory at once.

    Returns:
        List of dicts with document metadata: title, chunk_count, total_chars, pdf_path
    """
    try:
        try:
            client.get_collection(collection_name)
        except Exception as e:
            logger.error("Collection '%s' not found: %s", collection_name, e)
            return []

        doc_map: Dict[str, Dict] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=_FULL_SCAN_PAGE_SIZE,
                with_payload=["title", "pdf_path", "char_start", "char_end"],
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                title = payload.get("title")
                if not title:
                    continue
                entry = doc_map.setdefault(title, {
                    "title": title,
                    "pdf_path": payload.get("pdf_path"),
                    "chunk_count": 0,
                    "total_chars": 0,
                })
                entry["chunk_count"] += 1
                entry["total_chars"] += payload.get("char_end", 0) - payload.get("char_start", 0)
            if offset is None:
                break

        return sorted(doc_map.values(), key=lambda x: x["title"])

    except Exception as e:
        logger.error("Failed to get documents: %s", e)
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
    # Server-side filter on `title` (keyword-indexed) — pulls only matching points,
    # instead of scanning the whole collection client-side.
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    doc_info = None
    offset = None
    title_filter = Filter(
        must=[FieldCondition(key="title", match=MatchValue(value=document_title))]
    )
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=title_filter,
            limit=_SCROLL_PAGE_SIZE,
            with_payload=["doc_id", "page_start", "page_end", "text", "pdf_path"],
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            if doc_info is None:
                doc_info = {
                    "title": document_title,
                    "pdf_path": payload.get("pdf_path"),
                    "chunks": [],
                }
            doc_info["chunks"].append({
                "doc_id": payload.get("doc_id"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "char_count": len(payload.get("text", "")),
            })
        if offset is None:
            break

    return doc_info
