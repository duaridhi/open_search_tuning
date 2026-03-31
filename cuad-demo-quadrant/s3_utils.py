"""
s3_utils.py
───────────
S3/MinIO utilities for document storage and presigned URL generation.
"""

import logging
import os
import boto3
from botocore.config import Config
from typing import Optional

logger = logging.getLogger(__name__)


# Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_PUBLIC_ENDPOINT = os.getenv("MINIO_PUBLIC_ENDPOINT", MINIO_ENDPOINT)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = os.getenv("MINIO_BUCKET", "cuad-contracts")
PRESIGNED_EXPIRY = int(os.getenv("PRESIGNED_EXPIRY_SECONDS", "3600"))

# Cached S3 clients
_s3_internal: Optional[boto3.client] = None
_s3_public: Optional[boto3.client] = None


def init_s3_clients():
    """Initialize S3 clients for internal and public access."""
    global _s3_internal, _s3_public
    
    if _s3_internal is None:
        # Internal client for listing/accessing objects
        _s3_internal = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
        logger.info("S3 internal client initialized: %s", MINIO_ENDPOINT)
    
    if _s3_public is None:
        # Public client for presigned URLs (uses public endpoint)
        _s3_public = boto3.client(
            "s3",
            endpoint_url=MINIO_PUBLIC_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
        logger.info("S3 public client initialized: %s", MINIO_PUBLIC_ENDPOINT)
    
    return _s3_internal, _s3_public


def get_s3_clients():
    """Get cached S3 clients."""
    if _s3_internal is None or _s3_public is None:
        init_s3_clients()
    return _s3_internal, _s3_public


def generate_presigned_url(s3_key: str, expiry: Optional[int] = None) -> Optional[str]:
    """
    Generate presigned URL for a document in S3/MinIO.
    
    Args:
        s3_key: S3 object key (path in bucket)
        expiry: Expiry time in seconds (uses default if None)
    
    Returns:
        Presigned URL or None if generation fails
    """
    try:
        _, s3_public = get_s3_clients()
        
        url = s3_public.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET_NAME, "Key": s3_key},
            ExpiresIn=expiry or PRESIGNED_EXPIRY,
        )
        return url
    except Exception as e:
        logger.warning("Could not generate presigned URL for %s: %s", s3_key, e)
        return None


def list_s3_documents(prefix: str = "raw/") -> list[dict]:
    """
    List all PDF documents in S3 bucket.
    
    Args:
        prefix: S3 key prefix to list under
    
    Returns:
        List of document info dicts with name, key, size, and URLs
    """
    try:
        s3_internal, _ = get_s3_clients()
        documents = []
        
        paginator = s3_internal.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)
        
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                
                # Skip directory placeholders
                if key.endswith("/"):
                    continue
                
                # Extract document name
                name = key[len(prefix):] if key.startswith(prefix) else key
                if name.lower().endswith(".pdf"):
                    name = name[:-4]
                
                # Generate presigned URL
                pdf_url = generate_presigned_url(key)
                
                last_modified = obj.get("LastModified")
                documents.append({
                    "title": name,
                    "s3_key": key,
                    "size_bytes": obj.get("Size"),
                    "last_modified": last_modified.isoformat() if last_modified else None,
                    "pdf_url": pdf_url,
                })
        
        # Sort by name
        documents.sort(key=lambda d: d["title"].lower())
        return documents
    
    except Exception as e:
        logger.error("Failed to list S3 documents: %s", e)
        return []
