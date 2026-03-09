"""
Client for the Embedding Inference Service.
Handles communication with the standalone embedding service.
"""

import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")
EMBEDDING_SERVICE_TIMEOUT = int(os.getenv("EMBEDDING_SERVICE_TIMEOUT", "60"))


class EmbeddingServiceClient:
    """Client for calling the embedding inference service."""
    
    def __init__(self, base_url: str = EMBEDDING_SERVICE_URL, timeout: int = EMBEDDING_SERVICE_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
    
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client
    
    def health(self) -> dict:
        """Check if embedding service is running."""
        try:
            response = self._get_client().get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to embedding service at {self.base_url}: {e}")
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (list of floats)
        """
        try:
            response = self._get_client().post(
                f"{self.base_url}/embed",
                json={"texts": texts, "convert_to_numpy": False}
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except httpx.HTTPError as e:
            raise RuntimeError(f"Embedding service error: {e}")
    
    def info(self) -> dict:
        """Get embedding service information."""
        try:
            response = self._get_client().get(f"{self.base_url}/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Cannot get info from embedding service: {e}")
    
    def close(self):
        """Close HTTP client connection."""
        if self._client is not None:
            self._client.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Global client instance
_embedding_client: Optional[EmbeddingServiceClient] = None


def get_embedding_client() -> EmbeddingServiceClient:
    """Get or create global embedding service client."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingServiceClient()
    return _embedding_client


def embed(texts: list[str]) -> list[list[float]]:
    """Convenience function to get embeddings."""
    return get_embedding_client().embed(texts)
