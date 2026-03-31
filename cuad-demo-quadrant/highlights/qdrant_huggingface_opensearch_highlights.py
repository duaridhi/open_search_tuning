"""
Hugging Face Inference API-based Highlighting Module

Uses Hugging Face Inference API to highlight semantic sentences from text
using the OpenSearch semantic highlighter model, rather than running locally.

This approach:
- Reduces local resource usage (no GPU/CPU needed)
- Uses Hugging Face free/paid tier inference
- Provides error handling and timeouts
- Falls back gracefully on API failures
"""

import os
import re
from typing import Optional, List, Tuple
from pathlib import Path
import logging

import nltk
from huggingface_hub import InferenceClient

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("[WARN] HF_TOKEN not found in environment. HF highlighting will fail.")

MODEL_ID = "opensearch-project/opensearch-semantic-highlighter-v1"
INFERENCE_TIMEOUT = int(os.getenv("HF_INFERENCE_TIMEOUT", "30"))
MAX_BATCH_SIZE = int(os.getenv("HF_MAX_BATCH_SIZE", "10"))

# Initialize HF client
_hf_client: Optional[InferenceClient] = None


def initialize_hf_client() -> bool:
    """
    Initialize Hugging Face Inference client.
    
    Returns:
        True if successful, False otherwise
    """
    global _hf_client
    
    if not HF_TOKEN:
        logger.error("[ERROR] HF_TOKEN not set. Cannot initialize HF client.")
        return False
    
    try:
        _hf_client = InferenceClient(
            api_key=HF_TOKEN,
            timeout=INFERENCE_TIMEOUT,
        )
        logger.info("[INFO] HuggingFace Inference client initialized")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize HF client: {e}")
        return False


def get_hf_client() -> Optional[InferenceClient]:
    """Get or initialize HF client."""
    global _hf_client
    if _hf_client is None:
        initialize_hf_client()
    return _hf_client


def _tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    sentences = nltk.sent_tokenize(text)
    return sentences


def highlight_text(text: str) -> Tuple[List[str], List[int]]:
    """
    Highlight semantically important sentences using HF Inference API.
    
    This function:
    1. Splits text into sentences
    2. Calls HF API to classify importance of each sentence
    3. Returns highlighted sentences and their indices
    
    Args:
        text: Input text to highlight
        
    Returns:
        Tuple of (highlighted_sentences, sentence_indices)
        Returns ([], []) on failure with logging
    """
    if not text or not isinstance(text, str):
        logger.warning("[WARN] Invalid text input for highlighting")
        return [], []
    
    client = get_hf_client()
    if not client:
        logger.warning("[WARN] HF client not initialized. Cannot highlight.")
        return [], []
    
    try:
        # Tokenize into sentences
        sentences = _tokenize_sentences(text)
        if not sentences:
            logger.debug("[DEBUG] No sentences found in text")
            return [], []
        
        highlighted_indices = []
        
        # Process in batches
        for i in range(0, len(sentences), MAX_BATCH_SIZE):
            batch = sentences[i : i + MAX_BATCH_SIZE]
            
            try:
                # Call HF API for text classification
                results = client.text_classification(
                    text,
                    model=MODEL_ID,
                )
                
                # Extract high-confidence results
                # Model outputs labels like "LABEL_1" (highlighted) or "LABEL_0" (not highlighted)
                if isinstance(results, list):
                    for result_idx, result in enumerate(results):
                        if isinstance(result, dict):
                            label = result.get("label", "LABEL_0")
                            score = result.get("score", 0.0)
                            
                            # LABEL_1 typically means "important/highlighted"
                            if label == "LABEL_1" and score > 0.5:
                                # Map back to sentence index
                                sentence_idx = i + result_idx
                                if sentence_idx < len(sentences):
                                    highlighted_indices.append(sentence_idx)
                
                logger.debug(f"[DEBUG] Highlighted {len(highlighted_indices)} sentences from batch")
                
            except Exception as e:
                logger.warning(f"[WARN] HF API call failed for batch {i}: {e}")
                continue
        
        # Extract highlighted sentences
        highlighted_sentences = [sentences[idx] for idx in highlighted_indices if idx < len(sentences)]
        
        logger.debug(f"[DEBUG] Extracted {len(highlighted_sentences)} highlighted sentences")
        return highlighted_sentences, highlighted_indices
        
    except Exception as e:
        logger.error(f"[ERROR] Text highlighting failed: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def highlight_with_fallback(
    text: str,
    fallback_fn=None,
) -> Tuple[List[str], List[int]]:
    """
    Highlight text with fallback to local function on HF API failure.
    
    Args:
        text: Input text to highlight
        fallback_fn: Optional fallback function if HF API fails
        
    Returns:
        Tuple of (highlighted_sentences, sentence_indices)
    """
    try:
        highlighted, indices = highlight_text(text)
        if highlighted:
            return highlighted, indices
    except Exception as e:
        logger.warning(f"[WARN] HF highlighting failed: {e}")
    
    # Fallback to local function if provided
    if fallback_fn:
        try:
            logger.info("[INFO] Falling back to local highlighting function")
            return fallback_fn(text)
        except Exception as e:
            logger.error(f"[ERROR] Fallback highlighting also failed: {e}")
    
    return [], []


def get_top_sentences(
    text: str,
    top_k: int = 3,
) -> List[str]:
    """
    Get top-k most important sentences from text.
    
    Args:
        text: Input text
        top_k: Number of top sentences to return
        
    Returns:
        List of top-k sentences
    """
    if not text or top_k <= 0:
        return []
    
    try:
        sentences = _tokenize_sentences(text)
        if not sentences:
            return []
        
        client = get_hf_client()
        if not client:
            logger.warning("[WARN] HF client not available for sentence ranking")
            # Fallback: return first k sentences
            return sentences[:top_k]
        
        # Score all sentences
        sentence_scores = []
        try:
            result = client.text_classification(
                text,
                model=MODEL_ID,
            )
            
            # Extract scores
            if isinstance(result, list):
                for idx, item in enumerate(result[:len(sentences)]):
                    if isinstance(item, dict):
                        score = item.get("score", 0.0)
                        label = item.get("label", "LABEL_0")
                        # Higher score if LABEL_1
                        adjusted_score = score if label == "LABEL_1" else (1.0 - score)
                        sentence_scores.append((idx, sentences[idx], adjusted_score))
            
            # Sort by score and return top-k
            sentence_scores.sort(key=lambda x: x[2], reverse=True)
            top_sentences = [sent for _, sent, _ in sentence_scores[:top_k]]
            return top_sentences
        
        except Exception as e:
            logger.warning(f"[WARN] HF ranking failed: {e}. Returning first {top_k} sentences.")
            return sentences[:top_k]
    
    except Exception as e:
        logger.error(f"[ERROR] get_top_sentences failed: {e}")
        return []


def is_hf_available() -> bool:
    """Check if HF highlighting is available."""
    return bool(HF_TOKEN and get_hf_client())


if __name__ == "__main__":
    # Test the HF highlighting
    logging.basicConfig(level=logging.DEBUG)
    
    if initialize_hf_client():
        test_text = """
        The company entered into a termination agreement with the vendor.
        The agreement specifies the conditions for early termination.
        Both parties must provide 30 days notice before any action.
        """
        
        logger.info("Testing HF highlighting...")
        highlighted, indices = highlight_text(test_text)
        logger.info("Highlighted sentences: %s", highlighted)
        logger.info("Indices: %s", indices)
        
        logger.info("Testing top sentences...")
        top = get_top_sentences(test_text, top_k=2)
        logger.info("Top sentences: %s", top)
    else:
        logger.error("Failed to initialize HF client")
