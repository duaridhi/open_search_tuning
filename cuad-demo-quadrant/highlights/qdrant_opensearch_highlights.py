from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Dict, List, Sequence, Union

import nltk
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel


MODEL_ID = "opensearch-project/opensearch-semantic-highlighter-v1"
BASE_MODEL_ID = "bert-base-uncased"
DEFAULT_MAX_SEQ_LENGTH = 510
DEFAULT_STRIDE = 128


class BertTaggerForSentenceExtractionWithBackoff(BertPreTrainedModel):
    """Sentence-level BERT classifier with a confidence-backoff rule."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    @property
    def all_tied_weights_keys(self):
        return {}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_ids=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = self.dropout(outputs[0])

        def _get_agg_output(ids, seq_out):
            max_sentences = torch.max(ids) + 1
            d_model = seq_out.size(-1)

            agg_out, global_offsets, num_sents = [], [], []
            for i, sen_ids in enumerate(ids):
                out, local_ids = [], sen_ids.clone()
                mask = local_ids != -100
                offset = local_ids[mask].min()
                global_offsets.append(offset)
                local_ids[mask] -= offset
                n_sent = local_ids.max() + 1
                num_sents.append(n_sent)

                for j in range(int(n_sent)):
                    out.append(seq_out[i, local_ids == j].mean(dim=-2, keepdim=True))

                if max_sentences - n_sent:
                    padding = torch.zeros(
                        (int(max_sentences - n_sent), d_model), device=seq_out.device
                    )
                    out.append(padding)
                agg_out.append(torch.cat(out, dim=0))
            return torch.stack(agg_out), global_offsets, num_sents

        agg_output, offsets, num_sents_item = _get_agg_output(sentence_ids, sequence_output)
        logits = self.classifier(agg_output)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]

        def _get_preds(pp, offs, num_s, threshold=0.5, alpha=0.05):
            preds = []
            for p, off, ns in zip(pp, offs, num_s):
                rel_probs = p[:ns]
                hits = (rel_probs >= threshold).int()
                if hits.sum() == 0 and rel_probs.max().item() >= alpha:
                    hits[rel_probs.argmax()] = 1
                preds.append(torch.where(hits == 1)[0] + off)
            return preds

        return tuple(_get_preds(probs, offsets, num_sents_item))


@dataclass
class DataCollatorWithPadding:
    pad_kvs: Dict[str, Union[int, float]] = field(default_factory=dict)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        first = features[0]
        batch: Dict[str, Any] = {}

        for key, pad_value in self.pad_kvs.items():
            if key in first and first[key] is not None:
                batch[key] = pad_sequence(
                    [torch.tensor(f[key]) for f in features],
                    batch_first=True,
                    padding_value=pad_value,
                )

        for key, value in first.items():
            if key not in self.pad_kvs and value is not None and isinstance(value, torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])

        return batch


def prepare_input_features(
    tokenizer,
    examples,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    stride: int = DEFAULT_STRIDE,
    padding: bool = False,
):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=padding,
        is_split_into_words=True,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["sentence_ids"] = []

    for i, sample_index in enumerate(sample_mapping):
        word_ids = tokenized_examples.word_ids(i)
        word_level_sentence_ids = examples["word_level_sentence_ids"][sample_index]

        sequence_ids = tokenized_examples.sequence_ids(i)
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        sentence_ids = [-100] * token_start_index
        for word_idx in word_ids[token_start_index:]:
            if word_idx is not None:
                sentence_ids.append(word_level_sentence_ids[word_idx])
            else:
                sentence_ids.append(-100)

        tokenized_examples["sentence_ids"].append(sentence_ids)

    for key in ("input_ids", "token_type_ids", "attention_mask", "sentence_ids"):
        tokenized_examples[key] = [seq[:max_seq_length] for seq in tokenized_examples[key]]

    return [
        {
            "input_ids": tokenized_examples["input_ids"][index],
            "token_type_ids": tokenized_examples["token_type_ids"][index],
            "attention_mask": tokenized_examples["attention_mask"][index],
            "sentence_ids": tokenized_examples["sentence_ids"][index],
        }
        for index in range(len(tokenized_examples["input_ids"]))
    ]


def get_tokenizer():
    """Get tokenizer from embedding service global state."""
    try:
        from embeddings.embedding_service import _state
        tokenizer = _state.get("highlighter_tokenizer")
        if tokenizer is None:
            raise RuntimeError("Highlighter tokenizer not loaded in embedding service")
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to get tokenizer from embedding service: {e}")


def get_highlighter_model():
    """Get highlighter model from embedding service global state."""
    try:
        from embeddings.embedding_service import _state
        model = _state.get("highlighter_model")
        if model is None:
            raise RuntimeError("Highlighter model not loaded in embedding service")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to get highlighter model from embedding service: {e}")


def _split_sentences(document: str) -> List[str]:
    try:
        return nltk.sent_tokenize(document)
    except LookupError:
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9])"
        return [sentence.strip() for sentence in re.split(pattern, document) if sentence.strip()]


def _build_document_features(
    query: str,
    document: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> tuple[List[str], Dict[str, torch.Tensor]]:
    doc_sents = _split_sentences(document)
    if not doc_sents:
        return [], {}

    sentence_ids: List[int] = []
    context_words: List[str] = []
    for sid, sent in enumerate(doc_sents):
        words = sent.split()
        if not words:
            continue
        context_words.extend(words)
        sentence_ids.extend([sid] * len(words))

    if not context_words:
        return [], {}

    examples = {
        "question": [[query]],
        "context": [context_words],
        "word_level_sentence_ids": [sentence_ids],
    }
    features = prepare_input_features(
        tokenizer=get_tokenizer(),
        examples=examples,
        max_seq_length=max_seq_length,
        stride=stride,
    )
    if not features:
        return doc_sents, {}

    collator = DataCollatorWithPadding(
        pad_kvs={
            "input_ids": 0,
            "token_type_ids": 0,
            "attention_mask": 0,
            "sentence_ids": -100,
        }
    )
    return doc_sents, collator(features)


def highlight_document(
    query: str,
    document: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> Dict[str, Any]:
    """Run the OpenSearch semantic highlighter against arbitrary document text."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not document or not document.strip():
        raise ValueError("document must be a non-empty string")

    doc_sents, batch = _build_document_features(
        query=query,
        document=document,
        max_seq_length=max_seq_length,
        stride=stride,
    )
    if not doc_sents or not batch:
        return {
            "highlighted_sentences": [],
            "highlight_sentence_indexes": [],
            "sentences": doc_sents,
        }

    model = get_highlighter_model()
    max_len = model.config.max_position_embeddings
    for key in ("input_ids", "token_type_ids", "attention_mask", "sentence_ids"):
        batch[key] = batch[key][:, :max_len]

    with torch.inference_mode():
        predictions = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["sentence_ids"],
        )

    highlight_indexes = sorted(
        {
            int(sentence_index)
            for group in predictions
            for sentence_index in group.tolist()
            if int(sentence_index) < len(doc_sents)
        }
    )
    
    # Compute character offsets for each sentence
    sentence_offsets = _compute_sentence_offsets(document, doc_sents)
    
    # Extract offsets for highlighted sentences
    highlight_offsets = [
        sentence_offsets[idx] 
        for idx in highlight_indexes 
        if idx < len(sentence_offsets)
    ]
    
    return {
        "highlighted_sentences": [doc_sents[index] for index in highlight_indexes],
        "highlight_sentence_indexes": highlight_indexes,
        "highlight_offsets": highlight_offsets,  # [(start, end), ...]
        "sentences": doc_sents,
    }


def _compute_sentence_offsets(document: str, sentences: List[str]) -> List[tuple[int, int]]:
    """
    Compute character offsets (start, end) for each sentence in the document.
    
    Args:
        document: Full document text
        sentences: List of sentences
    
    Returns:
        List of (start_offset, end_offset) tuples for each sentence
    """
    offsets = []
    search_start = 0
    
    for sentence in sentences:
        # Find sentence in document starting from search_start
        start_idx = document.find(sentence, search_start)
        if start_idx == -1:
            # Fallback: sentence not found exactly, skip it
            offsets.append((search_start, search_start))
            continue
        
        end_idx = start_idx + len(sentence)
        offsets.append((start_idx, end_idx))
        search_start = end_idx  # Continue searching after this sentence
    
    return offsets


def _extract_qdrant_payload(qdrant_data_point: Any) -> tuple[Dict[str, Any], Any, Any]:
    if hasattr(qdrant_data_point, "payload"):
        payload = dict(getattr(qdrant_data_point, "payload") or {})
        return payload, getattr(qdrant_data_point, "id", None), getattr(qdrant_data_point, "score", None)

    if not isinstance(qdrant_data_point, dict):
        raise TypeError("qdrant_data_point must be a dict-like payload or a Qdrant point object")

    if isinstance(qdrant_data_point.get("payload"), dict):
        payload = dict(qdrant_data_point["payload"])
        return payload, qdrant_data_point.get("id"), qdrant_data_point.get("score")

    payload = dict(qdrant_data_point)
    return payload, payload.get("id") or payload.get("doc_id"), payload.get("score")


def highlight_qdrant_data_point(
    query: str,
    qdrant_data_point: Any,
    text_key: str = "text",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> Dict[str, Any]:
    """Execute the semantic highlighter for a Qdrant result or payload dict."""
    payload, point_id, score = _extract_qdrant_payload(qdrant_data_point)
    document = payload.get(text_key, "")
    if not isinstance(document, str) or not document.strip():
        raise ValueError(f"Qdrant payload is missing a non-empty '{text_key}' field")

    highlights = highlight_document(
        query=query,
        document=document,
        max_seq_length=max_seq_length,
        stride=stride,
    )
    return {
        "id": point_id,
        "doc_id": payload.get("doc_id", point_id),
        "title": payload.get("title"),
        "score": score,
        "text": document,
        **highlights,
    }
