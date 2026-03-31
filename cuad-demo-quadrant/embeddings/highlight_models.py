"""
Shared highlight model loader.
Manages loading and caching of highlighter models for use in both 
the main app and as fallback if the embedding service is unavailable.
"""

import logging
import os
from functools import lru_cache
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Model configuration
HIGHLIGHTER_MODEL_ID = os.getenv("HIGHLIGHTER_MODEL_ID", "opensearch-project/opensearch-semantic-highlighter-v1")
HIGHLIGHTER_BASE_MODEL_ID = os.getenv("HIGHLIGHTER_BASE_MODEL_ID", "bert-base-uncased")


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


# Global state
_models = {
    "tokenizer": None,
    "highlighter_model": None,
}


def set_tokenizer(tokenizer):
    """Set highlighter tokenizer (called by app at startup)."""
    _models["tokenizer"] = tokenizer


def set_highlighter_model(model):
    """Set highlighter model (called by app at startup)."""
    _models["highlighter_model"] = model


def get_highlighter_model_impl():
    """Get cached model. Will be pre-loaded at app startup."""
    if _models["highlighter_model"] is not None:
        return _models["highlighter_model"]
    # Fallback: load on demand (should not happen if app startup succeeded)
    logger.warning("Highlighter model not pre-loaded, loading on demand...")
    model = BertTaggerForSentenceExtractionWithBackoff.from_pretrained(HIGHLIGHTER_MODEL_ID)
    model.eval()
    _models["highlighter_model"] = model
    return model


def get_highlighter_tokenizer():
    """Get cached tokenizer. Will be pre-loaded at app startup."""
    return get_tokenizer()


def get_highlighter_model():
    """Get cached model. Will be pre-loaded at app startup."""
    return get_highlighter_model_impl()
