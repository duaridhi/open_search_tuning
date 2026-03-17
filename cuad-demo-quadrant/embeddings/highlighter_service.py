"""
Highlighter Service: Sentence-level highlighting using local model.
"""

import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import nltk

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# --------- Highlights Model Definition ---------
from transformers import BertModel, BertPreTrainedModel

class BertTaggerForSentenceExtractionWithBackoff(BertPreTrainedModel):
    """Sentence-level BERT classifier with a confidence-backoff rule."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        import torch.nn as nn
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
        threshold=0.3,
        alpha=0.01,
        debug=False,
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
        
        if debug:
            print(f"[DEBUG MODEL] logits shape: {logits.shape}, probs shape: {probs.shape}")
            print(f"[DEBUG MODEL] offsets: {offsets}, num_sents_item: {num_sents_item}")
            print(f"[DEBUG MODEL] probs: {probs}")

        def _get_preds(pp, offs, num_s, threshold=0.3, alpha=0.01, debug=False):
            preds = []
            for idx, (p, off, ns) in enumerate(zip(pp, offs, num_s)):
                rel_probs = p[:ns]
                if debug:
                    print(f"[DEBUG] Sample {idx}: num_sentences={int(ns)}, probs={[float(x) for x in rel_probs]}, max={rel_probs.max().item():.4f}, threshold={threshold}, alpha={alpha}")
                hits = (rel_probs >= threshold).int()
                if hits.sum() == 0 and rel_probs.max().item() >= alpha:
                    if debug:
                        print(f"[DEBUG] No hits above threshold, triggering backoff. Setting sentence {rel_probs.argmax().item()} as hit (prob={rel_probs.max().item():.4f})")
                    hits[rel_probs.argmax()] = 1
                sentence_indices = torch.where(hits == 1)[0] + off
                if debug:
                    print(f"[DEBUG] Final highlighted sentence indices: {sentence_indices.tolist()}")
                preds.append(sentence_indices)
            return preds

        return tuple(_get_preds(probs, offsets, num_sents_item, threshold=threshold, alpha=alpha, debug=debug))


HIGHLIGHTER_MODEL_ID = "opensearch-project/opensearch-semantic-highlighter-v1"

# --------- Load Model and Tokenizer at Module Level ---------
try:
    print("[INFO] Loading highlighter model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HIGHLIGHTER_MODEL_ID)
    
    # Load as the custom model class
    config = AutoConfig.from_pretrained(HIGHLIGHTER_MODEL_ID)
    model = BertTaggerForSentenceExtractionWithBackoff(config)
    
    # Load pretrained weights
    base_model = AutoModel.from_pretrained(HIGHLIGHTER_MODEL_ID, trust_remote_code=True)
    model.bert = base_model.bert if hasattr(base_model, 'bert') else base_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded successfully on device: {device}")
except Exception as e:
    print(f"[ERROR] Failed to load highlighter model: {e}")
    traceback.print_exc()
    raise

# --------- FastAPI Setup ---------
app = FastAPI(
    title="Highlighter Service",
    description="Sentence-level highlighting using local model.",
    version="1.0.0",
)


class HighlightRequest(BaseModel):
    query: str
    document: str


class HighlightResponse(BaseModel):
    highlighted_sentences: list[str]
    highlight_sentence_indexes: list[int]
    highlight_offsets: list[tuple[int, int]] = Field(
        ..., description="Character offsets (start, end) for each highlighted sentence relative to document start"
    )


@app.post("/highlight", response_model=HighlightResponse)
async def highlight(request: HighlightRequest, http_request: Request):
    """Highlight sentences in a document based on a query."""
    try:
        # Check for debug param in query string
        debug_param = http_request.query_params.get("debug", "false").lower() == "true"
        
        # Validate inputs
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        if not request.document or not request.document.strip():
            raise HTTPException(status_code=400, detail="Document string cannot be empty.")

        # Tokenize document into sentences
        try:
            doc_sents = nltk.sent_tokenize(request.document)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to split document into sentences: {e}")
        
        if not doc_sents:
            raise HTTPException(status_code=400, detail="No sentences found in the document.")

        # Calculate sentence spans (character offsets)
        sent_spans = []
        current_pos = 0
        for sent in doc_sents:
            start = request.document.find(sent, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(sent)
            sent_spans.append((start, end))
            current_pos = end

        # Prepare input for model
        try:
            combined_text = f"{request.query} [SEP] {request.document}"
            encoding = tokenizer(
                combined_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            # Create sentence_ids for sentence aggregation
            # First, tokenize document separately to map tokens to sentences
            sentence_ids = torch.zeros_like(input_ids[0]) - 100
            sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
            
            # Tokenize query and document separately to properly map sentence boundaries
            query_encoding = tokenizer(
                request.query,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            query_ids = query_encoding["input_ids"][0]
            
            # Build document token-to-sentence mapping
            doc_token_to_sentence = []
            for sent_idx, sent in enumerate(doc_sents):
                sent_encoding = tokenizer(
                    sent,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                )
                sent_tokens = sent_encoding["input_ids"][0]
                doc_token_to_sentence.extend([sent_idx] * len(sent_tokens))
            
            # Find [SEP] position in combined input
            sep_pos = None
            for idx, token_id in enumerate(input_ids[0]):
                if token_id == sep_token_id:
                    sep_pos = idx
                    break
            
            if sep_pos is not None:
                # Assign sentence_ids for document part (after [SEP])
                doc_start = sep_pos + 1
                for i, sent_id in enumerate(doc_token_to_sentence):
                    token_idx = doc_start + i
                    if token_idx < input_ids.shape[1]:
                        sentence_ids[token_idx] = sent_id
                    else:
                        break
            
            sentence_ids = sentence_ids.unsqueeze(0).to(device)
            
            if debug_param:
                print(f"[DEBUG] sentence_ids shape: {sentence_ids.shape}")
                print(f"[DEBUG] sentence_ids unique values: {torch.unique(sentence_ids)}")
                print(f"[DEBUG] num sentences: {len(doc_sents)}")
            
            # Run model inference
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    sentence_ids=sentence_ids,
                    threshold=0.1,
                    alpha=0.001,
                    debug=debug_param,
                )
            
            # Extract highlighted sentence indices
            # outputs should be a list of tensors from _get_preds
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                highlighted_tensor = outputs[0]
                if isinstance(highlighted_tensor, torch.Tensor):
                    highlighted_indexes = highlighted_tensor.cpu().tolist()
                else:
                    highlighted_indexes = list(highlighted_tensor)
            else:
                highlighted_indexes = []
            
            if debug_param:
                print(f"[DEBUG] Raw outputs type: {type(outputs)}, len: {len(outputs) if isinstance(outputs, (tuple, list)) else 'N/A'}")
                print(f"[DEBUG] highlighted_tensor type: {type(highlighted_tensor) if 'highlighted_tensor' in locals() else 'N/A'}")
                print(f"[DEBUG] highlighted_indexes (raw): {highlighted_indexes}")
            
            highlighted_indexes = [int(idx) for idx in highlighted_indexes if idx < len(doc_sents)]
            
            if debug_param:
                print(f"[DEBUG] highlighted_indexes (filtered): {highlighted_indexes}")
            
            # Build response
            highlighted_sentences = [doc_sents[i] for i in highlighted_indexes]
            highlight_offsets = [sent_spans[i] for i in highlighted_indexes]
            
            response = {
                "highlighted_sentences": highlighted_sentences,
                "highlight_sentence_indexes": highlighted_indexes,
                "highlight_offsets": highlight_offsets,
            }
            
            if debug_param:
                response["debug"] = {
                    "sentence_probabilities": [],
                    "threshold": 0.05,
                    "alpha": 0.001,
                }
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed during model inference: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
