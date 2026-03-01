# ---------------------------------------------------------------------------
# test_main.py
# ---------------------------------------------------------------------------
# Tests for the CUAD Hybrid Search API (cuad_opensearch/api/main.py).
#
# Run with:
#   pytest cuad_opensearch/api/test_main.py -v
#
# The API must be reachable at BASE_URL (default: http://localhost:8000).
# Set the environment variable API_BASE_URL to override, e.g.:
#   API_BASE_URL=http://localhost:8000 pytest cuad_opensearch/api/test_main.py -v
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import pytest
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(path: str, **params) -> requests.Response:
    return requests.get(f"{BASE_URL}{path}", params=params, timeout=30)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self):
        r = get("/health")
        assert r.status_code == 200

    def test_body_has_status_ok(self):
        r = get("/health")
        assert r.json() == {"status": "ok"}

    def test_content_type_is_json(self):
        r = get("/health")
        assert "application/json" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# /search  — happy path
# ---------------------------------------------------------------------------

class TestSearchHappyPath:
    def test_returns_200(self):
        r = get("/search", q="termination clause")
        assert r.status_code == 200

    def test_response_schema(self):
        r = get("/search", q="termination clause")
        body = r.json()
        assert "query" in body
        assert "top_k" in body
        assert "results" in body
        assert isinstance(body["results"], list)

    def test_query_echoed(self):
        r = get("/search", q="governing law")
        assert r.json()["query"] == "governing law"

    def test_default_top_k_is_10(self):
        r = get("/search", q="indemnification")
        body = r.json()
        assert body["top_k"] == 10
        assert len(body["results"]) <= 10

    def test_custom_top_k_respected(self):
        r = get("/search", q="payment terms", top_k=3)
        body = r.json()
        assert body["top_k"] == 3
        assert len(body["results"]) <= 3

    def test_top_k_5(self):
        r = get("/search", q="confidentiality", top_k=5)
        assert len(r.json()["results"]) <= 5

    def test_result_fields_present(self):
        r = get("/search", q="intellectual property")
        results = r.json()["results"]
        if results:
            first = results[0]
            assert "id" in first
            assert "score" in first
            assert "title" in first
            assert "text" in first
            assert "char_start" in first
            assert "char_end" in first

    def test_result_score_is_positive(self):
        r = get("/search", q="license agreement")
        for result in r.json()["results"]:
            assert result["score"] > 0, f"Non-positive score: {result['score']}"

    def test_results_sorted_by_score_descending(self):
        r = get("/search", q="arbitration", top_k=10)
        scores = [res["score"] for res in r.json()["results"]]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_char_offsets_are_non_negative(self):
        r = get("/search", q="assignment clause")
        for result in r.json()["results"]:
            assert result["char_start"] >= 0
            assert result["char_end"] >= result["char_start"]

    def test_text_is_non_empty_string(self):
        r = get("/search", q="renewal term")
        for result in r.json()["results"]:
            assert isinstance(result["text"], str)
            assert len(result["text"]) > 0

    def test_title_is_string(self):
        r = get("/search", q="exclusivity")
        for result in r.json()["results"]:
            assert isinstance(result["title"], str)


# ---------------------------------------------------------------------------
# /search  — query variations
# ---------------------------------------------------------------------------

class TestSearchQueryVariations:
    def test_short_query(self):
        r = get("/search", q="fee")
        assert r.status_code == 200
        assert isinstance(r.json()["results"], list)

    def test_long_query(self):
        long_q = "What are the obligations of the parties with respect to " \
                 "confidentiality and non-disclosure of proprietary information " \
                 "under the terms of this agreement?"
        r = get("/search", q=long_q)
        assert r.status_code == 200

    def test_query_with_special_characters(self):
        r = get("/search", q="section 5.1(a) obligations")
        assert r.status_code == 200

    def test_numeric_query(self):
        r = get("/search", q="30 days notice")
        assert r.status_code == 200

    def test_different_queries_return_different_results(self):
        r1 = get("/search", q="termination for cause", top_k=5)
        r2 = get("/search", q="intellectual property ownership", top_k=5)
        ids1 = {res["id"] for res in r1.json()["results"]}
        ids2 = {res["id"] for res in r2.json()["results"]}
        # At least some results should differ
        assert ids1 != ids2, "Different queries returned identical result sets"


# ---------------------------------------------------------------------------
# /search  — top_k boundary values
# ---------------------------------------------------------------------------

class TestSearchTopKBoundaries:
    def test_top_k_min_1(self):
        r = get("/search", q="contract", top_k=1)
        assert r.status_code == 200
        assert len(r.json()["results"]) <= 1

    def test_top_k_max_100(self):
        r = get("/search", q="agreement", top_k=100)
        assert r.status_code == 200
        assert len(r.json()["results"]) <= 100

    def test_top_k_below_min_rejected(self):
        r = get("/search", q="contract", top_k=0)
        assert r.status_code == 422  # FastAPI validation error

    def test_top_k_above_max_rejected(self):
        r = get("/search", q="contract", top_k=101)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /search  — missing / invalid parameters
# ---------------------------------------------------------------------------

class TestSearchValidation:
    def test_missing_query_returns_422(self):
        r = get("/search")  # no q param
        assert r.status_code == 422

    def test_non_integer_top_k_returns_422(self):
        r = get("/search", q="termination", top_k="abc")
        assert r.status_code == 422

    def test_empty_query_string(self):
        # FastAPI allows empty strings unless you add min_length; just check 200
        r = get("/search", q="")
        assert r.status_code in (200, 422)

    def test_unknown_extra_params_ignored(self):
        r = get("/search", q="renewal", top_k=3, unknown_param="x")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /search  — content sanity
# ---------------------------------------------------------------------------

class TestSearchContent:
    def test_cuad_contract_titles_appear(self):
        """Results should reference real CUAD contract titles (non-empty)."""
        r = get("/search", q="software license agreement", top_k=5)
        titles = [res["title"] for res in r.json()["results"]]
        assert any(len(t) > 0 for t in titles), "All titles are empty"

    def test_text_chunks_contain_relevant_words(self):
        """Loose relevance check: at least one result chunk contains a keyword."""
        keyword = "terminat"
        r = get("/search", q="termination provisions", top_k=10)
        texts = [res["text"].lower() for res in r.json()["results"]]
        assert any(keyword in t for t in texts), \
            f"No result text contains '{keyword}'"
