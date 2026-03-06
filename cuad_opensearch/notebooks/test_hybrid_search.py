"""
test_hybrid_search.py
---------------------
Tests for the functions in hybrid_search.py.

Test layers
───────────
1. Unit tests  — _rrf_score, _fuse_rrf: pure Python, no dependencies.
2. Mock tests  — _bm25_search, _knn_search: real query-body logic verified
                 against a fake OpenSearch client (unittest.mock).
3. Integration — _bm25_search, _knn_search, _fuse_rrf end-to-end using the
                 live OpenSearch cluster and the indexed CUAD document:
                 "LEGACYTECHNOLOGYHOLDINGS,INC_12_09_2005-EX-10.2-DISTRIBUTOR AGREEMENT"

Run all tests:          python -m pytest test_hybrid_search.py -v
Run unit tests only:    python -m pytest test_hybrid_search.py -v -m "not integration"
Run integration tests:  python -m pytest test_hybrid_search.py -v -m integration
"""

import sys
import os
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Make the notebooks directory importable ───────────────────────────────────
_notebooks_dir = Path(__file__).resolve().parent
if str(_notebooks_dir) not in sys.path:
    sys.path.insert(0, str(_notebooks_dir))

# Stub heavy modules before importing hybrid_search so the import doesn't
# trigger model downloads or an OpenSearch connection.
for _mod in ("sentence_transformers", "opensearchpy", "ranx"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Prevent the __main__ block from running during import
import builtins
_real_name = __name__

import hybrid_search as hs   # noqa: E402  (import after path setup)

# ── Sample document used across integration tests ────────────────────────────
SAMPLE_DOC = "LEGACYTECHNOLOGYHOLDINGS,INC_12_09_2005-EX-10.2-DISTRIBUTOR AGREEMENT"
SAMPLE_QUERY = "termination of agreement"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UNIT TESTS — _rrf_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestRrfScore:
    def test_rank_1_default_k(self):
        """Rank-1 with default k=60 should give 1/(60+1)."""
        assert hs._rrf_score(1) == pytest.approx(1 / 61)

    def test_rank_1_custom_k(self):
        assert hs._rrf_score(1, k=0) == pytest.approx(1.0)

    def test_higher_rank_lower_score(self):
        """Higher rank → lower RRF score."""
        assert hs._rrf_score(1) > hs._rrf_score(2) > hs._rrf_score(10)

    def test_score_always_positive(self):
        for rank in [1, 5, 100, 1000]:
            assert hs._rrf_score(rank) > 0

    def test_formula(self):
        """Explicit formula check: 1 / (k + rank)."""
        for k in [0, 10, 60]:
            for rank in [1, 3, 7]:
                assert hs._rrf_score(rank, k=k) == pytest.approx(1.0 / (k + rank))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UNIT TESTS — _fuse_rrf
# ═══════════════════════════════════════════════════════════════════════════════

class TestFuseRrf:

    def test_empty_inputs_return_empty(self):
        assert hs._fuse_rrf({}, {}, top_k=10) == []

    def test_single_list_ranking_preserved(self):
        """With only one non-empty list the order should match its ranking."""
        bm25 = {"doc_a": 0.9, "doc_b": 0.5, "doc_c": 0.1}
        result = hs._fuse_rrf(bm25, {}, top_k=3)
        ids = [r[0] for r in result]
        assert ids == ["doc_a", "doc_b", "doc_c"]

    def test_top_k_caps_results(self):
        bm25 = {f"doc_{i}": float(i) for i in range(20)}
        result = hs._fuse_rrf(bm25, {}, top_k=5)
        assert len(result) == 5

    def test_scores_sorted_descending(self):
        bm25 = {"doc_a": 0.9, "doc_b": 0.5}
        knn  = {"doc_b": 0.8, "doc_c": 0.3}
        result = hs._fuse_rrf(bm25, knn, top_k=3)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_doc_appearing_in_both_lists_boosted(self):
        """
        doc_shared appears in both lists at rank 2; doc_only_bm25 only in BM25
        at rank 1. After RRF, doc_shared should outscore doc_only_bm25 because
        it accumulates from two sources.
        """
        bm25 = {"doc_only_bm25": 10.0, "doc_shared": 5.0}
        knn  = {"doc_shared": 10.0, "doc_only_knn": 5.0}
        result = hs._fuse_rrf(bm25, knn, top_k=3)
        ids = [r[0] for r in result]
        assert ids[0] == "doc_shared"

    def test_all_scores_positive(self):
        bm25 = {"a": 1.0, "b": 0.5}
        knn  = {"c": 0.8}
        for _, score in hs._fuse_rrf(bm25, knn, top_k=5):
            assert score > 0

    def test_union_of_both_lists_returned(self):
        bm25 = {"doc_a": 1.0}
        knn  = {"doc_b": 1.0}
        result = hs._fuse_rrf(bm25, knn, top_k=10)
        ids = {r[0] for r in result}
        assert ids == {"doc_a", "doc_b"}

    def test_returns_list_of_tuples(self):
        result = hs._fuse_rrf({"x": 1.0}, {"y": 0.5}, top_k=5)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2

    def test_top_k_larger_than_union(self):
        """top_k > total unique docs should return all docs without error."""
        bm25 = {"a": 1.0}
        knn  = {"b": 0.5}
        result = hs._fuse_rrf(bm25, knn, top_k=100)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MOCK TESTS — _bm25_search query body construction
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_client(hits=None):
    """Return a mock OpenSearch client whose .search() returns `hits`."""
    if hits is None:
        hits = []
    mock = MagicMock()
    mock.search.return_value = {"hits": {"hits": hits}}
    return mock


class TestBm25SearchQueryBody:

    def test_no_document_filter_uses_simple_match(self):
        client = _make_mock_client()
        hs._bm25_search(client, "payment terms", top_k=5)
        body = client.search.call_args[1]["body"]
        assert "match" in body["query"]
        assert "bool" not in body["query"]

    def test_document_filter_uses_bool_with_term_filter(self):
        client = _make_mock_client()
        hs._bm25_search(client, "payment terms", top_k=5, document_name=SAMPLE_DOC)
        body = client.search.call_args[1]["body"]
        assert "bool" in body["query"]
        filters = body["query"]["bool"]["filter"]
        titles = [f["term"]["title"] for f in filters if "term" in f]
        assert SAMPLE_DOC in titles

    def test_highlight_block_always_present(self):
        for doc_name in [None, SAMPLE_DOC]:
            client = _make_mock_client()
            hs._bm25_search(client, "termination", top_k=5, document_name=doc_name)
            body = client.search.call_args[1]["body"]
            assert "highlight" in body
            assert "text" in body["highlight"]["fields"]

    def test_size_equals_top_k(self):
        client = _make_mock_client()
        hs._bm25_search(client, "exclusivity", top_k=7)
        body = client.search.call_args[1]["body"]
        assert body["size"] == 7

    def test_returns_scores_and_highlights(self):
        hits = [
            {"_id": "doc1", "_score": 1.5,
             "highlight": {"text": ["<mark>payment</mark> terms"]}},
            {"_id": "doc2", "_score": 0.8},
        ]
        client = _make_mock_client(hits)
        scores, highlights = hs._bm25_search(client, "payment", top_k=5)
        assert scores == {"doc1": 1.5, "doc2": 0.8}
        assert "doc1" in highlights
        assert "doc2" not in highlights

    def test_request_timeout_passed(self):
        client = _make_mock_client()
        hs._bm25_search(client, "indemnity", top_k=3)
        kwargs = client.search.call_args[1]
        assert kwargs.get("request_timeout") == 60


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MOCK TESTS — _knn_search query body construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestKnnSearchQueryBody:

    def test_no_filter_uses_neural_clause_directly(self):
        client = _make_mock_client()
        hs._knn_search(client, "exclusivity clause", top_k=5)
        body = client.search.call_args[1]["body"]
        assert "neural" in body["query"]

    def test_document_filter_wraps_in_bool(self):
        client = _make_mock_client()
        hs._knn_search(client, "governing law", top_k=5, document_name=SAMPLE_DOC)
        body = client.search.call_args[1]["body"]
        assert "bool" in body["query"]
        must_clauses = body["query"]["bool"]["must"]
        assert any("neural" in c for c in must_clauses)
        filters = body["query"]["bool"]["filter"]
        titles = [f["term"]["title"] for f in filters if "term" in f]
        assert SAMPLE_DOC in titles

    def test_semantic_highlight_only_without_document_filter(self):
        """
        Semantic highlight (type: semantic) must NOT be sent when a
        document_name filter is present — it causes a server-side NPE
        (clauseText is null) when the neural clause is nested in bool.
        Without a filter the semantic highlight block is safe and should exist.
        """
        # With document filter → no semantic highlight
        client = _make_mock_client()
        hs._knn_search(client, "termination", top_k=5, document_name=SAMPLE_DOC)
        body = client.search.call_args[1]["body"]
        highlight = body.get("highlight", {})
        for field_cfg in highlight.get("fields", {}).values():
            assert field_cfg.get("type") != "semantic", (
                "Semantic highlight must not be used when query is wrapped in bool"
            )

        # Without document filter → semantic highlight present
        client2 = _make_mock_client()
        hs._knn_search(client2, "termination", top_k=5)
        body2 = client2.search.call_args[1]["body"]
        if hs.HIGHLIGHT_MODEL_ID:
            assert "highlight" in body2
            field_cfgs = body2["highlight"].get("fields", {})
            assert any(
                v.get("type") == "semantic" for v in field_cfgs.values()
            ), "Semantic highlight should be present for top-level neural query"

    def test_document_filter_has_fallback_highlight(self):
        """With a document filter the response still includes a highlight block
        (plain fragment highlight) so the API can surface snippets."""
        client = _make_mock_client()
        hs._knn_search(client, "payment", top_k=5, document_name=SAMPLE_DOC)
        body = client.search.call_args[1]["body"]
        assert "highlight" in body
        assert "text" in body["highlight"]["fields"]

    def test_k_equals_top_k(self):
        client = _make_mock_client()
        hs._knn_search(client, "arbitration", top_k=8)
        body = client.search.call_args[1]["body"]
        neural = body["query"]["neural"]["embedding"]
        assert neural["k"] == 8

    def test_neural_model_id_is_set(self):
        client = _make_mock_client()
        hs._knn_search(client, "arbitration", top_k=3)
        body = client.search.call_args[1]["body"]
        neural = body["query"]["neural"]["embedding"]
        assert neural["model_id"] == hs.NEURAL_MODEL_ID

    def test_returns_empty_dicts_on_no_hits(self):
        client = _make_mock_client(hits=[])
        scores, highlights = hs._knn_search(client, "warranty", top_k=5)
        assert scores == {}
        assert highlights == {}

    def test_request_timeout_passed(self):
        client = _make_mock_client()
        hs._knn_search(client, "confidentiality", top_k=3)
        kwargs = client.search.call_args[1]
        assert kwargs.get("request_timeout") == 60


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION TESTS — live OpenSearch (skipped if cluster unreachable)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_live_client():
    """Try to connect to OpenSearch; return client or None."""
    try:
        from open_search_connect import connect
        c = connect()
        return c
    except Exception:
        return None


@pytest.fixture(scope="module")
def live_client():
    c = _get_live_client()
    if c is None:
        pytest.skip("OpenSearch cluster not reachable — skipping integration tests")
    return c


@pytest.mark.integration
class TestBm25SearchIntegration:

    def test_returns_results_for_known_query(self, live_client):
        scores, _ = hs._bm25_search(live_client, "termination", top_k=5)
        assert len(scores) > 0

    def test_document_filter_restricts_to_sample_doc(self, live_client):
        scores, _ = hs._bm25_search(
            live_client, "distributor", top_k=10, document_name=SAMPLE_DOC
        )
        # All returned doc IDs must belong to the sample document
        assert all(SAMPLE_DOC in doc_id for doc_id in scores)

    def test_document_filter_no_results_for_wrong_doc(self, live_client):
        scores, _ = hs._bm25_search(
            live_client, "termination", top_k=5,
            document_name="__nonexistent_document__"
        )
        assert scores == {}

    def test_highlights_contain_mark_tags(self, live_client):
        _, highlights = hs._bm25_search(
            live_client, "termination cause", top_k=5, document_name=SAMPLE_DOC
        )
        for hl in highlights.values():
            assert "<mark>" in hl

    def test_scores_are_positive_floats(self, live_client):
        scores, _ = hs._bm25_search(live_client, "payment", top_k=5)
        for score in scores.values():
            assert isinstance(score, float) and score > 0

    def test_top_k_respected(self, live_client):
        scores, _ = hs._bm25_search(live_client, "agreement", top_k=3)
        assert len(scores) <= 3


@pytest.mark.integration
class TestKnnSearchIntegration:

    def test_returns_results_for_known_query(self, live_client):
        scores, _ = hs._knn_search(live_client, "contract termination clause", top_k=5)
        assert len(scores) > 0

    def test_document_filter_restricts_to_sample_doc(self, live_client):
        scores, _ = hs._knn_search(
            live_client, "exclusive distribution rights", top_k=10,
            document_name=SAMPLE_DOC
        )
        assert all(SAMPLE_DOC in doc_id for doc_id in scores)

    def test_scores_are_positive_floats(self, live_client):
        scores, _ = hs._knn_search(live_client, "indemnification", top_k=5)
        for score in scores.values():
            assert isinstance(score, float) and score > 0

    def test_top_k_respected(self, live_client):
        scores, _ = hs._knn_search(live_client, "governing law", top_k=3)
        assert len(scores) <= 3


@pytest.mark.integration
class TestFuseRrfIntegration:
    """
    Run both searches against the live cluster and verify the fused result
    makes structural sense.
    """

    def test_fused_result_contains_sample_doc_chunks(self, live_client):
        bm25_scores, _ = hs._bm25_search(
            live_client, SAMPLE_QUERY, top_k=10, document_name=SAMPLE_DOC
        )
        knn_scores, _ = hs._knn_search(
            live_client, SAMPLE_QUERY, top_k=10, document_name=SAMPLE_DOC
        )
        fused = hs._fuse_rrf(bm25_scores, knn_scores, top_k=5)
        assert len(fused) > 0
        for doc_id, _ in fused:
            assert SAMPLE_DOC in doc_id

    def test_fused_scores_descending(self, live_client):
        bm25_scores, _ = hs._bm25_search(live_client, SAMPLE_QUERY, top_k=10)
        knn_scores, _  = hs._knn_search(live_client, SAMPLE_QUERY, top_k=10)
        fused = hs._fuse_rrf(bm25_scores, knn_scores, top_k=10)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_doc_in_both_lists_outscores_doc_in_one(self, live_client):
        """
        Artificially construct scores where one doc appears in both lists
        and verify it beats a doc that only appears in one.
        """
        # Use the live cluster to get real doc IDs for the sample document
        bm25_scores, _ = hs._bm25_search(
            live_client, "exclusive distributor", top_k=5, document_name=SAMPLE_DOC
        )
        knn_scores, _ = hs._knn_search(
            live_client, "exclusive distributor", top_k=5, document_name=SAMPLE_DOC
        )
        overlap = set(bm25_scores) & set(knn_scores)
        only_bm25 = set(bm25_scores) - set(knn_scores)
        only_knn  = set(knn_scores)  - set(bm25_scores)

        if not overlap or (not only_bm25 and not only_knn):
            pytest.skip("All docs appear in both lists for this query — can't test boost")

        fused = dict(hs._fuse_rrf(bm25_scores, knn_scores, top_k=20))
        shared_score = max(fused[d] for d in overlap if d in fused)
        single_score = max(
            fused[d] for d in (only_bm25 | only_knn) if d in fused
        )
        assert shared_score > single_score
