"""Tests for ingestion.rerank.

No cross-encoder is loaded here. What the real model scores is the model's
business; what this module has to get right is everything around it -- that the
cross-encoder's opinion actually replaces the cosine ordering, that the cosine
score survives it, and that the shortlist is cut *after* re-scoring rather than
before, which is the only reason reranking can rescue anything.

The fake encoder scores by chunk text, so a test can state the disagreement it
wants and assert the order that comes out.
"""

from __future__ import annotations

import pytest

from config import settings
from ingestion import rerank as rr
from ingestion.vector_store import Hit
from tools import rag_tools as rt


def hit(chunk_id: str, doc_id: str = "d1", score: float = 0.5) -> Hit:
    """A hit whose text is its id, so the fake encoder can be keyed on it."""
    return Hit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=chunk_id,
        score=score,
        metadata={"doc_id": doc_id},
    )


@pytest.fixture
def encoder(monkeypatch):
    """Install a cross-encoder that returns the scores a test asks for."""

    def _install(scores: dict[str, float]):
        class Encoder:
            def predict(self, pairs, **kwargs):
                return [scores[text] for _, text in pairs]

        monkeypatch.setattr(rr, "_cross_encoder", lambda name: Encoder())

    return _install


class TestRerank:
    def test_the_cross_encoder_overrules_the_cosine_order(self, encoder):
        encoder({"a": 0.1, "b": 0.9})
        hits = [hit("a", score=0.9), hit("b", score=0.1)]
        assert [h.chunk_id for h in rr.rerank("q", hits, name="fake")] == ["b", "a"]

    def test_the_cosine_score_is_kept_beside_the_new_one(self, encoder):
        # Losing it would make a bad answer untraceable: you could not tell
        # whether the dense stage or the reranker put the wrong chunk on top.
        encoder({"a": 0.2})
        (reranked,) = rr.rerank("q", [hit("a", score=0.87)], name="fake")
        assert reranked.score == 0.87
        assert reranked.rerank_score == 0.2

    def test_the_shortlist_is_cut_after_rescoring_not_before(self, encoder):
        # The whole point. Cutting first would leave the reranker reordering
        # the same two chunks the dense search already chose.
        encoder({"a": 0.1, "b": 0.2, "c": 0.9})
        hits = [hit("a", score=0.9), hit("b", score=0.8), hit("c", score=0.7)]
        assert [h.chunk_id for h in rr.rerank("q", hits, 2, name="fake")] == ["c", "b"]

    def test_nothing_to_rerank_loads_no_model(self, monkeypatch):
        monkeypatch.setattr(rr, "_cross_encoder", _explode)
        assert rr.rerank("q", [], name="fake") == []


class TestRankScore:
    def test_an_unreranked_hit_orders_by_its_cosine(self):
        assert hit("a", score=0.4).rank_score == 0.4

    def test_a_reranked_hit_orders_by_the_cross_encoder(self, encoder):
        encoder({"a": 0.9})
        (reranked,) = rr.rerank("q", [hit("a", score=0.4)], name="fake")
        assert reranked.rank_score == 0.9


class TestMaybeRerank:
    def test_an_unconfigured_reranker_leaves_the_order_alone(self, monkeypatch):
        monkeypatch.setattr(settings, "reranker_model", "")
        monkeypatch.setattr(rr, "_cross_encoder", _explode)
        hits = [hit("a", score=0.9), hit("b", score=0.1)]
        assert rr.maybe_rerank("q", hits) == hits

    def test_it_reranks_no_deeper_than_the_candidate_limit(
        self, monkeypatch, encoder
    ):
        # The cross-encoder is the expensive stage; paying for hits the
        # measurement says cannot change the answer is the one waste to avoid.
        monkeypatch.setattr(settings, "reranker_model", "fake")
        monkeypatch.setattr(rr, "CANDIDATES", 2)
        encoder({"a": 0.1, "b": 0.9})
        hits = [hit("a"), hit("b"), hit("c")]
        assert [h.chunk_id for h in rr.maybe_rerank("q", hits)] == ["b", "a"]

    def test_a_configured_one_runs(self, monkeypatch, encoder):
        monkeypatch.setattr(settings, "reranker_model", "fake")
        encoder({"a": 0.1, "b": 0.9})
        hits = [hit("a", score=0.9), hit("b", score=0.1)]
        assert [h.chunk_id for h in rr.maybe_rerank("q", hits)] == ["b", "a"]

    def test_whitespace_in_the_setting_is_not_a_reranker(self, monkeypatch):
        # An .env line left as `RERANKER_MODEL= ` should mean off, not a model
        # named " " that fails to load on every question.
        monkeypatch.setattr(settings, "reranker_model", "  ")
        assert not rr.is_enabled()


class TestFailures:
    def test_a_model_that_will_not_load_names_itself(self, monkeypatch):
        monkeypatch.setattr(
            rr, "_cross_encoder", rr._cross_encoder.__wrapped__
        )
        with pytest.raises(rr.RerankError, match="no/such/reranker"):
            rr.rerank("q", [hit("a")], name="no/such/reranker")


class TestItIsWiredIntoRetrieval:
    """The store is faked; the reranking and capping underneath are real."""

    @pytest.fixture
    def store(self, monkeypatch):
        hits = [
            hit("a", "d1", score=0.9),
            hit("b", "d2", score=0.8),
            hit("c", "d3", score=0.7),
        ]
        monkeypatch.setattr(rt, "search", lambda *args, **kwargs: hits)
        monkeypatch.setattr(settings, "reranker_model", "fake")
        return hits

    def test_the_agent_gets_the_reranked_order(self, store, encoder):
        encoder({"a": 0.1, "b": 0.2, "c": 0.9})
        result = rt.search_research("q")
        assert [c.chunk_id for c in result.chunks] == ["c", "b", "a"]

    def test_the_reported_score_is_the_one_that_decided_the_order(
        self, store, encoder
    ):
        # Otherwise the CLI prints a column that descends out of order and
        # looks like a ranking bug.
        encoder({"a": 0.1, "b": 0.2, "c": 0.9})
        scores = [c.score for c in rt.search_research("q").chunks]
        assert scores == sorted(scores, reverse=True)

    def test_a_reranker_that_will_not_load_is_a_retrieval_error(
        self, store, monkeypatch
    ):
        # A deployment fault, not an empty result. See tools/rag_tools.py.
        monkeypatch.setattr(rr, "_cross_encoder", _explode)
        with pytest.raises(rt.RagToolError):
            rt.search_research("q")


def _explode(name):
    raise rr.RerankError(f"should not have been loaded: {name}")
