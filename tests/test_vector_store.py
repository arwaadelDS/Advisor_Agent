"""Tests for ingestion.vector_store.

Most tests use a fake embedder -- deterministic hash-derived vectors -- because
what they check is wiring: the filter reaches Chroma, metadata survives the
round trip, a model mismatch is refused. Requiring a 2GB download to run the
suite would mean the suite stops being run.

``TestRealRetrieval`` does use the real model, since a fake cannot tell you
whether Arabic actually retrieves. It skips when the model is not cached.
"""

from __future__ import annotations

import hashlib

import pytest

from ingestion.catalog import load_catalog
from ingestion.chunk import chunk_corpus
from ingestion.extract import ARABIC, ENGLISH, extract_corpus
from ingestion import vector_store as vs

DIM = 32


def fake_embed(texts, model_name=None, *, as_query=False):
    """Deterministic unit vectors derived from the text.

    Identical text embeds identically, so an exact-match query ranks its own
    chunk first; everything else is effectively noise. Enough to test routing,
    filtering and metadata, and nothing more.
    """
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] - 127.5 for i in range(DIM)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        vectors.append([v / norm for v in raw])
    return vectors


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(vs, "embed", fake_embed)


@pytest.fixture(scope="module")
def corpus():
    documents = extract_corpus(load_catalog())
    return documents, chunk_corpus(documents)


@pytest.fixture
def index(tmp_path, fake, corpus):
    """A real Chroma index over the real corpus, with fake vectors."""
    documents, chunks = corpus
    vs.build_index(chunks, documents, tmp_path, model_name="fake-model")
    return tmp_path


class TestPrefixes:
    def test_e5_models_need_query_and_passage_prefixes(self):
        # Silently costs recall if forgotten, which is why it is a property of
        # the model id rather than something a caller passes.
        assert vs.prefixes("intfloat/multilingual-e5-base") == ("query: ", "passage: ")

    def test_bge_m3_and_unknown_models_need_none(self):
        assert vs.prefixes("BAAI/bge-m3") == ("", "")
        assert vs.prefixes("some/new-model") == ("", "")


class TestBuild:
    def test_it_writes_every_chunk(self, index, corpus):
        _, chunks = corpus
        assert vs.open_collection(index, "fake-model").count() == len(chunks)

    def test_rebuilding_does_not_duplicate(self, index, corpus, fake):
        documents, chunks = corpus
        vs.build_index(chunks, documents, index, model_name="fake-model")
        assert vs.open_collection(index, "fake-model").count() == len(chunks)

    def test_it_refuses_to_build_an_empty_index(self, tmp_path, fake):
        with pytest.raises(vs.VectorStoreError, match="empty"):
            vs.build_index([], [], tmp_path)

    def test_it_refuses_chunks_without_their_documents(self, tmp_path, fake, corpus):
        _, chunks = corpus
        with pytest.raises(vs.VectorStoreError, match="not passed in"):
            vs.build_index(chunks, [], tmp_path)

    def test_it_records_which_model_built_it(self, index):
        status = vs.index_status(index)
        assert status["built"] and status["embedding_model"] == "fake-model"


class TestTheModelMismatchGuard:
    """The failure this store exists to make loud."""

    def test_opening_with_another_model_is_refused(self, index):
        # Vectors from two models are not comparable, but nothing about the
        # query would error: it would return a confident ranking of nonsense.
        with pytest.raises(vs.VectorStoreError, match="not comparable") as exc:
            vs.open_collection(index, "a-different-model")
        # Both names, so the reader knows which one to change.
        assert "fake-model" in str(exc.value)
        assert "a-different-model" in str(exc.value)

    def test_a_missing_index_says_how_to_build_one(self, tmp_path):
        with pytest.raises(vs.VectorStoreError, match="build_vector_index"):
            vs.open_collection(tmp_path / "nothing-here")

    def test_status_reports_an_unbuilt_index_rather_than_raising(self, tmp_path):
        assert vs.index_status(tmp_path / "nothing-here")["built"] is False


class TestSearchFiltering:
    """The retrieval contract: only research covering what the client holds."""

    def test_an_unfiltered_search_can_reach_any_document(self, index, corpus):
        _, chunks = corpus
        hits = vs.search("petrochemical downcycle", None, k=5, path=index,
                         model_name="fake-model")
        assert hits
        assert {h.chunk_id for h in hits} <= {c.chunk_id for c in chunks}

    def test_a_filtered_search_only_returns_the_named_documents(self, index):
        catalog = load_catalog()
        doc_ids = catalog.doc_ids_for_tickers(["2010"])
        hits = vs.search("risks", doc_ids, k=10, path=index, model_name="fake-model")
        assert hits
        assert {h.doc_id for h in hits} <= set(doc_ids)

    def test_none_and_empty_holdings_are_not_the_same_thing(self, index):
        # The distinction that matters: a client holding nothing we cover must
        # not be answered from another client's research.
        assert vs.search("risks", [], k=5, path=index, model_name="fake-model") == []
        assert vs.search("risks", None, k=5, path=index, model_name="fake-model") != []

    def test_k_bounds_the_result_count(self, index):
        hits = vs.search("margins", None, k=3, path=index, model_name="fake-model")
        assert len(hits) == 3


class TestHits:
    def test_a_hit_carries_what_a_citation_needs(self, index):
        (hit,) = vs.search("risks", None, k=1, path=index, model_name="fake-model")
        assert hit.title and hit.metadata["published_at"] and hit.metadata["pages"]
        assert hit.metadata["isins"]
        assert hit.title in hit.citation and "page" in hit.citation

    def test_score_is_a_similarity_not_a_distance(self, index, corpus):
        # Chroma returns distance; the agent reasons about closeness. Embedding
        # a chunk's own text must score highest.
        _, chunks = corpus
        target = chunks[5]
        hits = vs.search(target.text, None, k=3, path=index, model_name="fake-model")
        assert hits[0].chunk_id == target.chunk_id
        assert hits[0].score > hits[-1].score

    def test_metadata_survives_the_round_trip(self, index, corpus):
        _, chunks = corpus
        target = chunks[5]
        (hit,) = vs.search(target.text, None, k=1, path=index, model_name="fake-model")
        assert hit.metadata["chunk_id"] == target.chunk_id
        assert hit.metadata["language"] == target.language
        assert hit.metadata["section"] == (target.section or "")


def model_is_cached() -> bool:
    """Whether the real embedding model can be loaded without a download."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False
    from config import settings

    try:
        repos = {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        return False
    return settings.embedding_model in repos


@pytest.fixture(scope="module")
def real_index(tmp_path_factory, corpus):
    if not model_is_cached():
        pytest.skip(
            "the embedding model is not cached; run "
            "`uv run python -m ingestion.build_vector_index` once to fetch it"
        )
    documents, chunks = corpus
    path = tmp_path_factory.mktemp("real-index")
    vs.build_index(chunks, documents, path)
    return path


class TestRealRetrieval:
    """The part a fake embedder cannot answer: does Arabic actually retrieve?"""

    def test_an_english_question_finds_the_right_company(self, real_index):
        hits = vs.search("What is the outlook for SABIC?", None, k=3, path=real_index)
        assert any("sabic" in h.doc_id for h in hits)

    def test_an_arabic_question_finds_the_right_company(self, real_index):
        # If normalisation or the embedder were wrong, this would still return
        # three confident hits -- just the wrong ones.
        hits = vs.search("ما هي مخاطر سابك؟", None, k=3, path=real_index)
        assert any("sabic" in h.doc_id for h in hits)

    def test_an_arabic_question_can_retrieve_arabic_text(self, real_index):
        hits = vs.search("ما هي المخاطر الرئيسية للقطاع؟", None, k=5, path=real_index)
        assert any(h.language == ARABIC for h in hits)

    def test_retrieval_crosses_the_language_boundary(self, real_index):
        # An advisor asking in Arabic should still reach the English half of a
        # bilingual note. If this fails, half the corpus is unreachable for
        # half the users and nothing says so.
        hits = vs.search("أرباح البنوك السعودية والهوامش", None, k=10, path=real_index)
        assert {h.language for h in hits} == {ARABIC, ENGLISH}

    def test_the_holdings_filter_changes_the_answer(self, real_index):
        catalog = load_catalog()
        question = "What are the main risks?"
        everything = vs.search(question, None, k=3, path=real_index)
        held = vs.search(question, catalog.doc_ids_for_tickers(["1120"]), k=3,
                         path=real_index)
        assert held and {h.doc_id for h in held} != {h.doc_id for h in everything}

    def test_a_question_about_a_rating_finds_the_key_facts_block(self, real_index):
        catalog = load_catalog()
        hits = vs.search(
            "What is the rating on SABIC?",
            catalog.doc_ids_for_tickers(["2010"]),
            k=5,
            path=real_index,
        )
        assert any("Underweight" in h.text for h in hits)
