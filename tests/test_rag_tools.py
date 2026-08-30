"""Tests for tools.rag_tools.

Most of this module is decision logic -- which tickers can be searched, how many
chunks any one document may contribute, what to say when the answer is nothing
-- and none of that needs an index. Those tests are pure and fast.

Only ``search_research`` touches the store, and it gets the same fake embedder
as tests/test_vector_store.py: the wiring is what is under test, not the
embeddings. The fake is duplicated rather than imported because these two files
should stay independently runnable.
"""

from __future__ import annotations

import hashlib

import pytest

from config import settings
from ingestion.catalog import load_catalog
from ingestion.chunk import chunk_corpus
from ingestion.extract import extract_corpus
from ingestion import vector_store as vs
from ingestion.vector_store import Hit
from schemas import ClientHolding, DocumentChunk, RAGSearchResult
from tools import rag_tools as rt

DIM = 32


def fake_embed(texts, model_name=None, *, as_query=False):
    """Deterministic unit vectors. Identical text embeds identically."""
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
    """A real Chroma index over the real corpus, with fake vectors.

    Labelled with the configured model name so the store's mismatch guard --
    which has its own tests in test_vector_store.py -- stays out of the way.
    """
    documents, chunks = corpus
    vs.build_index(chunks, documents, tmp_path, model_name=settings.embedding_model)
    return tmp_path


def hit(doc_id: str, score: float, chunk_id: str = "") -> Hit:
    return Hit(
        chunk_id=chunk_id or f"{doc_id}#{int(score * 100):02d}",
        doc_id=doc_id,
        text="body",
        score=score,
        metadata={"doc_id": doc_id},
    )


class TestTickersOf:
    """The loose read of whatever the SQL side hands over."""

    def test_bare_strings_pass_through(self):
        assert rt.tickers_of(["2010", "1120"]) == ["2010", "1120"]

    def test_it_reads_symbol_off_a_holding(self):
        holding = ClientHolding(
            symbol="1180", name_en="Al Rajhi Bank", name_ar="مصرف الراجحي",
            quantity=10, market_value=100, sector="Banking", asset_class="Equity",
        )
        assert rt.tickers_of([holding]) == ["1180"]

    def test_it_reads_dict_rows(self):
        assert rt.tickers_of([{"symbol": "2222", "quantity": 5}]) == ["2222"]

    def test_ticker_is_accepted_as_well_as_symbol(self):
        assert rt.tickers_of([{"ticker": "7010"}]) == ["7010"]

    def test_a_changed_holding_schema_does_not_break_it(self):
        # The real reason this function exists: ClientHolding gained name_en and
        # swapped asset_class for sector on the SQL branch. Only symbol is read,
        # so neither change reaches retrieval.
        class NewShape:
            symbol = "2010"
            name_en = "SABIC"
            sector = "Petrochemicals"

        assert rt.tickers_of([NewShape()]) == ["2010"]

    def test_duplicates_collapse_in_order(self):
        # The same name held in two accounts must not weight the filter twice.
        assert rt.tickers_of(["2010", "1120", "2010"]) == ["2010", "1120"]

    def test_whitespace_is_stripped_and_blanks_dropped(self):
        assert rt.tickers_of([" 2010 ", "", "  ", {"symbol": ""}]) == ["2010"]


class TestCoverage:
    """Covered, known-but-uncovered, and unknown are three different answers."""

    def test_a_covered_holding_is_searchable(self):
        covered, _, _ = rt.coverage(["2010"])
        assert covered == ["2010"]

    def test_an_unknown_ticker_is_reported_not_dropped(self):
        # A ticker the SQL side returned that instruments.csv has never heard of
        # is a data fault. Silently ignoring it is how the join rots unnoticed.
        _, _, unknown = rt.coverage(["9999"])
        assert unknown == ["9999"]

    def test_a_known_instrument_with_no_research_is_uncovered(self, monkeypatch):
        # Every instrument in the demo corpus has at least one document, so this
        # branch has to be forced. It is the normal case once the instrument
        # list outgrows the research.
        real = load_catalog()

        class NoDocuments:
            instrument_by_ticker = real.instrument_by_ticker

            @staticmethod
            def doc_ids_for_tickers(tickers):
                return []

        monkeypatch.setattr(rt, "catalog", lambda: NoDocuments())
        covered, uncovered, unknown = rt.coverage(["2010"])
        assert (covered, uncovered, unknown) == ([], ["2010"], [])


class TestDiversify:
    """One chatty document must not take every slot -- unless it is the only one."""

    def test_it_caps_chunks_per_document(self):
        hits = [hit("a", 0.9), hit("a", 0.8), hit("a", 0.7), hit("b", 0.6)]
        chosen = rt._diversify(hits, k=3, per_document=2)
        assert [h.doc_id for h in chosen] == ["a", "a", "b"]

    def test_it_tops_up_rather_than_starving_a_single_document(self):
        # A client holding one name has one relevant note. Returning two chunks
        # when five were asked for and available would be the cap misfiring.
        hits = [hit("a", 0.9 - i / 10) for i in range(5)]
        chosen = rt._diversify(hits, k=4, per_document=2)
        assert len(chosen) == 4

    def test_the_result_stays_in_score_order(self):
        hits = [hit("a", 0.9), hit("a", 0.8), hit("a", 0.7), hit("b", 0.6)]
        chosen = rt._diversify(hits, k=4, per_document=2)
        scores = [h.score for h in chosen]
        assert scores == sorted(scores, reverse=True)

    def test_k_still_bounds_the_result(self):
        hits = [hit(chr(97 + i), 0.9 - i / 10) for i in range(6)]
        assert len(rt._diversify(hits, k=3, per_document=2)) == 3


class TestSearchResearch:
    def test_a_filtered_search_stays_inside_the_holdings(self, index):
        result = rt.search_research("risks", ["2010"], k=5, path=index)
        doc_ids = load_catalog().doc_ids_for_tickers(["2010"])
        assert result.chunks
        assert {c.doc_id for c in result.chunks} <= set(doc_ids)

    def test_it_reports_which_tickers_it_searched(self, index):
        result = rt.search_research("outlook", ["2010", "1120"], path=index)
        assert result.searched_tickers == ["2010", "1120"]

    def test_k_bounds_the_chunk_count(self, index):
        assert len(rt.search_research("margins", None, k=3, path=index).chunks) == 3

    def test_holdings_accepts_the_sql_agents_own_objects(self, index):
        holding = ClientHolding(
            symbol="2010", name_en="SABIC", name_ar="سابك",
            quantity=1, market_value=1, sector="Petrochemicals", asset_class="Equity",
        )
        result = rt.search_research("risks", [holding], path=index)
        assert result.searched_tickers == ["2010"]


class TestTheEmptyAnswers:
    """Four ways to retrieve nothing, and they do not mean the same thing."""

    def test_no_holdings_searches_nothing(self, index):
        # Not the same as searching everything. A client with no positions must
        # not be answered from another client's research.
        result = rt.search_research("risks", [], path=index)
        assert result.chunks == []
        assert "no holdings" in result.note.lower()

    def test_none_holdings_searches_everything(self, index):
        assert rt.search_research("risks", None, path=index).chunks

    def test_holdings_we_cover_nothing_for_say_so(self, index):
        result = rt.search_research("risks", ["9999"], path=index)
        assert result.chunks == []
        assert "no research" in result.note.lower()
        assert "9999" in result.note
        assert result.uncovered_tickers == ["9999"]

    def test_the_note_tells_the_model_not_to_improvise(self, index):
        # This sentence is the whole defence against a confident answer with no
        # source behind it.
        note = rt.search_research("risks", ["9999"], path=index).note
        assert "general knowledge" in note


class TestTheUnscopedNote:
    """A corpus-wide search has no client, so the note must not invent one."""

    def test_it_does_not_claim_nothing_covers_the_client(self, index):
        # `covered` is empty both when a client's holdings cover nothing and
        # when there is no client at all. Reported as the first, the prompt
        # turns a perfectly good result into a refusal.
        result = rt.search_research("risks", None, path=index)
        assert result.chunks
        assert "client" not in result.note.lower()

    def test_it_does_not_tell_the_model_to_refuse(self, index):
        note = rt.search_research("risks", None, path=index).note
        assert "general knowledge" not in note

    def test_finding_nothing_unscoped_is_said_plainly(self, index, monkeypatch):
        monkeypatch.setattr(rt, "search", lambda *args, **kwargs: [])
        note = rt.search_research("risks", None, path=index).note
        assert "nothing in the index" in note.lower()


class TestOperationalFailuresRaise:
    """A broken index is a bug to fix, not an answer to give."""

    def test_a_missing_index_raises_rather_than_returning_nothing(self, tmp_path):
        # If this returned an empty result the agent would say "no research
        # found on SABIC" while the real problem is that nobody ran the build.
        with pytest.raises(rt.RagToolError, match="build_vector_index"):
            rt.search_research("risks", None, path=tmp_path / "nothing-here")

    def test_an_empty_question_raises(self, index):
        with pytest.raises(rt.RagToolError, match="empty"):
            rt.search_research("   ", None, path=index)


class TestChunksCarryTheirCitation:
    def test_a_chunk_can_be_cited_without_a_second_lookup(self, index):
        (chunk,) = rt.search_research("risks", None, k=1, path=index).chunks
        assert chunk.title and chunk.pages and chunk.published_at
        assert chunk.title in chunk.citation

    def test_isins_come_back_as_a_list(self, index):
        # Stored comma-joined because Chroma metadata must be scalar.
        (chunk,) = rt.search_research("banking", None, k=1, path=index).chunks
        assert isinstance(chunk.isins, list) and all(
            i.startswith("SA") for i in chunk.isins
        )

    def test_the_narrow_form_still_constructs(self):
        # The compatibility promise made to the SQL branch: the three original
        # fields are enough, and the widening is all optional.
        chunk = DocumentChunk(text="t", source="s", score=0.5)
        assert chunk.pages == "" and chunk.isins == []


class TestFormatContext:
    def test_blocks_are_numbered_for_the_answer_to_refer_to(self, index):
        text = rt.format_context(rt.search_research("risks", None, k=2, path=index))
        assert "[1]" in text and "[2]" in text

    def test_each_block_leads_with_its_citation(self, index):
        result = rt.search_research("risks", None, k=1, path=index)
        text = rt.format_context(result)
        # The source travels with the text; reattaching it afterwards from
        # memory is where invented page numbers come from.
        assert result.chunks[0].citation in text
        assert result.chunks[0].text in text

    def test_an_empty_result_formats_as_its_note(self, index):
        result = rt.search_research("risks", ["9999"], path=index)
        assert rt.format_context(result) == result.note

    def test_a_result_with_chunks_still_carries_the_note(self, index):
        result = rt.search_research("risks", ["2010", "9999"], k=2, path=index)
        assert result.chunks and "9999" in rt.format_context(result)

    def test_nothing_at_all_still_says_something(self):
        empty = RAGSearchResult(chunks=[])
        assert rt.format_context(empty).strip()


def test_partial_coverage_note_is_lexically_distinct_from_total_non_coverage():
    """A partial-coverage note must never contain the total-non-coverage phrase,
    since a shared prefix is what let the model discard real retrieved chunks."""
    from tools.rag_tools import _note

    partial = _note(covered=["2222"], uncovered=["1180", "2010"],
                     unknown=[], found=2, scoped=True)
    total = _note(covered=[], uncovered=[], unknown=[], found=0, scoped=True)

    assert "no research covers" not in partial.lower()
    assert "no research in the index covers any" in total.lower()