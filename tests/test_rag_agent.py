"""Tests for agents.rag_agent.

Nothing here calls Gemini. What is worth testing is not the model's prose but
the two things around it: that it is only handed research the client is
entitled to, and that the prompt forbids answering without a source.

Retrieval underneath uses the same fake embedder as the other suites, with
``settings.vector_store_path`` pointed at a temporary index so the default-path
resolution the graph really uses is exercised rather than bypassed.
"""

from __future__ import annotations

import hashlib

import pytest

from config import settings
from ingestion.catalog import load_catalog
from ingestion.chunk import chunk_corpus
from ingestion.extract import extract_corpus
from ingestion import vector_store as vs
from schemas import ClientHolding, RAGSearchResult, SQLQueryResult
from agents import rag_agent as ra

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


@pytest.fixture(scope="module")
def corpus():
    documents = extract_corpus(load_catalog())
    return documents, chunk_corpus(documents)


@pytest.fixture
def index(tmp_path, monkeypatch, corpus):
    """A fake-vector index, installed as the configured one."""
    monkeypatch.setattr(vs, "embed", fake_embed)
    documents, chunks = corpus
    vs.build_index(chunks, documents, tmp_path, model_name=settings.embedding_model)
    monkeypatch.setattr(settings, "vector_store_path", str(tmp_path))
    return tmp_path


@pytest.fixture
def stub_llm(monkeypatch):
    """Capture the prompt instead of sending it, and return a fixed answer."""
    sent: list = []

    class Response:
        content = "Risks are X [1]."

    class Model:
        def invoke(self, messages):
            sent.append(messages)
            return Response()

    monkeypatch.setattr(ra, "_llm", lambda: Model())
    return sent


def holding(symbol: str) -> ClientHolding:
    return ClientHolding(
        symbol=symbol, name_en="", quantity=1, market_value=1, sector=""
    )


def state_for(question: str, *symbols: str) -> dict:
    state: dict = {"messages": [{"role": "user", "content": question}]}
    if symbols:
        state["sql_result"] = SQLQueryResult(
            client_id="C1",
            holdings=[holding(s) for s in symbols],
            row_count=len(symbols),
        )
    return state


class TestQuestionOf:
    """The graph is being assembled by two people; the message shape may move."""

    def test_it_reads_a_dict_message(self):
        assert ra.question_of(state_for("what are the risks")) == "what are the risks"

    def test_it_reads_a_langchain_message(self):
        from langchain_core.messages import HumanMessage

        assert ra.question_of({"messages": [HumanMessage("risks?")]}) == "risks?"

    def test_it_reads_a_bare_string(self):
        assert ra.question_of({"messages": ["risks?"]}) == "risks?"

    def test_it_takes_the_latest_human_turn_and_skips_the_assistants(self):
        # Answering the model's last reply instead of the advisor's question is
        # a quiet, plausible-looking failure.
        state = {
            "messages": [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]
        }
        assert ra.question_of(state) == "new question"


class TestHoldingsOf:
    """None and [] are different all the way down; see tools/rag_tools.py."""

    def test_no_sql_result_means_no_portfolio_in_context(self):
        assert ra.holdings_of({"messages": []}) is None

    def test_a_sql_result_yields_its_holdings(self):
        held = ra.holdings_of(state_for("q", "2010", "1120"))
        assert [h.symbol for h in held] == ["2010", "1120"]

    def test_a_client_holding_nothing_is_an_empty_list_not_none(self):
        state = {"sql_result": SQLQueryResult(client_id="C1", holdings=[], row_count=0)}
        assert ra.holdings_of(state) == []

    def test_it_accepts_a_plain_dict_result(self):
        state = {"sql_result": {"holdings": [{"symbol": "2010"}]}}
        assert ra.holdings_of(state) == [{"symbol": "2010"}]


class TestRetrieve:
    def test_it_filters_to_the_clients_holdings(self, index):
        result = ra.retrieve(state_for("risks", "2010"), k=5)
        doc_ids = load_catalog().doc_ids_for_tickers(["2010"])
        assert result.chunks and {c.doc_id for c in result.chunks} <= set(doc_ids)

    def test_the_filter_is_not_something_the_model_chose(self, index):
        # Holdings come from state. Nothing in the question can widen them.
        result = ra.retrieve(state_for("tell me about Aramco and banks", "2010"), k=5)
        assert result.searched_tickers == ["2010"]

    def test_no_portfolio_searches_everything(self, index):
        assert ra.retrieve(state_for("petrochemical outlook"), k=5).chunks

    def test_an_unscoped_search_says_so_in_the_note(self, index):
        # Otherwise the model presents sector research as though it were about
        # this client.
        note = ra.retrieve(state_for("petrochemical outlook"), k=5).note
        assert "not restricted to any client" in note

    def test_a_scoped_search_does_not_carry_that_warning(self, index):
        assert "not restricted" not in ra.retrieve(state_for("risks", "2010")).note

    def test_a_client_holding_nothing_covered_retrieves_nothing(self, index):
        result = ra.retrieve(state_for("risks", "9999"))
        assert result.chunks == [] and "9999" in result.note

    def test_an_empty_question_raises(self, index):
        with pytest.raises(ra.RagAgentError, match="no advisor question"):
            ra.retrieve({"messages": []})


class TestThePrompt:
    """The prompt is the safety mechanism, so it gets tests of its own."""

    def test_the_context_reaches_the_model(self, index):
        result = ra.retrieve(state_for("risks", "2010"))
        _, (_, user) = ra.build_messages("risks", result)
        assert result.chunks[0].text in user

    def test_the_question_reaches_the_model(self, index):
        result = ra.retrieve(state_for("risks", "2010"))
        _, (_, user) = ra.build_messages("what are the risks", result)
        assert "what are the risks" in user

    def test_it_forbids_answering_without_a_source(self):
        (_, system), _ = ra.build_messages("q", RAGSearchResult(chunks=[]))
        assert "only from the numbered research extracts" in system
        assert "general knowledge" in system

    def test_it_requires_citations(self):
        (_, system), _ = ra.build_messages("q", RAGSearchResult(chunks=[]))
        assert "Cite every claim" in system

    def test_it_tells_the_model_to_answer_in_the_asking_language(self):
        # Half the corpus and half the advisors are Arabic; a model given Arabic
        # context still drifts to English unless told not to. Whitespace is
        # collapsed so rewrapping the prompt does not fail the test.
        (_, system), _ = ra.build_messages("q", RAGSearchResult(chunks=[]))
        assert "An Arabic question gets an Arabic answer" in " ".join(system.split())

    def test_refusal_is_named_as_the_correct_answer(self):
        (_, system), _ = ra.build_messages("q", RAGSearchResult(chunks=[]))
        assert "That is the correct answer, not a failure." in system

    def test_the_empty_case_still_sends_the_reason(self, index):
        # The model must be told *why* there is nothing, or it cannot say it.
        result = ra.retrieve(state_for("risks", "9999"))
        _, (_, user) = ra.build_messages("risks", result)
        assert "9999" in user


class TestRagNode:
    def test_it_returns_the_structured_context_not_just_prose(self, index, stub_llm):
        # A later node renders citations from this; scraping them back out of
        # the answer text would be guesswork.
        update = ra.rag_node(state_for("risks", "2010"))
        assert isinstance(update["rag_context"], RAGSearchResult)
        assert update["rag_context"].chunks

    def test_it_appends_the_answer_as_a_message(self, index, stub_llm):
        update = ra.rag_node(state_for("risks", "2010"))
        assert update["messages"][-1]["content"] == "Risks are X [1]."

    def test_it_keeps_the_conversation_so_far(self, index, stub_llm):
        # AdvisorState.messages is a plain list, so LangGraph overwrites it --
        # the node must return the whole conversation, not only its own turn.
        update = ra.rag_node(state_for("risks", "2010"))
        assert len(update["messages"]) == 2
        assert update["next_agent"] is None

    def test_the_model_only_sees_the_clients_own_research(self, index, stub_llm):
        ra.rag_node(state_for("risks", "2010"))
        (_, (_, user)), = [tuple(m) for m in stub_llm]
        allowed = load_catalog().doc_ids_for_tickers(["2010"])
        excluded = [
            entry.doc_id
            for entry in load_catalog().documents
            if entry.doc_id not in allowed
        ]
        assert not any(doc_id in user for doc_id in excluded)


class TestTheModelIsOptional:
    def test_retrieval_needs_no_api_key(self, index, monkeypatch):
        # An ingestion machine with no key must still be able to search.
        monkeypatch.setattr(settings, "google_api_key", "")
        assert ra.retrieve(state_for("risks", "2010")).chunks

    def test_a_missing_key_is_named_in_the_error(self, monkeypatch):
        monkeypatch.setattr(settings, "google_api_key", "")
        ra._llm.cache_clear()
        with pytest.raises(ra.RagAgentError, match="GOOGLE_API_KEY") as exc:
            ra._llm()
        # ... and points at the thing that still works without a key.
        assert "rag_tools" in str(exc.value)
        ra._llm.cache_clear()


class TestTheToolForm:
    def test_it_exposes_a_named_tool_a_model_can_understand(self):
        tool = ra.research_tool()
        assert tool.name == "search_research_documents"
        assert "Tadawul" in tool.description

    def test_it_returns_formatted_context(self, index):
        text = ra.research_tool().invoke({"question": "risks", "tickers": ["2010"]})
        assert "[1]" in text
