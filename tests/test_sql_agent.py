"""
Test suite for agents/sql_agent.py

Covers: synthesize_answer / synthesize_aggregate_answer (LLM mocked),
_wants_research_followup (pure regex), run_sql_agent orchestration
(rewrite_question / run_query_pipeline mocked), and sql_agent_node (the
LangGraph entrypoint: question_of / is_aggregate_question /
run_aggregate_query_pipeline / run_sql_agent all mocked at the module
boundary). No real DB, LLM, or graph execution anywhere in this file.

ASSUMPTIONS, both flagged since the real source wasn't available to
verify against:

1. tools.llm.text_of's real implementation is unknown -- patched here to
   a simple `lambda resp: resp.content` so these tests verify sql_agent's
   own orchestration, not text_of's behavior. If text_of does something
   more involved (e.g. handling multi-block Gemini responses), that
   deserves its own test file under tools/.

2. schemas.AggregateQueryResult's exact fields weren't available --
   inferred from tools/aggregate_sql_tools.py's usage (rows, row_count,
   query_used, error, needs_clarification, clarification_question),
   mirroring SQLQueryResult's shape.
"""

import pytest
from schemas import (
    ClientHolding, RewrittenQuery, SQLQueryResult, GeneratedQuery, PipelineResult,
    AggregateQueryResult,
)
from agents import sql_agent as agent


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_llm(monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def __init__(self):
            self.last_messages = None
            self.next_response = "some prose answer"

        def invoke(self, messages):
            self.last_messages = messages
            return FakeResponse(self.next_response)

    fake = FakeLLM()
    monkeypatch.setattr(agent, "get_llm", lambda: fake)
    monkeypatch.setattr(agent, "text_of", lambda resp: resp.content)
    return fake


def _holding(symbol="2222", name_en="Saudi Aramco", sector="Energy",
             quantity=100, market_value=5000.0):
    return ClientHolding(symbol=symbol, name_en=name_en, sector=sector,
                          quantity=quantity, market_value=market_value)


def _rewrite(needs_clarification=False, ambiguous=None, rewritten="show holdings for client C001"):
    return RewrittenQuery(
        original="original question",
        rewritten=rewritten,
        corrections=[],
        ambiguous=ambiguous or [],
        needs_clarification=needs_clarification,
    )


def _pipeline_result(question="show holdings for client C001", holdings=None, error=None,
                      needs_clarification=False, clarification_question=None):
    result = None
    if error is not None or holdings is not None:
        result = SQLQueryResult(
            client_id="C001",
            holdings=holdings or [],
            row_count=len(holdings) if holdings else 0,
            query_used="SELECT ...",
            error=error,
        )
    return PipelineResult(
        question=question,
        generated=GeneratedQuery(sql="SELECT ...", confidence=0.9),
        result=result,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )


def _agg_result(rows=None, error=None, needs_clarification=False, clarification_question=None):
    return AggregateQueryResult(
        rows=rows or [],
        row_count=len(rows) if rows else 0,
        query_used="SELECT ...",
        error=error,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )


# ---------------------------------------------------------------------------
# _format_holdings_for_prompt
# ---------------------------------------------------------------------------

def test_format_holdings_empty_list():
    assert agent._format_holdings_for_prompt([]) == "(no holdings found)"


def test_format_holdings_nonempty():
    holdings = [_holding(symbol="2222", name_en="Saudi Aramco", sector="Energy",
                          quantity=100, market_value=5000.0)]
    text = agent._format_holdings_for_prompt(holdings)
    assert "2222" in text
    assert "Saudi Aramco" in text
    assert "Energy" in text
    assert "100" in text
    assert "5000" in text


# ---------------------------------------------------------------------------
# _wants_research_followup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "what's your outlook on this sector",
    "any research on Saudi Aramco",
    "what's your opinion on these holdings",
    "would you recommend rebalancing",
    "how risky is this portfolio",
    "give me an analysis of these holdings",
    "should I be worried about this concentration",
    "what do you think about this allocation",
    "your view on Aramco",
    "what's your perspective here",
])
def test_wants_research_followup_matches_research_intent(question):
    assert agent._wants_research_followup(question) is True


@pytest.mark.parametrize("question", [
    "show holdings for client C001",
    "what does client C010 hold",
    "list portfolio holdings for C018",
])
def test_wants_research_followup_does_not_match_plain_listing_questions(question):
    assert agent._wants_research_followup(question) is False


# ---------------------------------------------------------------------------
# synthesize_answer — unit tests, no agent orchestration involved
# ---------------------------------------------------------------------------

def test_synthesize_answer_clarification_branch(fake_llm):
    result = agent.synthesize_answer(
        "show holdings for c01", None,
        clarification_question="'c01' is ambiguous between: C001, C010",
    )
    assert result == fake_llm.next_response
    system_msg, user_msg = fake_llm.last_messages[0]["content"], fake_llm.last_messages[1]["content"]
    assert system_msg == agent.ANSWER_SYSTEM_PROMPT
    assert "ambiguous" in user_msg.lower()
    assert "C001, C010" in user_msg


def test_synthesize_answer_none_result_branch(fake_llm):
    agent.synthesize_answer("show holdings for c999", None)
    user_msg = fake_llm.last_messages[1]["content"]
    assert "unknown error" in user_msg.lower()
    assert "could not be retrieved" in user_msg.lower()


def test_synthesize_answer_error_branch_includes_internal_note_and_instructs_llm_not_to_leak_it(fake_llm):
    sql_result = SQLQueryResult(client_id="C001", error="sqlite3.OperationalError: no such column: foo")
    agent.synthesize_answer("show holdings for c001", sql_result)
    system_msg, user_msg = fake_llm.last_messages[0]["content"], fake_llm.last_messages[1]["content"]
    assert "sqlite3.OperationalError" in user_msg
    assert "without repeating raw technical details" in user_msg
    assert "do not expose raw SQL or" in system_msg


def test_synthesize_answer_empty_holdings_branch(fake_llm):
    sql_result = SQLQueryResult(client_id="C001", holdings=[], row_count=0,
                                 query_used="SELECT ...", error=None)
    agent.synthesize_answer("show holdings for c001", sql_result)
    user_msg = fake_llm.last_messages[1]["content"]
    assert "(no holdings found)" in user_msg


def test_synthesize_answer_normal_holdings_branch(fake_llm):
    sql_result = SQLQueryResult(client_id="C001", holdings=[_holding()], row_count=1,
                                 query_used="SELECT ...", error=None)
    agent.synthesize_answer("show holdings for c001", sql_result)
    user_msg = fake_llm.last_messages[1]["content"]
    assert "Saudi Aramco" in user_msg
    assert "show holdings for c001" in user_msg


def test_synthesize_answer_clarification_takes_priority_over_sql_result(fake_llm):
    sql_result = SQLQueryResult(client_id="C001", holdings=[_holding()], row_count=1,
                                 query_used="SELECT ...", error=None)
    agent.synthesize_answer("show holdings for c001", sql_result,
                             clarification_question="matched two clients")
    user_msg = fake_llm.last_messages[1]["content"]
    assert "matched two clients" in user_msg
    assert "Saudi Aramco" not in user_msg


# ---------------------------------------------------------------------------
# synthesize_aggregate_answer
# ---------------------------------------------------------------------------

def test_synthesize_aggregate_answer_clarification_branch(fake_llm):
    result = _agg_result(needs_clarification=True, clarification_question="biggest by what metric?")
    agent.synthesize_aggregate_answer("what's the biggest holding", result)
    user_msg = fake_llm.last_messages[1]["content"]
    assert "biggest by what metric?" in user_msg
    assert "ambiguous" in user_msg.lower()


def test_synthesize_aggregate_answer_error_branch(fake_llm):
    result = _agg_result(error="query references disallowed table(s): sqlite_master")
    agent.synthesize_aggregate_answer("show everything", result)
    system_msg, user_msg = fake_llm.last_messages[0]["content"], fake_llm.last_messages[1]["content"]
    assert "sqlite_master" in user_msg
    assert "without repeating raw technical details" in user_msg
    assert "do not expose raw SQL or" in system_msg


def test_synthesize_aggregate_answer_normal_rows_branch(fake_llm):
    result = _agg_result(rows=[{"total_value": 1234567.0}])
    agent.synthesize_aggregate_answer("total portfolio value across all clients", result)
    user_msg = fake_llm.last_messages[1]["content"]
    assert "1234567" in user_msg


def test_synthesize_aggregate_answer_no_rows_branch(fake_llm):
    result = _agg_result(rows=[])
    agent.synthesize_aggregate_answer("total portfolio value", result)
    user_msg = fake_llm.last_messages[1]["content"]
    assert "(no rows)" in user_msg


# ---------------------------------------------------------------------------
# run_sql_agent — orchestration (single-client pipeline only; unaffected
# by the aggregate addition since sql_agent_node decides routing before
# run_sql_agent is ever called)
# ---------------------------------------------------------------------------

def test_run_sql_agent_short_circuits_on_ambiguous_rewrite(monkeypatch, fake_llm):
    rewrite = _rewrite(needs_clarification=True,
                        ambiguous=["'c01' is ambiguous between: C001, C010"])
    monkeypatch.setattr(agent, "rewrite_question", lambda q: rewrite)

    def _fail_if_called(*a, **kw):
        raise AssertionError("run_query_pipeline should not be called when rewrite needs clarification")
    monkeypatch.setattr(agent, "run_query_pipeline", _fail_if_called)

    result = agent.run_sql_agent("show holdings for c01")

    assert result.needs_clarification is True
    assert result.clarification_question == "'c01' is ambiguous between: C001, C010"
    assert result.question == "show holdings for c01"
    assert result.rewrite == rewrite
    assert result.generated.sql == ""
    assert result.generated.needs_clarification is True
    assert result.answer == fake_llm.next_response

    user_msg = fake_llm.last_messages[1]["content"]
    assert "'c01' is ambiguous between: C001, C010" in user_msg


def test_run_sql_agent_happy_path_wires_rewrite_and_preserves_original_question(monkeypatch, fake_llm):
    rewrite = _rewrite(needs_clarification=False, rewritten="show holdings for client C001")
    monkeypatch.setattr(agent, "rewrite_question", lambda q: rewrite)

    pipeline_result = _pipeline_result(question=rewrite.rewritten, holdings=[_holding()])

    captured = {}
    def _fake_pipeline(question, **kwargs):
        captured["question"] = question
        captured["kwargs"] = kwargs
        return pipeline_result
    monkeypatch.setattr(agent, "run_query_pipeline", _fake_pipeline)

    original_question = "shwo holdings for c001"
    result = agent.run_sql_agent(original_question)

    assert captured["question"] == rewrite.rewritten
    assert result.question == original_question
    user_msg = fake_llm.last_messages[1]["content"]
    assert original_question in user_msg

    assert result.rewrite == rewrite
    assert result.answer == fake_llm.next_response
    assert result.result is pipeline_result.result


def test_run_sql_agent_passes_through_repair_and_timeout_kwargs(monkeypatch, fake_llm):
    monkeypatch.setattr(agent, "rewrite_question", lambda q: _rewrite())

    captured = {}
    def _fake_pipeline(question, **kwargs):
        captured["kwargs"] = kwargs
        return _pipeline_result(holdings=[])
    monkeypatch.setattr(agent, "run_query_pipeline", _fake_pipeline)

    agent.run_sql_agent("some question", max_repair_attempts=5, timeout_seconds=9, max_rows=42)

    assert captured["kwargs"] == {
        "max_repair_attempts": 5,
        "timeout_seconds": 9,
        "max_rows": 42,
    }


def test_run_sql_agent_pipeline_error_still_synthesizes_answer(monkeypatch, fake_llm):
    monkeypatch.setattr(agent, "rewrite_question", lambda q: _rewrite())
    error_result = _pipeline_result(error="query exceeded 5s timeout")
    monkeypatch.setattr(agent, "run_query_pipeline", lambda q, **kw: error_result)

    result = agent.run_sql_agent("show holdings for c001")

    assert result.answer == fake_llm.next_response
    user_msg = fake_llm.last_messages[1]["content"]
    assert "query exceeded 5s timeout" in user_msg


def test_run_sql_agent_generation_level_clarification_overrides_fetched_holdings(monkeypatch, fake_llm):
    monkeypatch.setattr(agent, "rewrite_question", lambda q: _rewrite())

    pipeline_result = _pipeline_result(
        holdings=[_holding()],
        needs_clarification=True,
        clarification_question="matched two clients with similar names",
    )
    monkeypatch.setattr(agent, "run_query_pipeline", lambda q, **kw: pipeline_result)

    result = agent.run_sql_agent("show holdings for al-qahtani")

    assert result.needs_clarification is True
    assert result.clarification_question == "matched two clients with similar names"
    user_msg = fake_llm.last_messages[1]["content"]
    assert "matched two clients with similar names" in user_msg
    assert "Saudi Aramco" not in user_msg


def test_query_client_portfolio_tool_delegates_to_run_sql_agent(monkeypatch, fake_llm):
    monkeypatch.setattr(agent, "rewrite_question", lambda q: _rewrite())
    monkeypatch.setattr(agent, "run_query_pipeline", lambda q, **kw: _pipeline_result(holdings=[]))

    result = agent.query_client_portfolio.invoke({"question": "show holdings for c001"})

    assert result.question == "show holdings for c001"
    assert result.answer == fake_llm.next_response


# ---------------------------------------------------------------------------
# sql_agent_node — the LangGraph entrypoint: question_of / is_aggregate_question
# / run_aggregate_query_pipeline / run_sql_agent all mocked at the boundary
# ---------------------------------------------------------------------------

def test_sql_agent_node_empty_question_returns_apology_without_calling_anything(monkeypatch):
    monkeypatch.setattr(agent, "question_of", lambda state: "   ")

    def _fail_if_called(*a, **kw):
        raise AssertionError("nothing downstream should be called for an empty question")
    monkeypatch.setattr(agent, "is_aggregate_question", _fail_if_called)
    monkeypatch.setattr(agent, "run_sql_agent", _fail_if_called)

    result = agent.sql_agent_node({"messages": []})

    assert result["sql_result"] is None
    assert result["next_agent"] is None
    assert "didn't receive a question" in result["messages"][0].content


def test_sql_agent_node_routes_aggregate_questions_to_aggregate_pipeline(monkeypatch, fake_llm):
    monkeypatch.setattr(agent, "question_of", lambda state: "total portfolio value across all clients")
    monkeypatch.setattr(agent, "is_aggregate_question", lambda q: True)

    agg_result = _agg_result(rows=[{"total_value": 1000.0}])
    monkeypatch.setattr(agent, "run_aggregate_query_pipeline", lambda q: agg_result)

    called = {"run_sql_agent": False}
    def _fail_if_called(*a, **kw):
        called["run_sql_agent"] = True
        raise AssertionError("the single-client pipeline should not run for an aggregate question")
    monkeypatch.setattr(agent, "run_sql_agent", _fail_if_called)

    result = agent.sql_agent_node({"messages": []})

    assert called["run_sql_agent"] is False
    assert result["sql_result"] is None  # aggregate results aren't stored under sql_result -- confirm intentional
    assert result["next_agent"] is None
    assert result["messages"][0].content == fake_llm.next_response
    assert result["messages"][0].name == "sql_agent"


def test_sql_agent_node_aggregate_pipeline_exception_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(agent, "question_of", lambda state: "total portfolio value")
    monkeypatch.setattr(agent, "is_aggregate_question", lambda q: True)

    def _raise(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(agent, "run_aggregate_query_pipeline", _raise)

    result = agent.sql_agent_node({"messages": []})

    assert result["sql_result"] is None
    assert result["next_agent"] is None
    assert "issue computing" in result["messages"][0].content


def test_sql_agent_node_single_client_exception_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(agent, "question_of", lambda state: "show holdings for C001")
    monkeypatch.setattr(agent, "is_aggregate_question", lambda q: False)

    def _raise(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(agent, "run_sql_agent", _raise)

    result = agent.sql_agent_node({"messages": []})

    assert result["sql_result"] is None
    assert result["next_agent"] is None
    assert "issue retrieving the portfolio data" in result["messages"][0].content


@pytest.mark.parametrize("question,has_holdings,has_error,expected_next_agent", [
    ("show holdings for C001", True, False, None),                        # plain listing, no research intent
    ("what's your outlook on these holdings", True, False, "rag_agent"),  # research intent + real holdings
    ("what's your outlook on these holdings", False, False, None),        # research intent but no holdings
    ("what's your outlook on these holdings", True, True, None),          # research intent but result errored
])
def test_sql_agent_node_research_followup_routing(monkeypatch, question, has_holdings, has_error, expected_next_agent):
    monkeypatch.setattr(agent, "question_of", lambda state: question)
    monkeypatch.setattr(agent, "is_aggregate_question", lambda q: False)

    pipeline_result = _pipeline_result(
        holdings=([_holding()] if has_holdings else []) if not has_error else None,
        error="some error" if has_error else None,
    )
    monkeypatch.setattr(agent, "run_sql_agent", lambda q: pipeline_result)

    result = agent.sql_agent_node({"messages": []})

    assert result["next_agent"] == expected_next_agent
    assert result["sql_result"] is pipeline_result.result