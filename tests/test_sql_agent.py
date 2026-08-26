"""
Test suite for agents/sql_agent.py

This file only orchestrates: rewrite_question -> (stop if ambiguous) ->
run_query_pipeline -> synthesize_answer. Every test here mocks those three
boundaries directly (monkeypatching the names as imported into sql_agent's
own module namespace) rather than exercising the real DB, LLM, or
sql_tools pipeline -- those get their own test files.

Assumes this module lives at agents/sql_agent.py (per its own docstring's
reference to agents/rag_agent.py). Adjust the import below if it's
actually elsewhere (e.g. tools/sql_agent.py).
"""

import pytest
from schemas import (
    ClientHolding, RewrittenQuery, SQLQueryResult, GeneratedQuery, PipelineResult,
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
    # the raw error is passed as an internal note for the LLM to reason with...
    assert "sqlite3.OperationalError" in user_msg
    # ...but both the per-turn instruction and the system prompt tell it
    # not to repeat that to the advisor
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
    """clarification_question is checked first in synthesize_answer,
    ahead of sql_result -- so if both are ever passed together, fetched
    holdings are silently dropped from the prompt. Documenting that
    precedence explicitly since run_sql_agent's happy path can pass both
    at once (see test_run_sql_agent_generation_level_clarification_*
    below)."""
    sql_result = SQLQueryResult(client_id="C001", holdings=[_holding()], row_count=1,
                                 query_used="SELECT ...", error=None)
    agent.synthesize_answer("show holdings for c001", sql_result,
                             clarification_question="matched two clients")
    user_msg = fake_llm.last_messages[1]["content"]
    assert "matched two clients" in user_msg
    assert "Saudi Aramco" not in user_msg


# ---------------------------------------------------------------------------
# run_sql_agent — orchestration
# ---------------------------------------------------------------------------

def test_run_sql_agent_short_circuits_on_ambiguous_rewrite(monkeypatch, fake_llm):
    """When rewrite_question flags ambiguity, run_query_pipeline must
    never be called -- no SQL should be generated for a question the
    rewrite step hasn't resolved yet."""
    rewrite = _rewrite(needs_clarification=True,
                        ambiguous=["'c01' is ambiguous between: C001, C010"])
    monkeypatch.setattr(agent, "rewrite_question", lambda q: rewrite)

    def _fail_if_called(*a, **kw):
        raise AssertionError("run_query_pipeline should not be called when rewrite needs clarification")
    monkeypatch.setattr(agent, "run_query_pipeline", _fail_if_called)

    result = agent.run_sql_agent("show holdings for c01")

    assert result.needs_clarification is True
    assert result.clarification_question == "'c01' is ambiguous between: C001, C010"
    assert result.question == "show holdings for c01"          # original, not rewritten
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

    # the SQL pipeline only ever sees the rewritten text
    assert captured["question"] == rewrite.rewritten

    # but the original question is preserved on the final result and is
    # what's handed to the answer LLM, per the module's stated guarantee
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
    """rewrite_question isn't the only source of ambiguity: generate_query
    (inside run_query_pipeline) can independently set needs_clarification
    -- e.g. the rewritten question still doesn't uniquely identify one
    client. Confirming the actual behavior here: even when holdings were
    successfully fetched, synthesize_answer's clarification-first check
    means those holdings are silently dropped from the final prompt. If
    that's not the intended product behavior, it's worth deciding now
    rather than discovering it from a confusing advisor-facing answer
    later."""
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
    """Sanity check on the LangChain @tool wrapper -- it should be a thin
    pass-through, not reimplement any orchestration logic itself."""
    monkeypatch.setattr(agent, "rewrite_question", lambda q: _rewrite())
    monkeypatch.setattr(agent, "run_query_pipeline", lambda q, **kw: _pipeline_result(holdings=[]))

    result = agent.query_client_portfolio.invoke({"question": "show holdings for c001"})

    assert result.question == "show holdings for c001"
    assert result.answer == fake_llm.next_response