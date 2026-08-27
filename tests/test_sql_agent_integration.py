"""End-to-end tests: real query_rewriter, real generate_query/validate_sql/
execute (tools/sql_tools.py), real generate_aggregate_query/
run_aggregate_query_pipeline (tools/aggregate_sql_tools.py), real
synthesize_answer/synthesize_aggregate_answer -- all hitting the actual
Gemini model via tools.llm.get_llm() and the real seeded advisor_mock.db.
No mocking. Requires GOOGLE_API_KEY to be set (see tools/llm.py, which
loads it via dotenv).

Run with:
    uv run pytest tests/test_sql_agent_integration.py -m integration -v -s
"""

import pytest
from agents.sql_agent import run_sql_agent, sql_agent_node
from tools.aggregate_sql_tools import run_aggregate_query_pipeline

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Single-client pipeline (unchanged by the aggregate addition)
# ---------------------------------------------------------------------------

def test_clean_specific_client_question():
    result = run_sql_agent("show holdings for client C001")
    print(f"\nanswer: {result.answer}")

    assert result.needs_clarification is False
    assert result.result is not None
    assert result.result.client_id == "C001"
    assert result.result.error is None
    assert result.answer
    assert "sql" not in result.answer.lower()


def test_typo_riddled_question_still_resolves():
    result = run_sql_agent("shwo hodings for cleint c007")
    print(f"\nrewritten: {result.rewrite.rewritten}")
    print(f"answer: {result.answer}")

    assert result.needs_clarification is False
    assert result.result.client_id == "C007"
    assert result.rewrite.corrections


def test_ambiguous_client_name_asks_for_clarification():
    result = run_sql_agent("show holdings for Al-Sudairi")
    print(f"\nanswer: {result.answer}")

    assert result.needs_clarification is True
    assert result.result is None
    assert result.answer


def test_nonexistent_client_returns_graceful_answer_not_raw_error():
    result = run_sql_agent("show holdings for client C999")
    print(f"\nanswer: {result.answer}")

    assert result.result.error is not None or result.result.row_count == 0
    assert result.answer
    assert "OperationalError" not in result.answer
    assert "Traceback" not in result.answer


def test_answer_mentions_actual_retrieved_symbol():
    """Sanity check that the answer is actually grounded in real retrieved
    data, not a generic non-answer -- checks for at least one real ticker
    or sector word from the seeded data appearing in the response."""
    result = run_sql_agent("show holdings for client C001")
    print(f"\nretrieved: {[h.symbol for h in result.result.holdings]}")
    print(f"answer: {result.answer}")

    retrieved_symbols = {h.symbol for h in result.result.holdings}
    retrieved_names = {h.name_en for h in result.result.holdings}
    mentions_something_real = any(
        sym in result.answer or name in result.answer
        for sym, name in zip(retrieved_symbols, retrieved_names)
    ) or bool(result.result.holdings) is False

    assert mentions_something_real


# ---------------------------------------------------------------------------
# Aggregate pipeline (new) -- run_aggregate_query_pipeline directly, real
# LLM + real DB, no mocking. Tested at the same level of directness as the
# single-client pipeline above, so a failure here points straight at
# tools/aggregate_sql_tools.py rather than through sql_agent_node's
# routing layer.
# ---------------------------------------------------------------------------

def test_aggregate_total_across_all_clients_executes_cleanly():
    result = run_aggregate_query_pipeline("what is the total portfolio value across all clients")
    print(f"\nquery_used: {result.query_used}")
    print(f"rows: {result.rows}")

    assert result.needs_clarification is False
    assert result.error is None
    assert result.rows
    assert result.row_count == len(result.rows)


def test_aggregate_client_scoped_total_still_uses_aggregate_shape():
    """Per aggregate_sql_tools' own docstring: a question about ONE
    client's total is still an aggregate-shaped answer (SUM/COUNT), not a
    per-holding row list."""
    result = run_aggregate_query_pipeline("what is C001's total market value")
    print(f"\nquery_used: {result.query_used}")
    print(f"rows: {result.rows}")

    assert result.error is None
    assert result.rows


def test_aggregate_ambiguous_metric_asks_for_clarification_without_executing():
    """'Biggest' without a specified metric (value? quantity?) should be
    flagged, not guessed -- and per the needs_clarification short-circuit
    fix, no rows should come back attached to it."""
    result = run_aggregate_query_pipeline("what's the biggest thing in the portfolio")
    print(f"\nneeds_clarification: {result.needs_clarification}")
    print(f"clarification_question: {result.clarification_question}")
    print(f"rows: {result.rows}")

    assert result.needs_clarification is True
    assert result.rows == []


def test_aggregate_query_never_touches_disallowed_tables():
    """Adversarial-flavored aggregate question -- confirms validate_sql's
    shared table allow-list is equally effective here as it is for the
    single-client pipeline."""
    result = run_aggregate_query_pipeline("select count(*) from sqlite_master")
    print(f"\nerror: {result.error}")
    print(f"needs_clarification: {result.needs_clarification}")

    # either the model refuses to target sqlite_master and something else
    # happens, or validate_sql rejects it -- either way, no clean rows
    # from a disallowed table should ever come back
    assert result.error is not None or result.needs_clarification is True
    assert result.rows == [] or result.error is not None


# ---------------------------------------------------------------------------
# sql_agent_node routing (new top-level entrypoint) -- confirms an
# aggregate-shaped question reaches the aggregate pipeline end-to-end
# through the actual node the graph calls, not just through
# run_aggregate_query_pipeline directly.
#
# ASSUMPTION: the state shape below (a "messages" list containing a
# HumanMessage) is a best guess, since agents/rag_agent.py's question_of
# implementation wasn't available to verify against. If this fails to
# extract the question correctly on first run, adjust the state
# construction to match question_of's actual contract.
# ---------------------------------------------------------------------------

def test_sql_agent_node_routes_aggregate_question_end_to_end():
    from langchain_core.messages import HumanMessage

    state = {"messages": [HumanMessage(content="how many clients are Aggressive risk")]}
    result = sql_agent_node(state)

    print(f"\nmessages: {[m.content for m in result['messages']]}")
    assert result["sql_result"] is None
    assert result["next_agent"] is None
    assert result["messages"][0].content


def test_sql_agent_node_routes_single_client_question_end_to_end():
    from langchain_core.messages import HumanMessage

    state = {"messages": [HumanMessage(content="show holdings for client C001")]}
    result = sql_agent_node(state)

    print(f"\nmessages: {[m.content for m in result['messages']]}")
    assert result["sql_result"] is not None
    assert result["sql_result"].client_id == "C001"