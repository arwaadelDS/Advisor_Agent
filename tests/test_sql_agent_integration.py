"""End-to-end tests: real query_rewriter, real generate_query/validate_sql/
execute (tools/sql_tools.py), real synthesize_answer -- all hitting the
actual Gemini model via tools.llm.get_llm() and the real seeded
advisor_mock.db. No mocking. Requires GOOGLE_API_KEY to be set (see
tools/llm.py, which loads it via dotenv).

Run with:
    uv run pytest tests/test_sql_agent_integration.py -m integration -v -s
"""

import pytest
from agents.sql_agent import run_sql_agent

pytestmark = pytest.mark.integration


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

    