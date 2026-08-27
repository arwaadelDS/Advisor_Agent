"""
Test suite for tools/aggregate_sql_tools.py

is_aggregate_question is a pure regex check, tested directly.
generate_aggregate_query and run_aggregate_query_pipeline are tested with
generate_aggregate_query/validate_sql/_execute mocked at the module
boundary (as imported into aggregate_sql_tools' own namespace) -- same
approach as tests/test_sql_tools.py for the single-client pipeline. No
real LLM or DB calls.

ASSUMPTION: schemas.py's actual AggregateQuery/AggregateQueryResult
definitions weren't available to verify against -- inferred from usage in
aggregate_sql_tools.py, mirroring GeneratedQuery/SQLQueryResult's shape:

    class AggregateQuery(BaseModel):
        sql: str
        confidence: float
        needs_clarification: bool = False
        clarification_question: str | None = None

    class AggregateQueryResult(BaseModel):
        rows: list[dict] = Field(default_factory=list)
        row_count: int = 0
        query_used: str | None = None
        error: str | None = None
        needs_clarification: bool = False
        clarification_question: str | None = None

If the real fields differ, the fixtures below need matching adjustments.
"""

import pytest
from schemas import AggregateQuery, AggregateQueryResult
from tools import aggregate_sql_tools as agg


# ---------------------------------------------------------------------------
# is_aggregate_question
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "what is the total portfolio value across all clients",
    "how many clients are Aggressive risk",
    "which sector has the most holdings",
    "compare average holding value by sector",
    "give me a breakdown by risk profile",
    "what's the highest value holding overall",
    "count of HNW clients",
    "C009's total market value",  # scoped to one client, still an aggregate SHAPE
])
def test_is_aggregate_question_matches_aggregate_intent(question):
    assert agg.is_aggregate_question(question) is True


@pytest.mark.parametrize("question", [
    "show holdings for client C001",
    "what does client C010 hold",
    "list portfolio holdings for C018",
    "shwo hodings for cleint c007",
])
def test_is_aggregate_question_does_not_match_single_client_listing(question):
    assert agg.is_aggregate_question(question) is False


# ---------------------------------------------------------------------------
# generate_aggregate_query -- prompt construction, LLM mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_structured_llm(monkeypatch):
    class FakeStructuredLLM:
        def __init__(self):
            self.last_messages = None
            self.next_response = None

        def invoke(self, messages):
            self.last_messages = messages
            return self.next_response

    class FakeLLM:
        def __init__(self, structured):
            self._structured = structured

        def with_structured_output(self, schema):
            return self._structured

    fake_structured = FakeStructuredLLM()
    monkeypatch.setattr(agg, "get_llm", lambda: FakeLLM(fake_structured))
    return fake_structured


def test_generate_aggregate_query_sends_schema_and_question(fake_structured_llm):
    expected = AggregateQuery(sql="SELECT 1", confidence=0.9)
    fake_structured_llm.next_response = expected

    result = agg.generate_aggregate_query("total portfolio value across all clients")

    assert result is expected
    system_msg = fake_structured_llm.last_messages[0]["content"]
    user_msg = fake_structured_llm.last_messages[1]["content"]
    assert "aggregate or cross-client" in system_msg
    assert user_msg == "total portfolio value across all clients"
    assert "previous attempt failed" not in system_msg.lower()


def test_generate_aggregate_query_includes_repair_suffix_on_retry(fake_structured_llm):
    fake_structured_llm.next_response = AggregateQuery(sql="SELECT 1", confidence=0.5)

    agg.generate_aggregate_query(
        "total portfolio value",
        prior_error=("SELECT bad", "query references disallowed table(s): sqlite_master"),
    )

    system_msg = fake_structured_llm.last_messages[0]["content"]
    assert "previous attempt failed" in system_msg.lower()
    assert "SELECT bad" in system_msg
    assert "sqlite_master" in system_msg


# ---------------------------------------------------------------------------
# run_aggregate_query_pipeline -- generate_aggregate_query / validate_sql /
# _execute mocked at the module boundary
# ---------------------------------------------------------------------------

VALID_AGG_SQL = "SELECT SUM(holdings.market_value) AS total_value FROM holdings"


def _agg_generated(sql=VALID_AGG_SQL, confidence=0.9, needs_clarification=False,
                    clarification_question=None):
    return AggregateQuery(sql=sql, confidence=confidence,
                           needs_clarification=needs_clarification,
                           clarification_question=clarification_question)


def test_pipeline_short_circuits_when_model_flags_needs_clarification(monkeypatch):
    """Regression test for the reintroduced version of the bug fixed in
    sql_tools.run_query_pipeline: a model-flagged ambiguity must stop
    before validate_sql/_execute ever run, so no real aggregate row can
    be attached to a question the model itself didn't resolve (e.g.
    'biggest' without a specified metric)."""
    gq = _agg_generated(needs_clarification=True, clarification_question="biggest by what metric?")
    monkeypatch.setattr(agg, "generate_aggregate_query", lambda q, prior_error=None: gq)

    def _fail_if_called(*a, **kw):
        raise AssertionError("validate_sql/_execute should not run when the model flagged ambiguity")
    monkeypatch.setattr(agg, "validate_sql", _fail_if_called)
    monkeypatch.setattr(agg, "_execute", _fail_if_called)

    result = agg.run_aggregate_query_pipeline("what's the biggest holding")

    assert result.needs_clarification is True
    assert result.clarification_question == "biggest by what metric?"
    assert result.rows == []
    assert result.error is None


def test_pipeline_happy_path_returns_rows(monkeypatch):
    monkeypatch.setattr(agg, "generate_aggregate_query", lambda q, prior_error=None: _agg_generated())
    monkeypatch.setattr(agg, "validate_sql", lambda sql, max_rows: (sql, None))
    rows = [{"total_value": 1234567.0}]
    monkeypatch.setattr(agg, "_execute", lambda sql, timeout_seconds: (rows, None))

    result = agg.run_aggregate_query_pipeline("total portfolio value across all clients")

    assert result.error is None
    assert result.rows == rows
    assert result.row_count == 1
    assert result.needs_clarification is False


def test_pipeline_validation_failure_exhausts_repair_then_returns_error(monkeypatch):
    calls = {"n": 0}
    def _fake_generate(q, prior_error=None):
        calls["n"] += 1
        return _agg_generated(sql="SELECT * FROM sqlite_master")
    monkeypatch.setattr(agg, "generate_aggregate_query", _fake_generate)
    monkeypatch.setattr(
        agg, "validate_sql",
        lambda sql, max_rows: (None, "query references disallowed table(s): sqlite_master"),
    )
    def _fail_if_called(*a, **kw):
        raise AssertionError("_execute should never run for a query that never validated")
    monkeypatch.setattr(agg, "_execute", _fail_if_called)

    result = agg.run_aggregate_query_pipeline("show everything", max_repair_attempts=2)

    assert result.error == "query references disallowed table(s): sqlite_master"
    assert result.rows == []
    assert calls["n"] == 3  # initial + 2 repairs


def test_pipeline_stops_early_on_repeated_identical_execution_error(monkeypatch):
    calls = {"n": 0}
    def _fake_generate(q, prior_error=None):
        calls["n"] += 1
        return _agg_generated()
    monkeypatch.setattr(agg, "generate_aggregate_query", _fake_generate)
    monkeypatch.setattr(agg, "validate_sql", lambda sql, max_rows: (sql, None))
    monkeypatch.setattr(agg, "_execute", lambda sql, timeout_seconds: (None, "same error every time"))

    result = agg.run_aggregate_query_pipeline("total portfolio value", max_repair_attempts=5)

    assert result.error == "same error every time"
    assert calls["n"] == 2  # first attempt sets last_error, second matches it and stops


def test_pipeline_passes_through_max_rows_and_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(agg, "generate_aggregate_query", lambda q, prior_error=None: _agg_generated())

    def _fake_validate(sql, max_rows):
        captured["max_rows"] = max_rows
        return sql, None
    monkeypatch.setattr(agg, "validate_sql", _fake_validate)

    def _fake_execute(sql, timeout_seconds):
        captured["timeout_seconds"] = timeout_seconds
        return [], None
    monkeypatch.setattr(agg, "_execute", _fake_execute)

    agg.run_aggregate_query_pipeline("total value", timeout_seconds=9, max_rows=42)

    assert captured["max_rows"] == 42
    assert captured["timeout_seconds"] == 9