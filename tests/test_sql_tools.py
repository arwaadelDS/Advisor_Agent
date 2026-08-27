"""
Test suite for tools/sql_tools.py

Same three-tier structure as test_query_rewriter.py:

1. PURE UNIT — validate_sql, _extract_client_id, _map_rows need no DB or
   LLM at all. These run fast and pin down the exact regression cases
   already found by hand (the 'DROP the ball' literal false-positive,
   stacked-statement injection, the KeyError-class row-mapping bug).

2. GENERATION — generate_query / repair-loop behavior, using a fake LLM
   fixture that returns a queued sequence of GeneratedQuery responses, so
   multi-attempt repair-loop paths can be scripted deterministically.

3. INTEGRATION — run_query_pipeline against the real read-only DB via
   tools.db.agent_engine, with the LLM still faked (so tests are
   deterministic) but execution, row-mapping, and client_id extraction
   are real. Requires a live, seeded DB (same mock data as
   test_query_rewriter.py: clients C001-C018).
"""

import pytest
from tools import sql_tools as st
from schemas import GeneratedQuery, ClientHolding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeStructuredLLM:
    """Returned by FakeLLM.with_structured_output(...). Pops the next
    queued response each call so repair-loop sequences can be scripted;
    repeats the last response if the queue runs out."""

    def __init__(self, outer):
        self._outer = outer

    def invoke(self, messages):
        self._outer.last_messages = messages
        self._outer.call_count += 1
        if self._outer.responses:
            self._outer.last_response = self._outer.responses.pop(0)
        return self._outer.last_response


class FakeLLM:
    def __init__(self):
        self.responses: list[GeneratedQuery] = []
        self.last_response: GeneratedQuery | None = None
        self.last_messages = None
        self.call_count = 0

    def with_structured_output(self, schema):
        assert schema is GeneratedQuery
        return _FakeStructuredLLM(self)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(st, "get_llm", lambda: fake)
    return fake


# A known-good query matching the fixed required projection, for a real
# seeded client. Assumes C001 exists (mock generator always creates
# C001-C018) and has at least one holding — if C001 happens to have zero
# holdings for a given seed, swap to another C0xx.
GOOD_SQL_C001 = (
    "SELECT holdings.ticker AS symbol, instruments.name_en AS name_en, "
    "holdings.quantity AS quantity, holdings.market_value AS market_value, "
    "instruments.sector AS sector "
    "FROM holdings "
    "JOIN clients ON holdings.client_id = clients.client_id "
    "JOIN instruments ON holdings.ticker = instruments.ticker "
    "WHERE clients.client_id = 'C001'"
)


# ---------------------------------------------------------------------------
# 1. PURE UNIT — validate_sql
# ---------------------------------------------------------------------------

def test_validate_sql_accepts_clean_select():
    safe_sql, error = st.validate_sql("SELECT * FROM clients")
    assert error is None
    assert "LIMIT 200" in safe_sql


def test_validate_sql_injects_limit_when_missing():
    safe_sql, error = st.validate_sql("SELECT * FROM clients", max_rows=50)
    assert error is None
    assert "LIMIT 50" in safe_sql


def test_validate_sql_clamps_oversized_limit():
    safe_sql, error = st.validate_sql("SELECT * FROM clients LIMIT 5000", max_rows=200)
    assert error is None
    assert "LIMIT 200" in safe_sql
    assert "5000" not in safe_sql


def test_validate_sql_keeps_limit_under_cap():
    safe_sql, error = st.validate_sql("SELECT * FROM clients LIMIT 10", max_rows=200)
    assert error is None
    assert "LIMIT 10" in safe_sql


def test_validate_sql_rejects_stacked_statements():
    """Regression: multi-statement injection via `; DROP TABLE ...`"""
    safe_sql, error = st.validate_sql("SELECT * FROM clients; DROP TABLE clients;")
    assert safe_sql is None
    assert "one statement" in error


@pytest.mark.parametrize("sql", [
    "DROP TABLE clients",
    "DELETE FROM clients",
    "UPDATE clients SET name = 'x'",
    "INSERT INTO clients VALUES (1,2,3,4)",
    "PRAGMA table_info(clients)",
    "ATTACH DATABASE '/etc/passwd' AS x",
])
def test_validate_sql_rejects_non_select_statements(sql):
    safe_sql, error = st.validate_sql(sql)
    assert safe_sql is None
    assert "SELECT" in error


def test_validate_sql_rejects_disallowed_table():
    safe_sql, error = st.validate_sql("SELECT * FROM secrets")
    assert safe_sql is None
    assert "secrets" in error


def test_validate_sql_regression_no_false_positive_on_keyword_in_literal():
    """Regression: a raw keyword-substring pre-filter would reject this
    on the word 'DROP' appearing inside a string literal, not SQL syntax.
    validate_sql must accept it since it's a structurally clean SELECT."""
    sql = "SELECT * FROM clients WHERE name = 'DROP the ball'"
    safe_sql, error = st.validate_sql(sql)
    assert error is None
    assert "DROP the ball" in safe_sql


def test_validate_sql_regression_no_false_positive_on_insert_in_literal():
    sql = "SELECT * FROM clients WHERE name = 'Insert Coin Al-Otaibi'"
    safe_sql, error = st.validate_sql(sql)
    assert error is None


def test_validate_sql_rejects_malformed_sql():
    safe_sql, error = st.validate_sql("SELECT FROM WHERE ***")
    assert safe_sql is None
    assert error is not None


# ---------------------------------------------------------------------------
# 1. PURE UNIT — _extract_client_id
# ---------------------------------------------------------------------------

def test_extract_client_id_simple_equality():
    sql = "SELECT * FROM clients WHERE clients.client_id = 'C007'"
    assert st._extract_client_id(sql) == "C007"


def test_extract_client_id_unqualified_column():
    sql = "SELECT * FROM clients WHERE client_id = 'C012'"
    assert st._extract_client_id(sql) == "C012"


def test_extract_client_id_from_full_join_query():
    assert st._extract_client_id(GOOD_SQL_C001) == "C001"


def test_extract_client_id_returns_none_when_absent():
    sql = "SELECT * FROM clients WHERE risk_profile = 'Aggressive'"
    assert st._extract_client_id(sql) is None


def test_extract_client_id_diagnostic_reversed_literal():
    """Diagnostic, not necessarily pass/fail depending on desired scope:
    a reversed equality ('C007' = client_id) currently is NOT matched
    since the extraction only checks Column-on-the-left. Prints the
    result so this gap is a visible, tracked decision rather than a
    silent gap."""
    sql = "SELECT * FROM clients WHERE 'C007' = clients.client_id"
    result = st._extract_client_id(sql)
    print(f"\nreversed-literal extraction result: {result!r} "
          f"(currently expected to be None — widen the EQ check if the "
          f"model is observed producing this form)")


# ---------------------------------------------------------------------------
# 1. PURE UNIT — _map_rows
# ---------------------------------------------------------------------------

def test_map_rows_happy_path():
    rows = [
        {"symbol": "2222", "name_en": "Saudi Aramco", "quantity": 1000,
         "market_value": 50000, "sector": "Energy"},
        {"symbol": "1120", "name_en": "Al Rajhi Bank", "quantity": 500,
         "market_value": 25000, "sector": "Banking"},
    ]
    holdings, error = st._map_rows(rows)
    assert error is None
    assert len(holdings) == 2
    assert isinstance(holdings[0], ClientHolding)
    assert holdings[0].symbol == "2222"


def test_map_rows_empty_result_is_not_an_error():
    holdings, error = st._map_rows([])
    assert error is None
    assert holdings == []


def test_map_rows_regression_missing_columns_reported_not_raised():
    """Regression: this is exactly the KeyError class of bug from the
    eval. A wrong/missing column must come back as an error string for
    the repair loop, never as an unhandled exception."""
    rows = [{"ticker": "2222", "name_en": "Saudi Aramco"}]  # wrong keys
    holdings, error = st._map_rows(rows)
    assert holdings is None
    assert error is not None
    assert "symbol" in error
    assert "quantity" in error


def test_map_rows_partial_missing_column():
    rows = [{"symbol": "2222", "name_en": "Saudi Aramco", "quantity": 1000,
              "market_value": 50000}]  # sector missing
    holdings, error = st._map_rows(rows)
    assert holdings is None
    assert "sector" in error


def test_map_rows_wrong_type_reported_as_error():
    rows = [{"symbol": "2222", "name_en": "Saudi Aramco",
              "quantity": "not-a-number", "market_value": 50000, "sector": "Energy"}]
    holdings, error = st._map_rows(rows)
    assert holdings is None
    assert error is not None


# ---------------------------------------------------------------------------
# 2. GENERATION — generate_query
# ---------------------------------------------------------------------------

def test_generate_query_sends_question_and_schema(fake_llm):
    fake_llm.responses = [GeneratedQuery(sql=GOOD_SQL_C001, confidence=0.95)]
    result = st.generate_query("show holdings for client C001")

    assert result.sql == GOOD_SQL_C001
    assert result.confidence == 0.95
    system_msg = fake_llm.last_messages[0]["content"]
    assert "clients(" in system_msg  # schema snapshot was injected
    assert fake_llm.last_messages[1]["content"] == "show holdings for client C001"


def test_generate_query_includes_prior_error_in_repair_prompt(fake_llm):
    fake_llm.responses = [GeneratedQuery(sql=GOOD_SQL_C001, confidence=0.9)]
    st.generate_query("show holdings for client C001",
                       prior_error=("SELECT * FROM secrets", "disallowed table"))

    system_msg = fake_llm.last_messages[0]["content"]
    assert "SELECT * FROM secrets" in system_msg
    assert "disallowed table" in system_msg


# ---------------------------------------------------------------------------
# 3. INTEGRATION — run_query_pipeline (real DB, faked LLM)
# ---------------------------------------------------------------------------

def test_pipeline_happy_path_real_db(fake_llm):
    fake_llm.responses = [GeneratedQuery(sql=GOOD_SQL_C001, confidence=0.95)]
    pipeline_result = st.run_query_pipeline("show holdings for client C001")

    assert pipeline_result.result is not None
    assert pipeline_result.result.error is None
    assert pipeline_result.result.client_id == "C001"
    assert pipeline_result.repair_attempts == 0
    assert pipeline_result.needs_clarification is False
    for h in pipeline_result.result.holdings:
        assert isinstance(h, ClientHolding)


def test_pipeline_never_executes_when_generation_flags_clarification(fake_llm):
    """An ambiguous generated query must never be executed against the DB
    -- the model's own uncertainty is reason enough not to touch real data,
    even read-only. The advisor sees the clarification question either way."""
    fake_llm.responses = [GeneratedQuery(
        sql=GOOD_SQL_C001, confidence=0.4,
        needs_clarification=True,
        clarification_question="Did you mean client C001 or C010?",
    )]
    pipeline_result = st.run_query_pipeline("show holdings for that client")

    assert pipeline_result.needs_clarification is True
    assert pipeline_result.clarification_question == "Did you mean client C001 or C010?"
    assert pipeline_result.result is None  # never executed


def test_pipeline_repairs_validation_error_real_db(fake_llm):
    """First attempt references a disallowed table; second attempt is
    the valid query. Should self-correct within budget."""
    fake_llm.responses = [
        GeneratedQuery(sql="SELECT * FROM secrets", confidence=0.5),
        GeneratedQuery(sql=GOOD_SQL_C001, confidence=0.9),
    ]
    pipeline_result = st.run_query_pipeline("show holdings for client C001")

    assert pipeline_result.result.error is None
    assert pipeline_result.repair_attempts == 1
    assert fake_llm.call_count == 2


def test_pipeline_repairs_row_mapping_error_real_db(fake_llm):
    """First attempt has wrong column aliases (the KeyError-class bug);
    second attempt is correct. Confirms the repair loop is actually wired
    to the row-mapping check, not just validation/execution errors."""
    bad_sql = (
        "SELECT holdings.ticker, instruments.name_en, holdings.quantity "
        "FROM holdings "
        "JOIN clients ON holdings.client_id = clients.client_id "
        "JOIN instruments ON holdings.ticker = instruments.ticker "
        "WHERE clients.client_id = 'C001'"
    )
    fake_llm.responses = [
        GeneratedQuery(sql=bad_sql, confidence=0.7),
        GeneratedQuery(sql=GOOD_SQL_C001, confidence=0.9),
    ]
    pipeline_result = st.run_query_pipeline("show holdings for client C001")

    assert pipeline_result.result.error is None
    assert pipeline_result.repair_attempts == 1


def test_pipeline_stops_early_on_repeated_identical_error_real_db(fake_llm):
    """If the model produces the exact same failing SQL twice in a row,
    the loop must stop after the first repair attempt rather than
    burning the full budget on a query that isn't converging."""
    bad_sql = "SELECT holdings.nonexistent_column FROM holdings WHERE holdings.client_id = 'C001'"
    fake_llm.responses = [
        GeneratedQuery(sql=bad_sql, confidence=0.5),
        GeneratedQuery(sql=bad_sql, confidence=0.5),  # identical failure again
        GeneratedQuery(sql=bad_sql, confidence=0.5),  # should never be reached
    ]
    pipeline_result = st.run_query_pipeline("show holdings for client C001", max_repair_attempts=3)

    assert pipeline_result.result.error is not None
    assert pipeline_result.repair_attempts == 1
    assert fake_llm.call_count == 2  # initial + one repair, not three


def test_pipeline_stops_at_max_attempts_with_differing_errors_real_db(fake_llm):
    """When errors keep changing (so the identical-error shortcut never
    fires), the loop must still respect the hard attempt cap."""
    fake_llm.responses = [
        GeneratedQuery(sql="SELECT holdings.bad_col_a FROM holdings WHERE holdings.client_id='C001'", confidence=0.5),
        GeneratedQuery(sql="SELECT holdings.bad_col_b FROM holdings WHERE holdings.client_id='C001'", confidence=0.5),
        GeneratedQuery(sql="SELECT holdings.bad_col_c FROM holdings WHERE holdings.client_id='C001'", confidence=0.5),
    ]
    pipeline_result = st.run_query_pipeline("show holdings for client C001", max_repair_attempts=2)

    assert pipeline_result.result.error is not None
    assert pipeline_result.repair_attempts == 2
    assert fake_llm.call_count == 3  # initial + 2 repairs, capped


def test_pipeline_extracts_correct_client_id_for_different_client_real_db(fake_llm):
    sql_c002 = GOOD_SQL_C001.replace("C001", "C002")
    fake_llm.responses = [GeneratedQuery(sql=sql_c002, confidence=0.9)]
    pipeline_result = st.run_query_pipeline("show holdings for client C002")

    assert pipeline_result.result.client_id == "C002"


# ---------------------------------------------------------------------------
# Schema grounding sanity check (real DB)
# ---------------------------------------------------------------------------

def test_schema_snapshot_contains_all_tables_real_db():
    snapshot = st._schema_snapshot()
    for table in ("clients", "instruments", "holdings"):
        assert f"{table}(" in snapshot


def test_schema_snapshot_is_cached():
    first = st._schema_snapshot()
    second = st._schema_snapshot()
    assert first is second  # lru_cache returns the same object, not just equal