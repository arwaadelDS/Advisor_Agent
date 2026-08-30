"""Unit tests for tools/sql_tools.py.

Each layer is tested against its own boundary: validate_sql and _shape_rows
need no mocking at all (pure functions); generate_query mocks _generator_llm;
execute_sql/fetch_client_profile mock agent_engine; run_query_pipeline mocks
generate_query/validate_sql/execute_sql/fetch_client_profile directly, since
its own contract is "wire these four together with a repair budget," not
"re-verify what each of them does alone."
"""

import pytest

from schemas import ClientHolding, ClientProfile, GeneratedQuery, PipelineResult, SQLQueryResult
from tools import sql_tools as st


# ---------------------------------------------------------------------------
# validate_sql
# ---------------------------------------------------------------------------

class TestValidateSql:
    def test_valid_client_scoped_query_passes(self):
        sql = "SELECT holdings.ticker AS symbol, name_en, name_ar, sector, asset_class, quantity, market_value " \
              "FROM holdings JOIN instruments ON holdings.ticker = instruments.ticker " \
              "WHERE holdings.client_id = :client_id LIMIT 50"
        assert st.validate_sql(sql, client_id="C001") == []

    def test_valid_unscoped_aggregate_passes(self):
        sql = "SELECT risk_profile, AVG(market_value) FROM clients " \
              "JOIN holdings ON clients.client_id = holdings.client_id " \
              "GROUP BY risk_profile LIMIT 10"
        assert st.validate_sql(sql, client_id=None) == []

    def test_aggregate_query_with_client_selected_but_not_filtered_passes(self):
        """The core fix: a client_id in session no longer forces a filter --
        an aggregate question asked while a client happens to be selected
        must be allowed to legitimately span all clients."""
        sql = "SELECT COUNT(*) FROM clients WHERE risk_profile = 'Aggressive' LIMIT 10"
        assert st.validate_sql(sql, client_id="C001") == []

    def test_client_scoped_query_with_filter_still_passes(self):
        """The other legitimate reading with a client selected -- filtering
        is still allowed, just no longer required."""
        sql = "SELECT * FROM holdings WHERE client_id = :client_id LIMIT 10"
        assert st.validate_sql(sql, client_id="C001") == []

    def test_unscoped_query_with_client_filter_rejected(self):
        errors = st.validate_sql(
            "SELECT * FROM holdings WHERE client_id = :client_id LIMIT 10", client_id=None
        )
        assert any("must not reference :client_id" in e for e in errors)

    def test_non_select_rejected(self):
        errors = st.validate_sql("DELETE FROM holdings LIMIT 1", client_id=None)
        assert any("Only SELECT" in e for e in errors)

    def test_multi_statement_rejected(self):
        errors = st.validate_sql("SELECT * FROM clients LIMIT 1; DROP TABLE clients", client_id=None)
        assert any("Multi-statement" in e for e in errors)

    def test_non_allowlisted_table_rejected(self):
        errors = st.validate_sql("SELECT * FROM secret_table LIMIT 10", client_id=None)
        assert any("non-allowlisted tables" in e for e in errors)

    def test_missing_limit_rejected(self):
        errors = st.validate_sql("SELECT * FROM clients", client_id=None)
        assert any("LIMIT is required" in e for e in errors)

    def test_unparseable_sql_rejected(self):
        errors = st.validate_sql("SELEKT nonsense FROM (((", client_id=None)
        assert any("parse error" in e for e in errors)

    def test_unknown_qualified_column_rejected(self):
        errors = st.validate_sql("SELECT clients.ssn FROM clients LIMIT 10", client_id=None)
        assert any("Unknown column clients.ssn" in e for e in errors)

    def test_ambiguous_unqualified_column_in_join_rejected(self):
        """The exact bug the integration test hit: 'ticker' exists on both
        joined tables and is left unqualified."""
        sql = "SELECT ticker, name_en FROM holdings JOIN instruments " \
              "ON holdings.ticker = instruments.ticker LIMIT 10"
        errors = st.validate_sql(sql, client_id=None)
        assert any("Ambiguous unqualified columns" in e for e in errors)
        assert any("ticker" in e for e in errors)

    def test_qualified_column_in_join_not_flagged_ambiguous(self):
        sql = "SELECT holdings.ticker, name_en FROM holdings JOIN instruments " \
              "ON holdings.ticker = instruments.ticker LIMIT 10"
        errors = st.validate_sql(sql, client_id=None)
        assert not any("Ambiguous" in e for e in errors)

    def test_single_table_query_never_flagged_ambiguous(self):
        """Only relevant once two+ tables are joined -- a single-table query
        can't have a genuinely ambiguous column."""
        errors = st.validate_sql("SELECT ticker FROM instruments LIMIT 10", client_id=None)
        assert not any("Ambiguous" in e for e in errors)

    def test_multiple_errors_all_reported(self):
        errors = st.validate_sql("DELETE FROM secret_table", client_id=None)
        assert len(errors) >= 2

    def test_group_by_client_identity_still_rejected(self):
        """Regression: GROUP BY name still discloses per-client data, a
        SUM() wrapper does not make it safe."""
        sql = "SELECT name, SUM(market_value) AS total FROM clients " \
              "JOIN holdings ON clients.client_id = holdings.client_id " \
              "GROUP BY name LIMIT 100"
        errors = st.validate_sql(sql, client_id=None)
        assert any("must be aggregated" in e for e in errors)

    def test_select_star_on_clients_rejected_unscoped(self):
        """Regression: SELECT * bypassed detection entirely -- sqlglot
        represents it as exp.Star, not exp.Column."""
        errors = st.validate_sql("SELECT * FROM clients LIMIT 100", client_id=None)
        assert any("must be aggregated" in e for e in errors)

    def test_group_by_sector_not_client_identity_still_passes(self):
        """Confirms the fix isn't overbroad: grouping by a non-identity
        column is still legitimate aggregation."""
        sql = "SELECT risk_profile, AVG(market_value) FROM clients " \
              "JOIN holdings ON clients.client_id = holdings.client_id " \
              "GROUP BY risk_profile LIMIT 10"
        assert st.validate_sql(sql, client_id=None) == []

    def test_ddl_via_alternate_casing_still_rejected(self):
        """Regression: the old check was regex-on-raw-text; this confirms
        detection now survives things a naive text scan could miss."""
        errors = st.validate_sql("dRoP table clients", client_id=None)
        assert any("Only SELECT" in e for e in errors)


# ---------------------------------------------------------------------------
# _shape_rows
# ---------------------------------------------------------------------------

class TestShapeRows:
    def test_holding_shaped_columns_become_client_holdings(self):
        columns = ["symbol", "name_en", "name_ar", "sector", "asset_class", "quantity", "market_value"]
        rows = [("2010", "SABIC", "سابك", "Petrochemicals", "Equity", 1000, 45000.0)]

        holdings, generic_rows = st._shape_rows(columns, rows)

        assert generic_rows == []
        assert len(holdings) == 1
        assert isinstance(holdings[0], ClientHolding)
        assert holdings[0].symbol == "2010"
        assert holdings[0].market_value == 45000.0

    def test_holding_shaped_columns_in_different_order_still_match(self):
        columns = ["market_value", "symbol", "quantity", "name_en", "sector", "asset_class", "name_ar"]
        rows = [(45000.0, "2010", 1000, "SABIC", "Petrochemicals", "Equity", "سابك")]

        holdings, generic_rows = st._shape_rows(columns, rows)

        assert generic_rows == []
        assert holdings[0].symbol == "2010"

    def test_non_holding_shape_becomes_generic_rows(self):
        columns = ["risk_profile", "AVG(market_value)"]
        rows = [("Aggressive", 52000.0), ("Conservative", 31000.0)]

        holdings, generic_rows = st._shape_rows(columns, rows)

        assert holdings == []
        assert generic_rows == [
            {"risk_profile": "Aggressive", "AVG(market_value)": 52000.0},
            {"risk_profile": "Conservative", "AVG(market_value)": 31000.0},
        ]

    def test_empty_rows_returns_empty_generic(self):
        holdings, generic_rows = st._shape_rows(["risk_profile"], [])
        assert holdings == []
        assert generic_rows == []


# ---------------------------------------------------------------------------
# generate_query
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, response):
        self._response = response

    def invoke(self, messages):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.fixture(autouse=True)
def clear_llm_cache():
    st._generator_llm.cache_clear()
    yield
    st._generator_llm.cache_clear()


def _patch_generator_llm(monkeypatch, response):
    monkeypatch.setattr(st, "_generator_llm", lambda: FakeLLM(response))


class TestGenerateQuery:
    def test_empty_question_raises(self):
        with pytest.raises(st.SQLToolError, match="cannot be empty"):
            st.generate_query("   ")

    def test_successful_generation_returns_generated_query(self, monkeypatch):
        fake = GeneratedQuery(sql="SELECT * FROM clients LIMIT 10", confidence=0.9)
        _patch_generator_llm(monkeypatch, fake)

        result = st.generate_query("list all clients")

        assert result.sql == "SELECT * FROM clients LIMIT 10"

    def test_non_generated_query_return_raises(self, monkeypatch):
        _patch_generator_llm(monkeypatch, "not a GeneratedQuery")
        with pytest.raises(st.SQLToolError, match="did not return a GeneratedQuery"):
            st.generate_query("some question")

    def test_llm_exception_wrapped(self, monkeypatch):
        _patch_generator_llm(monkeypatch, RuntimeError("quota exceeded"))
        with pytest.raises(st.SQLToolError, match="SQL generation failed"):
            st.generate_query("some question")


# ---------------------------------------------------------------------------
# execute_sql / fetch_client_profile
# ---------------------------------------------------------------------------

class FakeResult:
    def __init__(self, columns, rows):
        self._columns = columns
        self._rows = rows

    def fetchmany(self, n):
        return self._rows[:n]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def keys(self):
        return self._columns


class FakeConnection:
    def __init__(self, columns=None, rows=None, raise_on_execute=None):
        self._columns = columns or []
        self._rows = rows or []
        self._raise_on_execute = raise_on_execute

    def execute(self, stmt, params=None):
        if self._raise_on_execute:
            raise self._raise_on_execute
        return FakeResult(self._columns, self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeEngine:
    def __init__(self, connection=None, raise_on_connect=None):
        self._connection = connection
        self._raise_on_connect = raise_on_connect

    def connect(self):
        if self._raise_on_connect:
            raise self._raise_on_connect
        return self._connection


class TestExecuteSql:
    def test_holdings_shaped_success(self, monkeypatch):
        columns = ["symbol", "name_en", "name_ar", "sector", "asset_class", "quantity", "market_value"]
        rows = [("2010", "SABIC", "سابك", "Petrochemicals", "Equity", 1000, 45000.0)]
        monkeypatch.setattr(st, "agent_engine", FakeEngine(FakeConnection(columns, rows)))

        result = st.execute_sql("SELECT ... LIMIT 50", client_id="C001")

        assert result.error is None
        assert result.client_id == "C001"
        assert result.row_count == 1
        assert len(result.holdings) == 1
        assert result.rows == []

    def test_aggregate_shaped_success(self, monkeypatch):
        columns = ["risk_profile", "AVG(market_value)"]
        rows = [("Aggressive", 52000.0)]
        monkeypatch.setattr(st, "agent_engine", FakeEngine(FakeConnection(columns, rows)))

        result = st.execute_sql("SELECT ... LIMIT 10", client_id=None)

        assert result.error is None
        assert result.client_id is None
        assert result.holdings == []
        assert result.rows == [{"risk_profile": "Aggressive", "AVG(market_value)": 52000.0}]

    def test_db_error_returns_result_with_error_not_raise(self, monkeypatch):
        monkeypatch.setattr(
            st, "agent_engine",
            FakeEngine(FakeConnection(raise_on_execute=RuntimeError("no such column: foo"))),
        )

        result = st.execute_sql("SELECT foo FROM clients LIMIT 10", client_id=None)

        assert result.error is not None
        assert "no such column" in result.error
        assert result.holdings == []
        assert result.rows == []


class TestFetchClientProfile:
    def test_found_client_returns_profile(self, monkeypatch):
        row = ("C001", "Faisal Al-Otaibi", "Aggressive", "HNW")
        conn = FakeConnection(rows=[row])
        monkeypatch.setattr(st, "agent_engine", FakeEngine(conn))

        profile = st.fetch_client_profile("C001")

        assert profile == ClientProfile(
            client_id="C001", name="Faisal Al-Otaibi", risk_profile="Aggressive", aum_tier="HNW"
        )

    def test_unknown_client_returns_none(self, monkeypatch):
        conn = FakeConnection(rows=[])
        monkeypatch.setattr(st, "agent_engine", FakeEngine(conn))

        assert st.fetch_client_profile("C999") is None

    def test_db_error_raises_sql_tool_error(self, monkeypatch):
        monkeypatch.setattr(
            st, "agent_engine",
            FakeEngine(raise_on_connect=RuntimeError("db locked")),
        )
        with pytest.raises(st.SQLToolError, match="client profile lookup failed"):
            st.fetch_client_profile("C001")


# ---------------------------------------------------------------------------
# run_query_pipeline -- mocks the four layers it wires together
# ---------------------------------------------------------------------------

class TestRunQueryPipeline:
    def test_aggregate_question_with_client_selected_is_not_forced_to_scope(self, monkeypatch):
        gen = GeneratedQuery(
            sql="SELECT COUNT(*) AS n FROM clients WHERE risk_profile = 'Aggressive' LIMIT 10",
            confidence=0.9,
        )
        monkeypatch.setattr(st, "generate_query", lambda *a, **k: gen)
        monkeypatch.setattr(
            st, "execute_sql",
            lambda sql, client_id: SQLQueryResult(client_id=client_id, rows=[{"n": 4}], row_count=1),
        )

        outcome = st.run_query_pipeline("how many clients have Aggressive risk profile?", client_id="C001")

        assert outcome.validation_error is None
        assert outcome.result.rows == [{"n": 4}]

    def test_needs_clarification_short_circuits_before_validation(self, monkeypatch):
        gen = GeneratedQuery(
            sql="", confidence=0.2, needs_clarification=True,
            clarification_question="Which time period do you mean?",
        )
        monkeypatch.setattr(st, "generate_query", lambda *a, **k: gen)
        monkeypatch.setattr(st, "validate_sql", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("validate_sql should not be called")
        ))

        outcome = st.run_query_pipeline("some vague question")

        assert outcome.needs_clarification is True
        assert outcome.clarification_question == "Which time period do you mean?"
        assert outcome.result is None

    def test_successful_first_attempt_no_repair(self, monkeypatch):
        gen = GeneratedQuery(sql="SELECT * FROM clients LIMIT 10", confidence=0.9)
        monkeypatch.setattr(st, "generate_query", lambda *a, **k: gen)
        monkeypatch.setattr(st, "validate_sql", lambda *a, **k: [])
        monkeypatch.setattr(
            st, "execute_sql",
            lambda *a, **k: SQLQueryResult(client_id=None, rows=[{"n": 18}], row_count=1),
        )

        outcome = st.run_query_pipeline("how many clients")

        assert outcome.repair_attempts == 0
        assert outcome.validation_error is None
        assert outcome.result.rows == [{"n": 18}]

    def test_validation_failure_retries_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def fake_generate(question, client_id=None, prior_error=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return GeneratedQuery(sql="SELECT bogus_col FROM clients LIMIT 10", confidence=0.5)
            return GeneratedQuery(sql="SELECT client_id FROM clients LIMIT 10", confidence=0.9)

        def fake_validate(sql, client_id):
            return ["Unknown column clients.bogus_col"] if "bogus_col" in sql else []

        monkeypatch.setattr(st, "generate_query", fake_generate)
        monkeypatch.setattr(st, "validate_sql", fake_validate)
        monkeypatch.setattr(
            st, "execute_sql",
            lambda *a, **k: SQLQueryResult(client_id=None, rows=[{"client_id": "C001"}], row_count=1),
        )

        outcome = st.run_query_pipeline("list client ids")

        assert calls["n"] == 2
        assert outcome.repair_attempts == 1
        assert outcome.validation_error is None
        assert outcome.result is not None

    def test_validation_failure_exhausts_repair_budget(self, monkeypatch):
        gen = GeneratedQuery(sql="SELECT bogus_col FROM clients LIMIT 10", confidence=0.5)
        monkeypatch.setattr(st, "generate_query", lambda *a, **k: gen)
        monkeypatch.setattr(st, "validate_sql", lambda *a, **k: ["Unknown column clients.bogus_col"])
        monkeypatch.setattr(
            st, "execute_sql",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("execute_sql should not run"))
        )

        outcome = st.run_query_pipeline("a persistently bad question")

        assert outcome.repair_attempts == st.MAX_REPAIR_ATTEMPTS
        assert outcome.validation_error == "Unknown column clients.bogus_col"
        assert outcome.result is None

    def test_execution_failure_retries_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def fake_execute(sql, client_id):
            calls["n"] += 1
            if calls["n"] == 1:
                return SQLQueryResult(client_id=client_id, error="no such column: foo")
            return SQLQueryResult(client_id=client_id, rows=[{"ok": True}], row_count=1)

        monkeypatch.setattr(st, "generate_query", lambda *a, **k: GeneratedQuery(sql="SELECT ... LIMIT 10", confidence=0.9))
        monkeypatch.setattr(st, "validate_sql", lambda *a, **k: [])
        monkeypatch.setattr(st, "execute_sql", fake_execute)

        outcome = st.run_query_pipeline("some question")

        assert calls["n"] == 2
        assert outcome.repair_attempts == 1
        assert outcome.result.error is None

    def test_client_profile_attached_only_on_client_scoped_success(self, monkeypatch):
        monkeypatch.setattr(st, "generate_query", lambda *a, **k: GeneratedQuery(
            sql="SELECT * FROM holdings WHERE client_id = :client_id LIMIT 10", confidence=0.9))
        monkeypatch.setattr(st, "validate_sql", lambda *a, **k: [])
        monkeypatch.setattr(
            st, "execute_sql",
            lambda *a, **k: SQLQueryResult(client_id="C001", holdings=[], row_count=0),
        )
        fake_profile = ClientProfile(client_id="C001", name="Faisal Al-Otaibi", risk_profile="Aggressive", aum_tier="HNW")
        monkeypatch.setattr(st, "fetch_client_profile", lambda client_id: fake_profile)

        outcome = st.run_query_pipeline("this client's holdings", client_id="C001")

        assert outcome.result.client_profile == fake_profile

    def test_client_profile_not_attached_when_unscoped(self, monkeypatch):
        monkeypatch.setattr(st, "generate_query", lambda *a, **k: GeneratedQuery(sql="SELECT ... LIMIT 10", confidence=0.9))
        monkeypatch.setattr(st, "validate_sql", lambda *a, **k: [])
        monkeypatch.setattr(
            st, "execute_sql",
            lambda *a, **k: SQLQueryResult(client_id=None, rows=[{"n": 18}], row_count=1),
        )
        monkeypatch.setattr(
            st, "fetch_client_profile",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called when unscoped"))
        )

        outcome = st.run_query_pipeline("how many clients total")

        assert outcome.result.client_profile is None

    def test_client_profile_not_attached_on_execution_error(self, monkeypatch):
        monkeypatch.setattr(st, "generate_query", lambda *a, **k: GeneratedQuery(sql="SELECT ... LIMIT 10", confidence=0.9))
        monkeypatch.setattr(st, "validate_sql", lambda *a, **k: [])
        monkeypatch.setattr(
            st, "execute_sql",
            lambda *a, **k: SQLQueryResult(client_id="C001", error="db down"),
        )
        monkeypatch.setattr(
            st, "fetch_client_profile",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called on execution error"))
        )

        outcome = st.run_query_pipeline("this client's holdings", client_id="C001")

        assert outcome.result.error == "db down"

    def test_client_profile_only_attached_when_sql_actually_filters_on_client(self, monkeypatch):
        """Regression: a client selected in session must not get their profile
        attached to a result that was intentionally unscoped (e.g. an aggregate
        across all clients) -- the answer LLM will otherwise narrate the
        aggregate as if it were about that one client."""
        monkeypatch.setattr(st, "generate_query", lambda *a, **k: GeneratedQuery(
            sql="SELECT SUM(market_value) AS total FROM holdings LIMIT 1", confidence=0.9))
        monkeypatch.setattr(st, "validate_sql", lambda *a, **k: [])
        monkeypatch.setattr(st, "execute_sql", lambda *a, **k: SQLQueryResult(
            client_id="C001", rows=[{"total": 42989894.0}], row_count=1))
        monkeypatch.setattr(st, "fetch_client_profile", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should not be called -- SQL never filtered on client_id")))

        outcome = st.run_query_pipeline("total portfolio value across all clients", client_id="C001")

        assert outcome.result.client_profile is None
    def test_group_by_client_identity_rejected_even_with_client_selected(self):
        """The actual injection gap: a client_id being selected in session
        must not disable the disclosure guard for a query that doesn't
        filter by that client. Regression for the has_client_filter fix."""
        sql = "SELECT name, SUM(market_value) AS total FROM clients " \
              "JOIN holdings ON clients.client_id = holdings.client_id " \
              "GROUP BY name LIMIT 100"
        errors = st.validate_sql(sql, client_id="C007")
        assert any("must be aggregated" in e for e in errors)

    def test_select_star_on_clients_rejected_even_with_client_selected(self):
        """Same gap, SELECT * form -- confirms the fix isn't narrowly tied
        to the GROUP BY case."""
        errors = st.validate_sql("SELECT * FROM clients LIMIT 100", client_id="C007")
        assert any("must be aggregated" in e for e in errors)

    def test_legitimate_scoped_query_with_identity_columns_still_passes(self):
        """The fix must not overcorrect: a query that DOES filter by the
        selected client, and legitimately returns that one client's own
        name/id, is not a disclosure -- has_client_filter should let it
        through same as before."""
        sql = "SELECT name, client_id FROM clients WHERE client_id = :client_id LIMIT 1"
        assert st.validate_sql(sql, client_id="C007") == []