"""Integration test: real DB, real validator, real executor -- only the LLM
boundary is faked, since these tests should catch real SQL/schema mismatches
without needing an API key.

Requires the mock DB to exist (tools/db.py's agent_engine) and be seeded from
data/mock/generate_data.py. If client IDs referenced here don't exist in your
seeded DB, adjust to a client_id you know is present.
"""

import pytest
from langchain_core.messages import HumanMessage

from schemas import GeneratedQuery, RewrittenQuery
from tools import query_rewriter as qr
from tools import sql_tools as st
from agents import sql_agent as sa

KNOWN_CLIENT_ID = "C001"  # adjust if your seeded DB differs


class FakeStructuredLLM:
    """Returns a fixed structured object regardless of the prompt -- these
    tests aim the model's output deliberately, they don't test what a real
    model would produce (that's the eval harness's job)."""
    def __init__(self, response):
        self._response = response

    def invoke(self, messages):
        return self._response


@pytest.fixture(autouse=True)
def clear_caches():
    qr._raw_grounding.cache_clear()
    qr._rewriter_llm.cache_clear()
    st._generator_llm.cache_clear()
    yield
    qr._raw_grounding.cache_clear()
    qr._rewriter_llm.cache_clear()
    st._generator_llm.cache_clear()


class TestSingleClientLookupIntegration:
    def test_client_holdings_end_to_end(self, monkeypatch):
        monkeypatch.setattr(qr, "_rewriter_llm", lambda: FakeStructuredLLM(
            RewrittenQuery(original="", rewritten="this client's holdings")))
        monkeypatch.setattr(st, "_generator_llm", lambda: FakeStructuredLLM(
    GeneratedQuery(
        sql=("SELECT holdings.ticker AS symbol, name_en, name_ar, sector, asset_class, "
             "quantity, market_value FROM holdings JOIN instruments "
             "ON holdings.ticker = instruments.ticker "
             "WHERE holdings.client_id = :client_id LIMIT 50"),
        confidence=0.9,
    )))
        monkeypatch.setattr(sa, "answer", lambda question, result: "Summary stub.")

        result = sa.sql_agent_node({
            "messages": [HumanMessage(content="this client's holdings")],
            "client_id": KNOWN_CLIENT_ID,
        })

        assert result["sql_result"] is not None
        assert result["sql_result"].error is None
        assert result["sql_result"].client_profile is not None
        assert result["sql_result"].client_profile.client_id == KNOWN_CLIENT_ID
        # Real DB, real join: every row should genuinely be a ClientHolding.
        for h in result["sql_result"].holdings:
            assert h.symbol and h.quantity >= 0


class TestAggregateIntegration:
    def test_average_market_value_by_risk_profile(self, monkeypatch):
        monkeypatch.setattr(qr, "_rewriter_llm", lambda: FakeStructuredLLM(
            RewrittenQuery(original="", rewritten="average market value by risk profile")))
        monkeypatch.setattr(st, "_generator_llm", lambda: FakeStructuredLLM(
            GeneratedQuery(
                sql=("SELECT risk_profile, AVG(market_value) AS avg_value FROM clients "
                     "JOIN holdings ON clients.client_id = holdings.client_id "
                     "GROUP BY risk_profile LIMIT 10"),
                confidence=0.9,
            )))
        monkeypatch.setattr(sa, "answer", lambda question, result: "Summary stub.")

        result = sa.sql_agent_node({
            "messages": [HumanMessage(content="average market value by risk profile")],
            "client_id": None,
        })

        assert result["sql_result"].error is None
        assert result["sql_result"].holdings == []
        assert len(result["sql_result"].rows) > 0
        assert result["sql_result"].client_profile is None  # unscoped -- no profile fetch


class TestValidatorCatchesRealBadSql:
    def test_generated_dml_is_blocked_before_touching_the_db(self, monkeypatch):
        monkeypatch.setattr(qr, "_rewriter_llm", lambda: FakeStructuredLLM(
            RewrittenQuery(original="", rewritten="delete everything")))
        # Model misbehaves; validator's job is to stop this regardless of
        # what a real model might someday be coaxed into producing.
        call_count = {"n": 0}

        def bad_then_stuck(*a, **k):
            call_count["n"] += 1
            return FakeStructuredLLM(
                GeneratedQuery(sql="DELETE FROM holdings LIMIT 1", confidence=0.5)
            ).invoke(a)
        monkeypatch.setattr(st, "_generator_llm",
                             lambda: type("_", (), {"invoke": staticmethod(bad_then_stuck)})())

        result = sa.sql_agent_node({
            "messages": [HumanMessage(content="delete everything")],
            "client_id": None,
        })

        assert result["sql_result"] is None
        assert "safe query" in result["messages"][0].content.lower()
        # Repair budget was spent retrying, not a single silent pass-through.
        assert call_count["n"] == st.MAX_REPAIR_ATTEMPTS + 1