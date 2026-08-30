"""Unit tests for agents/sql_agent.py.

Mocks rewrite_query, run_query_pipeline, and answer -- sql_agent_node's own
contract is "call these three in order, stop at the first failure/
clarification, degrade instead of raising." Each of those three already has
its own test file; this one only tests the wiring between them.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from schemas import ClientProfile, PipelineResult, RewrittenQuery, SQLQueryResult
from tools.query_rewriter import QueryRewriterError
from tools.sql_tools import SQLToolError
from agents import sql_agent as sa


def _state(question, client_id=None):
    return {"messages": [HumanMessage(content=question)], "client_id": client_id}


def _clean_rewrite(question, rewritten=None):
    return RewrittenQuery(original=question, rewritten=rewritten or question)


class TestQuestionOf:
    def test_reads_last_human_message(self):
        state = {"messages": [
            AIMessage(content="hi"),
            HumanMessage(content="how many clients are there"),
        ]}
        assert sa.question_of(state) == "how many clients are there"

    def test_no_messages_returns_empty(self):
        assert sa.question_of({"messages": []}) == ""
        assert sa.question_of({}) == ""


class TestSqlNode:
    def test_empty_question_returns_clarifying_message_no_calls(self, monkeypatch):
        monkeypatch.setattr(sa, "rewrite_query", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("rewrite_query should not be called")))
        result = sa.sql_agent_node(_state("   "))
        assert result["sql_result"] is None
        assert "rephrase" in result["messages"][0].content.lower()

    def test_rewrite_failure_degrades(self, monkeypatch):
        monkeypatch.setattr(sa, "rewrite_query",
                             lambda q, **k: (_ for _ in ()).throw(QueryRewriterError("db down")))
        result = sa.sql_agent_node(_state("this client's holdings", client_id="C001"))
        assert result["sql_result"] is None
        assert "try again" in result["messages"][0].content.lower()

    def test_rewrite_needs_clarification_short_circuits_before_pipeline(self, monkeypatch):
        rewritten = RewrittenQuery(
            original="q", rewritten="q",
            ambiguous=["شركة وهمية -> Fictional Corp"],
            needs_clarification=True,
        )
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: rewritten)
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("run_query_pipeline should not be called")))

        result = sa.sql_agent_node(_state("سؤال عن شركة وهمية"))

        assert result["sql_result"] is None
        assert "شركة وهمية" in result["messages"][0].content

    def test_pipeline_tool_error_degrades(self, monkeypatch):
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: _clean_rewrite(q))
        monkeypatch.setattr(sa, "run_query_pipeline",
                             lambda *a, **k: (_ for _ in ()).throw(SQLToolError("no model configured")))
        result = sa.sql_agent_node(_state("how many clients"))
        assert result["sql_result"] is None
        assert "try again" in result["messages"][0].content.lower()

    def test_pipeline_needs_clarification(self, monkeypatch):
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: _clean_rewrite(q))
        from schemas import GeneratedQuery
        pipeline = PipelineResult(
            question="q",
            generated=GeneratedQuery(
                sql="", confidence=0.1, needs_clarification=True,
                clarification_question="Which time period?"),
            needs_clarification=True,
            clarification_question="Which time period?",
        )
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: pipeline)
        result = sa.sql_agent_node(_state("recent performance"))
        assert result["sql_result"] is None
        assert result["messages"][0].content == "Which time period?"

    def test_validation_exhausted_degrades(self, monkeypatch):
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: _clean_rewrite(q))
        from schemas import GeneratedQuery
        pipeline = PipelineResult(
            question="q",
            generated=GeneratedQuery(sql="bad sql", confidence=0.3),
            validation_error="Unknown column clients.ssn",
            repair_attempts=2,
        )
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: pipeline)
        result = sa.sql_agent_node(_state("client ssn"))
        assert result["sql_result"] is None
        assert "safe query" in result["messages"][0].content.lower()

    def test_execution_error_after_repair_budget_keeps_errored_result(self, monkeypatch):
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: _clean_rewrite(q))
        from schemas import GeneratedQuery
        errored = SQLQueryResult(client_id=None, error="database is locked")
        pipeline = PipelineResult(
            question="q", generated=GeneratedQuery(sql="SELECT ...", confidence=0.9),
            result=errored, repair_attempts=2,
        )
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: pipeline)
        result = sa.sql_agent_node(_state("some question"))
        assert result["sql_result"] is errored
        assert "couldn't recover" in result["messages"][0].content.lower()

    def test_success_calls_answer_and_returns_message(self, monkeypatch):
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: _clean_rewrite(q))
        from schemas import GeneratedQuery
        good_result = SQLQueryResult(client_id=None, rows=[{"n": 18}], row_count=1)
        pipeline = PipelineResult(
            question="q", generated=GeneratedQuery(sql="SELECT COUNT(*) LIMIT 1", confidence=0.9),
            result=good_result,
        )
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: pipeline)
        monkeypatch.setattr(sa, "answer", lambda question, result: "There are 18 clients.")

        result = sa.sql_agent_node(_state("how many clients"))

        assert result["sql_result"] is good_result
        assert result["messages"][0].content == "There are 18 clients."
        assert result["next_agent"] is None

    def test_wants_research_with_holdings_hands_off_to_rag(self, monkeypatch):
        """Regression: the graph already has a sql_agent -> rag_agent edge
        (see graph/workflow.py) but sql_agent_node never took it -- this
        locks in that a question asking for research on retrieved holdings
        actually continues to rag_agent within the same turn."""
        rewritten = RewrittenQuery(
            original="q", rewritten="q", wants_research=True,
        )
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: rewritten)
        from schemas import ClientHolding, GeneratedQuery
        holding = ClientHolding(symbol="2280", name_en="Almarai", name_ar="المراعي",
                                 sector="Consumer", asset_class="Equity", quantity=100, market_value=5000.0)
        good_result = SQLQueryResult(client_id="C008", holdings=[holding])
        pipeline = PipelineResult(
            question="q", generated=GeneratedQuery(sql="SELECT ...", confidence=0.9),
            result=good_result,
        )
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: pipeline)
        monkeypatch.setattr(sa, "answer", lambda question, result: "Client holds Almarai.")

        result = sa.sql_agent_node(
            _state("What does this client hold, and what's the research saying?", client_id="C008")
        )

        assert result["next_agent"] == "rag_agent"

    def test_wants_research_with_no_holdings_does_not_hand_off(self, monkeypatch):
        """An aggregate result has nothing for rag_agent.holdings_of() to
        filter research by, so the handoff must not fire even when
        wants_research is true."""
        rewritten = RewrittenQuery(original="q", rewritten="q", wants_research=True)
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: rewritten)
        from schemas import GeneratedQuery
        good_result = SQLQueryResult(client_id=None, rows=[{"n": 18}])
        pipeline = PipelineResult(
            question="q", generated=GeneratedQuery(sql="SELECT ...", confidence=0.9),
            result=good_result,
        )
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: pipeline)
        monkeypatch.setattr(sa, "answer", lambda question, result: "18 clients.")

        result = sa.sql_agent_node(_state("How many clients, and any research on them?"))

        assert result["next_agent"] is None

    def test_no_research_wanted_does_not_hand_off(self, monkeypatch):
        """A plain data question, even with holdings present, must not
        trigger the RAG handoff -- wants_research is what gates it."""
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: _clean_rewrite(q))
        from schemas import GeneratedQuery
        good_result = SQLQueryResult(client_id="C001", rows=[{"n": 18}])
        pipeline = PipelineResult(
            question="q", generated=GeneratedQuery(sql="SELECT ...", confidence=0.9),
            result=good_result,
        )
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: pipeline)
        monkeypatch.setattr(sa, "answer", lambda question, result: "18 clients.")

        result = sa.sql_agent_node(_state("How many clients are there"))

        assert result["next_agent"] is None

    def test_answer_synthesis_failure_still_returns_sql_result(self, monkeypatch):
        """The data was retrieved successfully; only the prose summary
        failed -- the caller should still get the structured result, same
        principle as rag_node's own note about degrading only the summary."""
        monkeypatch.setattr(sa, "rewrite_query", lambda q, **k: _clean_rewrite(q))
        from schemas import GeneratedQuery
        good_result = SQLQueryResult(client_id="C001", holdings=[])
        pipeline = PipelineResult(
            question="q", generated=GeneratedQuery(sql="SELECT ...", confidence=0.9),
            result=good_result,
        )
        monkeypatch.setattr(sa, "run_query_pipeline", lambda *a, **k: pipeline)
        monkeypatch.setattr(sa, "answer",
                             lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rate limited")))

        result = sa.sql_agent_node(_state("this client's holdings", client_id="C001"))

        assert result["sql_result"] is good_result
        assert "below" in result["messages"][0].content.lower()


class TestFormatResult:
    def test_error_result(self):
        assert "failed" in sa.format_result(SQLQueryResult(client_id=None, error="boom")).lower()

    def test_empty_result(self):
        assert "No rows matched" in sa.format_result(SQLQueryResult(client_id=None))

    def test_profile_and_holdings_both_rendered(self):
        from schemas import ClientHolding
        result = SQLQueryResult(
            client_id="C001",
            client_profile=ClientProfile(client_id="C001", name="Faisal", risk_profile="Aggressive", aum_tier="HNW"),
            holdings=[ClientHolding(symbol="2010", name_en="SABIC", name_ar="سابك",
                                     sector="Petrochemicals", asset_class="Equity",
                                     quantity=1000, market_value=45000.0)],
        )
        text = sa.format_result(result)
        assert "Faisal" in text
        assert "2010" in text