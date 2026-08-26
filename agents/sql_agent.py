"""Top-level SQL agent: wires query rewriting into the SQL sub-pipeline,
composes a final prose answer (mirroring agents/rag_agent.py's pattern of
"structured data in, cited/grounded prose out"), and exposes a single
@tool entrypoint to the Supervisor.

Orchestration boundary:
- tools/query_rewriter.py -> normalizes/corrects the raw question
- tools/sql_tools.py -> generate/validate/execute/repair sub-pipeline
  (a cohesive, independently-testable unit; left where it is)
- this file -> decides the order (rewrite first), decides what happens
  when rewriting itself surfaces ambiguity (stop before any SQL is
  generated), composes the final advisor-facing answer, and is the only
  place wrapped as a LangChain @tool for the Supervisor to call.

Answer synthesis rule, same principle as rag_agent.py: the LLM composing
the final sentence only ever sees the already-retrieved holdings data (or
the error/ambiguity state) -- never the raw question alone -- so it can't
invent figures that aren't in the retrieved rows.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.llm import get_llm
from tools.query_rewriter import rewrite_question
from tools.sql_tools import (
    run_query_pipeline,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_REPAIR_ATTEMPTS,
)
from schemas import PipelineResult, GeneratedQuery, SQLQueryResult, ClientHolding


ANSWER_SYSTEM_PROMPT = """You are a portfolio assistant for wealth advisors.

Answer the advisor's question using ONLY the holdings data given to you below.
Do not invent figures, tickers, sectors, or holdings that are not listed.

Rules:
- If the holdings list is empty, say plainly that the client has no matching
  holdings. That is a correct answer, not a failure.
- If the data could not be retrieved, explain briefly and professionally that
  you weren't able to complete the request -- do not expose raw SQL or
  internal error details to the advisor.
- If the question was ambiguous, ask the advisor to clarify, briefly and
  specifically about what's ambiguous.
- Reply in the language of the advisor's question.
- Be concise -- an advisor is reading this between calls.
"""


def _format_holdings_for_prompt(holdings: list[ClientHolding]) -> str:
    if not holdings:
        return "(no holdings found)"
    return "\n".join(
        f"- {h.symbol} ({h.name_en}, {h.sector}): qty {h.quantity}, value {h.market_value}"
        for h in holdings
    )


def synthesize_answer(question: str, sql_result: SQLQueryResult | None,
                       clarification_question: str | None = None) -> str:
    """Composes the final prose answer. Mirrors rag_agent.answer()'s role:
    structured data/decisions happen upstream; this is the one place that
    turns them into something an advisor actually reads."""
    if clarification_question:
        user_content = (
            f"The advisor asked: {question}\n\n"
            f"This question is ambiguous: {clarification_question}\n"
            f"Ask the advisor to clarify, briefly."
        )
    elif sql_result is None or sql_result.error:
        err = sql_result.error if sql_result else "unknown error"
        user_content = (
            f"The advisor asked: {question}\n\n"
            f"The data could not be retrieved. Internal note: {err}\n"
            f"Explain briefly and professionally that this couldn't be "
            f"retrieved, without repeating raw technical details."
        )
    else:
        holdings_text = _format_holdings_for_prompt(sql_result.holdings)
        user_content = (
            f"The advisor asked: {question}\n\n"
            f"Holdings data:\n{holdings_text}\n\n"
            f"Answer the question using only this data."
        )

    resp = get_llm().invoke([
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    return str(getattr(resp, "content", resp)).strip()


def run_sql_agent(question: str,
                   max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
                   timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                   max_rows: int = DEFAULT_MAX_ROWS) -> PipelineResult:
    """Full agent entrypoint: rewrite -> (stop here if ambiguous) -> SQL
    pipeline -> synthesize final answer.

    The original (pre-rewrite) question is always preserved on the
    returned PipelineResult.question; the RewrittenQuery itself is
    attached under .rewrite for audit/logging. The SQL pipeline only ever
    sees the rewritten text; the final .answer is always composed from
    what was actually retrieved (or the actual error/ambiguity), never
    from the raw question alone.
    """
    rewrite = rewrite_question(question)

    if rewrite.needs_clarification:
        clarification_text = "; ".join(rewrite.ambiguous)
        placeholder_generated = GeneratedQuery(
            sql="", confidence=0.0, needs_clarification=True,
            clarification_question=clarification_text,
        )
        answer_text = synthesize_answer(
            question, None, clarification_question=clarification_text
        )
        return PipelineResult(
            question=question,
            generated=placeholder_generated,
            rewrite=rewrite,
            needs_clarification=True,
            clarification_question=clarification_text,
            answer=answer_text,
        )

    result = run_query_pipeline(
        rewrite.rewritten,
        max_repair_attempts=max_repair_attempts,
        timeout_seconds=timeout_seconds,
        max_rows=max_rows,
    )
    result.rewrite = rewrite
    result.question = question  
    result.answer = synthesize_answer(
        question, result.result, clarification_question=result.clarification_question
    )
    return result

class QueryClientPortfolioInput(BaseModel):
    question: str = Field(
        ..., description="Natural-language question about a client's portfolio holdings"
    )


@tool("query_client_portfolio", args_schema=QueryClientPortfolioInput)
def query_client_portfolio(question: str) -> PipelineResult:
    """Retrieve a client's portfolio holdings and answer the advisor's
    question about them in plain language. Use this when the question
    concerns one specific client's holdings (quantities, values, sectors)."""
    return run_sql_agent(question)