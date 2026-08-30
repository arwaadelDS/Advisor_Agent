"""The SQL agent: answer from the client's own data, or ask what's unclear.

The node graph/state.py declares -- reads client_id and messages off
AdvisorState, writes sql_result.

Two grounding steps run before any SQL is generated, in this order:
rewrite_query() first (Arabic references resolved against real DB values),
then run_query_pipeline() (generate -> validate -> execute -> repair). Either
one can independently decide the question is unanswerable as asked --
rewrite_query() when a reference in the question matches nothing in the DB,
run_query_pipeline() when the question itself is ambiguous even after a clean
rewrite -- and either is checked before the next step runs. There is no point
generating SQL against a question that still has an unresolved reference in
it, and no point retrying SQL generation against a question the model itself
says it cannot answer.

Like rag_node, this degrades instead of raising: a rewrite failure, a
generation failure, or an answer-synthesis failure all produce an apologetic
message rather than a traceback, because advisors see this, not logs.

sql_result is written as the raw SQLQueryResult, not the wrapping
PipelineResult -- that's the exact shape rag_agent.holdings_of() and
tools.rag_tools.tickers_of() already read off AdvisorState, so the RAG
handoff needs no changes here; see schemas.py's own docstring on why
ClientHolding stayed exactly as RAG expects it.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from ingestion.extract import ARABIC, ENGLISH, classify
from schemas import PipelineResult, SQLQueryResult
from tools.llm import get_llm, text_of
from tools.query_rewriter import QueryRewriterError, rewrite_query
from tools.sql_tools import SQLToolError, run_query_pipeline

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {ARABIC: "Arabic", ENGLISH: "English"}

SYSTEM_PROMPT = """\
You are a data assistant for wealth advisors at SNB Capital.

You answer only from the query result given to you below. It is the only
source you may use -- you have no other knowledge of this client or these
figures.

Rules:
- State only what the result contains. Do not add, round, or infer any
  number that is not directly present in the data.
- If the result is empty, say plainly that nothing matched, and do not guess
  why.
- Be brief and direct -- an advisor is reading this between calls, not a
  report.
- Reply in the language of the advisor's question.
"""

USER_PROMPT = """\
Advisor's question:
{question}

Query result:
{result}
{language}"""


class SqlAgentError(RuntimeError):
    """Raised when the agent cannot run -- no question found in state."""


def question_of(state: dict[str, Any]) -> str:
    """The advisor's question: the last human turn in messages.

    Same tolerant walk as rag_agent.question_of -- LangChain message objects,
    dicts, or bare strings -- kept as a separate copy rather than a shared
    import, since the two agents' state-reading needs are allowed to drift
    (see agents/sql_agent.py's own node not being a shared graph node).
    """
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, str):
            return message
        role = getattr(message, "type", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if role in ("human", "user") and content:
            return str(content)
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content")
        return str(content if content is not None else last)
    return ""


def language_instruction(question: str) -> str:
    """A line naming the language to answer in -- same reasoning as
    rag_agent.language_instruction: naming it outright holds where "reply in
    the question's language" alone does not, especially once the result data
    itself is a mix of English column values and Arabic instrument names."""
    name = LANGUAGE_NAMES.get(classify(question))
    return f"\nWrite your answer in {name}.\n" if name else ""


def format_result(result: SQLQueryResult) -> str:
    """The structured result as text a model can answer from.

    Kept deliberately plain (no markdown table, no reformatting) -- the model
    is asked to state only what's here, and a plain list of what's present
    makes an invented figure easier to catch on review than a table would.
    """
    if result.error:
        return f"The query failed: {result.error}"

    lines: list[str] = []
    if result.client_profile:
        p = result.client_profile
        lines.append(
            f"Client: {p.name} (id={p.client_id}, risk_profile={p.risk_profile}, aum_tier={p.aum_tier})"
        )
    if result.holdings:
        lines.append(f"Holdings ({len(result.holdings)}):")
        for h in result.holdings:
            lines.append(
                f"- {h.symbol} {h.name_en} / {h.name_ar} | sector={h.sector} "
                f"asset_class={h.asset_class} qty={h.quantity} value={h.market_value}"
            )
    if result.rows:
        lines.append(f"Rows ({len(result.rows)}):")
        for row in result.rows:
            lines.append(f"- {row}")
    if not lines:
        lines.append("No rows matched.")
    return "\n".join(lines)


def build_messages(question: str, result: SQLQueryResult) -> list[tuple[str, str]]:
    """The exact prompt for answer synthesis -- separated from the call so
    the grounding rules can be tested without a network round trip, same
    reasoning as rag_agent.build_messages."""
    return [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT.format(
            question=question,
            result=format_result(result),
            language=language_instruction(question),
        )),
    ]


def answer(question: str, result: SQLQueryResult) -> str:
    """Compose a plain-language answer over a structured query result."""
    return text_of(get_llm().invoke(build_messages(question, result)))


def sql_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """The LangGraph node. Rewrites, queries, answers, updates AdvisorState.

    Three points this can stop at, each returning its own apologetic or
    clarifying message rather than falling through to the next step:
    rewrite failure, an unresolved reference the rewriter flagged, or a
    question the SQL pipeline itself could not resolve even after a clean
    rewrite. Only a clean run past all three reaches answer synthesis.

    On success, next_agent is set to "rag_agent" rather than None when the
    rewriter detected the question also wants research/opinion on the
    holdings just retrieved (RewrittenQuery.wants_research), and there are
    actual holdings for RAG to filter by -- an aggregate result has nothing
    for rag_agent.holdings_of() to search against. See graph/workflow.py's
    docstring for the sql_agent -> rag_agent edge this relies on.
    """
    question = question_of(state)
    client_id = state.get("client_id")

    if not question.strip():
        logger.error("sql_node found no advisor question in state['messages']")
        return {
            "sql_result": None,
            "messages": [AIMessage(
                content="I didn't catch a question to look up. Could you rephrase?",
                name="sql_agent",
            )],
            "next_agent": None,
        }

    try:
        rewritten = rewrite_query(question, client_selected=bool(client_id))
    except QueryRewriterError as exc:
        logger.error("sql_node rewrite failed: %s", exc, exc_info=True)
        return {
            "sql_result": None,
            "messages": [AIMessage(
                content="I couldn't look up the data just now. Please try again shortly.",
                name="sql_agent",
            )],
            "next_agent": None,
        }

    if rewritten.needs_clarification:
        clarification = (
            "I couldn't match: " + ", ".join(rewritten.ambiguous)
            if rewritten.ambiguous else
            "Could you clarify what you mean?"
        )
        return {
            "sql_result": None,
            "messages": [AIMessage(content=clarification, name="sql_agent")],
            "next_agent": None,
        }

    try:
        pipeline: PipelineResult = run_query_pipeline(rewritten.rewritten, client_id)
    except SQLToolError as exc:
        logger.error("sql_node query pipeline failed: %s", exc, exc_info=True)
        return {
            "sql_result": None,
            "messages": [AIMessage(
                content="I couldn't run that query just now. Please try again shortly.",
                name="sql_agent",
            )],
            "next_agent": None,
        }

    if pipeline.needs_clarification:
        return {
            "sql_result": None,
            "messages": [AIMessage(
                content=pipeline.clarification_question or "Could you clarify what you mean?",
                name="sql_agent",
            )],
            "next_agent": None,
        }

    if pipeline.validation_error:
        logger.error("sql_node validation failed after repair budget: %s", pipeline.validation_error)
        return {
            "sql_result": None,
            "messages": [AIMessage(
                content="I couldn't build a safe query for that. Could you rephrase?",
                name="sql_agent",
            )],
            "next_agent": None,
        }

    result = pipeline.result
    if result is None or result.error:
        logger.error("sql_node execution failed after repair budget: %s",
                     result.error if result else "no result")
        return {
            "sql_result": result,
            "messages": [AIMessage(
                content="I found an issue running that query and couldn't recover. "
                        "Please try rephrasing.",
                name="sql_agent",
            )],
            "next_agent": None,
        }

    try:
        text = answer(question, result)
    except Exception as exc:
        logger.error("sql_node could not compose an answer: %s", exc, exc_info=True)
        text = "I retrieved the data but couldn't summarise it just now. Details are below."

    next_agent = "rag_agent" if (rewritten.wants_research and result.holdings) else None

    return {
        "sql_result": result,
        "messages": [AIMessage(content=text, name="sql_agent")],
        "next_agent": next_agent,
    }


def main(argv: list[str] | None = None) -> int:
    """Ask the SQL agent a question from the command line.

        uv run python -m agents.sql_agent "average market value by risk profile"
        uv run python -m agents.sql_agent "this client's holdings" --client C001
    """
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="agents.sql_agent")
    parser.add_argument("question")
    parser.add_argument("--client", default=None, dest="client_id")
    args = parser.parse_args(argv)

    state: dict[str, Any] = {
        "messages": [{"role": "user", "content": args.question}],
        "client_id": args.client_id,
    }
    outcome = sql_agent_node(state)
    for message in outcome["messages"]:
        print(message.content)
    if outcome["sql_result"]:
        print()
        print(format_result(outcome["sql_result"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())