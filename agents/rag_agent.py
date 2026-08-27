"""The research agent: answer from the client's own research, or say there is none.

The node ``graph/state.py`` declares -- reads ``sql_result`` and ``messages``
off ``AdvisorState``, writes ``rag_context``.

Retrieval runs first, every time, rather than being offered to the model as a
tool it may skip: a model that answers "SABIC looks well positioned" without
retrieving has made an unsourced claim about a real security, and nothing
downstream can tell that from a sourced one. ``research_tool`` exports the
tool-calling form for a supervisor that wants it.

Holdings come from state, never from the model, so a paraphrase ("their tech
exposure") cannot widen the filter to names the client does not hold. With no
``sql_result`` the search is unfiltered, and the context block says so.

The prompt carries two rules the corpus makes necessary: answer in the language
of the question, and treat "no research covers this" as the correct answer
rather than something to paper over. Both are asserted in the tests.

The language rule needs more than the prompt. Retrieval is cross-lingual, so an
English question often comes back with mostly Arabic extracts and the model
answers in their language instead. ``language_instruction`` decides the answer's
language here, from the question, and says so outright.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage

from config import settings
from ingestion.extract import ARABIC, ENGLISH, classify
from schemas import RAGSearchResult
from tools.citations import check_citations
from tools.llm import text_of
from tools.rag_tools import RagToolError, format_context, search_research

logger = logging.getLogger(__name__)

DEFAULT_K = 5

# What to call each language in the prompt. ``classify`` can also return "und"
# for a question with no letters in it ("2010?"), which falls back to naming no
# language at all rather than guessing one.
LANGUAGE_NAMES = {ARABIC: "Arabic", ENGLISH: "English"}

SYSTEM_PROMPT = """\
You are a research assistant for wealth advisors at SNB Capital.

You answer only from the numbered research extracts given to you. They are the
only source you may use. You have no other knowledge of these companies.

Rules:
- Cite every claim with the bracketed number of the extract it came from, like
  [1] or [2]. A sentence with no citation must not appear.
- If the extracts do not answer the question, say so plainly and stop. Do not
  fall back on general knowledge, and do not soften it into a vague answer that
  sounds informed.
- If the Note at the end of the context says no research covers the client's
  holdings, say that. That is the correct answer, not a failure.
- Reply in the language of the advisor's question. An Arabic question gets an
  Arabic answer, using the Arabic terms the extracts themselves use. Half this
  corpus is in the other language from the question, so expect to translate
  what you cite. The question fixes the language of the answer; the extracts
  never do.
- Be brief. An advisor is reading this between calls.
"""

USER_PROMPT = """\
Advisor's question:
{question}

Research extracts:
{context}
{language}"""


class RagAgentError(RuntimeError):
    """Raised when the agent cannot run -- no model configured, or no question."""


def question_of(state: dict[str, Any]) -> str:
    """The advisor's question: the last human turn in ``messages``.

    Tolerates LangChain message objects, ``{"role": ..., "content": ...}`` dicts
    and bare strings, because the graph is being assembled by two people and the
    message shape is not settled yet. Falls back to the last message of any kind
    rather than raising, so a single-turn call still works.
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


def holdings_of(state: dict[str, Any]) -> list[Any] | None:
    """The client's positions, or ``None`` when there is no portfolio in play.

    ``None`` and ``[]`` mean different things all the way down this stack: no
    client in context versus a client who holds nothing. See tools/rag_tools.py.
    """
    result = state.get("sql_result")
    if result is None:
        return None
    holdings = getattr(result, "holdings", None)
    if holdings is None and isinstance(result, dict):
        holdings = result.get("holdings")
    return list(holdings) if holdings is not None else []


def retrieve(state: dict[str, Any], k: int = DEFAULT_K) -> RAGSearchResult:
    """Retrieve the research this state entitles the advisor to see."""
    question = question_of(state)
    if not question.strip():
        raise RagAgentError("no advisor question found in state['messages']")

    holdings = holdings_of(state)
    result = search_research(question, holdings, k=k)
    if holdings is None:
        # Said in the context block, not just in the logs: without it the model
        # would present sector research as though it were about this client.
        scope = (
            "This search was not restricted to any client's holdings, so it may "
            "cover securities the client does not own."
        )
        result = result.model_copy(
            update={"note": f"{result.note} {scope}".strip()}
        )
    return result


@lru_cache(maxsize=1)
def _llm():
    """The chat model, built once. Imported lazily so retrieval needs no key."""
    if not settings.google_api_key:
        raise RagAgentError(
            "GOOGLE_API_KEY is not set, so the agent cannot compose an answer.\n"
            "Retrieval itself needs no key -- try "
            '`uv run python -m tools.rag_tools "your question"`.'
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RagAgentError(
            "langchain-google-genai is not installed. Run `uv sync`."
        ) from exc
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def language_instruction(question: str) -> str:
    """A line naming the language to answer in, or "" if the script is unclear.

    The system prompt already says to answer in the question's language, and on
    its own that is not enough: retrieval is cross-lingual on purpose, so an
    English question routinely returns mostly Arabic extracts, and the model
    then follows the evidence rather than the instruction. Naming the language
    outright, decided here rather than inferred there, is what actually holds.
    """
    name = LANGUAGE_NAMES.get(classify(question))
    return f"\nWrite your answer in {name}.\n" if name else ""


def build_messages(question: str, result: RAGSearchResult) -> list[tuple[str, str]]:
    """The exact prompt sent to the model, as a value a test can inspect.

    Kept separate from the call so the grounding rules can be tested without a
    network round trip -- the prompt is the safety mechanism here, so it needs
    tests of its own.
    """
    return [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT.format(
            question=question,
            context=format_context(result),
            language=language_instruction(question),
        )),
    ]


def answer(question: str, result: RAGSearchResult) -> str:
    """Compose a cited answer from retrieved research."""
    return text_of(_llm().invoke(build_messages(question, result)))


def rag_node(state: dict[str, Any]) -> dict[str, Any]:
    """The LangGraph node. Retrieves, answers, and updates ``AdvisorState``.

    Returns both the structured ``rag_context`` -- so a later node can render
    citations properly instead of scraping them out of prose -- and the answer
    as a message.

    Degrades instead of raising, like the other three nodes: a rate limit or a
    dropped connection here would otherwise take down the whole turn, and the
    advisor would see a traceback where an apology belongs.

    The two failures are kept apart because they leave different things usable.
    If retrieval failed there is nothing to show. If retrieval worked and only
    the model call failed, ``rag_context`` still holds the chunks and their
    citations, so the interface can show the research even when no one managed
    to summarise it.
    """
    question = question_of(state)
    try:
        result = retrieve(state)
    except (RagAgentError, RagToolError) as exc:
        logger.error("rag_node retrieval failed: %s", exc, exc_info=True)
        return {
            "rag_context": None,
            "messages": [AIMessage(
                content="I couldn't search the research library just now. "
                        "Please try again shortly.",
                name="rag_agent",
            )],
            "next_agent": None,
        }

    try:
        text = answer(question, result)
    except Exception as exc:
        logger.error("rag_node could not compose an answer: %s", exc, exc_info=True)
        text = ("I found research on this but couldn't summarise it just now. "
                "The extracts are listed below.")

    # A regex over a string already in memory, so this runs on every answer.
    # Logged rather than acted on: the rest of the answer and the extracts under
    # it are still worth showing.
    report = check_citations(text, result)
    if not report.ok:
        logger.error("rag_node answer cites unretrieved extracts: %s",
                     report.summary())

    return {
        "rag_context": result,
        "messages": [AIMessage(content=text, name="rag_agent")],
        "next_agent": None,
    }


def research_tool():
    """The retrieval step as a LangChain tool, for a tool-calling supervisor.

    Built on demand rather than at import so that importing this module does not
    require langchain-core. Note it takes tickers explicitly: a model calling
    this cannot widen the filter beyond what it is given, because the caller
    decides what to pass.
    """
    from langchain_core.tools import tool

    @tool
    def search_research_documents(question: str, tickers: list[str] | None = None) -> str:
        """Search SNB Capital equity research, optionally restricted to holdings.

        Args:
            question: what the advisor wants to know.
            tickers: Tadawul codes the client holds, e.g. ["2010", "1120"].
                Omit to search all research.

        Returns numbered extracts with citations, or a note saying why there
        are none.
        """
        return format_context(search_research(question, tickers))

    return search_research_documents


def main(argv: list[str] | None = None) -> int:
    """Ask the research agent a question from the command line.

        uv run python -m agents.rag_agent "ما هي مخاطر سابك؟" --ticker 2010
    """
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="agents.rag_agent",
        description="Answer an advisor question from the research corpus.",
    )
    parser.add_argument("question")
    parser.add_argument("--ticker", action="append", default=[], metavar="CODE",
                        help="a holding to restrict to; repeatable")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompt instead of calling the model")
    args = parser.parse_args(argv)

    from schemas import ClientHolding, SQLQueryResult

    state: dict[str, Any] = {"messages": [{"role": "user", "content": args.question}]}
    if args.ticker:
        state["sql_result"] = SQLQueryResult(
            client_id="cli",
            holdings=[
                # Only symbol reaches retrieval; the rest of the SQL agent's
                # shape is filled in so the model validates.
                ClientHolding(symbol=t, name_en="", quantity=0,
                              market_value=0, sector="")
                for t in args.ticker
            ],
            row_count=len(args.ticker),
        )

    try:
        result = retrieve(state, k=args.k)
        if args.dry_run:
            for role, content in build_messages(args.question, result):
                print(f"----- {role} -----")
                print(content)
            return 0
        print(answer(args.question, result))
        print()
        for number, chunk in enumerate(result.chunks, start=1):
            print(f"[{number}] {chunk.citation}")
    except (RagAgentError, RagToolError) as exc:
        print("FAILED")
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
