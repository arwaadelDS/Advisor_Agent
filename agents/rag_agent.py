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
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from config import settings
from schemas import RAGSearchResult
from tools.rag_tools import RagToolError, format_context, search_research

DEFAULT_K = 5

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
  Arabic answer, using the Arabic terms the extracts themselves use.
- Be brief. An advisor is reading this between calls.
"""

USER_PROMPT = """\
Advisor's question:
{question}

Research extracts:
{context}
"""


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


def build_messages(question: str, result: RAGSearchResult) -> list[tuple[str, str]]:
    """The exact prompt sent to the model, as a value a test can inspect.

    Kept separate from the call so the grounding rules can be tested without a
    network round trip -- the prompt is the safety mechanism here, so it needs
    tests of its own.
    """
    return [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT.format(question=question, context=format_context(result))),
    ]


def answer(question: str, result: RAGSearchResult) -> str:
    """Compose a cited answer from retrieved research."""
    response = _llm().invoke(build_messages(question, result))
    return str(getattr(response, "content", response)).strip()


def rag_node(state: dict[str, Any]) -> dict[str, Any]:
    """The LangGraph node. Retrieves, answers, and updates ``AdvisorState``.

    Returns both the structured ``rag_context`` -- so a later node can render
    citations properly instead of scraping them out of prose -- and the answer
    as a message.
    """
    question = question_of(state)
    result = retrieve(state)
    try:
        text = answer(question, result)
    except RagToolError as exc:
        raise RagAgentError(str(exc)) from exc
    return {
        "rag_context": result,
        "messages": [*(state.get("messages") or []), {"role": "assistant",
                                                      "content": text}],
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
