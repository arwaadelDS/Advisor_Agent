"""Compiles the multi-agent graph and provides the turn-level entrypoint
that wires a session's thread_id into the checkpointer.

Routing: supervisor decides next_agent; rag_agent and search_agent run
once and return to END for this turn. sql_agent may instead continue to
rag_agent within the same turn when the question implies wanting research
on the retrieved holdings (see agents/sql_agent.py's sql_agent_node) --
so a single turn can be supervisor -> sql_agent -> rag_agent -> END.

Checkpointer uses an explicit allowed_msgpack_modules list so the custom
Pydantic types stored in AdvisorState (SQLQueryResult, ClientHolding,
RAGSearchResult) deserialize without the "unregistered type" warning --
see langgraph's checkpoint deserialization security notes (CVE-2026-28277)
for why this allowlist exists. MemorySaver here is purely in-process, so
the actual attack surface (a malicious checkpoint store) doesn't apply,
but registering the types is the correct fix rather than ignoring the
warning.
"""

import logging

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from graph.state import AdvisorState
from agents.supervisor import supervisor_node
from agents.search_agent import search_agent_node
from agents.rag_agent import rag_node as rag_agent_node
from agents.sql_agent import sql_agent_node

logger = logging.getLogger(__name__)

_checkpoint_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("schemas", "SQLQueryResult"),
        ("schemas", "ClientHolding"),
        ("schemas", "RAGSearchResult"),
    ]
)


def route_next(state: AdvisorState) -> str:
    return state.get("next_agent") or "end"


def build_graph():
    graph = StateGraph(AdvisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("sql_agent", sql_agent_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "search_agent": "search_agent",
            "end": END,
        },
    )
    graph.add_edge("search_agent", END)
    graph.add_edge("rag_agent", END)
    graph.add_conditional_edges(
        "sql_agent",
        route_next,
        {"rag_agent": "rag_agent", "end": END},
    )

    return graph.compile(checkpointer=MemorySaver(serde=_checkpoint_serde))


app = build_graph()


def run_turn(user_message: str, thread_id: str, client_id: str = "unspecified") -> dict:
    """Runs one turn of the conversation on a given thread_id.

    Only messages/client_id/session_id are passed on every call -- the
    checkpointer merges this onto whatever state already exists for
    thread_id (confirmed behavior: partial input is merged, not
    overwritten, and different thread_id values are isolated from each
    other), so sql_result/rag_context/search_result set on a prior turn
    remain available to later nodes in the same thread without needing to
    be re-passed here.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "client_id": client_id,
            "session_id": thread_id,
        },
        config=config,
    )
    return result


def _cli_loop():
    """Manual interactive testing: uv run python -m graph.workflow"""
    import sys

    thread_id = input("thread_id (blank for 'manual-session'): ").strip() or "manual-session"
    client_id = input("client_id (blank for 'C001'): ").strip() or "C001"
    print(f"\nSession thread_id={thread_id!r}, client_id={client_id!r}. Ctrl+C to exit.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
        if not user_message:
            continue

        result = run_turn(user_message, thread_id=thread_id, client_id=client_id)
        last = result["messages"][-1]
        print(f"Assistant: {getattr(last, 'content', last)}\n")


if __name__ == "__main__":
    _cli_loop()