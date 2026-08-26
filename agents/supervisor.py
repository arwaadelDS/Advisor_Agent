import logging
from functools import lru_cache

from graph.state import AdvisorState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from tools.llm import get_llm
from schemas import RouteDecision

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """Route the advisor's request to the correct worker:
⚬	search_agent: Live market data, real-time news, macro trends.
⚬	rag_agent: Internal SNB Capital research, reports, strategy notes.
⚬	sql_agent: Client portfolios, account holdings, balances.
⚬	end: Greetings, pleasantries, or general non-specialist queries.
"""


@lru_cache(maxsize=1)
def _get_supervisor_llm():
    return get_llm().with_structured_output(RouteDecision)


@lru_cache(maxsize=1)
def _get_general_chat_llm():
    return get_llm()


def supervisor_node(state: AdvisorState) -> dict:
    """Routes the user request to the right agent or replies directly."""
    messages = state.get("messages", [])
    client_id = state.get("client_id", "Not specified")

    if not messages:
        return {
            "messages": [AIMessage(content="Hello! How can I assist you today?")],
            "next_agent": "end",
        }

    context_prefix = f"[Active Context: Client ID = {client_id}]\n"
    augmented_messages: list[BaseMessage] = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=context_prefix),
        *messages,
    ]

    try:
        decision: RouteDecision = _get_supervisor_llm().invoke(augmented_messages)
    except Exception as e:
        logger.error(f"Supervisor routing failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I encountered an issue analyzing your request. How can I assist you?")],
            "next_agent": "end",
        }

    logger.info(f"Supervisor Route: {decision.next_agent} | Rationale: {decision.reasoning}")

    if decision.next_agent == "end":
        direct_prompt = [
            SystemMessage(
                content="You assist SNB Capital advisors. Reply politely and keep it brief."
            ),
            *messages,
        ]
        chat_response = _get_general_chat_llm().invoke(direct_prompt)
        return {
            "messages": [chat_response],
            "next_agent": "end",
        }
    return {
        "next_agent": decision.next_agent,
    }