import logging
from functools import lru_cache

from graph.state import AdvisorState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tools.llm import get_llm
from tools.web_search_tools import search_web

logger = logging.getLogger(__name__)

SEARCH_AGENT_PROMPT = """You are a market news helper for SNB Capital advisors.
Summarize the search results simply."""


@lru_cache(maxsize=1)
def _get_search_llm():
    return get_llm()


def search_agent_node(state: AdvisorState) -> dict:
    """Worker node that executes live search queries"""
    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": [AIMessage(content="No search query provided to the Market Intelligence Agent.")],
            "search_result": None,
        }

    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    try:
        raw_search = search_web.invoke({"query": query})
    except Exception as e:
        logger.error(f"Failed invoking search_web tool: {e}")
        raw_search = "Unable to retrieve real-time search context due to a connection error."

    prompt = f"Advisor Query: {query}\n\nSearch Context:\n{raw_search}\n\nProvide an executive briefing:"

    response = _get_search_llm().invoke([
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        "messages": [response],
        "search_result": raw_search,
    }