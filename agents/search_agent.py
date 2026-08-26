import logging
from config import settings
from graph.state import AdvisorState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.web_search_tools import search_web

logger = logging.getLogger(__name__)

SEARCH_AGENT_PROMPT = """You are a market news helper for SNB Capital advisors.
Summarize the search results simply."""

_search_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.inference_endpoint or settings.search_api_key,
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    max_tokens=500
)


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
    
    response = _search_llm.invoke([
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        "messages": [response],
        "search_result": raw_search,
    }
