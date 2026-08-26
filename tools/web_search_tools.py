import logging
from functools import lru_cache

from config import settings
from langchain_core.tools import tool
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_search_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.search_api_key,
        timeout=15.0,
    )


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=8),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def fetch_web_results(query: str) -> str:
    """Queries OpenRouter web plugin for real-time market and news updates."""
    response = _get_search_client().chat.completions.create(
        model=settings.search_model,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a real-time financial market search assistant. "
                    "Return concise factual findings, exact numbers, dates, and credible sources."
                ),
            },
            {"role": "user", "content": query},
        ],
        extra_body={"plugins": [{"id": "web"}]},
    )

    if not response.choices:
        return "No search results returned by the provider."

    return response.choices[0].message.content or "No relevant findings found."


@tool
def search_web(query: str) -> str:
    """Searches the web for live financial news, market trends, and real-time world events."""
    try:
        return fetch_web_results(query)
    except Exception as e:
        logger.error(f"Search tool execution failed: {e}", exc_info=True)
        return f"Error executing web search: {str(e)}"