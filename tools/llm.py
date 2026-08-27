"""The chat model, and the one safe way to read a reply out of it."""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.retry import call_with_retry

load_dotenv()


class Retrying:
    """A chat model that retries its own calls. tools/retry.py decides which.

    Wrapped here rather than at each call site, since ``get_llm`` is what the
    supervisor, both SQL paths, the rewriter and the search agent all call.
    LangChain's own ``.with_retry()`` cannot sit here: it returns a runnable
    with no ``with_structured_output``, which two of those callers need.
    """

    def __init__(self, model):
        self._model = model

    def invoke(self, *args, **kwargs):
        return call_with_retry(lambda: self._model.invoke(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs) -> "Retrying":
        # Wrapped again on the way out, so the retry sits outside the parsing.
        return Retrying(self._model.with_structured_output(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._model, name)


@lru_cache(maxsize=1)
def get_llm():
    """The shared chat model, built once.

    Building one costs about 1.6s -- the client sets up three SSL contexts and
    each loads the system trust store -- and a turn calls this eight times. The
    model is fixed at first use, so a test that changes the env afterwards needs
    ``get_llm.cache_clear()``.
    """
    return Retrying(
        ChatGoogleGenerativeAI(
            model=os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", 0.2)),
            google_api_key=os.environ["GOOGLE_API_KEY"],
            # An attempt count, not a retry count: 1 sends the request once and
            # leaves the SDK's own retry off. Its default of 6 retries 429s
            # underneath us, spending half a minute on a dead daily quota
            # before tools/retry.py is even asked.
            max_retries=1,
        )
    )


def text_of(response: Any) -> str:
    """The prose out of a chat response, whichever shape the model returned.

    Gemini 2.5 puts a plain string on ``.content``; Gemini 3.x puts a list of
    typed blocks there, each carrying an opaque ``signature`` alongside the
    text. Neither ``str(...)`` nor ``.strip()`` on that list does what it looks
    like it does -- one shows the advisor a Python repr full of base64, the
    other raises. Every caller reads replies through here so the next model
    that changes shape breaks in one place.
    """
    # Checked as a string before as a callable: langchain-core returns a str
    # subclass that is still callable for back-compat, and invoking it warns.
    text = getattr(response, "text", None)
    if isinstance(text, str):
        if text.strip():
            return text.strip()
    elif callable(text):
        text = text()
        if isinstance(text, str) and text.strip():
            return text.strip()

    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(content).strip()