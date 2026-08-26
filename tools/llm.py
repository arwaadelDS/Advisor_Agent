"""The chat model, and the one safe way to read a reply out of it."""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", 0.2)),
        google_api_key=os.environ["GOOGLE_API_KEY"],
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