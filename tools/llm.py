import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", 0.2)),
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )