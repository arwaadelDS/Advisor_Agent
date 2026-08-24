from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2

    # Read from GOOGLE_API_KEY in .env. Needed for the LLM only -- embeddings
    # run locally, so ingestion and retrieval work with no key at all.
    google_api_key: str = ""

    # Written by ingestion/seed_mock_db.py, which names it advisor_mock.db.
    sql_db_uri: str = "sqlite:///data/mock/advisor_mock.db"

    vector_store_path: str = "data/index"

    # A sentence-transformers model id, not an API model. Multilingual on
    # purpose: half this corpus is Arabic, and an English-only embedder would
    # return plausible-looking nonsense for it rather than failing.
    embedding_model: str = "BAAI/bge-m3"

    # Cross-encoder that re-scores the shortlist before the model sees it.
    # Empty means no reranking: it is a separate download, and retrieval has to
    # work without it. "BAAI/bge-reranker-v2-m3" is the multilingual one that
    # matches the embedder above -- see ingestion/rerank.py.
    reranker_model: str = ""

    search_api_key: str = ""

    inference_endpoint: str = ""

    # Absolute path to a pdftotext executable. Leave empty to auto-discover.
    # Set this when the wrong pdftotext shadows the right one on PATH --
    # see ingestion/pdf_backend.py for why that matters here.
    poppler_path: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
