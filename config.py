from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2

    sql_db_uri: str = "sqlite:///data/mock/advisor.db"

    vector_store_path: str = "data/index"
    embedding_model: str = "text-embedding-3-small"

    search_api_key: str = ""

    inference_endpoint: str = ""

    # Absolute path to a pdftotext executable. Leave empty to auto-discover.
    # Set this when the wrong pdftotext shadows the right one on PATH --
    # see ingestion/pdf_backend.py for why that matters here.
    poppler_path: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
