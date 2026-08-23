from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2

    sql_db_uri: str="sqlite:///mock.db"

    vector_store_path: str="./data/chroma"
    embedding_model: str = "text-embedding-3-small"

    search_api_key: str=""

    inference_endpoint: str = ""

    class Config:
        env_file = ".env"
        extra="ignore"


settings = Settings()