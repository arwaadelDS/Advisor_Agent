from pydantic import BaseModel
from typing import Literal


class RouteDecision(BaseModel):
    """Supervisor's structured routing output."""
    next_agent: Literal["sql_agent", "rag_agent", "search_agent", "end"]
    reasoning: str


class ClientHolding(BaseModel):
    """A single position in a client's portfolio."""
    symbol: str
    quantity: float
    market_value: float
    asset_class: str


class SQLQueryResult(BaseModel):
    """Output of the SQL agent."""
    client_id: str
    holdings: list[ClientHolding]
    row_count: int


class DocumentChunk(BaseModel):
    """A single retrieved chunk from the vector store."""
    text: str
    source: str
    score: float


class RAGSearchResult(BaseModel):
    """Output of the RAG agent."""
    chunks: list[DocumentChunk]


class SearchResult(BaseModel):
    """Output of the web search agent."""
    summary: str
    sources: list[str]