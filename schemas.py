from pydantic import BaseModel
from typing import Literal


class RouteDecision(BaseModel):
    next_agent: Literal["sql_agent", "rag_agent", "search_agent", "end"]
    reasoning: str


class ClientHolding(BaseModel):
    symbol: str
    name_en: str
    quantity: float
    market_value: float
    sector: str


class SQLQueryResult(BaseModel):
    client_id: str
    holdings: list[ClientHolding]
    row_count: int


class SectorExposure(BaseModel):
    sector: str
    total_value: float


class PortfolioSummary(BaseModel):
    client_id: str
    total_market_value: float
    sector_breakdown: list[SectorExposure]


class DocumentChunk(BaseModel):
    text: str
    source: str
    score: float


class RAGSearchResult(BaseModel):
    chunks: list[DocumentChunk]


class SearchResult(BaseModel):
    summary: str
    sources: list[str]


# ---------------------------------------------------------------------------
# New: schemas for the three additional SQL tools
# ---------------------------------------------------------------------------


class SecurityInfo(BaseModel):
    """One row from `instruments`, as returned by a ticker or name search."""

    ticker: str
    name_en: str
    sector: str


class SecuritySearchResult(BaseModel):
    """Result of get_security_info -- zero, one, or many matches."""

    query: str
    matches: list[SecurityInfo]
    match_count: int


class ClientSummary(BaseModel):
    """One client row, as returned by a tier lookup."""

    client_id: str
    name: str
    aum_tier: str


class ClientsByTierResult(BaseModel):
    """Result of get_clients_by_tier."""

    tier: str
    clients: list[ClientSummary]
    client_count: int


class StockHolder(BaseModel):
    """One client's position in a single stock."""

    client_id: str
    quantity: float
    market_value: float


class StockHoldersResult(BaseModel):
    """Result of get_holders_of_stock -- who owns a given ticker."""

    ticker: str
    holders: list[StockHolder]
    holder_count: int