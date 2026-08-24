from pydantic import BaseModel, Field
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
    """A single retrieved chunk from the vector store.

    ``text``, ``source`` and ``score`` are the original three fields and are
    still all a caller strictly needs. The rest carry the citation down from the
    index, so an answer can say "Saudi Basic Industries -- Equity Research,
    page 2 (2026-08-06)" instead of naming a file. Every added field has a
    default, so code that builds the three-field form keeps working.

    Why bother: research is only useful to an advisor if they can turn to the
    page it came from. A chunk that cannot be cited is a chunk the compliance
    side has to take on trust.
    """
    text: str
    source: str
    score: float

    # Citation, carried from chunk metadata rather than looked up again.
    chunk_id: str = ""
    doc_id: str = ""
    title: str = ""
    section: str = ""
    pages: str = ""            # "2" or "1-2"; a chunk can span a page break
    published_at: str = ""     # ISO date
    language: str = ""         # "ar" | "en" | "und"
    isins: list[str] = Field(default_factory=list)
    citation: str = ""         # preformatted, for display under an answer


class RAGSearchResult(BaseModel):
    """Output of the RAG agent.

    Beyond the chunks, this reports *what was searched*. An empty ``chunks``
    means one of two very different things -- "nothing in the covered research
    answers this" or "we hold no research on what this client owns" -- and an
    agent that cannot tell them apart will answer the second case by inventing
    something. ``note`` says which it was, in a sentence meant to be read by the
    model composing the reply.
    """
    chunks: list[DocumentChunk]
    query: str = ""
    searched_tickers: list[str] = Field(default_factory=list)
    uncovered_tickers: list[str] = Field(default_factory=list)
    note: str = ""


class SearchResult(BaseModel):
    """Output of the web search agent."""
    summary: str
    sources: list[str]