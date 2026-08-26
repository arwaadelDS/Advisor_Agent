"""Shared pydantic contracts between pipeline steps (query_rewriter ->
sql_tools) and consumers outside this module (the RAG branch). Keeping
these here instead of inline in each tool file so the output shape a
downstream consumer depends on can't silently drift when one file changes
without the other being updated to match.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ClientHolding(BaseModel):
    symbol: str
    name_en: str
    quantity: int
    market_value: float
    sector: str


class RewrittenQuery(BaseModel):
    original: str
    rewritten: str
    corrections: list[str] = []
    ambiguous: list[str] = []
    needs_clarification: bool = False


class SQLQueryResult(BaseModel):
    client_id: str
    holdings: list[ClientHolding] = Field(default_factory=list)
    row_count: int = 0
    query_used: Optional[str] = None
    error: Optional[str] = None


class GeneratedQuery(BaseModel):
    sql: str
    confidence: float
    needs_clarification: bool = False
    clarification_question: str | None = None


class PipelineResult(BaseModel):
    question: str
    generated: GeneratedQuery
    validation_error: str | None = None
    result: SQLQueryResult | None = None
    repair_attempts: int = 0
    needs_clarification: bool = False
    clarification_question: str | None = None
    rewrite: RewrittenQuery | None = None
    answer: str | None = None