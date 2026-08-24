"""Read-only query functions over the mock advisor database.

Every function here does the same three things, in the same order:

1. validate the input shape (is client_id a real string? is ticker known?)
2. run one parameterized SELECT
3. map the rows into a typed Pydantic model

That repetition is deliberate rather than lazy -- an LLM calling these tools
needs predictable failure modes more than it needs clever code. `handle_db_errors`
and the two `_require_known_*` helpers exist so every function fails the same
way for the same class of problem, instead of each one growing its own
ad-hoc try/except.

Two kinds of "no result" are treated differently on purpose:

- a well-formed lookup for something that doesn't exist (unknown client,
  unknown ticker, unknown tier) is a *caller mistake* -> raise
- a well-formed lookup for something that exists but has nothing attached
  (a real client with no holdings, a real ticker nobody owns) is a
  *valid empty answer* -> return a typed result with zero rows

Getting this distinction right is what lets the agent say "I don't recognise
that client ID" instead of "this client owns nothing," which is a very
different thing to tell an advisor.
"""

from __future__ import annotations

import functools

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config import settings
from schemas import (
    ClientHolding,
    ClientsByTierResult,
    ClientSummary,
    PortfolioSummary,
    SecurityInfo,
    SecuritySearchResult,
    SectorExposure,
    SQLQueryResult,
    StockHolder,
    StockHoldersResult,
)

engine = create_engine(settings.sql_db_uri)  
 
class SqlToolError(RuntimeError):
    """Base class for every error this module raises."""


class ClientNotFoundError(SqlToolError):
    """Raised when client_id is well-formed but not in the clients table."""


class UnknownTickerError(SqlToolError):
    """Raised when ticker is well-formed but not in the instruments table."""


class InvalidTierError(SqlToolError):
    """Raised when the requested aum_tier doesn't match any tier on file."""

def handle_db_errors(func):
    """Catch SQLAlchemyError and re-raise as SqlToolError, with context."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SQLAlchemyError as exc:
            raise SqlToolError(f"{func.__name__} failed: {exc}") from exc

    return wrapper

#input validation 

def _require_nonempty_str(value, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SqlToolError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


def _require_known_client(client_id: str) -> None:
    client_id = _require_nonempty_str(client_id, field_name="client_id")
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM clients WHERE client_id = :client_id"),
            {"client_id": client_id},
        ).first()
    if row is None:
        raise ClientNotFoundError(f"no client with id {client_id!r}")


def _require_known_ticker(ticker: str) -> None:
    ticker = _require_nonempty_str(ticker, field_name="ticker")
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM instruments WHERE ticker = :ticker"),
            {"ticker": ticker},
        ).first()
    if row is None:
        raise UnknownTickerError(f"no instrument with ticker {ticker!r}")

#queries to retreive from the db (relationships in db)
#1- get the client's info from holdings(all the stocks the own)
@handle_db_errors
def get_client_holdings(client_id: str) -> SQLQueryResult:
    """Return all holdings for a given client, joined with instruments details."""
    _require_known_client(client_id)

    query = text("""
        SELECT s.ticker, s.name_en, h.quantity, h.market_value, s.sector
        FROM holdings h
        JOIN instruments s ON h.ticker = s.ticker
        WHERE h.client_id = :client_id
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"client_id": client_id}).mappings().all()

    holdings = [
        ClientHolding(
            symbol=row["ticker"],
            name_en=row["name_en"],
            quantity=row["quantity"],
            market_value=row["market_value"],
            sector=row["sector"],
        )
        for row in rows
    ]
    return SQLQueryResult(client_id=client_id, holdings=holdings, row_count=len(holdings))

#2-get the client's total invesments devided by company sector
@handle_db_errors
def get_portfolio_summary(client_id: str) -> PortfolioSummary:
    """Return total portfolio value and a sector-by-sector breakdown for a client."""
    _require_known_client(client_id)

    query = text("""
        SELECT s.sector, SUM(h.market_value) AS total_value
        FROM holdings h
        JOIN instruments s ON h.ticker = s.ticker
        WHERE h.client_id = :client_id
        GROUP BY s.sector
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"client_id": client_id}).mappings().all()

    breakdown = [SectorExposure(sector=r["sector"], total_value=r["total_value"]) for r in rows]
    total = sum(r.total_value for r in breakdown)
    return PortfolioSummary(client_id=client_id, total_market_value=total, sector_breakdown=breakdown)


@handle_db_errors
def get_security_info(query: str) -> SecuritySearchResult:
    """Find instruments by ticker, (partial) name, or sector.

    Three ways this matches, all in one query:
      - exact ticker   e.g. "2010"
      - partial name   e.g. "sabic" -> SABIC (case-insensitive)
      - exact sector   e.g. "Energy" -> every instrument in that sector
                       (also case-insensitive, so "energy" works too)

    Returns a list because a sector or name search can legitimately match
    many rows. Zero matches is a normal outcome, not an error.
    """
    raw_query = _require_nonempty_str(query, field_name="query")

    sql = text("""
        SELECT ticker, name_en, sector
        FROM instruments
        WHERE ticker = :exact
           OR name_en LIKE :pattern
           OR LOWER(sector) = LOWER(:exact)
    """)
    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {"exact": raw_query, "pattern": f"%{raw_query}%"},
        ).mappings().all()

    matches = [
        SecurityInfo(ticker=row["ticker"], name_en=row["name_en"], sector=row["sector"])
        for row in rows
    ]
    return SecuritySearchResult(query=raw_query, matches=matches, match_count=len(matches))



@handle_db_errors
def get_clients_by_tier(tier: str) -> ClientsByTierResult:
    """Return every client in a given aum_tier"""
    raw_tier = _require_nonempty_str(tier, field_name="tier")

    with engine.connect() as conn:
        known_tiers = [
            row[0] for row in conn.execute(text("SELECT DISTINCT aum_tier FROM clients"))
        ]

    match = next((t for t in known_tiers if t.lower() == raw_tier.lower()), None)
    if match is None:
        valid = ", ".join(sorted(known_tiers)) or "(no tiers found)"
        raise InvalidTierError(f"unknown tier {raw_tier!r}. Known tiers: {valid}")

    query = text("""
        SELECT client_id, name, aum_tier
        FROM clients
        WHERE aum_tier = :tier
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"tier": match}).mappings().all()

    clients = [
        ClientSummary(client_id=row["client_id"], name=row["name"], aum_tier=row["aum_tier"])
        for row in rows
    ]
    return ClientsByTierResult(tier=match, clients=clients, client_count=len(clients))


#get clients invested in specific company 
@handle_db_errors
def get_holders_of_stock(ticker: str) -> StockHoldersResult:
    """Return every client holding a given ticker, with quantity and value."""
    raw_ticker = _require_nonempty_str(ticker, field_name="ticker")
    _require_known_ticker(raw_ticker)

    query = text("""
        SELECT client_id, quantity, market_value
        FROM holdings
        WHERE ticker = :ticker
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"ticker": raw_ticker}).mappings().all()

    holders = [
        StockHolder(
            client_id=row["client_id"],
            quantity=row["quantity"],
            market_value=row["market_value"],
        )
        for row in rows
    ]
    return StockHoldersResult(ticker=raw_ticker, holders=holders, holder_count=len(holders))

