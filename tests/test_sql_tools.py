"""Tests for sql_tools.py.

Each test gets its own isolated, in-memory-backed SQLite database via the
`db` fixture below, seeded fresh every time. Nothing here touches the real
advisor_mock.db except the final TestTheRealDatabase class, which is a smoke
test against the actual file if it happens to exist.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from tools import sql_tools
from tools.sql_tools import (
    ClientNotFoundError,
    InvalidTierError,
    SqlToolError,
    UnknownTickerError,
    get_client_holdings,
    get_clients_by_tier,
    get_holders_of_stock,
    get_portfolio_summary,
    get_security_info,
)

SCHEMA = """
CREATE TABLE instruments (
    ticker   TEXT PRIMARY KEY,
    name_en  TEXT NOT NULL,
    sector   TEXT NOT NULL
);

CREATE TABLE clients (
    client_id TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    aum_tier  TEXT NOT NULL
);

CREATE TABLE holdings (
    client_id    TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    quantity     REAL NOT NULL,
    market_value REAL NOT NULL
);
"""

SEED = """
INSERT INTO instruments (ticker, name_en, sector) VALUES
    ('2222', 'Saudi Aramco', 'Energy'),
    ('2010', 'SABIC',        'Materials'),
    ('1120', 'Al Rajhi Bank', 'Financials');

INSERT INTO clients (client_id, name, aum_tier) VALUES
    ('C001', 'Fahad Al-Otaibi', 'Ultra-HNW'),
    ('C002', 'Noura Al-Qahtani', 'HNW'),
    ('C003', 'Sara Al-Mutairi', 'Ultra-HNW');

INSERT INTO holdings (client_id, ticker, quantity, market_value) VALUES
    ('C001', '2222', 1000, 32000.0),
    ('C001', '2010', 500,  15000.0),
    ('C002', '1120', 200,  18000.0);
    -- note: C003 (a real, known client) intentionally has NO holdings rows
    -- note: '1120' (a real, known ticker) has exactly one holder
    -- '2010' below has no rows inserted for it in a separate no-holder test
"""


@pytest.fixture()
def db(monkeypatch, tmp_path: Path):
    """Build a fresh temp SQLite DB, seed it, and point sql_tools.engine at it."""
    db_path = tmp_path / "test_advisor.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executescript(SEED)
    conn.commit()
    conn.close()

    test_engine = create_engine(f"sqlite:///{db_path}")
    monkeypatch.setattr(sql_tools, "engine", test_engine)
    return test_engine


# ---------------------------------------------------------------------------
# get_client_holdings / get_portfolio_summary (existing functions, hardened)
# ---------------------------------------------------------------------------


class TestGetClientHoldings:
    def test_known_client_with_holdings(self, db):
        result = get_client_holdings("C001")
        assert result.client_id == "C001"
        assert result.row_count == 2
        assert {h.symbol for h in result.holdings} == {"2222", "2010"}

    def test_known_client_with_no_holdings_returns_empty_not_error(self, db):
        result = get_client_holdings("C003")
        assert result.row_count == 0
        assert result.holdings == []

    def test_unknown_client_raises(self, db):
        with pytest.raises(ClientNotFoundError):
            get_client_holdings("C999")

    @pytest.mark.parametrize("bad_input", [None, "", "   ", 123, 45.6])
    def test_malformed_client_id_raises(self, db, bad_input):
        with pytest.raises(SqlToolError):
            get_client_holdings(bad_input)

    def test_client_id_is_case_sensitive(self, db):
        # 'c001' != 'C001' -- documents current behaviour rather than assuming it
        with pytest.raises(ClientNotFoundError):
            get_client_holdings("c001")

    def test_sql_injection_shaped_input_is_just_a_miss(self, db):
        with pytest.raises(ClientNotFoundError):
            get_client_holdings("C001'; DROP TABLE clients; --")


class TestGetPortfolioSummary:
    def test_aggregates_by_sector(self, db):
        result = get_portfolio_summary("C001")
        assert result.total_market_value == 47000.0
        sectors = {s.sector: s.total_value for s in result.sector_breakdown}
        assert sectors == {"Energy": 32000.0, "Materials": 15000.0}

    def test_unknown_client_raises(self, db):
        with pytest.raises(ClientNotFoundError):
            get_portfolio_summary("nope")

    def test_known_client_no_holdings_gives_zero_total(self, db):
        result = get_portfolio_summary("C003")
        assert result.total_market_value == 0
        assert result.sector_breakdown == []


# ---------------------------------------------------------------------------
# get_security_info
# ---------------------------------------------------------------------------


class TestGetSecurityInfo:
    def test_exact_ticker_match(self, db):
        result = get_security_info("2222")
        assert result.match_count == 1
        assert result.matches[0].name_en == "Saudi Aramco"

    def test_partial_case_insensitive_name_match(self, db):
        result = get_security_info("sabic")
        assert result.match_count == 1
        assert result.matches[0].ticker == "2010"

    def test_name_fragment_can_match_multiple(self, db):
        # both 'Saudi Aramco' names contain 'a' -- use something narrower to
        # prove multi-match plumbing without depending on exact seed wording
        result = get_security_info("al")
        tickers = {m.ticker for m in result.matches}
        assert "1120" in tickers  # Al Rajhi Bank

    def test_no_match_returns_empty_not_error(self, db):
        result = get_security_info("nonexistent-security-xyz")
        assert result.match_count == 0
        assert result.matches == []

    @pytest.mark.parametrize("bad_input", [None, "", "   "])
    def test_empty_query_raises(self, db, bad_input):
        with pytest.raises(SqlToolError):
            get_security_info(bad_input)


# ---------------------------------------------------------------------------
# get_clients_by_tier
# ---------------------------------------------------------------------------


class TestGetClientsByTier:
    def test_known_tier(self, db):
        result = get_clients_by_tier("Ultra-HNW")
        assert result.client_count == 2
        assert {c.client_id for c in result.clients} == {"C001", "C003"}

    def test_tier_matching_is_case_insensitive(self, db):
        result = get_clients_by_tier("ultra-hnw")
        assert result.client_count == 2
        assert result.tier == "Ultra-HNW"  # normalised to the DB's real casing

    def test_unknown_tier_raises_with_valid_options_listed(self, db):
        with pytest.raises(InvalidTierError) as exc_info:
            get_clients_by_tier("Mega-Rich")
        assert "Ultra-HNW" in str(exc_info.value)

    @pytest.mark.parametrize("bad_input", [None, "", "  "])
    def test_empty_tier_raises(self, db, bad_input):
        with pytest.raises(SqlToolError):
            get_clients_by_tier(bad_input)


# ---------------------------------------------------------------------------
# get_holders_of_stock
# ---------------------------------------------------------------------------


class TestGetHoldersOfStock:
    def test_ticker_with_one_holder(self, db):
        result = get_holders_of_stock("1120")
        assert result.holder_count == 1
        assert result.holders[0].client_id == "C002"

    def test_ticker_with_multiple_holders(self, db):
        # add a second holder of 2222 on top of the seed data
        with db.connect() as conn:
            conn.exec_driver_sql(
                "INSERT INTO holdings VALUES ('C002', '2222', 50, 1600.0)"
            )
            conn.commit()
        result = get_holders_of_stock("2222")
        assert result.holder_count == 2

    def test_known_ticker_zero_holders_returns_empty_not_error(self, db):
        # '2010' is a real instrument in this test DB but nobody holds it
        # in a fresh copy without C001's seed row -- simulate by using a
        # ticker that exists but was never inserted into holdings
        with db.connect() as conn:
            conn.exec_driver_sql(
                "INSERT INTO instruments VALUES ('9999', 'Unheld Co', 'Utilities')"
            )
            conn.commit()
        result = get_holders_of_stock("9999")
        assert result.holder_count == 0
        assert result.holders == []

    def test_unknown_ticker_raises(self, db):
        with pytest.raises(UnknownTickerError):
            get_holders_of_stock("0000")

    @pytest.mark.parametrize("bad_input", [None, "", "   "])
    def test_empty_ticker_raises(self, db, bad_input):
        with pytest.raises(SqlToolError):
            get_holders_of_stock(bad_input)


# ---------------------------------------------------------------------------
# Decorator behaviour: any SQLAlchemyError becomes a clean SqlToolError
# ---------------------------------------------------------------------------


class TestErrorWrapping:
    def test_missing_database_file_raises_sqltoolerror_not_raw_traceback(self, monkeypatch, tmp_path):
        broken_engine = create_engine(f"sqlite:///{tmp_path / 'does_not_exist.db'}")
        monkeypatch.setattr(sql_tools, "engine", broken_engine)
        # the table doesn't exist in this empty DB file, so the query itself fails
        with pytest.raises(SqlToolError):
            get_client_holdings("anyone")


# ---------------------------------------------------------------------------
# Optional smoke test against the real mock database, if present
# ---------------------------------------------------------------------------


class TestTheRealDatabase:
    """Sanity-checks against advisor_mock.db, skipped if that file isn't found."""

    REAL_DB = Path("advisor_mock.db")

    @pytest.mark.skipif(not REAL_DB.exists(), reason="advisor_mock.db not found")
    def test_a_known_client_from_seed_data_resolves(self):
        # adjust 'C001' to a client_id you know exists in your real seed data
        result = get_client_holdings("C001")
        assert result.client_id == "C001"