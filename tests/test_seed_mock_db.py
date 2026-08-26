import pytest
from pathlib import Path
from sqlalchemy import create_engine, text, inspect

from ingestion import seed_mock_db


@pytest.fixture
def seeded_engine(tmp_path, monkeypatch):
    """Run seed() against a throwaway DB file so we never touch the real one."""
    test_db_path = tmp_path / "test_advisor_mock.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}")

    monkeypatch.setattr(seed_mock_db, "engine", test_engine)
    monkeypatch.setattr(seed_mock_db, "DB_PATH", test_db_path)

    seed_mock_db.seed()
    return test_engine


def test_tables_created(seeded_engine):
    inspector = inspect(seeded_engine)
    tables = set(inspector.get_table_names())
    assert {"instruments", "clients", "holdings"}.issubset(tables)


def test_view_does_not_exist(seeded_engine):
    inspector = inspect(seeded_engine)
    views = inspector.get_view_names()
    assert "client_portfolio_view" not in views
    assert len(views) == 0


def test_row_counts_match_csv(seeded_engine):
    with seeded_engine.connect() as conn:
        instr_count = conn.execute(text("SELECT COUNT(*) FROM instruments")).scalar()
        client_count = conn.execute(text("SELECT COUNT(*) FROM clients")).scalar()
        holdings_count = conn.execute(text("SELECT COUNT(*) FROM holdings")).scalar()

    assert instr_count == 12   
    assert client_count == 18  
    assert holdings_count > 0


def test_foreign_key_values_are_valid(seeded_engine):
    """Every holding should reference a real client_id and ticker — catches
    seeding order bugs or CSV mismatches early."""
    with seeded_engine.connect() as conn:
        orphan_clients = conn.execute(text("""
            SELECT COUNT(*) FROM holdings h
            LEFT JOIN clients c ON h.client_id = c.client_id
            WHERE c.client_id IS NULL
        """)).scalar()
        orphan_tickers = conn.execute(text("""
            SELECT COUNT(*) FROM holdings h
            LEFT JOIN instruments i ON h.ticker = i.ticker
            WHERE i.ticker IS NULL
        """)).scalar()

    assert orphan_clients == 0
    assert orphan_tickers == 0


def test_rerunning_seed_is_idempotent(seeded_engine):
    """Running seed() twice shouldn't duplicate rows or error."""
    seed_mock_db.seed()
    with seeded_engine.connect() as conn:
        client_count = conn.execute(text("SELECT COUNT(*) FROM clients")).scalar()
    assert client_count == 18