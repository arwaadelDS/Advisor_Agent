# building the database 3 tables and connecting them together in 
# holdings by (client_id,ticker) to be foreign keys
#creating view for the agent 
import csv
from pathlib import Path
from sqlalchemy import create_engine, text

CSV_DIR = Path(__file__).parent.parent / "data" / "mock"
DB_PATH = CSV_DIR / "advisor_mock.db"

engine = create_engine(f"sqlite:///{DB_PATH}")

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS instruments (
    ticker TEXT PRIMARY KEY, isin TEXT, name_en TEXT, name_ar TEXT,
    sector TEXT, asset_class TEXT, shariah_flag TEXT
);
CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY, name TEXT, risk_profile TEXT, aum_tier TEXT
);
CREATE TABLE IF NOT EXISTS holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT, ticker TEXT, quantity INTEGER, market_value REAL
);
"""

def read_csv(name):
    with open(CSV_DIR / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def seed():
    instruments = read_csv("instruments.csv")
    clients = read_csv("clients.csv")
    holdings = read_csv("holdings.csv")

    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS client_portfolio_view"))

        for stmt in CREATE_TABLES.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
        conn.execute(text("DELETE FROM instruments"))
        conn.execute(text("DELETE FROM clients"))
        conn.execute(text("DELETE FROM holdings"))

        for i in instruments:
            conn.execute(text(
                "INSERT INTO instruments VALUES "
                "(:ticker,:isin,:name_en,:name_ar,:sector,:asset_class,:shariah_flag)"
            ), i)
        for c in clients:
            conn.execute(text("INSERT INTO clients VALUES (:client_id,:name,:risk_profile,:aum_tier)"), c)
        for h in holdings:
            conn.execute(text(
                "INSERT INTO holdings (client_id,ticker,quantity,market_value) "
                "VALUES (:client_id,:ticker,:quantity,:market_value)"), h)

    print(f"Seeded {DB_PATH.name}: {len(instruments)} instruments, {len(clients)} clients, {len(holdings)} holdings.")

if __name__ == "__main__":
    seed()