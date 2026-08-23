"""Generates mock instruments, clients, and holdings data.

generate three tables: clients; including client id ,name,tier and risk_profile 
instruments: changed this table to suit the rag retreival to contain the isin for the companies,ticker,name in both 
arabic&english, sector ,asset,and shariah->empty for now
holdings: what connects them twoe together(foriegn keys) ,tickers,quantity and market value

Run once with:  uv run python data/mock/generate_data.py
"""
import csv
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent

# shariah_flag left empty: the RAG side's loader treats "" as None (unpopulated),
INSTRUMENTS = [
    {"ticker": "2222", "isin": "SA14TG012N13", "name_en": "Saudi Aramco",
     "name_ar": "أرامكو السعودية", "sector": "Energy", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "1120", "isin": "SA0007879113", "name_en": "Al Rajhi Bank",
     "name_ar": "مصرف الراجحي", "sector": "Banking", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "1180", "isin": "SA13L050IE10", "name_en": "Saudi National Bank",
     "name_ar": "البنك الأهلي السعودي", "sector": "Banking", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "1010", "isin": "SA0007879048", "name_en": "Riyad Bank",
     "name_ar": "بنك الرياض", "sector": "Banking", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "2010", "isin": "SA0007879121", "name_en": "SABIC",
     "name_ar": "سابك", "sector": "Petrochemicals", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "2350", "isin": "SA000A0MQCJ2", "name_en": "Saudi Kayan",
     "name_ar": "كيان السعودية", "sector": "Petrochemicals", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "7010", "isin": "SA0007879543", "name_en": "stc",
     "name_ar": "الاتصالات السعودية", "sector": "Telecom", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "7020", "isin": "SA000A0DM9P2", "name_en": "Etihad Etisalat (Mobily)",
     "name_ar": "اتحاد اتصالات - موبايلي", "sector": "Telecom", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "1211", "isin": "SA123GA0ITH7", "name_en": "Ma'aden",
     "name_ar": "معادن", "sector": "Materials", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "5110", "isin": "SA0007879550", "name_en": "Saudi Electricity",
     "name_ar": "السعودية للكهرباء", "sector": "Utilities", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "2280", "isin": "SA000A0ETHT1", "name_en": "Almarai",
     "name_ar": "المراعي", "sector": "Consumer", "asset_class": "Equity", "shariah_flag": ""},
    {"ticker": "4013", "isin": "SA1510P1UMH1", "name_en": "Dr. Sulaiman Al Habib",
     "name_ar": "د. سليمان الحبيب", "sector": "Healthcare", "asset_class": "Equity", "shariah_flag": ""},
]

FIRST_NAMES = ["Faisal","Noura","Khalid","Reem","Abdullah","Sara","Turki","Lama","Fahad","Hind",
               "Majed","Dana","Omar","Aisha","Saud","Munira","Bandar","Alia"]
LAST_NAMES = ["Al-Otaibi","Al-Sudairi","Al-Ghamdi","Al-Qahtani","Al-Harbi","Al-Dosari",
              "Al-Shehri","Al-Zahrani","Al-Mutairi","Al-Anazi"]
RISK_PROFILES = ["Conservative","Balanced","Aggressive"]
AUM_TIERS = ["HNW","Ultra-HNW"]


def generate():
    random.seed(42)

    clients = []
    for i in range(1, 19):
        clients.append({"client_id": f"C{i:03d}",
                         "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
                         "risk_profile": random.choice(RISK_PROFILES),
                         "aum_tier": random.choice(AUM_TIERS)})

    holdings = []
    for c in clients:
        chosen = random.sample(INSTRUMENTS, random.randint(2, 6))
        for sec in chosen:
            qty = random.randint(1000, 30000)
            price = random.randint(20, 60)
            holdings.append({"client_id": c["client_id"], "ticker": sec["ticker"],
                              "quantity": qty, "market_value": qty * price})

    _write_csv("instruments.csv", INSTRUMENTS,
               ["ticker", "isin", "name_en", "name_ar", "sector", "asset_class", "shariah_flag"])
    _write_csv("clients.csv", clients, ["client_id", "name", "risk_profile", "aum_tier"])
    _write_csv("holdings.csv", holdings, ["client_id", "ticker", "quantity", "market_value"])

    print(f"Generated into {OUT_DIR}: "
          f"{len(INSTRUMENTS)} instruments, {len(clients)} clients, {len(holdings)} holdings.")


def _write_csv(filename, rows, fieldnames):
    with open(OUT_DIR / filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate()