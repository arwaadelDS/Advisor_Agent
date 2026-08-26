from pathlib import Path
from sqlalchemy import create_engine

DB_PATH = Path(__file__).parent.parent / "data" / "mock" / "advisor_mock.db"

# read-only connection 
agent_engine = create_engine(f"sqlite:///file:{DB_PATH}?mode=ro&uri=true")