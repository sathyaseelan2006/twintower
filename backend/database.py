import os
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Detect Vercel serverless environment (read-only filesystem except /tmp)
if os.environ.get("VERCEL"):
    DATABASE_PATH = Path(tempfile.gettempdir()) / "telemetry.db"
else:
    data_dir = BASE_DIR / "data"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        DATABASE_PATH = data_dir / "telemetry.db"
    except OSError:
        DATABASE_PATH = Path(tempfile.gettempdir()) / "telemetry.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
