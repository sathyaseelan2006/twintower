from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float

from backend.database import Base

# -----------------------------
# SQLAlchemy Database Model
# -----------------------------

class TelemetryDB(Base):

    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True)

    device_id = Column(String, index=True)
    timestamp = Column(String)

    temperature = Column(Float)
    voltage = Column(Float)
    current = Column(Float)
    vibration = Column(Float)

    rpm = Column(Integer)
    status = Column(String)
    ml_prediction = Column(String, nullable=True)
    anomaly_score = Column(Float, nullable=True)


# -----------------------------
# Pydantic API Model
# -----------------------------

class TelemetryData(BaseModel):

    device_id: str
    timestamp: str

    temperature: float
    voltage: float
    current: float
    vibration: float

    rpm: int
    status: str