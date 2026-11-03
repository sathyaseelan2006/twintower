from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy import Column, Integer, String, Float
from backend.database import Base

# ============================================================
# SQLAlchemy Database Models
# ============================================================

class MicrogridTelemetryDB(Base):
    __tablename__ = "microgrid_telemetry"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    timestamp = Column(String)

    # Solar PV Channels
    solar_voltage = Column(Float)
    solar_current = Column(Float)
    solar_power = Column(Float)

    # Battery Storage Channels
    battery_voltage = Column(Float)
    battery_level = Column(Float)       # SoC (%) from Pin D35
    battery_temp = Column(Float)
    battery_soh = Column(Float)         # State of Health (%)
    battery_rul_cycles = Column(Integer) # Remaining Useful Life (Cycles)

    # Load Consumption Channels
    load_current = Column(Float)        # ACS712 Amperes
    load_power = Column(Float)          # Watts

    # Microgrid Actuator & Operating States
    mode = Column(String)               # SOLAR, BATTERY, AUTO
    load_status = Column(String)        # ON, OFF
    servo_angle = Column(Integer)       # Pin 23 angle
    led_status = Column(String)         # Pin 27 status

    # AI / Analytics Results
    ml_prediction = Column(String)      # OPTIMAL, WARNING, ANOMALY
    anomaly_score = Column(Float)
    optimization_action = Column(String) # Power routing recommendation


class ControlStateDB(Base):
    __tablename__ = "control_state"

    id = Column(Integer, primary_key=True)
    mode = Column(String, default="BATTERY")     # SOLAR, BATTERY, AUTO
    load_status = Column(String, default="ON")   # ON, OFF
    servo_angle = Column(Integer, default=0)    # Pin 23
    led_status = Column(String, default="OFF")   # Pin 27
    updated_at = Column(String)


# ============================================================
# Pydantic Inbound & Outbound Models
# ============================================================

class MicrogridTelemetryData(BaseModel):
    device_id: str = "ESP32-MICROGRID-001"
    timestamp: Optional[str] = None

    solar_voltage: float = 10.20
    solar_current: Optional[float] = 0.0
    solar_power: Optional[float] = None

    battery_voltage: float = 16.83
    battery_level: float = 100.0        # Pin D35 (%)
    battery_temp: Optional[float] = 27.5

    load_current: float = 0.17          # ACS712 (A)
    load_power: float = 2.84            # (W)

    mode: str = "BATTERY"               # SOLAR, BATTERY, AUTO
    load_status: str = "ON"             # ON, OFF
    servo_angle: int = 0                # Pin 23
    led_status: str = "OFF"             # Pin 27


class ModeControlRequest(BaseModel):
    mode: str = Field(..., description="SOLAR, BATTERY, or AUTO")


class LoadControlRequest(BaseModel):
    load_status: str = Field(..., description="ON or OFF")


class ServoControlRequest(BaseModel):
    angle: int = Field(..., ge=0, le=180, description="Servo angle in degrees")


class LedControlRequest(BaseModel):
    led_status: str = Field(..., description="ON or OFF")
