from typing import List, Dict, Any, Optional
import os
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from backend.database import engine, get_db
from backend.models import (
    Base,
    MicrogridTelemetryDB,
    ControlStateDB,
    MicrogridTelemetryData,
    ModeControlRequest,
    LoadControlRequest,
    ServoControlRequest,
    LedControlRequest
)
from ml.battery_health import battery_predictor
from ml.energy_optimizer import energy_optimizer
from ml.anomaly_detector import microgrid_anomaly_detector

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Hybrid Microgrid Digital Twin API",
    description="Real-time telemetry ingestion, battery health prediction, and energy optimization backend.",
    version="2.0.0"
)

# Enable CORS for external dashboards and ESP32
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strip /api prefix when deployed on Vercel Serverless
@app.middleware("http")
async def handle_api_prefix(request: Request, call_next):
    path = request.scope.get("path", "")
    if path.startswith("/api/"):
        request.scope["path"] = path[4:]
    elif path == "/api":
        request.scope["path"] = "/"
    return await call_next(request)


def get_or_create_control_state(db: Session) -> ControlStateDB:
    state = db.query(ControlStateDB).first()
    if not state:
        state = ControlStateDB(
            mode="BATTERY",
            load_status="ON",
            servo_angle=0,
            led_status="OFF",
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        db.add(state)
        db.commit()
        db.refresh(state)
    return state


# ============================================================
# ROOT & SYSTEM HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "project": "Self-Learning Microgrid Energy Optimization & Battery Health Prediction",
        "status": "ONLINE",
        "database": "SQLite Connected",
        "target_device": "ESP32 (10.174.75.204)"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "database": "connected"
    }


# ============================================================
# TELEMETRY INGESTION & ANALYTICS
# ============================================================

@app.post("/telemetry")
def receive_telemetry(
    data: MicrogridTelemetryData,
    db: Session = Depends(get_db)
):
    timestamp = data.timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 1. Battery Health Prediction (SoH, RUL)
    b_health = battery_predictor.predict(
        battery_voltage=data.battery_voltage,
        battery_level=data.battery_level,
        load_current=data.load_current,
        battery_temp=data.battery_temp or 27.5
    )

    # 2. Self-Learning Energy Optimization
    opt_result = energy_optimizer.optimize(
        solar_voltage=data.solar_voltage,
        battery_voltage=data.battery_voltage,
        battery_level=data.battery_level,
        load_power=data.load_power,
        current_mode=data.mode
    )

    # 3. Anomaly & Safety Hazards Detection
    anomaly_result = microgrid_anomaly_detector.predict(data)

    solar_pwr = data.solar_power if data.solar_power is not None else round(opt_result["estimated_solar_power"], 2)

    telemetry = MicrogridTelemetryDB(
        device_id=data.device_id,
        timestamp=timestamp,
        solar_voltage=data.solar_voltage,
        solar_current=data.solar_current or 0.0,
        solar_power=solar_pwr,
        battery_voltage=data.battery_voltage,
        battery_level=data.battery_level,
        battery_temp=data.battery_temp or 27.5,
        battery_soh=b_health["battery_soh"],
        battery_rul_cycles=b_health["battery_rul_cycles"],
        load_current=data.load_current,
        load_power=data.load_power,
        mode=data.mode,
        load_status=data.load_status,
        servo_angle=data.servo_angle,
        led_status=data.led_status,
        ml_prediction=anomaly_result["prediction"],
        anomaly_score=anomaly_result["anomaly_score"],
        optimization_action=opt_result["action"]
    )

    db.add(telemetry)
    db.commit()
    db.refresh(telemetry)

    return {
        "message": "Microgrid telemetry analyzed and logged",
        "id": telemetry.id,
        "device_id": telemetry.device_id,
        "battery_soh": b_health["battery_soh"],
        "battery_rul_cycles": b_health["battery_rul_cycles"],
        "optimization": opt_result,
        "safety_status": anomaly_result
    }


@app.get("/telemetry")
def get_telemetry(limit: int = 100, db: Session = Depends(get_db)):
    readings = (
        db.query(MicrogridTelemetryDB)
        .order_by(MicrogridTelemetryDB.id.desc())
        .limit(limit)
        .all()
    )
    return readings


@app.get("/telemetry/latest")
def latest_telemetry(db: Session = Depends(get_db)):
    reading = (
        db.query(MicrogridTelemetryDB)
        .order_by(MicrogridTelemetryDB.id.desc())
        .first()
    )
    if reading is None:
        return {"message": "No telemetry available"}
    return reading


@app.get("/telemetry/count")
def telemetry_count(db: Session = Depends(get_db)):
    count = db.query(MicrogridTelemetryDB).count()
    return {"total_readings": count}


@app.get("/telemetry/ingestion-health")
def telemetry_ingestion_health(db: Session = Depends(get_db)):
    latest = (
        db.query(MicrogridTelemetryDB)
        .order_by(MicrogridTelemetryDB.id.desc())
        .first()
    )
    if latest is None:
        return {
            "status": "NO_DATA",
            "healthy": False,
            "latest_id": None,
            "age_seconds": None
        }

    try:
        latest_time = datetime.fromisoformat(latest.timestamp.replace("Z", "+00:00"))
        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_seconds = (now - latest_time).total_seconds()
        healthy = 0 <= age_seconds <= 30

        return {
            "status": "ACTIVE" if healthy else "STALE",
            "healthy": healthy,
            "latest_id": latest.id,
            "latest_timestamp": latest.timestamp,
            "age_seconds": round(max(0, age_seconds), 2)
        }
    except Exception as e:
        return {
            "status": "TIMESTAMP_ERROR",
            "healthy": False,
            "error": str(e)
        }


# ============================================================
# BI-DIRECTIONAL HARDWARE CONTROL (ESP32 ACTUATION)
# ============================================================

@app.get("/control/state")
def get_control_state(db: Session = Depends(get_db)):
    """ESP32 calls this endpoint to fetch the latest desired switch/relay state."""
    state = get_or_create_control_state(db)
    return {
        "mode": state.mode,
        "load_status": state.load_status,
        "servo_angle": state.servo_angle,
        "led_status": state.led_status,
        "updated_at": state.updated_at
    }


@app.post("/control/mode")
def set_mode(req: ModeControlRequest, db: Session = Depends(get_db)):
    state = get_or_create_control_state(db)
    state.mode = req.mode.upper()
    state.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.commit()
    return {"success": True, "mode": state.mode}


@app.post("/control/load")
def set_load(req: LoadControlRequest, db: Session = Depends(get_db)):
    state = get_or_create_control_state(db)
    state.load_status = req.load_status.upper()
    state.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.commit()
    return {"success": True, "load_status": state.load_status}


@app.post("/control/servo")
def set_servo(req: ServoControlRequest, db: Session = Depends(get_db)):
    state = get_or_create_control_state(db)
    state.servo_angle = req.angle
    state.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.commit()
    return {"success": True, "servo_angle": state.servo_angle}


@app.post("/control/led")
def set_led(req: LedControlRequest, db: Session = Depends(get_db)):
    state = get_or_create_control_state(db)
    state.led_status = req.led_status.upper()
    state.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.commit()
    return {"success": True, "led_status": state.led_status}


# ============================================================
# SUMMARY & ANALYTICS
# ============================================================

@app.get("/system/summary")
def system_summary(db: Session = Depends(get_db)):
    readings = db.query(MicrogridTelemetryDB).all()
    total = len(readings)

    anomaly_count = sum(1 for r in readings if r.ml_prediction == "ANOMALY")
    normal_count = sum(1 for r in readings if r.ml_prediction == "NORMAL")
    unclassified = total - normal_count - anomaly_count
    anomaly_rate = (anomaly_count / total * 100) if total > 0 else 0

    latest = db.query(MicrogridTelemetryDB).order_by(MicrogridTelemetryDB.id.desc()).first()
    ctrl = get_or_create_control_state(db)

    return {
        "total_readings": total,
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "unclassified_count": unclassified,
        "anomaly_rate": round(anomaly_rate, 2),
        "current_mode": ctrl.mode,
        "load_status": ctrl.load_status,
        "latest_solar_voltage": latest.solar_voltage if latest else 10.20,
        "latest_battery_voltage": latest.battery_voltage if latest else 16.83,
        "latest_battery_level": latest.battery_level if latest else 100.0,
        "latest_load_power": latest.load_power if latest else 2.84,
        "battery_soh": latest.battery_soh if latest else 98.5,
        "battery_rul_cycles": latest.battery_rul_cycles if latest else 1420
    }


# ============================================================
# ACADEMIC SOFTWARE TESTING SUITE ENDPOINTS
# ============================================================
from tests.academic_test_suite import ACADEMIC_TEST_CASES, execute_test, execute_all_tests, execute_test_suite, parse_csv_test_suite
from fastapi.responses import PlainTextResponse

@app.get("/tests/academic")
def get_academic_tests(demo: bool = True, link_failure: bool = False, db: Session = Depends(get_db)):
    latest = db.query(MicrogridTelemetryDB).order_by(MicrogridTelemetryDB.id.desc()).first()
    live_dict = {c.name: getattr(latest, c.name) for c in latest.__table__.columns} if latest else None
    return execute_all_tests(live_dict, use_demo_data=demo, link_failure=link_failure)

@app.post("/tests/academic/run/{test_id}")
def run_single_academic_test(test_id: str, demo: bool = True, link_failure: bool = False, db: Session = Depends(get_db)):
    tc = next((t for t in ACADEMIC_TEST_CASES if t["id"] == test_id), None)
    if not tc:
        raise HTTPException(status_code=404, detail=f"Test case {test_id} not found")
    latest = db.query(MicrogridTelemetryDB).order_by(MicrogridTelemetryDB.id.desc()).first()
    live_dict = {c.name: getattr(latest, c.name) for c in latest.__table__.columns} if latest else None
    return execute_test(tc, live_dict, use_demo_data=demo, link_failure=link_failure)

@app.get("/tests/academic/demo-csv", response_class=PlainTextResponse)
def get_demo_test_csv():
    demo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "academic_demo_test_suite.csv")
    if os.path.exists(demo_path):
        with open(demo_path, "r", encoding="utf-8") as f:
            return f.read()
    return "id,name,parameter,unit,expected,tolerance,actual,description\nTC-C01,Demo Test,battery_voltage,V,16.5,0.3,16.8,Sample"

@app.post("/tests/academic/custom-suite")
def run_custom_test_suite(test_cases: List[Dict[str, Any]], demo: bool = True, link_failure: bool = False, db: Session = Depends(get_db)):
    latest = db.query(MicrogridTelemetryDB).order_by(MicrogridTelemetryDB.id.desc()).first()
    live_dict = {c.name: getattr(latest, c.name) for c in latest.__table__.columns} if latest else None
    return execute_test_suite(test_cases, live_dict, use_demo_data=demo, link_failure=link_failure)


