from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from backend.database import engine, get_db
from backend.models import Base, TelemetryDB, TelemetryData
from ml.anomaly_detector import anomaly_detector
from datetime import datetime, timezone

# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Embedded Digital Twin API",
    description=(
        "Backend API for real-time embedded device "
        "telemetry and fault monitoring."
    ),
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "project": "Embedded Digital Twin",
        "status": "API running",
        "database": "SQLite connected"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "database": "connected"
    }


@app.post("/telemetry")
def receive_telemetry(
    data: TelemetryData,
    db: Session = Depends(get_db)
):

    # Analyze incoming telemetry using the trained LOF model
    ml_result = anomaly_detector.predict(data)

    telemetry = TelemetryDB(
        device_id=data.device_id,
        timestamp=data.timestamp,
        temperature=data.temperature,
        voltage=data.voltage,
        current=data.current,
        vibration=data.vibration,
        rpm=data.rpm,
        status=data.status,
        ml_prediction=ml_result["prediction"],
        anomaly_score=ml_result["anomaly_score"]
    )

    db.add(telemetry)
    db.commit()
    db.refresh(telemetry)

    return {
        "message": "Telemetry analyzed and stored successfully",
        "id": telemetry.id,
        "device_id": telemetry.device_id,
        "sensor_status": telemetry.status,
        "ml_prediction": telemetry.ml_prediction,
        "anomaly_score": telemetry.anomaly_score
    }


@app.get("/telemetry")
def get_telemetry(
    limit: int = 100,
    db: Session = Depends(get_db)
):

    readings = (
        db.query(TelemetryDB)
        .order_by(TelemetryDB.id.desc())
        .limit(limit)
        .all()
    )

    return readings


@app.get("/telemetry/latest")
def latest_telemetry(
    db: Session = Depends(get_db)
):

    reading = (
        db.query(TelemetryDB)
        .order_by(TelemetryDB.id.desc())
        .first()
    )

    if reading is None:
        return {
            "message": "No telemetry available"
        }

    return reading


@app.get("/telemetry/count")
def telemetry_count(
    db: Session = Depends(get_db)
):

    count = db.query(TelemetryDB).count()

    return {
        "total_readings": count
    }
@app.post("/ml/train")
def train_ml_model(
    db: Session = Depends(get_db)
):
    readings = db.query(TelemetryDB).all()

    if len(readings) < 10:
        return {
            "success": False,
            "message": "At least 10 telemetry readings are required.",
            "available_readings": len(readings)
        }

    trained_count = anomaly_detector.train(readings)

    return {
        "success": True,
        "message": "Anomaly detection model trained successfully",
        "training_samples": trained_count,
        "algorithm": "Local Outlier Factor"
    }    

@app.get("/ml/status")
def ml_status():
    return {
        "algorithm": "Local Outlier Factor",
        "trained": anomaly_detector.is_trained,
        "model_loaded": anomaly_detector.is_trained
    }


@app.get("/ml/predict/latest")
def predict_latest(
    db: Session = Depends(get_db)
):
    reading = (
        db.query(TelemetryDB)
        .order_by(TelemetryDB.id.desc())
        .first()
    )

    if reading is None:
        return {
            "message": "No telemetry available"
        }

    result = anomaly_detector.predict(reading)

    return {
        "device_id": reading.device_id,
        "timestamp": reading.timestamp,
        "sensor_status": reading.status,
        "ml_prediction": result["prediction"],
        "anomaly_score": result["anomaly_score"]
    }
@app.get("/system/status")
def system_status(
    db: Session = Depends(get_db)
):
    try:
        total_readings = db.query(TelemetryDB).count()

        return {
            "system": "Embedded Digital Twin",
            "backend": "ONLINE",
            "database": "CONNECTED",
            "ml_algorithm": "Local Outlier Factor",
            "ml_model": "LOADED" if anomaly_detector.is_trained else "NOT_LOADED",
            "total_readings": total_readings,
            "mqtt_enabled": True
        }

    except Exception as e:
        return {
            "system": "Embedded Digital Twin",
            "backend": "ONLINE",
            "database": "ERROR",
            "error": str(e)
        }
@app.get("/system/summary")
def system_summary(
    db: Session = Depends(get_db)
):
    readings = db.query(TelemetryDB).all()

    total = len(readings)

    anomaly_count = sum(
        1 for reading in readings
        if reading.ml_prediction == "ANOMALY"
    )

    normal_count = sum(
        1 for reading in readings
        if reading.ml_prediction == "NORMAL"
    )
    unclassified_count = total - normal_count - anomaly_count
    
    anomaly_rate = (
        (anomaly_count / total) * 100
        if total > 0 else 0
    )

    latest = (
        db.query(TelemetryDB)
        .order_by(TelemetryDB.id.desc())
        .first()
    )

    return {
        "total_readings": total,
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "unclassified_count": unclassified_count,
        "anomaly_rate": round(anomaly_rate, 2),
        "latest_status": latest.status if latest else "UNKNOWN",
        "latest_ml_prediction": (
            latest.ml_prediction if latest else "UNKNOWN"
        )
    }    
@app.get("/telemetry/latest-id")
def latest_telemetry_id(
    db: Session = Depends(get_db)
):
    latest = (
        db.query(TelemetryDB)
        .order_by(TelemetryDB.id.desc())
        .first()
    )

    return {
        "latest_id": latest.id if latest else None,
        "telemetry_available": latest is not None
    }

@app.get("/telemetry/ingestion-health")
def telemetry_ingestion_health(
    db: Session = Depends(get_db)
):
    latest = (
        db.query(TelemetryDB)
        .order_by(TelemetryDB.id.desc())
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
        latest_time = datetime.fromisoformat(
            latest.timestamp.replace("Z", "+00:00")
    )

        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(
                tzinfo=timezone.utc
            )

        now = datetime.now(timezone.utc)

        age_seconds = (
            now - latest_time
        ).total_seconds()

        # Consider telemetry stale if no new reading
        # has arrived for more than 30 seconds.
        healthy = age_seconds <= 30

        return {
            "status": "ACTIVE" if healthy else "STALE",
            "healthy": healthy,
            "latest_id": latest.id,
            "latest_timestamp": latest.timestamp,
            "age_seconds": round(age_seconds, 2)
        }

    except Exception as e:
        return {
            "status": "TIMESTAMP_ERROR",
            "healthy": False,
            "latest_id": latest.id,
            "latest_timestamp": latest.timestamp,
            "error": str(e)
        }