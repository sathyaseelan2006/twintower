from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "Embedded Digital Twin"
    assert data["status"] == "API running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200


def test_get_telemetry():
    response = client.get("/telemetry")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_telemetry_count():
    response = client.get("/telemetry/count")

    assert response.status_code == 200

    data = response.json()

    assert "total_readings" in data
    assert isinstance(data["total_readings"], int)


def test_latest_telemetry():
    response = client.get("/telemetry/latest")

    assert response.status_code == 200

    data = response.json()

    assert "device_id" in data
    assert "temperature" in data
    assert "voltage" in data
    assert "current" in data
    assert "vibration" in data
    assert "rpm" in data


def test_ml_status():
    response = client.get("/ml/status")

    assert response.status_code == 200

    data = response.json()

    assert data["algorithm"] == "Local Outlier Factor"
    assert data["model_loaded"] is True


def test_post_telemetry():

    telemetry = {
        "device_id": "TEST-DEVICE-001",
        "timestamp": "2026-07-26T22:45:00",
        "temperature": 30.0,
        "voltage": 12.0,
        "current": 1.5,
        "vibration": 0.2,
        "rpm": 1500,
        "status": "NORMAL"
    }

    response = client.post(
        "/telemetry",
        json=telemetry
    )

    assert response.status_code == 200

    data = response.json()

    assert data["device_id"] == "TEST-DEVICE-001"

    assert data["ml_prediction"] in [
        "NORMAL",
        "ANOMALY"
    ]

    assert "anomaly_score" in data


def test_invalid_telemetry():

    invalid_data = {
        "device_id": "TEST-DEVICE-002"
    }

    response = client.post(
        "/telemetry",
        json=invalid_data
    )

    assert response.status_code == 422
def test_system_status():
    response = client.get("/system/status")

    assert response.status_code == 200

    data = response.json()

    assert data["system"] == "Embedded Digital Twin"
    assert data["backend"] == "ONLINE"
    assert data["database"] == "CONNECTED"
    assert data["ml_algorithm"] == "Local Outlier Factor"
    assert "ml_model" in data
    assert "total_readings" in data
    assert "mqtt_enabled" in data 

def test_system_summary():
    response = client.get("/system/summary")

    assert response.status_code == 200

    data = response.json()

    assert "total_readings" in data
    assert "normal_count" in data
    assert "anomaly_count" in data
    assert "unclassified_count" in data
    assert "anomaly_rate" in data
    assert "latest_status" in data
    assert "latest_ml_prediction" in data

    assert data["total_readings"] >= 0
    assert data["normal_count"] >= 0
    assert data["anomaly_count"] >= 0
    assert data["unclassified_count"] >= 0

    assert (
        data["normal_count"]
        + data["anomaly_count"]
        + data["unclassified_count"]
        == data["total_readings"]
    )

def test_latest_telemetry_id():
    response = client.get("/telemetry/latest-id")

    assert response.status_code == 200

    data = response.json()

    assert "latest_id" in data
    assert "telemetry_available" in data
    assert isinstance(data["telemetry_available"], bool)

    if data["telemetry_available"]:
        assert data["latest_id"] is not None
        assert data["latest_id"] >= 1
def test_ingestion_health():
    response = client.get("/telemetry/ingestion-health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "healthy" in data
    assert "latest_id" in data

    assert data["status"] in {
        "ACTIVE",
        "STALE",
        "NO_DATA",
        "TIMESTAMP_ERROR"
    }

    assert isinstance(data["healthy"], bool)

    if data["latest_id"] is not None:
        assert data["latest_id"] >= 1