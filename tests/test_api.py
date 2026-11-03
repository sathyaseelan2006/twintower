from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert data["status"] == "ONLINE"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_control_state_defaults():
    response = client.get("/control/state")
    assert response.status_code == 200
    data = response.json()
    assert "mode" in data
    assert "load_status" in data

def test_post_microgrid_telemetry():
    packet = {
        "device_id": "TEST-ESP32-001",
        "solar_voltage": 12.5,
        "battery_voltage": 16.2,
        "battery_level": 95.0,
        "load_current": 0.20,
        "load_power": 3.24,
        "mode": "AUTO",
        "load_status": "ON",
        "servo_angle": 30,
        "led_status": "OFF"
    }
    response = client.post("/telemetry", json=packet)
    assert response.status_code == 200
    data = response.json()
    assert "battery_soh" in data
    assert "optimization" in data
    assert "safety_status" in data

def test_control_mode_switch():
    response = client.post("/control/mode", json={"mode": "SOLAR"})
    assert response.status_code == 200
    assert response.json()["mode"] == "SOLAR"

    # Verify updated in state
    state = client.get("/control/state").json()
    assert state["mode"] == "SOLAR"

def test_control_load_toggle():
    response = client.post("/control/load", json={"load_status": "OFF"})
    assert response.status_code == 200
    assert response.json()["load_status"] == "OFF"

    state = client.get("/control/state").json()
    assert state["load_status"] == "OFF"

def test_control_servo():
    response = client.post("/control/servo", json={"angle": 90})
    assert response.status_code == 200
    assert response.json()["servo_angle"] == 90

def test_telemetry_history():
    response = client.get("/telemetry?limit=5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_system_summary():
    response = client.get("/system/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_readings" in data
    assert "current_mode" in data
