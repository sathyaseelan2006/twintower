# Embedded Digital Twin for Real-Time Fault Monitoring

A real-time Embedded Digital Twin system for monitoring simulated embedded-device telemetry, detecting abnormal operating conditions, and visualizing device health through an interactive dashboard.

The system combines MQTT communication, FastAPI, SQLite, machine-learning-based anomaly detection using Local Outlier Factor (LOF), Streamlit visualization, automated API testing, and Docker containerization.

---

## Features

- Real-time embedded-device telemetry simulation
- MQTT-based telemetry communication
- FastAPI REST backend
- SQLite telemetry storage
- Local Outlier Factor (LOF) anomaly detection
- Model retraining using stored telemetry
- Real-time fault detection
- Streamlit Digital Twin dashboard
- Live telemetry ingestion-health monitoring
- Historical sensor analytics
- Fault and anomaly analytics
- Digital Twin health score
- CSV telemetry export
- Docker Compose deployment
- Docker service health checks
- Automated API testing with Pytest

---

## System Architecture

```text
Embedded Device Simulator
        |
        | MQTT
        v
Eclipse Mosquitto Broker
        |
        v
MQTT Subscriber
        |
        | HTTP
        v
FastAPI Backend
        |
        +--------------------+
        |                    |
        v                    v
SQLite Database       LOF ML Model
        |                    |
        +---------+----------+
                  |
                  v
          Streamlit Dashboard
```

---

## Telemetry Parameters

The simulated embedded device produces:

- Temperature
- Voltage
- Current
- Vibration
- RPM
- Device status
- Timestamp
- Device ID

The simulator can inject:

- `OVERHEATING`
- `HIGH_VIBRATION`
- `OVERCURRENT`
- `LOW_VOLTAGE`
- `NORMAL`

---

## Machine Learning

The project uses **Local Outlier Factor (LOF)** for anomaly detection.

Input features:

```text
Temperature
Voltage
Current
Vibration
RPM
```

Telemetry is standardized using `StandardScaler` before being processed by the LOF model.

The system returns:

```text
NORMAL
ANOMALY
```

along with an anomaly score.

The trained model and scaler are stored as:

```text
ml/lof_model.pkl
ml/scaler.pkl
```

The model can also be retrained using telemetry stored in the database.

---

## Technology Stack

### Backend

- Python
- FastAPI
- SQLAlchemy
- SQLite
- Uvicorn

### IoT / Messaging

- MQTT
- Eclipse Mosquitto
- Paho MQTT

### Machine Learning

- Scikit-learn
- Local Outlier Factor
- StandardScaler
- Pandas
- NumPy
- Joblib

### Dashboard

- Streamlit
- Plotly
- Pandas

### DevOps & Testing

- Docker
- Docker Compose
- Pytest
- Git

---

## Project Structure

```text
Embedded_Digital_Twin/
|
|-- backend/
|   |-- __init__.py
|   |-- database.py
|   |-- main.py
|   |-- models.py
|   `-- mqtt_subscriber.py
|
|-- dashboard/
|   |-- __init__.py
|   `-- app.py
|
|-- data/
|   `-- training_dataset.csv
|
|-- ml/
|   |-- __init__.py
|   |-- anomaly_detector.py
|   |-- isolation_forest_model.pkl
|   |-- lof_model.pkl
|   |-- model_comparison.py
|   |-- scaler.pkl
|   `-- train_model.py
|
|-- simulator/
|   |-- __init__.py
|   |-- dataset_generator.py
|   |-- device_simulator.py
|   `-- mqtt_device_simulator.py
|
|-- tests/
|   `-- test_api.py
|
|-- .dockerignore
|-- .env.example
|-- .gitignore
|-- Dockerfile
|-- docker-compose.yml
|-- mosquitto.conf
|-- requirements.txt
`-- README.md
```

---

## Running with Docker

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Embedded_Digital_Twin
```

### 2. Create environment configuration

Copy:

```text
.env.example
```

to:

```text
.env
```

Never commit `.env` containing private configuration.

### 3. Build and start the system

```bash
docker compose up -d --build
```

### 4. Check container status

```bash
docker compose ps
```

Expected services:

```text
digital-twin-mosquitto
digital-twin-backend
digital-twin-mqtt-subscriber
digital-twin-dashboard
```

Mosquitto, FastAPI, and the dashboard include Docker health checks.

---

## Start the MQTT Device Simulator

From the project directory:

```bash
python simulator/mqtt_device_simulator.py
```

The simulator publishes telemetry to:

```text
uday/digital-twin/DT-DEVICE-001/telemetry
```

---

## Web Interfaces

### Digital Twin Dashboard

```text
http://localhost:8501
```

### FastAPI

```text
http://localhost:8000
```

### FastAPI Documentation

```text
http://localhost:8000/docs
```

---

## Important API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend health |
| POST | `/telemetry` | Store and analyze telemetry |
| GET | `/telemetry` | Retrieve telemetry history |
| GET | `/telemetry/latest` | Latest telemetry |
| GET | `/telemetry/count` | Number of stored readings |
| GET | `/telemetry/latest-id` | Latest telemetry ID |
| GET | `/telemetry/ingestion-health` | Live ingestion health |
| GET | `/ml/status` | ML model status |
| GET | `/ml/predict/latest` | Analyze latest reading |
| POST | `/ml/train` | Retrain LOF model |
| GET | `/system/status` | Overall system status |
| GET | `/system/summary` | Operational statistics |

---

## Telemetry Ingestion Health

The backend monitors how recently telemetry was received.

Possible states:

```text
ACTIVE
STALE
NO_DATA
TIMESTAMP_ERROR
```

`ACTIVE` indicates recent telemetry is reaching the system.

`STALE` indicates that no new telemetry has arrived within the configured threshold.

Telemetry timestamps are generated in UTC.

---

## Dashboard

The Streamlit dashboard provides:

- System status
- Live telemetry health
- Operational summary
- Live device telemetry
- Real-time fault alerts
- LOF anomaly prediction
- Anomaly score
- ML model retraining
- Digital Twin health score
- Fault distribution
- ML anomaly distribution
- Anomaly score trends
- Historical sensor graphs
- Fault event log
- CSV telemetry export

---

## ML Model Retraining

The LOF model can be retrained through the API:

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:8000/ml/train
```

or directly from the Streamlit dashboard using:

```text
Retrain LOF Model
```

---

## Automated Testing

Run:

```bash
python -m pytest tests/test_api.py -v
```

The project includes automated tests for core backend functionality, including system status, operational summary, telemetry endpoints, and ingestion-health monitoring.

---

## Stop the System

```bash
docker compose down
```

Stored SQLite telemetry is persisted through the project's data volume configuration.

---

## Screenshots

Add dashboard screenshots here before publishing the repository.

Suggested screenshots:

1. System Status and Live Telemetry Health
2. Live Device Telemetry
3. AI/ML Fault Detection
4. Digital Twin Health
5. Fault & Anomaly Analytics
6. Historical Sensor Analytics
7. Fault Event Log
8. Telemetry CSV Export

---

## Future Enhancements

Potential extensions include:

- Real ESP32/STM32 sensor integration
- Cloud MQTT deployment
- Multiple Digital Twin devices
- WebSocket-based dashboard updates
- Predictive maintenance
- Remaining Useful Life estimation
- Email/SMS fault notifications
- Time-series databases
- Kubernetes deployment

---

## Author

**Uday Chandra S**

Electronics and Communication Engineering

---

## Project Status

Core implementation complete.

The system supports real-time MQTT telemetry ingestion, persistent storage, machine-learning anomaly detection, Digital Twin visualization, live ingestion monitoring, model retraining, automated testing, and Docker-based deployment.