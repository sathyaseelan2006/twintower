import os
from pathlib import Path
from dotenv import load_dotenv
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_BASE_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="Digital Twin Control Center",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# API FUNCTIONS
# ============================================================

def api_get(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=5
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None


def get_latest():
    return api_get("/telemetry/latest")


def get_history(limit=500):
    data = api_get(
        "/telemetry",
        params={"limit": limit}
    )
    return data if isinstance(data, list) else []


def get_count():
    data = api_get("/telemetry/count")

    if data:
        return data.get("total_readings", 0)

    return 0


def get_ml_status():
    return api_get("/ml/status")
def get_system_status():
    return api_get("/system/status")
def get_system_summary():
    return api_get("/system/summary")
def get_ingestion_health():
    return api_get("/telemetry/ingestion-health")
def train_ml_model():
    try:
        response = requests.post(
            f"{API_BASE_URL}/ml/train",
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        st.error(
            f"ML training failed: {e}"
        )
        return None
# ============================================================
# HEADER
# ============================================================

st.title("⚙️ Embedded Digital Twin Control Center")

st.caption(
    "MQTT telemetry • FastAPI • SQLite • "
    "Local Outlier Factor anomaly detection"
)


# ============================================================
# LOAD DATA
# ============================================================

latest = get_latest()
history = get_history()
total_readings = get_count()
ml_status = get_ml_status()
system_status = get_system_status()
system_summary = get_system_summary()
ingestion_health = get_ingestion_health()

if latest is None:
    st.error(
        "🔴 FastAPI backend is offline. "
        "Start the backend on port 8000."
    )
    st.stop()


if "message" in latest:
    st.warning(latest["message"])
    st.stop()


df = pd.DataFrame(history)


if not df.empty:

    if "id" in df.columns:
        df = df.sort_values("id")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.subheader("System Status")

if system_status:

    status1, status2, status3, status4 = st.columns(4)

    status1.metric(
        "Backend",
        system_status.get("backend", "UNKNOWN")
    )

    status2.metric(
        "Database",
        system_status.get("database", "UNKNOWN")
    )

    status3.metric(
        "ML Model",
        system_status.get("ml_model", "UNKNOWN")
    )

    status4.metric(
        "Stored Telemetry",
        system_status.get("total_readings", total_readings)
    )

    mqtt_enabled = system_status.get("mqtt_enabled", False)

    st.caption(
        f"ML Algorithm: "
        f"{system_status.get('ml_algorithm', 'Unknown')} | "
        f"MQTT: {'Enabled' if mqtt_enabled else 'Disabled'} | "
        f"Device: {latest.get('device_id', 'Unknown')}"
    )

else:
    st.warning(
        "System status endpoint is currently unavailable."
    )

# ============================================================
# LIVE INGESTION HEALTH
# ============================================================

st.subheader("Live Telemetry Health")

if ingestion_health:

    ingestion_status = ingestion_health.get("status", "UNKNOWN")
    ingestion_healthy = ingestion_health.get("healthy", False)
    latest_id = ingestion_health.get("latest_id", "N/A")
    age_seconds = ingestion_health.get("age_seconds")

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Ingestion Status",
        ingestion_status
    )

    h2.metric(
        "Latest Telemetry ID",
        latest_id
    )

    h3.metric(
        "Reading Age",
        f"{age_seconds:.1f} sec"
        if isinstance(age_seconds, (int, float))
        else "N/A"
    )

    if ingestion_healthy:
        st.success("Live telemetry ingestion is ACTIVE.")
    elif ingestion_status == "STALE":
        st.error("Telemetry is STALE — no recent readings received.")
    else:
        st.warning(
            f"Telemetry ingestion status: {ingestion_status}"
        )

else:
    st.error("Unable to retrieve telemetry ingestion health.")

# ============================================================
# OPERATIONAL SUMMARY
# ============================================================


if system_summary:

    op1, op2, op3, op4, op5 = st.columns(5)

    op1.metric(
        "Total Readings",
        system_summary.get("total_readings", 0)
    )

    op2.metric(
        "Normal Predictions",
        system_summary.get("normal_count", 0)
    )

    op3.metric(
        "ML Anomalies",
        system_summary.get("anomaly_count", 0)
    )

    op4.metric(
        "Anomaly Rate",
        f"{system_summary.get('anomaly_rate', 0):.2f}%"
    )

    op5.metric(
    "Unclassified",
    system_summary.get("unclassified_count", 0)
    )   

    latest_status = system_summary.get(
        "latest_status",
        "UNKNOWN"
    )

    latest_prediction = system_summary.get(
        "latest_ml_prediction",
        "UNKNOWN"
    )

    st.caption(
        f"Latest Device Status: {latest_status} | "
        f"Latest ML Prediction: {latest_prediction}"
    )

else:
    st.warning(
        "Operational summary is currently unavailable."
    )

     
# ============================================================
# LIVE SENSOR DATA
# ============================================================

st.subheader("Live Device Telemetry")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "🌡️ Temperature",
    f"{latest.get('temperature', 0):.2f} °C"
)

c2.metric(
    "⚡ Voltage",
    f"{latest.get('voltage', 0):.2f} V"
)

c3.metric(
    "🔌 Current",
    f"{latest.get('current', 0):.2f} A"
)

c4.metric(
    "📳 Vibration",
    f"{latest.get('vibration', 0):.3f}"
)

c5.metric(
    "🔄 RPM",
    latest.get("rpm", 0)
)

# ============================================================
# REAL-TIME FAULT ALERT
# ============================================================

if latest:

    device_status = latest.get("status", "UNKNOWN")

    if device_status == "OVERHEATING":
        st.error(
            "⚠️ OVERHEATING detected — temperature is above "
            "the normal operating range."
        )

    elif device_status == "HIGH_VIBRATION":
        st.error(
            "⚠️ HIGH VIBRATION detected — inspect the motor "
            "or mechanical assembly."
        )

    elif device_status == "OVERCURRENT":
        st.error(
            "⚠️ OVERCURRENT detected — electrical current "
            "exceeds the normal range."
        )

    elif device_status == "LOW_VOLTAGE":
        st.warning(
            "⚠️ LOW VOLTAGE detected — supply voltage is "
            "below the expected range."
        )

    elif device_status == "NORMAL":
        st.success(
            "Device operating normally."
        )

    else:
        st.warning(
            f"Unknown device status: {device_status}"
        )
# ============================================================
# ML FAULT DETECTION
# ============================================================

st.subheader("AI / ML Fault Detection")

sensor_status = latest.get(
    "status",
    "UNKNOWN"
)

ml_prediction = latest.get(
    "ml_prediction",
    "UNKNOWN"
)

anomaly_score = latest.get(
    "anomaly_score"
)

st.subheader("Operational Summary")
m1, m2, m3 = st.columns(3)

m1.metric(
    "Simulator Status",
    sensor_status
)

m2.metric(
    "LOF Prediction",
    ml_prediction
)

m3.metric(
    "Anomaly Score",
    (
        f"{anomaly_score:.4f}"
        if anomaly_score is not None
        else "N/A"
    )
)


if ml_prediction == "ANOMALY":

    st.error(
        f"⚠️ ML ANOMALY DETECTED — "
        f"Device status: {sensor_status}"
    )

elif ml_prediction == "NORMAL":

    st.success(
        "✅ Device operating inside the learned normal region."
    )

else:

    st.warning(
        "ML prediction unavailable."
    )

st.subheader("ML Model Control")

if st.button("Retrain LOF Model"):

    with st.spinner("Training LOF model..."):

        training_result = train_ml_model()

    if training_result and training_result.get("success"):

        st.success(
            "LOF model retrained successfully "
            f"using {training_result.get('training_samples', 0)} samples."
        )

    else:
        st.error("LOF model retraining failed.")
# ============================================================
# DEVICE HEALTH GAUGE
# ============================================================

st.subheader("Digital Twin Health")

if ml_prediction == "ANOMALY":

    health_score = 30

elif anomaly_score is not None:

    health_score = int(
        max(
            50,
            min(
                100,
                85 + (anomaly_score * 100)
            )
        )
    )

else:

    health_score = 50


health_figure = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={
            "text": "Device Health Score"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            }
        }
    )
)

health_figure.update_layout(
    height=300
)

st.plotly_chart(
    health_figure,
    width="stretch"
)

st.caption(
    "Health Score is a monitoring heuristic derived from "
    "the LOF result and anomaly score. It is not a "
    "remaining-useful-life prediction."
)


# ============================================================
# ANOMALY STATISTICS
# ============================================================

st.subheader("Fault & Anomaly Analytics")


if not df.empty:

    ml_anomalies = df[
        df["ml_prediction"] == "ANOMALY"
    ]

    normal_predictions = df[
        df["ml_prediction"] == "NORMAL"
    ]

    total_ml_records = (
        len(ml_anomalies)
        + len(normal_predictions)
    )

    anomaly_rate = (
        len(ml_anomalies)
        / total_ml_records
        * 100
        if total_ml_records > 0
        else 0
    )


    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "ML Anomalies",
        len(ml_anomalies)
    )

    a2.metric(
        "ML Normal",
        len(normal_predictions)
    )

    a3.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )

    if "status" in df.columns:

        injected_faults = (
            df["status"] != "NORMAL"
        ).sum()

    else:

        injected_faults = 0

    a4.metric(
        "Simulator Faults",
        injected_faults
    )


# ============================================================
# FAULT DISTRIBUTION
# ============================================================

    fault_data = df[
        df["status"] != "NORMAL"
    ]


    if not fault_data.empty:

        fault_counts = (
            fault_data["status"]
            .value_counts()
            .reset_index()
        )

        fault_counts.columns = [
            "Fault",
            "Count"
        ]


        fault_chart = px.bar(
            fault_counts,
            x="Fault",
            y="Count",
            title="Fault Distribution"
        )

        st.plotly_chart(
            fault_chart,
            width="stretch"
        )


# ============================================================
# ANOMALY SCORE TREND
# ============================================================

    score_data = df.dropna(
        subset=["anomaly_score"]
    )


    if not score_data.empty:

        score_chart = px.line(
            score_data,
            x="timestamp",
            y="anomaly_score",
            title="LOF Anomaly Score Trend"
        )

        score_chart.add_hline(
            y=0,
            line_dash="dash",
            annotation_text="Decision Boundary"
        )

        st.plotly_chart(
            score_chart,
            width="stretch"
        )


# ============================================================
# SENSOR HISTORY
# ============================================================

st.subheader("Historical Sensor Analytics")


if not df.empty:

    temperature_chart = px.line(
        df,
        x="timestamp",
        y="temperature",
        title="Temperature"
    )

    st.plotly_chart(
        temperature_chart,
        width="stretch"
    )


    electrical_data = df.melt(
        id_vars=["timestamp"],
        value_vars=[
            "voltage",
            "current"
        ],
        var_name="Sensor",
        value_name="Value"
    )

    electrical_chart = px.line(
        electrical_data,
        x="timestamp",
        y="Value",
        color="Sensor",
        title="Voltage & Current"
    )

    st.plotly_chart(
        electrical_chart,
        width="stretch"
    )


    vibration_chart = px.line(
        df,
        x="timestamp",
        y="vibration",
        title="Vibration"
    )

    st.plotly_chart(
        vibration_chart,
        width="stretch"
    )


    rpm_chart = px.line(
        df,
        x="timestamp",
        y="rpm",
        title="Motor RPM"
    )

    st.plotly_chart(
        rpm_chart,
        width="stretch"
    )

# ============================================================
# TELEMETRY DATA EXPORT
# ============================================================

st.subheader("Telemetry Data Export")

if history:

    export_df = pd.DataFrame(history)

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Telemetry CSV",
        data=csv_data,
        file_name="digital_twin_telemetry.csv",
        mime="text/csv"
    )

    st.caption(
        f"{len(export_df)} telemetry records available for export."
    )

else:
    st.warning("No telemetry data available for export.")
    
# ============================================================
# FAULT EVENT TABLE
# ============================================================

st.subheader("Fault Event Log")


if not df.empty:

    event_data = df[
        (
            df["status"] != "NORMAL"
        )
        |
        (
            df["ml_prediction"] == "ANOMALY"
        )
    ]


    if not event_data.empty:

        event_columns = [
            column
            for column in [
                "id",
                "timestamp",
                "temperature",
                "voltage",
                "current",
                "vibration",
                "rpm",
                "status",
                "ml_prediction",
                "anomaly_score"
            ]
            if column in event_data.columns
        ]


        st.dataframe(
            event_data[
                event_columns
            ].sort_values(
                "id",
                ascending=False
            ),
            width="stretch",
            hide_index=True
        )

    else:

        st.success(
            "No fault events recorded."
        )


# ============================================================
# RECENT TELEMETRY
# ============================================================

with st.expander(
    "View Recent Raw Telemetry"
):

    if not df.empty:

        st.dataframe(
            df.tail(50).sort_values(
                "id",
                ascending=False
            ),
            width="stretch",
            hide_index=True
        )


# ============================================================
# AUTO REFRESH
# ============================================================

st.divider()

r1, r2 = st.columns(2)

with r1:

    refresh_seconds = st.selectbox(
        "Refresh Interval",
        [2, 5, 10, 30],
        index=1
    )

with r2:

    auto_refresh = st.toggle(
        "Automatic Refresh",
        value=True
    )


st.caption(
    f"Latest telemetry: "
    f"{latest.get('timestamp', 'Unknown')}"
)


if auto_refresh:

    time.sleep(refresh_seconds)
    st.rerun()

# ============================================================
# ML MODEL CONTROL
# ============================================================

st.subheader("ML Model Control")

if st.button("Retrain LOF Model"):

    with st.spinner("Training LOF model..."):

        training_result = train_ml_model()

    if training_result and training_result.get("success"):

        st.success(
            "LOF model retrained successfully "
            f"using {training_result.get('training_samples', 0)} samples."
        )

    else:
        st.error(
            "LOF model retraining failed."
        )