import random
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Embedded Digital Twin",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "telemetry_history" not in st.session_state:
    st.session_state.telemetry_history = []

if "reading_id" not in st.session_state:
    st.session_state.reading_id = 0


# ============================================================
# TELEMETRY GENERATOR
# ============================================================

def generate_reading():

    st.session_state.reading_id += 1

    temperature = random.uniform(28, 35)
    voltage = random.uniform(11.7, 12.3)
    current = random.uniform(1.2, 2.0)
    vibration = random.uniform(0.10, 0.35)
    rpm = random.randint(1450, 1550)

    status = "NORMAL"
    ml_prediction = "NORMAL"
    anomaly_score = random.uniform(0.02, 0.20)

    fault = random.random()

    if fault < 0.04:

        temperature = random.uniform(60, 80)

        status = "OVERHEATING"
        ml_prediction = "ANOMALY"

        anomaly_score = random.uniform(
            -0.80,
            -0.20
        )

    elif fault < 0.08:

        vibration = random.uniform(1.5, 3.5)

        status = "HIGH_VIBRATION"
        ml_prediction = "ANOMALY"

        anomaly_score = random.uniform(
            -0.80,
            -0.20
        )

    elif fault < 0.12:

        current = random.uniform(4.0, 6.5)

        status = "OVERCURRENT"
        ml_prediction = "ANOMALY"

        anomaly_score = random.uniform(
            -0.80,
            -0.20
        )

    elif fault < 0.16:

        voltage = random.uniform(7.5, 9.5)

        status = "LOW_VOLTAGE"
        ml_prediction = "ANOMALY"

        anomaly_score = random.uniform(
            -0.80,
            -0.20
        )

    return {
        "id": st.session_state.reading_id,
        "device_id": "DT-DEVICE-001",
        "timestamp": datetime.now(),
        "temperature": round(temperature, 2),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "vibration": round(vibration, 3),
        "rpm": rpm,
        "status": status,
        "ml_prediction": ml_prediction,
        "anomaly_score": round(
            anomaly_score,
            4
        )
    }


# ============================================================
# GENERATE NEW READING
# ============================================================

new_reading = generate_reading()

st.session_state.telemetry_history.append(
    new_reading
)

st.session_state.telemetry_history = (
    st.session_state.telemetry_history[-100:]
)

history = st.session_state.telemetry_history

latest = history[-1]

df = pd.DataFrame(history)


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚙️ Embedded Digital Twin Control Center"
)

st.caption(
    "Real-Time Telemetry • Fault Monitoring • "
    "Machine Learning Anomaly Detection"
)

st.info(
    "Public Demonstration Mode — "
    "simulated embedded-device telemetry"
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.subheader("System Status")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "System",
    "ONLINE"
)

c2.metric(
    "Telemetry",
    "ACTIVE"
)

c3.metric(
    "ML Model",
    "LOADED"
)

c4.metric(
    "Readings",
    len(history)
)


# ============================================================
# LIVE DEVICE TELEMETRY
# ============================================================

st.subheader("Live Device Telemetry")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Temperature",
    f"{latest['temperature']} °C"
)

m2.metric(
    "Voltage",
    f"{latest['voltage']} V"
)

m3.metric(
    "Current",
    f"{latest['current']} A"
)

m4.metric(
    "Vibration",
    latest["vibration"]
)

m5.metric(
    "RPM",
    latest["rpm"]
)


# ============================================================
# DEVICE STATUS
# ============================================================

st.subheader("Device Status")

s1, s2, s3 = st.columns(3)

s1.metric(
    "Device",
    latest["device_id"]
)

s2.metric(
    "Sensor Status",
    latest["status"]
)

s3.metric(
    "ML Prediction",
    latest["ml_prediction"]
)


if latest["status"] == "NORMAL":

    st.success(
        "Device operating normally."
    )

else:

    st.error(
        f"⚠️ Fault detected: "
        f"{latest['status']}"
    )


# ============================================================
# AI / ML FAULT DETECTION
# ============================================================

st.subheader("AI / ML Fault Detection")

ml1, ml2, ml3 = st.columns(3)

ml1.metric(
    "Algorithm",
    "Local Outlier Factor"
)

ml2.metric(
    "Prediction",
    latest["ml_prediction"]
)

ml3.metric(
    "Anomaly Score",
    latest["anomaly_score"]
)


if latest["ml_prediction"] == "ANOMALY":

    st.error(
        "⚠️ ML ANOMALY DETECTED"
    )

else:

    st.success(
        "✅ Device operating inside "
        "the normal region."
    )


# ============================================================
# DIGITAL TWIN HEALTH
# ============================================================

st.subheader("Digital Twin Health")

if latest["ml_prediction"] == "ANOMALY":
    health_score = 35
else:
    health_score = 95


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
            },

            "steps": [

                {
                    "range": [0, 40]
                },

                {
                    "range": [40, 70]
                },

                {
                    "range": [70, 100]
                }

            ]
        }
    )
)

health_figure.update_layout(
    height=300
)

st.plotly_chart(
    health_figure,
    use_container_width=True
)


# ============================================================
# LIVE SENSOR WAVEFORMS
# ============================================================

st.subheader("📈 Live Sensor Waveforms")

wave_df = df.tail(50)


temperature_chart = px.line(
    wave_df,
    x="timestamp",
    y="temperature",
    title="Temperature Waveform"
)

st.plotly_chart(
    temperature_chart,
    use_container_width=True
)


electrical_chart = go.Figure()

electrical_chart.add_trace(

    go.Scatter(
        x=wave_df["timestamp"],
        y=wave_df["voltage"],
        mode="lines",
        name="Voltage"
    )
)

electrical_chart.add_trace(

    go.Scatter(
        x=wave_df["timestamp"],
        y=wave_df["current"],
        mode="lines",
        name="Current"
    )
)

electrical_chart.update_layout(
    title="Voltage & Current Waveform",
    xaxis_title="Time"
)

st.plotly_chart(
    electrical_chart,
    use_container_width=True
)


vibration_chart = px.line(
    wave_df,
    x="timestamp",
    y="vibration",
    title="Vibration Waveform"
)

st.plotly_chart(
    vibration_chart,
    use_container_width=True
)


rpm_chart = px.line(
    wave_df,
    x="timestamp",
    y="rpm",
    title="Motor RPM Waveform"
)

st.plotly_chart(
    rpm_chart,
    use_container_width=True
)


# ============================================================
# FAULT ANALYTICS
# ============================================================

st.subheader("Fault Analytics")

normal_count = sum(
    1
    for reading in history
    if reading["ml_prediction"] == "NORMAL"
)

anomaly_count = sum(
    1
    for reading in history
    if reading["ml_prediction"] == "ANOMALY"
)

total = len(history)

anomaly_rate = (
    (anomaly_count / total) * 100
    if total > 0
    else 0
)


a1, a2, a3 = st.columns(3)

a1.metric(
    "Normal",
    normal_count
)

a2.metric(
    "Anomalies",
    anomaly_count
)

a3.metric(
    "Anomaly Rate",
    f"{anomaly_rate:.2f}%"
)


# ============================================================
# FAULT EVENT LOG
# ============================================================

st.subheader("Fault Event Log")

fault_df = df[
    df["status"] != "NORMAL"
]

if fault_df.empty:

    st.success(
        "No fault events recorded."
    )

else:

    st.dataframe(

        fault_df[
            [
                "timestamp",
                "status",
                "temperature",
                "voltage",
                "current",
                "vibration",
                "rpm",
                "ml_prediction"
            ]
        ],

        use_container_width=True
    )


# ============================================================
# TELEMETRY DATA
# ============================================================

with st.expander(
    "View Telemetry Data"
):

    st.dataframe(
        df.sort_values(
            "id",
            ascending=False
        ),
        use_container_width=True
    )


# ============================================================
# CSV EXPORT
# ============================================================

st.subheader("Telemetry Export")

csv_data = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Telemetry CSV",
    csv_data,
    "digital_twin_demo_telemetry.csv",
    "text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Embedded Digital Twin • "
    "IoT Condition Monitoring • "
    "Machine Learning Fault Detection"
)


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(2)

st.rerun()