import os
from pathlib import Path
from dotenv import load_dotenv
import json
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))

DEVICE_ID = os.getenv(
    "DEVICE_ID",
    "DT-DEVICE-001"
)

TOPIC = os.getenv(
    "MQTT_TOPIC",
    f"uday/digital-twin/{DEVICE_ID}/telemetry"
)


def generate_reading():

    temperature = random.uniform(28, 35)
    voltage = random.uniform(11.7, 12.3)
    current = random.uniform(1.2, 2.0)
    vibration = random.uniform(0.1, 0.35)
    rpm = random.randint(1450, 1550)

    status = "NORMAL"

    # Randomly inject faults
    fault_chance = random.random()

    if fault_chance < 0.05:

        temperature = random.uniform(60, 80)
        status = "OVERHEATING"

    elif fault_chance < 0.10:

        vibration = random.uniform(1.5, 3.5)
        status = "HIGH_VIBRATION"

    elif fault_chance < 0.15:

        current = random.uniform(4.0, 6.5)
        status = "OVERCURRENT"

    elif fault_chance < 0.20:

        voltage = random.uniform(7.5, 9.5)
        status = "LOW_VOLTAGE"

    return {
        "device_id": DEVICE_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        ),
        "temperature": round(temperature, 2),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "vibration": round(vibration, 3),
        "rpm": rpm,
        "status": status
    }


def on_connect(client, userdata, flags, reason_code, properties):

    if reason_code == 0:
        print("Connected to MQTT broker successfully.")
    else:
        print(
            f"MQTT connection failed: {reason_code}"
        )


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect = on_connect

print("======================================")
print(" Embedded Digital Twin MQTT Simulator")
print("======================================")

print(f"Broker : {BROKER}")
print(f"Topic  : {TOPIC}")

print("\nConnecting...")

client.connect(
    BROKER,
    PORT,
    60
)

client.loop_start()


try:

    while True:

        telemetry = generate_reading()

        payload = json.dumps(telemetry)

        result = client.publish(
            TOPIC,
            payload,
            qos=1
        )

        result.wait_for_publish()

        print(
            f"Published | "
            f"Temp: {telemetry['temperature']}°C | "
            f"Voltage: {telemetry['voltage']}V | "
            f"Current: {telemetry['current']}A | "
            f"Vibration: {telemetry['vibration']} | "
            f"RPM: {telemetry['rpm']} | "
            f"Status: {telemetry['status']}"
        )

        time.sleep(2)


except KeyboardInterrupt:

    print("\nStopping MQTT simulator...")


finally:

    client.loop_stop()
    client.disconnect()

    print("MQTT simulator disconnected.")