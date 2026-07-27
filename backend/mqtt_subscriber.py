import json
import requests
import paho.mqtt.client as mqtt
import os
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))

TOPIC = os.getenv(
    "MQTT_TOPIC",
    "uday/digital-twin/DT-DEVICE-001/telemetry"
)

FASTAPI_BASE_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000"
)

FASTAPI_URL = f"{FASTAPI_BASE_URL}/telemetry"


def on_connect(client, userdata, flags, reason_code, properties):

    if reason_code == 0:

        print("Connected to MQTT broker successfully.")

        client.subscribe(
            TOPIC,
            qos=1
        )

        print(f"Subscribed to: {TOPIC}")

    else:

        print(
            f"MQTT connection failed: {reason_code}"
        )


def on_message(client, userdata, message):

    try:

        payload = message.payload.decode("utf-8")

        telemetry = json.loads(payload)

        print(
            f"\nMQTT Received | "
            f"Device: {telemetry['device_id']} | "
            f"Status: {telemetry['status']}"
        )

        response = requests.post(
            FASTAPI_URL,
            json=telemetry,
            timeout=5
        )

        if response.status_code == 200:

            result = response.json()

            print(
                f"FastAPI Stored | "
                f"ID: {result['id']} | "
                f"ML: {result['ml_prediction']} | "
                f"Score: {result['anomaly_score']}"
            )

        else:

            print(
                f"FastAPI error: "
                f"{response.status_code}"
            )

    except json.JSONDecodeError:

        print("Invalid MQTT JSON received.")

    except requests.RequestException as error:

        print(
            f"FastAPI connection error: {error}"
        )

    except Exception as error:

        print(
            f"Processing error: {error}"
        )


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect = on_connect
client.on_message = on_message


print("======================================")
print(" Digital Twin MQTT Ingestion Service")
print("======================================")

print(f"Broker : {BROKER}")
print(f"Topic  : {TOPIC}")

print("\nConnecting to MQTT broker...")


try:

    client.connect(
        BROKER,
        PORT,
        60
    )

    client.loop_forever()


except KeyboardInterrupt:

    print("\nMQTT ingestion service stopped.")

    client.disconnect()