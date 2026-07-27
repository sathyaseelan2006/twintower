import random
import csv
from datetime import datetime, timedelta
from pathlib import Path


TOTAL_SAMPLES = 2000
FAULT_RATIO = 0.15

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "training_dataset.csv"
)


def generate_normal():
    return {
        "temperature": round(random.uniform(25, 42), 2),
        "voltage": round(random.uniform(11.2, 12.8), 2),
        "current": round(random.uniform(0.8, 2.5), 2),
        "vibration": round(random.uniform(0.05, 0.8), 3),
        "rpm": random.randint(1200, 1800),
        "label": "NORMAL",
        "fault_type": "NONE"
    }


def generate_fault():
    reading = generate_normal()

    fault = random.choice([
        "OVERHEATING",
        "OVERCURRENT",
        "HIGH_VIBRATION",
        "LOW_VOLTAGE"
    ])

    if fault == "OVERHEATING":
        reading["temperature"] = round(
            random.uniform(55, 80), 2
        )

    elif fault == "OVERCURRENT":
        reading["current"] = round(
            random.uniform(4.0, 7.0), 2
        )

    elif fault == "HIGH_VIBRATION":
        reading["vibration"] = round(
            random.uniform(1.5, 4.0), 3
        )

    elif fault == "LOW_VOLTAGE":
        reading["voltage"] = round(
            random.uniform(7.0, 9.5), 2
        )

    reading["label"] = "ANOMALY"
    reading["fault_type"] = fault

    return reading


def generate_dataset():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    start_time = datetime.now()

    rows = []

    for i in range(TOTAL_SAMPLES):

        if random.random() < FAULT_RATIO:
            reading = generate_fault()
        else:
            reading = generate_normal()

        reading["timestamp"] = (
            start_time + timedelta(seconds=i * 2)
        ).isoformat(timespec="seconds")

        reading["device_id"] = "DT-DEVICE-001"

        rows.append(reading)

    random.shuffle(rows)

    fieldnames = [
        "device_id",
        "timestamp",
        "temperature",
        "voltage",
        "current",
        "vibration",
        "rpm",
        "label",
        "fault_type"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    normal_count = sum(
        row["label"] == "NORMAL"
        for row in rows
    )

    anomaly_count = (
        TOTAL_SAMPLES - normal_count
    )

    print("--------------------------------")
    print("ML Dataset Generated Successfully")
    print("--------------------------------")
    print(f"Total samples : {TOTAL_SAMPLES}")
    print(f"Normal        : {normal_count}")
    print(f"Anomalies     : {anomaly_count}")
    print(f"Saved to      : {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()