import random
import time
import requests
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


# FastAPI telemetry endpoint
API_URL = "http://127.0.0.1:8000/telemetry"


@dataclass
class SensorReading:
    device_id: str
    timestamp: str
    temperature: float
    voltage: float
    current: float
    vibration: float
    rpm: int
    status: str


class EmbeddedDeviceSimulator:

    def __init__(self, device_id="DT-DEVICE-001"):
        self.device_id = device_id

        self.temperature = 30.0
        self.voltage = 12.0
        self.current = 1.5
        self.vibration = 0.25
        self.rpm = 1500

    def _update_normal_values(self):
        """Simulate realistic normal operating fluctuations with mean-reversion."""
        self.temperature += 0.05 * (32.0 - self.temperature) + random.uniform(-0.25, 0.25)
        self.voltage += 0.05 * (12.0 - self.voltage) + random.uniform(-0.05, 0.05)
        self.current += 0.05 * (1.5 - self.current) + random.uniform(-0.05, 0.05)
        self.vibration += 0.05 * (0.22 - self.vibration) + random.uniform(-0.015, 0.015)
        self.rpm += int(0.05 * (1500 - self.rpm)) + random.randint(-15, 15)

        self.temperature = max(25.0, min(self.temperature, 40.0))
        self.voltage = max(11.2, min(self.voltage, 12.8))
        self.current = max(1.0, min(self.current, 2.2))
        self.vibration = max(0.10, min(self.vibration, 0.50))
        self.rpm = max(1400, min(self.rpm, 1600))
    def _inject_fault(self):
        """Generate a simulated device fault."""

        fault = random.choice([
            "OVERHEATING",
            "OVERCURRENT",
            "HIGH_VIBRATION",
            "LOW_VOLTAGE"
        ])

        if fault == "OVERHEATING":
            self.temperature = random.uniform(55, 80)

        elif fault == "OVERCURRENT":
            self.current = random.uniform(4.0, 7.0)

        elif fault == "HIGH_VIBRATION":
            self.vibration = random.uniform(1.5, 4.0)

        elif fault == "LOW_VOLTAGE":
            self.voltage = random.uniform(7.0, 9.5)

        return fault

    def generate_reading(self):
        """Generate one telemetry packet."""

        self._update_normal_values()

        status = "NORMAL"

        if random.random() < 0.05:
            status = self._inject_fault()

        reading = SensorReading(
            device_id=self.device_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            temperature=round(self.temperature, 2),
            voltage=round(self.voltage, 2),
            current=round(self.current, 2),
            vibration=round(self.vibration, 3),
            rpm=self.rpm,
            status=status
        )

        return asdict(reading)


def send_to_backend(reading):
    """Send telemetry to the FastAPI backend."""

    try:
        response = requests.post(
            API_URL,
            json=reading,
            timeout=5
        )

        if response.status_code == 200:
            print(
                f"[OK] Sent | "
                f"Temp: {reading['temperature']}C | "
                f"Voltage: {reading['voltage']}V | "
                f"Current: {reading['current']}A | "
                f"Vibration: {reading['vibration']} | "
                f"RPM: {reading['rpm']} | "
                f"Status: {reading['status']}"
            )
        else:
            print(f"[ERROR] API error: {response.status_code}")

    except requests.exceptions.RequestException as error:
        print(f"[ERROR] Backend connection failed: {error}")


if __name__ == "__main__":

    device = EmbeddedDeviceSimulator()

    print("Embedded Digital Twin Simulator")
    print("Sending telemetry to FastAPI...")
    print("--------------------------------")

    try:
        while True:

            reading = device.generate_reading()

            send_to_backend(reading)

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")