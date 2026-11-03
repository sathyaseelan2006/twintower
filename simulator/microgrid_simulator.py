"""
ESP32 Smart Hybrid Microgrid Simulator
Simulates the physical hardware at IP 10.174.75.204:
- Solar Voltage & Panel Yield
- Battery Voltage & Level (Pin D35)
- ACS712 Current & Load Power
- Servo (Pin 23) & LED (Pin 27)
- Syncs bi-directionally with backend control commands
"""

import time
import random
import requests
from datetime import datetime, timezone

API_BASE_URL = "http://127.0.0.1:8000"
TELEMETRY_URL = f"{API_BASE_URL}/telemetry"
CONTROL_URL = f"{API_BASE_URL}/control/state"

class ESP32MicrogridSimulator:
    def __init__(self, device_id="ESP32-MICROGRID-001"):
        self.device_id = device_id
        
        # Hardware baseline matching project presentation
        self.solar_voltage = 10.20
        self.battery_voltage = 16.83
        self.battery_level = 100.0   # Pin D35 (%)
        self.battery_temp = 27.5     # °C
        self.load_current = 0.17     # ACS712 Amperes
        self.load_power = 2.84       # Watts
        
        # Operating States
        self.mode = "BATTERY"        # SOLAR, BATTERY, AUTO
        self.load_status = "ON"      # ON, OFF
        self.servo_angle = 0         # Pin 23
        self.led_status = "OFF"      # Pin 27

    def sync_control_commands(self):
        """Fetch remote user directives (relay toggle, mode switch) from backend."""
        try:
            res = requests.get(CONTROL_URL, timeout=2)
            if res.status_code == 200:
                data = res.json()
                self.mode = data.get("mode", self.mode)
                self.load_status = data.get("load_status", self.load_status)
                self.servo_angle = data.get("servo_angle", self.servo_angle)
                self.led_status = data.get("led_status", self.led_status)
        except Exception:
            pass

    def update_physics(self):
        """Simulate microgrid electrical dynamics."""
        # 1. Solar variations (drifts between 9.5V and 18.2V)
        self.solar_voltage += 0.05 * (11.5 - self.solar_voltage) + random.uniform(-0.15, 0.15)
        self.solar_voltage = max(8.0, min(20.0, self.solar_voltage))

        # 2. Load dynamics
        if self.load_status == "ON":
            self.load_current = round(max(0.12, min(0.35, 0.17 + random.uniform(-0.02, 0.02))), 2)
            self.load_power = round(self.battery_voltage * self.load_current, 2)
        else:
            self.load_current = 0.0
            self.load_power = 0.0

        # 3. Battery dynamics
        if self.mode == "SOLAR" and self.solar_voltage > 12.0:
            # Solar charging battery
            self.battery_level = min(100.0, self.battery_level + 0.05)
            self.battery_voltage = round(15.5 + (self.battery_level / 100.0) * 1.33, 2)
        elif self.load_status == "ON":
            # Discharging battery
            self.battery_level = max(10.0, self.battery_level - 0.02)
            self.battery_voltage = round(13.8 + (self.battery_level / 100.0) * 3.03, 2)

        # Servo rotation simulation in tracking mode
        if self.mode == "SOLAR":
            self.servo_angle = (self.servo_angle + 5) % 180

    def generate_packet(self):
        return {
            "device_id": self.device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "solar_voltage": round(self.solar_voltage, 2),
            "solar_current": round(max(0.0, (self.solar_voltage - 8.0) * 0.12), 2),
            "solar_power": round(max(0.0, (self.solar_voltage - 8.0) * 0.12 * self.solar_voltage), 2),
            "battery_voltage": round(self.battery_voltage, 2),
            "battery_level": round(self.battery_level, 1),
            "battery_temp": round(self.battery_temp + random.uniform(-0.1, 0.1), 1),
            "load_current": self.load_current,
            "load_power": self.load_power,
            "mode": self.mode,
            "load_status": self.load_status,
            "servo_angle": self.servo_angle,
            "led_status": self.led_status
        }

    def run(self):
        print("==================================================")
        print(" ESP32 Smart Hybrid Microgrid Simulator Started")
        print(" Matching Hardware Profile: IP 10.174.75.204")
        print("==================================================")
        while True:
            self.sync_control_commands()
            self.update_physics()
            pkt = self.generate_packet()
            try:
                res = requests.post(TELEMETRY_URL, json=pkt, timeout=3)
                if res.status_code == 200:
                    data = res.json()
                    print(f"[OK] Mode: {pkt['mode']} | Solar: {pkt['solar_voltage']}V | Bat: {pkt['battery_voltage']}V ({pkt['battery_level']}%) | Load: {pkt['load_power']}W | SoH: {data.get('battery_soh')}%")
                else:
                    print(f"[ERR] Status {res.status_code}")
            except Exception as e:
                print(f"[ERR] Failed connecting to backend: {e}")
            time.sleep(2)

if __name__ == "__main__":
    sim = ESP32MicrogridSimulator()
    sim.run()
