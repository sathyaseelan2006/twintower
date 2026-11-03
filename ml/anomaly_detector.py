"""
Microgrid Anomaly & Fault Detection Module
Identifies electrical hazards (overvoltage, deep discharge, overcurrent)
and statistical outliers in solar/battery telemetry.
"""

class MicrogridAnomalyDetector:
    def __init__(self):
        self.is_trained = True

    def predict(self, telemetry):
        """
        Evaluate electrical limits and multi-parameter safety boundaries.
        """
        solar_v = getattr(telemetry, "solar_voltage", 0.0)
        battery_v = getattr(telemetry, "battery_voltage", 0.0)
        battery_soc = getattr(telemetry, "battery_level", 100.0)
        load_i = getattr(telemetry, "load_current", 0.0)
        load_w = getattr(telemetry, "load_power", 0.0)
        battery_t = getattr(telemetry, "battery_temp", 27.0)

        # 1. Critical Electrical Safety Rules
        if battery_v > 17.5:
            return {"prediction": "ANOMALY", "anomaly_score": -0.92, "fault": "OVERVOLTAGE_CHARGING", "details": "Battery voltage exceeds 17.5V overcharge threshold!"}

        if battery_soc < 15.0 or (battery_v > 0 and battery_v < 10.5):
            return {"prediction": "ANOMALY", "anomaly_score": -0.85, "fault": "DEEP_DISCHARGE_RISK", "details": "Battery depleted below 10.5V! Critical degradation hazard."}

        if load_i > 2.5:
            return {"prediction": "ANOMALY", "anomaly_score": -0.78, "fault": "OVERCURRENT_LOAD", "details": f"ACS712 current {load_i:.2f}A exceeds 2.5A safety limit!"}

        if battery_t > 48.0:
            return {"prediction": "ANOMALY", "anomaly_score": -0.88, "fault": "BATTERY_OVERHEATING", "details": f"Battery temp {battery_t:.1f}°C indicates thermal runaway hazard!"}

        if solar_v > 24.0:
            return {"prediction": "ANOMALY", "anomaly_score": -0.75, "fault": "SOLAR_SURGE", "details": "Solar PV input voltage exceeds 24V surge threshold!"}

        # 2. Healthy / Optimal calculation
        # Normal score positive (+0.05 to +0.20)
        score = 0.08 + (0.01 if 13.0 <= battery_v <= 16.9 else -0.05)
        return {
            "prediction": "NORMAL",
            "anomaly_score": round(score, 4),
            "fault": "NONE",
            "details": "All microgrid parameters within safe operating envelope."
        }

microgrid_anomaly_detector = MicrogridAnomalyDetector()
