"""
Battery Health Prediction Engine
Calculates State of Health (SoH) and Remaining Useful Life (RUL in cycles)
for Lithium-Ion / Lead-Acid microgrid battery storage.
"""

class BatteryHealthPredictor:
    def __init__(self, nominal_capacity_ah=7.0, rated_cycles=1500):
        self.nominal_capacity_ah = nominal_capacity_ah
        self.rated_cycles = rated_cycles
        self.accumulated_cycles = 124  # Base baseline cycles

    def predict(self, battery_voltage: float, battery_level: float, load_current: float, battery_temp: float = 27.0):
        """
        Estimate SoH and RUL based on voltage under load, DoD stress, and temperature.
        """
        # 1. Voltage sag / Internal resistance estimation:
        # Healthy battery: minimal voltage drop under current
        # Degraded battery: higher internal resistance causes voltage dip
        baseline_ir = 0.08  # Ohms for fresh pack
        estimated_ir = max(0.05, (16.8 - battery_voltage) / max(0.1, load_current + 0.5))
        ir_degradation = min(1.0, max(0.0, (estimated_ir - baseline_ir) / 0.25))

        # 2. Temperature degradation factor (Arrhenius law effect)
        # Optimal battery temp: 20°C - 30°C. >45°C accelerates degradation exponentially.
        if battery_temp > 35.0:
            temp_penalty = (battery_temp - 35.0) * 0.8
        elif battery_temp < 10.0:
            temp_penalty = (10.0 - battery_temp) * 0.5
        else:
            temp_penalty = 0.0

        # 3. State of Health (SoH %)
        # 100% = Factory new, 80% = End of Life (EOL) for energy storage
        base_soh = 98.5 - (self.accumulated_cycles / self.rated_cycles) * 15.0
        calculated_soh = max(60.0, min(100.0, base_soh - (ir_degradation * 4.0) - temp_penalty))

        # 4. Remaining Useful Life (RUL in charge cycles)
        # Assuming battery retirement at 80% SoH
        remaining_soh_capacity = max(0.0, calculated_soh - 80.0)
        rul_cycles = int((remaining_soh_capacity / 20.0) * self.rated_cycles)

        return {
            "battery_soh": round(calculated_soh, 1),
            "battery_rul_cycles": max(0, rul_cycles),
            "internal_resistance_ohm": round(estimated_ir, 3),
            "health_grade": "EXCELLENT" if calculated_soh >= 92 else ("GOOD" if calculated_soh >= 82 else "REPLACE_SOON")
        }

battery_predictor = BatteryHealthPredictor()
