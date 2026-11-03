"""
Microgrid Automated Test Suite
Executes standard engineering scenarios across Energy Optimization,
Battery Health Prediction, and Electrical Safety Hazards.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from ml.battery_health import battery_predictor
from ml.energy_optimizer import energy_optimizer
from ml.anomaly_detector import microgrid_anomaly_detector
from backend.models import MicrogridTelemetryData

BENCHMARK_CASES = [
    {
        "id": "TC-01",
        "name": "Peak Solar Surplus & Battery Top-Up",
        "inputs": {
            "solar_voltage": 17.5,
            "battery_voltage": 15.5,
            "battery_level": 75.0,
            "load_current": 0.15,
            "load_power": 2.3,
            "mode": "AUTO",
            "battery_temp": 26.0
        },
        "expected": {
            "action": "SOLAR_SURPLUS_CHARGING",
            "safety": "NORMAL",
            "min_soh": 90.0
        }
    },
    {
        "id": "TC-02",
        "name": "Night Operation (Solar Depleted, Battery Operational)",
        "inputs": {
            "solar_voltage": 0.0,
            "battery_voltage": 15.0,
            "battery_level": 70.0,
            "load_current": 0.18,
            "load_power": 2.7,
            "mode": "AUTO",
            "battery_temp": 25.0
        },
        "expected": {
            "action": "BATTERY_AUTO_DISCHARGE",
            "safety": "NORMAL",
            "min_soh": 85.0
        }
    },
    {
        "id": "TC-03",
        "name": "Critical Deep Discharge Prevention (Load Shedding)",
        "inputs": {
            "solar_voltage": 4.2,
            "battery_voltage": 10.2,
            "battery_level": 11.0,
            "load_current": 0.18,
            "load_power": 1.84,
            "mode": "AUTO",
            "battery_temp": 28.0
        },
        "expected": {
            "action": "AUTO_LOAD_SHEDDING",
            "safety": "ANOMALY",
            "fault": "DEEP_DISCHARGE_RISK"
        }
    },
    {
        "id": "TC-04",
        "name": "ACS712 Overcurrent Trip Hazard",
        "inputs": {
            "solar_voltage": 12.0,
            "battery_voltage": 16.0,
            "battery_level": 90.0,
            "load_current": 2.85,
            "load_power": 45.6,
            "mode": "BATTERY",
            "battery_temp": 30.0
        },
        "expected": {
            "safety": "ANOMALY",
            "fault": "OVERCURRENT_LOAD"
        }
    },
    {
        "id": "TC-05",
        "name": "Battery Overvoltage Overcharge Hazard",
        "inputs": {
            "solar_voltage": 18.0,
            "battery_voltage": 18.2,
            "battery_level": 100.0,
            "load_current": 0.10,
            "load_power": 1.82,
            "mode": "SOLAR",
            "battery_temp": 28.0
        },
        "expected": {
            "safety": "ANOMALY",
            "fault": "OVERVOLTAGE_CHARGING"
        }
    },
    {
        "id": "TC-06",
        "name": "High Temperature Thermal Degradation Hazard",
        "inputs": {
            "solar_voltage": 12.5,
            "battery_voltage": 14.8,
            "battery_level": 60.0,
            "load_current": 0.20,
            "load_power": 2.96,
            "mode": "AUTO",
            "battery_temp": 52.0
        },
        "expected": {
            "safety": "ANOMALY",
            "fault": "BATTERY_OVERHEATING"
        }
    },
    {
        "id": "TC-07",
        "name": "Hybrid Power Sharing (Solar Supplementing Peak Demand)",
        "inputs": {
            "solar_voltage": 12.0,
            "battery_voltage": 15.0,
            "battery_level": 80.0,
            "load_current": 0.30,
            "load_power": 4.5,
            "mode": "AUTO",
            "battery_temp": 27.0
        },
        "expected": {
            "action": "HYBRID_PEAK_SHAVING",
            "safety": "NORMAL"
        }
    }
]

def run_test_case(tc):
    inp = tc["inputs"]
    exp = tc["expected"]

    # 1. Energy Optimization
    opt = energy_optimizer.optimize(
        solar_voltage=inp["solar_voltage"],
        battery_voltage=inp["battery_voltage"],
        battery_level=inp["battery_level"],
        load_power=inp["load_power"],
        current_mode=inp["mode"]
    )

    # 2. Battery Health
    health = battery_predictor.predict(
        battery_voltage=inp["battery_voltage"],
        battery_level=inp["battery_level"],
        load_current=inp["load_current"],
        battery_temp=inp.get("battery_temp", 27.0)
    )

    # 3. Anomaly & Safety
    t_obj = MicrogridTelemetryData(**inp)
    anomaly = microgrid_anomaly_detector.predict(t_obj)

    # Assertions
    passed = True
    fail_reasons = []

    if "action" in exp and opt["action"] != exp["action"]:
        passed = False
        fail_reasons.append(f"Expected action {exp['action']} but got {opt['action']}")

    if "safety" in exp and anomaly["prediction"] != exp["safety"]:
        passed = False
        fail_reasons.append(f"Expected safety {exp['safety']} but got {anomaly['prediction']}")

    if "fault" in exp and anomaly.get("fault") != exp["fault"]:
        passed = False
        fail_reasons.append(f"Expected fault {exp['fault']} but got {anomaly.get('fault')}")

    if "min_soh" in exp and health["battery_soh"] < exp["min_soh"]:
        passed = False
        fail_reasons.append(f"Expected SoH >= {exp['min_soh']} but got {health['battery_soh']}")

    return {
        "id": tc["id"],
        "name": tc["name"],
        "passed": passed,
        "opt_action": opt["action"],
        "opt_desc": opt["description"],
        "safety_pred": anomaly["prediction"],
        "fault": anomaly.get("fault", "NONE"),
        "soh": health["battery_soh"],
        "rul": health["battery_rul_cycles"],
        "fail_reasons": fail_reasons
    }

def main():
    print("======================================================================")
    print(" MICROGRID BENCHMARK TEST SUITE")
    print(" System: Self-Learning Energy Optimization & Battery Health Prediction")
    print("======================================================================")

    total = len(BENCHMARK_CASES)
    passed_count = 0

    for tc in BENCHMARK_CASES:
        res = run_test_case(tc)
        status = "[PASS]" if res["passed"] else "[FAIL]"
        if res["passed"]:
            passed_count += 1

        print(f"\n{status} {res['id']}: {res['name']}")
        print(f"       Optimization Verdict : {res['opt_action']}")
        print(f"       Safety Verdict       : {res['safety_pred']} (Fault: {res['fault']})")
        print(f"       Battery Health       : SoH {res['soh']}% | RUL {res['rul']} Cycles")
        if not res["passed"]:
            print(f"       Failure Details      : {'; '.join(res['fail_reasons'])}")

    print("\n----------------------------------------------------------------------")
    print(f" SUMMARY: {passed_count}/{total} Test Cases Passed ({(passed_count/total)*100:.1f}%)")
    print("======================================================================")

if __name__ == "__main__":
    main()
