import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lof_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ============================================================
# ANOMALY DETECTOR
# ============================================================

class AnomalyDetector:

    def __init__(self):

        self.model = None
        self.scaler = None
        self.is_trained = False

        self.load_model()


    # ========================================================
    # LOAD EXISTING MODEL
    # ========================================================

    def load_model(self):
        """Load the trained LOF model and scaler."""

        if not MODEL_PATH.exists():
            print("LOF model not found.")
            return

        if not SCALER_PATH.exists():
            print("Scaler not found.")
            return

        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)

            self.is_trained = True

            print(
                "LOF anomaly detection model "
                "loaded successfully."
            )

        except Exception as e:

            self.model = None
            self.scaler = None
            self.is_trained = False

            print(
                f"Failed to load LOF model: {e}"
            )


    # ========================================================
    # TRAIN / RETRAIN MODEL
    # ========================================================

    def train(self, readings):
        """Train and save the LOF anomaly detection model."""

        if len(readings) < 10:
            raise ValueError(
                "At least 10 telemetry readings are required."
            )

        # Convert database telemetry into a DataFrame
        features = pd.DataFrame(
            [
                {
                    "temperature": reading.temperature,
                    "voltage": reading.voltage,
                    "current": reading.current,
                    "vibration": reading.vibration,
                    "rpm": reading.rpm
                }
                for reading in readings
            ]
        )

        # Remove incomplete historical readings
        features = features.dropna()

        if len(features) < 10:
            raise ValueError(
                "At least 10 valid telemetry readings are required."
            )

        # Create and train scaler
        self.scaler = StandardScaler()

        scaled_features = self.scaler.fit_transform(
            features
        )

        # Create LOF model
        self.model = LocalOutlierFactor(
            n_neighbors=min(
                20,
                len(features) - 1
            ),
            contamination="auto",
            novelty=True
        )

        # Train LOF
        self.model.fit(
            scaled_features
        )

        # Save trained model
        joblib.dump(
            self.model,
            MODEL_PATH
        )

        # Save scaler
        joblib.dump(
            self.scaler,
            SCALER_PATH
        )

        self.is_trained = True

        print(
            "LOF anomaly detection model "
            "trained successfully."
        )

        print(
            f"Training samples: {len(features)}"
        )

        return len(features)


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, reading):
        """Predict whether a telemetry reading is anomalous."""

        if not self.is_trained:

            return {
                "prediction": "MODEL_NOT_LOADED",
                "anomaly_score": None
            }

        # Use same feature names and order used during training
        features = pd.DataFrame(
            [
                {
                    "temperature": reading.temperature,
                    "voltage": reading.voltage,
                    "current": reading.current,
                    "vibration": reading.vibration,
                    "rpm": reading.rpm
                }
            ]
        )

        # Scale incoming telemetry
        scaled_features = self.scaler.transform(
            features
        )

        # LOF prediction
        raw_prediction = self.model.predict(
            scaled_features
        )[0]

        # LOF anomaly score
        score = self.model.decision_function(
            scaled_features
        )[0]

        prediction = (
            "NORMAL"
            if raw_prediction == 1
            else "ANOMALY"
        )

        return {
            "prediction": prediction,
            "anomaly_score": round(
                float(score),
                4
            )
        }


# ============================================================
# GLOBAL ANOMALY DETECTOR INSTANCE
# ============================================================

anomaly_detector = AnomalyDetector()