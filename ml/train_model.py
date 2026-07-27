import pandas as pd
import joblib
from pathlib import Path

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "training_dataset.csv"

MODEL_PATH = BASE_DIR / "ml" / "lof_model.pkl"
SCALER_PATH = BASE_DIR / "ml" / "scaler.pkl"


FEATURES = [
    "temperature",
    "voltage",
    "current",
    "vibration",
    "rpm"
]


print("========================================")
print(" Digital Twin - Final LOF Model Training")
print("========================================")


# Load dataset
df = pd.read_csv(DATASET_PATH)

print(f"Dataset loaded: {len(df)} samples")


# Separate normal and anomalous samples
normal_data = df[df["label"] == "NORMAL"].copy()
anomaly_data = df[df["label"] == "ANOMALY"].copy()


# Train only using normal operating behaviour
normal_train, normal_test = train_test_split(
    normal_data,
    test_size=0.25,
    random_state=42
)


X_train = normal_train[FEATURES]


# Build evaluation dataset
test_data = pd.concat(
    [normal_test, anomaly_data],
    ignore_index=True
)

X_test = test_data[FEATURES]

y_test = (
    test_data["label"]
    .map({
        "NORMAL": 0,
        "ANOMALY": 1
    })
    .values
)


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Final Local Outlier Factor model
# novelty=True allows prediction on future/live readings.
model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.15,
    novelty=True
)


print("\nTraining LOF model...")

model.fit(X_train_scaled)


# Evaluate
raw_predictions = model.predict(X_test_scaled)

y_pred = [
    0 if prediction == 1 else 1
    for prediction in raw_predictions
]


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n------------- RESULTS -------------")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "NORMAL",
            "ANOMALY"
        ],
        zero_division=0
    )
)


# Save final model and scaler
joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print("\n========================================")
print(" FINAL MODEL SAVED")
print("========================================")

print(f"Model  : {MODEL_PATH}")
print(f"Scaler : {SCALER_PATH}")

print("\nAlgorithm: Local Outlier Factor")
print("Ready for live FastAPI integration.")