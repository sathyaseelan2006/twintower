import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "training_dataset.csv"

FEATURES = [
    "temperature",
    "voltage",
    "current",
    "vibration",
    "rpm"
]


# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv(DATASET_PATH)

normal_data = df[df["label"] == "NORMAL"].copy()
anomaly_data = df[df["label"] == "ANOMALY"].copy()


normal_train, normal_test = train_test_split(
    normal_data,
    test_size=0.25,
    random_state=42
)


X_train = normal_train[FEATURES]

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


# -------------------------
# Scaling
# -------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------
# Models
# -------------------------

models = {
    "Isolation Forest": IsolationForest(
        n_estimators=300,
        contamination=0.15,
        random_state=42,
        n_jobs=-1
    ),

    "One-Class SVM": OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.15
    ),

    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.15,
        novelty=True
    )
}


results = []


print("\n========================================")
print("     ANOMALY DETECTION MODEL TEST")
print("========================================")


for name, model in models.items():

    model.fit(X_train_scaled)

    raw_predictions = model.predict(
        X_test_scaled
    )

    y_pred = [
        0 if prediction == 1 else 1
        for prediction in raw_predictions
    ]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


# -------------------------
# Results Table
# -------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)


print("\n========================================")
print("             RANKING")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


best_model = results_df.iloc[0]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print(best_model["Model"])

print(
    f"F1 Score: "
    f"{best_model['F1']:.4f}"
)