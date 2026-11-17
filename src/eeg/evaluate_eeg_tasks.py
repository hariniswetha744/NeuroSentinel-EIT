
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"C:\NeuroSentinel")
DATA = ROOT / "data"

INPUT = DATA / "NeuroSentinel_EEG_TaskBandPower_S001R04.csv"
df = pd.read_csv(INPUT)

FEATURES = ["Delta", "Theta", "Alpha", "Beta"]
X = df[FEATURES]
y = df["Task"]

if len(df) < 5 or y.nunique() != 2:
    raise ValueError("Expected at least 5 windows and 2 task classes.")

if X.isna().any().any() or not X.map(
    lambda value: pd.notna(value) and abs(value) != float("inf")
).all().all():
    raise ValueError("Features contain missing or non-finite values.")

cv = LeaveOneOut()

models = {
    "Most-frequent baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
}

all_predictions = []
all_metrics = []

for name, model in models.items():
    predictions = cross_val_predict(model, X, y, cv=cv)

    all_predictions.append(
        pd.DataFrame({
            "Model": name,
            "Window": df["Window"],
            "Actual_Task": y,
            "Predicted_Task": predictions,
        })
    )

    all_metrics.append({
        "Model": name,
        "Windows": len(y),
        "Accuracy": accuracy_score(y, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y, predictions),
        "Macro_F1": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
    })

    print(f"\n{name}")
    print("Confusion matrix (rows/columns: T1, T2):")
    print(confusion_matrix(y, predictions, labels=["T1", "T2"]))

metrics = pd.DataFrame(all_metrics)
predictions = pd.concat(all_predictions, ignore_index=True)

metrics.to_csv(
    DATA / "NeuroSentinel_EEG_TaskModel_Metrics.csv",
    index=False
)
predictions.to_csv(
    DATA / "NeuroSentinel_EEG_TaskModel_Predictions.csv",
    index=False
)

print("\nEXPLORATORY RESULTS")
print(metrics.to_string(index=False))
print("\nSaved metrics and predictions to:", DATA)
print("Warning: one participant and 15 windows; results are preliminary.")
