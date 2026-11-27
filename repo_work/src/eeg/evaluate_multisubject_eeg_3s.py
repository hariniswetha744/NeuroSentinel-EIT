
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(r"C:\NeuroSentinel\data")

INPUT_CSV = DATA_DIR / "NeuroSentinel_MultiSubject_EEG_3s_Features.csv"
METRICS_CSV = DATA_DIR / "NeuroSentinel_MultiSubject_EEG_3s_Grouped_Metrics.csv"
PREDICTIONS_CSV = DATA_DIR / "NeuroSentinel_MultiSubject_EEG_3s_Grouped_Predictions.csv"
CONFUSION_CSV = DATA_DIR / "NeuroSentinel_MultiSubject_EEG_3s_Grouped_Confusion.csv"

FEATURES = ["Delta", "Theta", "Alpha", "Beta"]
LABELS = ["T1", "T2"]


def main():
    df = pd.read_csv(INPUT_CSV)

    required = ["Participant", "Task"] + FEATURES
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[required].isna().any().any():
        raise ValueError("Missing values detected. Fix the dataset first.")

    if set(df["Task"].unique()) != set(LABELS):
        raise ValueError("Expected exactly the T1 and T2 task labels.")

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["Task"].to_numpy()
    groups = df["Participant"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Non-finite feature values detected.")

    participants = np.unique(groups)
    if len(participants) < 2:
        raise ValueError("At least two participants are required.")

    splitter = LeaveOneGroupOut()

    models = {
        "Majority-class baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    }

    prediction_rows = []

    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}")

        for train_idx, test_idx in splitter.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # Explicitly verify that participants do not overlap.
            if train_groups.intersection(test_groups):
                raise RuntimeError("Participant leakage detected!")

            if len(np.unique(y[train_idx])) < 2:
                raise RuntimeError("Training fold lacks one task class.")

            model.fit(X[train_idx], y[train_idx])
            predicted = model.predict(X[test_idx])

            for idx, prediction in zip(test_idx, predicted):
                prediction_rows.append({
                    "Model": model_name,
                    "Participant": df.iloc[idx]["Participant"],
                    "Run": df.iloc[idx]["Run"],
                    "Window": int(df.iloc[idx]["Window"]),
                    "Actual_Task": y[idx],
                    "Predicted_Task": prediction,
                })

        print("  Completed participant-held-out folds.")

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    metric_rows = []
    confusion_rows = []

    for model_name, subset in predictions.groupby("Model", sort=False):
        actual = subset["Actual_Task"]
        predicted = subset["Predicted_Task"]

        metric_rows.append({
            "Model": model_name,
            "Windows": len(subset),
            "Participants": subset["Participant"].nunique(),
            "Accuracy": accuracy_score(actual, predicted),
            "Balanced_Accuracy": balanced_accuracy_score(actual, predicted),
            "Macro_F1": f1_score(
                actual, predicted, labels=LABELS,
                average="macro", zero_division=0
            ),
        })

        matrix = confusion_matrix(
            actual, predicted, labels=LABELS
        )

        for i, actual_label in enumerate(LABELS):
            for j, predicted_label in enumerate(LABELS):
                confusion_rows.append({
                    "Model": model_name,
                    "Actual_Task": actual_label,
                    "Predicted_Task": predicted_label,
                    "Count": int(matrix[i, j]),
                })

    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)

    metrics.to_csv(METRICS_CSV, index=False)
    confusion.to_csv(CONFUSION_CSV, index=False)

    # Confirm every participant was held out for each model.
    counts = predictions.groupby("Model").agg(
        Predictions=("Actual_Task", "size"),
        Participants=("Participant", "nunique"),
    )

    print("\nPARTICIPANT-LEVEL EVALUATION COMPLETE")
    print("\nInput rows:", len(df))
    print("Unique participants:", len(participants))
    print("\nMetrics:\n", metrics.round(4).to_string(index=False))
    print("\nPrediction coverage:\n", counts.to_string())
    print("\nConfusion matrices (rows=actual, columns=predicted):")

    for model_name in models:
        subset = predictions[predictions["Model"] == model_name]
        matrix = confusion_matrix(
            subset["Actual_Task"],
            subset["Predicted_Task"],
            labels=LABELS,
        )
        print(f"\n{model_name} — label order {LABELS}:")
        print(matrix)

    print("\nSaved files:")
    print(METRICS_CSV)
    print(PREDICTIONS_CSV)
    print(CONFUSION_CSV)


if __name__ == "__main__":
    main()
