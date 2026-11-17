
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
INPUT_CSV = DATA_DIR / "NeuroSentinel_MultiSubject_EEG_Features.csv"
FEATURES = ["Delta", "Theta", "Alpha", "Beta"]
LABELS = ["T1", "T2"]


def main():
    data = pd.read_csv(INPUT_CSV)

    X_raw = data[FEATURES].to_numpy(dtype=float)
    y = data["Task"].to_numpy()
    groups = data["Participant"].to_numpy()

    if len(data) == 0:
        raise ValueError("Input dataset is empty.")

    if not np.isfinite(X_raw).all() or (X_raw < 0).any():
        raise ValueError("Band-power features must be finite and nonnegative.")

    if set(np.unique(y)) != set(LABELS):
        raise ValueError("Expected T1 and T2 labels.")

    totals = X_raw.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Cannot calculate relative power for zero-total rows.")

    # Each feature is divided by the sum of the four band powers.
    X = X_raw / totals[:, None]

    if not np.allclose(X.sum(axis=1), 1.0):
        raise RuntimeError("Relative-power row-sum validation failed.")

    models = {
        "Majority-class baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression (relative power)": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    }

    cv = LeaveOneGroupOut()
    metric_rows = []
    prediction_rows = []
    participant_rows = []
    confusion_rows = []

    for model_name, model in models.items():
        actual_all = []
        predicted_all = []

        for train_idx, test_idx in cv.split(X, y, groups):
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise RuntimeError("Participant leakage detected.")

            model.fit(X[train_idx], y[train_idx])
            predictions = model.predict(X[test_idx])

            actual_all.extend(y[test_idx])
            predicted_all.extend(predictions)

            participant = str(groups[test_idx][0])
            participant_rows.append({
                "Model": model_name,
                "Participant": participant,
                "Windows": len(test_idx),
                "Accuracy": accuracy_score(y[test_idx], predictions),
                "Balanced_Accuracy": balanced_accuracy_score(
                    y[test_idx], predictions
                ),
                "Macro_F1": f1_score(
                    y[test_idx], predictions,
                    labels=LABELS, average="macro", zero_division=0
                ),
            })

            for idx, actual, predicted in zip(
                test_idx, y[test_idx], predictions
            ):
                prediction_rows.append({
                    "Model": model_name,
                    "Participant": groups[idx],
                    "Window": data.iloc[idx]["Window"],
                    "Actual_Task": actual,
                    "Predicted_Task": predicted,
                })

        actual_all = np.asarray(actual_all)
        predicted_all = np.asarray(predicted_all)

        metric_rows.append({
            "Model": model_name,
            "Windows": len(actual_all),
            "Participants": len(np.unique(groups)),
            "Accuracy": accuracy_score(actual_all, predicted_all),
            "Balanced_Accuracy": balanced_accuracy_score(
                actual_all, predicted_all
            ),
            "Macro_F1": f1_score(
                actual_all, predicted_all,
                labels=LABELS, average="macro", zero_division=0
            ),
        })

        cm = confusion_matrix(
            actual_all, predicted_all, labels=LABELS
        )
        for i, actual_label in enumerate(LABELS):
            for j, predicted_label in enumerate(LABELS):
                confusion_rows.append({
                    "Model": model_name,
                    "Actual_Task": actual_label,
                    "Predicted_Task": predicted_label,
                    "Count": int(cm[i, j]),
                })

    outputs = {
        "Grouped_Metrics": metric_rows,
        "Grouped_Predictions": prediction_rows,
        "Grouped_Confusion": confusion_rows,
        "Participant_Metrics": participant_rows,
    }

    for suffix, rows in outputs.items():
        output_path = DATA_DIR / (
            f"NeuroSentinel_MultiSubject_EEG_RelativePower_{suffix}.csv"
        )
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print("Saved:", output_path)

    print("\nRELATIVE-POWER EVALUATION COMPLETE")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nLabels represent motor tasks, not tumour diagnoses.")


if __name__ == "__main__":
    main()
