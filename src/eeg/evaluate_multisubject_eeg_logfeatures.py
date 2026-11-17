
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

    if data.empty:
        raise ValueError("Input dataset is empty.")

    X_raw = data[FEATURES].to_numpy(dtype=float)
    y = data["Task"].to_numpy()
    groups = data["Participant"].to_numpy()

    if not np.isfinite(X_raw).all():
        raise ValueError("Feature data contains non-finite values.")

    if (X_raw <= 0).any():
        raise ValueError("Log10 requires strictly positive features.")

    if set(np.unique(y)) != set(LABELS):
        raise ValueError("Expected both T1 and T2 task labels.")

    # Transform features; labels and participant groups remain unchanged.
    X = np.log10(X_raw)

    models = {
        "Majority-class baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression (log features)": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    }

    cv = LeaveOneGroupOut()
    prediction_rows = []
    metric_rows = []
    participant_rows = []
    confusion_rows = []

    for model_name, model in models.items():
        all_actual = []
        all_predicted = []

        for train_idx, test_idx in cv.split(X, y, groups):
            train_people = set(groups[train_idx])
            test_people = set(groups[test_idx])

            if train_people & test_people:
                raise RuntimeError("Participant leakage detected.")

            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])

            all_actual.extend(y[test_idx])
            all_predicted.extend(pred)

            participant = str(groups[test_idx][0])

            participant_rows.append({
                "Model": model_name,
                "Participant": participant,
                "Windows": len(test_idx),
                "Accuracy": accuracy_score(y[test_idx], pred),
                "Balanced_Accuracy": balanced_accuracy_score(
                    y[test_idx], pred
                ),
                "Macro_F1": f1_score(
                    y[test_idx], pred,
                    labels=LABELS, average="macro",
                    zero_division=0,
                ),
            })

            for idx, actual, predicted in zip(
                test_idx, y[test_idx], pred
            ):
                prediction_rows.append({
                    "Model": model_name,
                    "Participant": groups[idx],
                    "Window": data.iloc[idx]["Window"],
                    "Actual_Task": actual,
                    "Predicted_Task": predicted,
                })

        all_actual = np.asarray(all_actual)
        all_predicted = np.asarray(all_predicted)

        metric_rows.append({
            "Model": model_name,
            "Windows": len(all_actual),
            "Participants": len(np.unique(groups)),
            "Accuracy": accuracy_score(all_actual, all_predicted),
            "Balanced_Accuracy": balanced_accuracy_score(
                all_actual, all_predicted
            ),
            "Macro_F1": f1_score(
                all_actual, all_predicted,
                labels=LABELS, average="macro",
                zero_division=0,
            ),
        })

        cm = confusion_matrix(
            all_actual, all_predicted, labels=LABELS
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
        path = DATA_DIR / (
            f"NeuroSentinel_MultiSubject_EEG_LogFeatures_{suffix}.csv"
        )
        pd.DataFrame(rows).to_csv(path, index=False)
        print("Saved:", path)

    print("\nLOG-FEATURE EVALUATION COMPLETE")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nParticipant-wise results saved separately.")
    print("Labels are motor tasks (T1/T2), not tumour diagnoses.")


if __name__ == "__main__":
    main()
