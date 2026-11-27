
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from scipy.signal import welch
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


EDF_ROOT = Path(
    r"C:\Users\harin\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0"
)
DATA_DIR = Path(r"C:\NeuroSentinel\data")

FEATURE_FILE = DATA_DIR / "NeuroSentinel_C3C4_MuBeta_3s_Features.csv"
METRICS_FILE = DATA_DIR / "NeuroSentinel_C3C4_MuBeta_LOSO_Metrics.csv"
PREDICTIONS_FILE = DATA_DIR / "NeuroSentinel_C3C4_MuBeta_LOSO_Predictions.csv"
CONFUSION_FILE = DATA_DIR / "NeuroSentinel_C3C4_MuBeta_LOSO_Confusion.csv"

OUTPUT_FILES = [
    FEATURE_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    CONFUSION_FILE,
]

LABELS = ["T1", "T2"]
BANDS = {
    "Mu": (8.0, 13.0),
    "Beta": (13.0, 30.0),
}

BUFFER_SECONDS = 0.5
WINDOW_END_SECONDS = 3.5


def find_channel(raw, target):
    """Match C3/C4 despite punctuation in EDF channel names."""
    for name in raw.ch_names:
        normalized = "".join(
            char for char in name.upper() if char.isalnum()
        )
        if normalized == target:
            return name
    raise ValueError(
        f"{target} not found. Available channels: {raw.ch_names}"
    )


def band_power(signal, sfreq, low, high):
    freqs, psd = welch(
        signal,
        fs=sfreq,
        nperseg=min(len(signal), int(2 * sfreq)),
    )
    mask = (freqs >= low) & (freqs < high)

    if mask.sum() < 2:
        raise ValueError(
            f"Insufficient frequency bins for {low}-{high} Hz"
        )

    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_features():
    recordings = sorted(EDF_ROOT.rglob("*R04.edf"))

    if not recordings:
        raise FileNotFoundError("No R04 EDF recordings found.")

    rows = []

    for edf_path in recordings:
        participant = edf_path.parent.name
        print(f"Processing {participant}: {edf_path.name}")

        raw = mne.io.read_raw_edf(
            str(edf_path), preload=True, verbose="ERROR"
        )

        try:
            sfreq = float(raw.info["sfreq"])
            c3_name = find_channel(raw, "C3")
            c4_name = find_channel(raw, "C4")

            events, event_ids = mne.events_from_annotations(
                raw, verbose="ERROR"
            )

            for task in LABELS:
                if task not in event_ids:
                    raise ValueError(
                        f"{task} annotation missing in {edf_path}"
                    )

                task_events = events[
                    events[:, 2] == event_ids[task]
                ]

                for window_number, event in enumerate(
                    task_events, start=1
                ):
                    onset = event[0] / sfreq
                    start = onset + BUFFER_SECONDS
                    stop = onset + WINDOW_END_SECONDS

                    start_sample = int(round(start * sfreq))
                    stop_sample = int(round(stop * sfreq))

                    if stop_sample > raw.n_times:
                        continue

                    segment = raw.get_data(
                        picks=[c3_name, c4_name],
                        start=start_sample,
                        stop=stop_sample,
                    )

                    if segment.shape[1] < int(2 * sfreq):
                        continue

                    c3_signal = segment[0]
                    c4_signal = segment[1]

                    row = {
                        "Participant": participant,
                        "Run": "R04",
                        "Task": task,
                        "Window": window_number,
                        "Onset_seconds": round(onset, 3),
                        "Duration_seconds": round(
                            segment.shape[1] / sfreq, 3
                        ),
                        "C3_Channel": c3_name,
                        "C4_Channel": c4_name,
                    }

                    for band_name, (low, high) in BANDS.items():
                        row[f"C3_{band_name}_Power"] = band_power(
                            c3_signal, sfreq, low, high
                        )
                        row[f"C4_{band_name}_Power"] = band_power(
                            c4_signal, sfreq, low, high
                        )

                    rows.append(row)

        finally:
            raw.close()

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No EEG feature windows extracted.")

    power_columns = [
        "C3_Mu_Power",
        "C4_Mu_Power",
        "C3_Beta_Power",
        "C4_Beta_Power",
    ]

    if df[power_columns].isna().any().any():
        raise ValueError("Missing power values detected.")

    if not np.isfinite(df[power_columns].to_numpy()).all():
        raise ValueError("Non-finite power values detected.")

    if (df[power_columns] <= 0).any().any():
        raise ValueError("Non-positive power values detected.")

    df["C3_Mu_Log10"] = np.log10(df["C3_Mu_Power"])
    df["C4_Mu_Log10"] = np.log10(df["C4_Mu_Power"])
    df["C3_Beta_Log10"] = np.log10(df["C3_Beta_Power"])
    df["C4_Beta_Log10"] = np.log10(df["C4_Beta_Power"])

    return df


def evaluate_loso(df):
    features = [
        "C3_Mu_Log10",
        "C4_Mu_Log10",
        "C3_Beta_Log10",
        "C4_Beta_Log10",
    ]

    required = ["Participant", "Task"] + features

    if df[required].isna().any().any():
        raise ValueError("Missing evaluation values.")

    X = df[features].to_numpy(dtype=float)
    y = df["Task"].to_numpy()
    groups = df["Participant"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Non-finite classifier input.")

    if set(np.unique(y)) != set(LABELS):
        raise ValueError("Expected T1 and T2 labels.")

    if len(np.unique(groups)) < 2:
        raise ValueError("At least two participants are required.")

    splitter = LeaveOneGroupOut()

    models = {
        "Majority-class baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "C3/C4 Mu-Beta Logistic Regression": make_pipeline(
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

            if train_groups.intersection(test_groups):
                raise RuntimeError("Participant leakage detected.")

            if len(np.unique(y[train_idx])) != 2:
                raise RuntimeError("Training fold lacks a task class.")

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

    predictions = pd.DataFrame(prediction_rows)

    metric_rows = []
    confusion_rows = []

    for model_name, subset in predictions.groupby(
        "Model", sort=False
    ):
        actual = subset["Actual_Task"]
        predicted = subset["Predicted_Task"]

        metric_rows.append({
            "Model": model_name,
            "Windows": len(subset),
            "Participants": subset["Participant"].nunique(),
            "Accuracy": accuracy_score(actual, predicted),
            "Balanced_Accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "Macro_F1": f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
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

    return (
        predictions,
        pd.DataFrame(metric_rows),
        pd.DataFrame(confusion_rows),
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = [str(path) for path in OUTPUT_FILES if path.exists()]
    if existing:
        raise FileExistsError(
            "To protect existing files, outputs were not written:\n"
            + "\n".join(existing)
        )

    print("EXTRACTING C3/C4 MU-BETA FEATURES")
    df = extract_features()

    print("\nRUNNING PARTICIPANT-INDEPENDENT EVALUATION")
    predictions, metrics, confusion = evaluate_loso(df)

    # Write only new, uniquely named outputs.
    df.to_csv(FEATURE_FILE, index=False)
    metrics.to_csv(METRICS_FILE, index=False)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    confusion.to_csv(CONFUSION_FILE, index=False)

    print("\nC3/C4 MU-BETA PIPELINE COMPLETE")
    print("Feature rows:", len(df))
    print("Participants:", df["Participant"].nunique())
    print("Task counts:\n", df["Task"].value_counts().to_string())
    print("\nMetrics:\n", metrics.round(4).to_string(index=False))
    print("\nConfusion matrices (actual rows, predicted columns):")

    for model_name in metrics["Model"]:
        subset = predictions[predictions["Model"] == model_name]
        matrix = confusion_matrix(
            subset["Actual_Task"],
            subset["Predicted_Task"],
            labels=LABELS,
        )
        print(f"\n{model_name}:")
        print(matrix)

    print("\nNew files saved:")
    for path in OUTPUT_FILES:
        print(path)


if __name__ == "__main__":
    main()
