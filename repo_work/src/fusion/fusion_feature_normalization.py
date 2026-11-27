from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


ROOT = Path(r"C:\NeuroSentinel")

INPUT_DIR = ROOT / "fusion" / "prepared_features"
OUTPUT_DIR = ROOT / "fusion" / "normalized_features"

EEG_FILE = INPUT_DIR / "EEG_Prepared_Features.csv"
EIT_FILE = INPUT_DIR / "EIT_Prepared_Features.csv"

EEG_FEATURES = ["Delta", "Theta", "Alpha", "Beta"]

EIT_QA_ONLY = ["Region_Mask_Element_Count_QA_Only"]


def normalize_dataset(frame, feature_columns, name):
    values = frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if values.empty:
        raise ValueError(f"{name} dataset is empty.")

    if values.isna().any().any():
        raise ValueError(f"{name} has missing or invalid values.")

    array = values.to_numpy(dtype=float)

    if not np.isfinite(array).all():
        raise ValueError(f"{name} has non-finite values.")

    scaler = StandardScaler()
    normalized = scaler.fit_transform(array)

    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} normalization failed.")

    result = pd.DataFrame(
        normalized,
        columns=feature_columns,
    )

    return result, scaler


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("NEUROSENTINEL FEATURE NORMALIZATION")

    eeg = pd.read_csv(EEG_FILE)
    eit = pd.read_csv(EIT_FILE)

    missing_eeg = [
        col for col in EEG_FEATURES
        if col not in eeg.columns
    ]

    if missing_eeg:
        raise ValueError(
            f"Missing EEG features: {missing_eeg}"
        )

    eit_features = [
        col for col in eit.columns
        if col not in EIT_QA_ONLY
    ]

    if not eit_features:
        raise ValueError("No EIT model features available.")

    eeg_normalized, eeg_scaler = normalize_dataset(
        eeg,
        EEG_FEATURES,
        "EEG",
    )

    eit_normalized, eit_scaler = normalize_dataset(
        eit,
        eit_features,
        "EIT",
    )

    eeg_output = OUTPUT_DIR / "EEG_Normalized_Features.csv"
    eit_output = OUTPUT_DIR / "EIT_Normalized_Features.csv"

    eeg_normalized.to_csv(eeg_output, index=False)
    eit_normalized.to_csv(eit_output, index=False)

    joblib.dump(
        eeg_scaler,
        OUTPUT_DIR / "EEG_StandardScaler.joblib",
    )

    joblib.dump(
        eit_scaler,
        OUTPUT_DIR / "EIT_StandardScaler.joblib",
    )

    report = {
        "EEG_rows": len(eeg_normalized),
        "EEG_features": EEG_FEATURES,
        "EIT_rows": len(eit_normalized),
        "EIT_features": eit_features,
        "normalization": "StandardScaler",
        "EEG_scaler_separate": True,
        "EIT_scaler_separate": True,
        "artificial_pairing_performed": False,
        "clinical_prediction_performed": False,
        "clinical_training_ready": False,
        "status": "NORMALIZATION_PASSED_PAIRING_NOT_READY",
        "limitations": [
            "EEG and EIT are normalized independently.",
            "EIT currently contains only one row.",
            "One-row EIT normalization is not statistically reliable.",
            "No clinical EEG-EIT pairing or prediction is established.",
        ],
    }

    report_path = (
        OUTPUT_DIR /
        "NeuroSentinel_Normalization_Report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nEEG rows normalized:", len(eeg_normalized))
    print("EEG features:", len(EEG_FEATURES))

    print("\nEIT rows normalized:", len(eit_normalized))
    print("EIT features:", len(eit_features))

    print("\nSeparate EEG scaler: saved")
    print("Separate EIT scaler: saved")
    print("Artificial pairing: False")
    print("Clinical prediction: False")

    print("\nSTATUS: NORMALIZATION PIPELINE PASSED")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()