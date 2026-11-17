from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(r"C:\NeuroSentinel")

EEG_FILE = (
    ROOT / "data" /
    "NeuroSentinel_MultiSubject_EEG_Features.csv"
)

EIT_FILE = (
    ROOT / "eit" / "features" /
    "EIT_Feature_Summary.csv"
)

OUTPUT_DIR = ROOT / "fusion" / "prepared_features"

EEG_METADATA = [
    "Participant",
    "Run",
    "Task",
    "Window",
    "Onset_seconds",
    "Duration_seconds",
]

EEG_FEATURES = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
]

EIT_QA_ONLY = [
    "Region_Mask_Element_Count_QA_Only",
]


def validate_features(frame, name, feature_columns):
    if frame.empty:
        raise ValueError(f"{name} dataset is empty.")

    missing = [
        col for col in feature_columns
        if col not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing required features: {missing}"
        )

    values = frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if values.isna().any().any():
        raise ValueError(
            f"{name} contains missing or non-numeric features."
        )

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            f"{name} contains non-finite feature values."
        )

    return values.astype(float)


def prepare_eeg():
    frame = pd.read_csv(EEG_FILE)

    required_metadata = [
        col for col in EEG_METADATA
        if col not in frame.columns
    ]

    if required_metadata:
        raise ValueError(
            f"EEG metadata missing: {required_metadata}"
        )

    features = validate_features(
        frame,
        "EEG",
        EEG_FEATURES,
    )

    prepared = pd.concat(
        [
            frame[EEG_METADATA].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )

    return prepared


def prepare_eit():
    frame = pd.read_csv(EIT_FILE)

    excluded = [
        col for col in EIT_QA_ONLY
        if col in frame.columns
    ]

    feature_columns = [
        col for col in frame.columns
        if col not in EIT_QA_ONLY
    ]

    features = validate_features(
        frame,
        "EIT",
        feature_columns,
    )

    return features, excluded


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("NEUROSENTINEL DUAL-ENCODER PREPARATION")

    eeg = prepare_eeg()
    eit, excluded = prepare_eit()

    eeg_output = OUTPUT_DIR / "EEG_Prepared_Features.csv"
    eit_output = OUTPUT_DIR / "EIT_Prepared_Features.csv"

    eeg.to_csv(eeg_output, index=False)
    eit.to_csv(eit_output, index=False)

    report = {
        "EEG_rows": len(eeg),
        "EEG_feature_columns": EEG_FEATURES,
        "EEG_metadata_columns": EEG_METADATA,
        "EIT_rows": len(eit),
        "EIT_feature_columns": list(eit.columns),
        "EIT_excluded_QA_columns": excluded,
        "artificial_pairing_performed": False,
        "clinical_fusion_training_ready": False,
        "status": "FEATURE_PREPARATION_PASSED_PAIRING_NOT_READY",
    }

    report_path = (
        OUTPUT_DIR /
        "NeuroSentinel_Feature_Preparation_Report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    print("\nEEG rows prepared:", len(eeg))
    print("EEG features:", EEG_FEATURES)

    print("\nEIT rows prepared:", len(eit))
    print("EIT features:", len(eit.columns))
    print("Excluded QA columns:", excluded)

    print("\nArtificial pairing:", False)
    print("Clinical fusion training ready:", False)

    print("\nSTATUS: FEATURE PREPARATION PASSED")
    print("EEG output:", eeg_output)
    print("EIT output:", eit_output)
    print("Report:", report_path)


if __name__ == "__main__":
    main()