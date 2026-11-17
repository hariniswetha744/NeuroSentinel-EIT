from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(r"C:\NeuroSentinel")

EEG_FILE = (
    ROOT / "fusion" / "prepared_features" /
    "EEG_Prepared_Features.csv"
)

EIT_FILE = (
    ROOT / "fusion" / "prepared_features" /
    "EIT_Prepared_Features.csv"
)

OUTPUT_FILE = (
    ROOT / "fusion" /
    "NeuroSentinel_Fusion_Architecture_Audit.json"
)

EEG_FEATURES = ["Delta", "Theta", "Alpha", "Beta"]

EIT_QA_ONLY = ["Region_Mask_Element_Count_QA_Only"]

MINIMUM_OBSERVATIONS = 2


def inspect_dataset(path, name, features):
    result = {
        "file_exists": path.exists(),
        "rows": 0,
        "features": features,
        "missing_values": None,
        "finite_numeric_values": False,
        "sample_id_available": False,
        "valid": False,
    }

    if not path.exists():
        return result

    frame = pd.read_csv(path)

    result["rows"] = len(frame)
    result["missing_values"] = int(
        frame.isna().sum().sum()
    )

    missing_features = [
        col for col in features
        if col not in frame.columns
    ]

    if missing_features or frame.empty:
        return result

    numeric = frame[features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    result["finite_numeric_values"] = bool(
        np.isfinite(numeric.to_numpy()).all()
    )

    result["sample_id_available"] = (
        "Sample_ID" in frame.columns
    )

    result["valid"] = (
        result["missing_values"] == 0
        and result["finite_numeric_values"]
    )

    return result


def main():
    eeg_frame = pd.read_csv(EEG_FILE)
    eit_frame = pd.read_csv(EIT_FILE)

    eeg_features = EEG_FEATURES

    eit_features = [
        col for col in eit_frame.columns
        if col not in EIT_QA_ONLY
    ]

    eeg = inspect_dataset(
        EEG_FILE,
        "EEG",
        eeg_features,
    )

    eit = inspect_dataset(
        EIT_FILE,
        "EIT",
        eit_features,
    )

    eeg_ids_available = "Sample_ID" in eeg_frame.columns
    eit_ids_available = "Sample_ID" in eit_frame.columns

    matching_ids_verified = False

    if eeg_ids_available and eit_ids_available:
        eeg_ids = eeg_frame["Sample_ID"]
        eit_ids = eit_frame["Sample_ID"]

        matching_ids_verified = (
            not eeg_ids.isna().any()
            and not eit_ids.isna().any()
            and not eeg_ids.duplicated().any()
            and not eit_ids.duplicated().any()
            and set(eeg_ids) == set(eit_ids)
        )

    enough_eeg = len(eeg_frame) >= MINIMUM_OBSERVATIONS
    enough_eit = len(eit_frame) >= MINIMUM_OBSERVATIONS

    fusion_ready = (
        eeg["valid"]
        and eit["valid"]
        and enough_eeg
        and enough_eit
        and matching_ids_verified
    )

    report = {
        "EEG": eeg,
        "EIT": eit,
        "minimum_observations_required": MINIMUM_OBSERVATIONS,
        "EEG_observation_requirement_met": enough_eeg,
        "EIT_observation_requirement_met": enough_eit,
        "EEG_sample_ID_available": eeg_ids_available,
        "EIT_sample_ID_available": eit_ids_available,
        "verified_matching_sample_IDs": matching_ids_verified,
        "artificial_pairing_performed": False,
        "clinical_fusion_training_ready": False,
        "fusion_architecture_data_ready": fusion_ready,
        "status": (
            "READY_FOR_FUSION_DATA_VALIDATION"
            if fusion_ready
            else "BLOCKED_CLINICAL_DATA_REQUIREMENTS"
        ),
        "limitations": [
            "Current EEG data represents public motor-imagery recordings.",
            "Current EIT data represents simulation-derived features.",
            "The current datasets have no verified shared Sample_ID.",
            "A valid software architecture does not establish clinical validity.",
        ],
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    print("NEUROSENTINEL FUSION ARCHITECTURE AUDIT")
    print("\nEEG rows:", eeg["rows"])
    print("EEG valid:", eeg["valid"])

    print("\nEIT rows:", eit["rows"])
    print("EIT valid:", eit["valid"])

    print("\nEEG observation requirement:", enough_eeg)
    print("EIT observation requirement:", enough_eit)

    print("\nEEG Sample_ID available:", eeg_ids_available)
    print("EIT Sample_ID available:", eit_ids_available)

    print("Verified matching IDs:", matching_ids_verified)
    print("Artificial pairing:", False)

    print("\nFusion architecture data ready:", fusion_ready)
    print("Clinical fusion training ready:", False)

    print("\nSTATUS:", report["status"])
    print("Report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()