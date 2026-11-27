
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(r"C:\NeuroSentinel")

EEG_FILE = ROOT / "data" / "NeuroSentinel_MultiSubject_EEG_Features.csv"
EIT_FILE = ROOT / "eit" / "features" / "EIT_Feature_Summary.csv"
OUTPUT_FILE = ROOT / "fusion" / "NeuroSentinel_Feature_Audit.json"

EEG_KEYS = ["Participant", "Run", "Task", "Window"]
EEG_FEATURES = ["Delta", "Theta", "Alpha", "Beta"]
EIT_QA_ONLY = ["Region_Mask_Element_Count_QA_Only"]


def audit_table(path, required_columns, numeric_columns, key_columns=None):
    report = {
        "file": str(path),
        "exists": path.exists(),
        "rows": 0,
        "columns": [],
        "missing_required_columns": [],
        "missing_values": 0,
        "duplicate_key_rows": None,
        "non_numeric_values": {},
        "non_finite_values": {},
        "passed": False,
    }

    if not path.exists():
        return report, None

    frame = pd.read_csv(path)
    report["rows"] = len(frame)
    report["columns"] = list(frame.columns)
    report["missing_required_columns"] = [
        col for col in required_columns if col not in frame.columns
    ]
    report["missing_values"] = int(frame.isna().sum().sum())

    if key_columns and all(col in frame.columns for col in key_columns):
        report["duplicate_key_rows"] = int(
            frame.duplicated(subset=key_columns, keep=False).sum()
        )

    for col in numeric_columns:
        if col not in frame.columns:
            continue

        converted = pd.to_numeric(frame[col], errors="coerce")
        report["non_numeric_values"][col] = int(
            (frame[col].notna() & converted.isna()).sum()
        )
        report["non_finite_values"][col] = int(
            sum(
                not math.isfinite(float(value))
                for value in converted.dropna()
            )
        )

    numeric_ok = (
        all(v == 0 for v in report["non_numeric_values"].values())
        and all(v == 0 for v in report["non_finite_values"].values())
    )
    duplicates_ok = (
        report["duplicate_key_rows"] in (None, 0)
    )

    report["passed"] = bool(
        len(frame) > 0
        and not report["missing_required_columns"]
        and report["missing_values"] == 0
        and duplicates_ok
        and numeric_ok
    )

    return report, frame


def main():
    eeg_required = EEG_KEYS + [
        "Onset_seconds",
        "Duration_seconds",
    ] + EEG_FEATURES

    eeg_numeric = [
        "Window",
        "Onset_seconds",
        "Duration_seconds",
    ] + EEG_FEATURES

    eit_numeric = [
        "Measurement_Count",
        "Baseline_Voltage_Mean",
        "Baseline_Voltage_Std",
        "Anomaly_Voltage_Mean",
        "Anomaly_Voltage_Std",
        "Voltage_Change_Mean",
        "Voltage_Change_Std",
        "Voltage_Change_MAE",
        "Voltage_Change_RMS",
        "Reconstruction_Element_Count",
        "Reconstruction_Min",
        "Reconstruction_Max",
        "Reconstruction_Mean",
        "Reconstruction_Std",
        "Reconstruction_RMS",
        "Positive_Reconstruction_Count",
        "Negative_Reconstruction_Count",
        "Region_Mask_Element_Count_QA_Only",
    ]

    eeg_report, eeg = audit_table(
        EEG_FILE, eeg_required, eeg_numeric, EEG_KEYS
    )

    eit_report, eit = audit_table(
        EIT_FILE, eit_numeric, eit_numeric
    )

    participants = []
    task_counts = {}

    if eeg is not None:
        if "Participant" in eeg.columns:
            participants = sorted(
                eeg["Participant"].dropna().astype(str).unique().tolist()
            )
        if "Task" in eeg.columns:
            task_counts = {
                str(k): int(v)
                for k, v in eeg["Task"].value_counts().items()
            }

    report = {
        "project": "NeuroSentinel",
        "purpose": "Independent EEG-EIT feature integrity audit",
        "EEG_audit": eeg_report,
        "EIT_audit": eit_report,
        "observation_levels": {
            "EEG": "Individual motor-imagery windows",
            "EIT": "One simulated reconstruction summary row",
        },
        "EEG_participants": participants,
        "EEG_task_counts": task_counts,
        "QA_only_columns_excluded_from_model_features": EIT_QA_ONLY,
        "paired_sample_ids_verified": False,
        "artificial_pairing_performed": False,
        "clinical_AI_training_ready": False,
        "status": (
            "FEATURE_AUDIT_PASSED_PAIRING_NOT_READY"
            if eeg_report["passed"] and eit_report["passed"]
            else "FEATURE_AUDIT_REQUIRES_ATTENTION"
        ),
        "warnings": [
            "EEG task labels represent motor-imagery tasks, not tumour status.",
            "EIT features represent a simulation, not a verified patient.",
            "EEG windows and EIT summary have no verified shared sample IDs.",
            "The EIT QA-only region-mask count must not be used as a model feature or tumour label.",
            "No clinical tumour diagnosis or recurrence prediction is established.",
        ],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("NEUROSENTINEL FEATURE AUDIT")
    print("EEG rows:", eeg_report["rows"])
    print("EEG audit passed:", eeg_report["passed"])
    print("EIT rows:", eit_report["rows"])
    print("EIT audit passed:", eit_report["passed"])
    print("EEG participants:", len(participants))
    print("EEG task counts:", task_counts)
    print("Artificial pairing performed: False")
    print("Clinical AI training ready: False")
    print("Status:", report["status"])
    print("Report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
