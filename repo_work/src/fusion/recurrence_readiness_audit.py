
from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"C:\NeuroSentinel")
OUTPUT_DIR = ROOT / "fusion"
REPORT_FILE = OUTPUT_DIR / "NeuroSentinel_Recurrence_Readiness.json"

# These are required fields for a future longitudinal dataset.
REQUIRED_COLUMNS = [
    "Subject_ID",
    "Session_ID",
    "Observation_Date",
    "EEG_Record_ID",
    "EIT_Record_ID",
    "Verified_Tumour_Status",
    "Verified_Recurrence_Status",
    "Outcome_Verification_Source",
]

# No clinical dataset is assumed or fabricated.
DATA_FILE = ROOT / "fusion" / "longitudinal_clinical_data.csv"


def main():
    report = {
        "project": "NeuroSentinel",
        "purpose": "Longitudinal recurrence data readiness audit",
        "input_file": str(DATA_FILE),
        "input_exists": DATA_FILE.is_file(),
        "records": 0,
        "required_columns": REQUIRED_COLUMNS,
        "missing_columns": REQUIRED_COLUMNS.copy(),
        "missing_values": None,
        "invalid_dates": None,
        "duplicate_subject_session_pairs": None,
        "subjects_with_multiple_sessions": None,
        "verified_outcomes_available": False,
        "paired_eeg_eit_records_available": False,
        "recurrence_model_evaluation_ready": False,
        "status": "DATASET_NOT_PROVIDED",
        "warnings": [],
    }

    if DATA_FILE.is_file():
        try:
            frame = pd.read_csv(DATA_FILE)
            report["records"] = int(len(frame))
            report["missing_columns"] = [
                col for col in REQUIRED_COLUMNS
                if col not in frame.columns
            ]

            if not report["missing_columns"]:
                report["missing_values"] = int(
                    frame[REQUIRED_COLUMNS].isna().sum().sum()
                )

                dates = pd.to_datetime(
                    frame["Observation_Date"],
                    errors="coerce"
                )
                report["invalid_dates"] = int(dates.isna().sum())

                report["duplicate_subject_session_pairs"] = int(
                    frame.duplicated(
                        subset=["Subject_ID", "Session_ID"],
                        keep=False
                    ).sum()
                )

                sessions = frame.groupby(
                    "Subject_ID"
                )["Session_ID"].nunique()

                report["subjects_with_multiple_sessions"] = int(
                    (sessions >= 2).sum()
                )

                verified = (
                    frame["Verified_Tumour_Status"].notna()
                    & frame["Verified_Recurrence_Status"].notna()
                    & frame["Outcome_Verification_Source"].notna()
                )

                report["verified_outcomes_available"] = bool(
                    verified.any()
                )

                paired = (
                    frame["EEG_Record_ID"].notna()
                    & frame["EIT_Record_ID"].notna()
                )

                report["paired_eeg_eit_records_available"] = bool(
                    paired.any()
                )

                if (
                    len(frame) > 0
                    and report["missing_values"] == 0
                    and report["invalid_dates"] == 0
                    and report["duplicate_subject_session_pairs"] == 0
                    and report["subjects_with_multiple_sessions"] > 0
                    and report["verified_outcomes_available"]
                    and report["paired_eeg_eit_records_available"]
                ):
                    report["status"] = (
                        "STRUCTURAL_CHECKS_PASSED_REQUIRES_CLINICAL_REVIEW"
                    )
                    report["warnings"].append(
                        "Structural checks do not establish clinical validity."
                    )
                else:
                    report["status"] = "DATASET_REQUIRES_ATTENTION"

            else:
                report["status"] = "REQUIRED_COLUMNS_MISSING"

        except Exception as exc:
            report["status"] = "AUDIT_ERROR"
            report["warnings"].append(str(exc))

    else:
        report["warnings"].append(
            "No longitudinal clinical dataset was supplied."
        )

    report["warnings"].extend([
        "Never invent subject IDs, session IDs or clinical outcomes.",
        "EEG motor-imagery labels are not tumour labels.",
        "The current EIT summary represents a simulation.",
        "Recurrence evaluation requires verified longitudinal clinical outcomes.",
        "A structural audit is not clinical validation.",
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("NEUROSENTINEL RECURRENCE READINESS AUDIT")
    print("-" * 45)
    print(json.dumps(report, indent=2))
    print("\nReport saved:", REPORT_FILE)


if __name__ == "__main__":
    main()
