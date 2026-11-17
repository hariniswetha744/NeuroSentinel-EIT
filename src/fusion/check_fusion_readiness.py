from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"C:\NeuroSentinel")
EEG_FILE = ROOT / "data" / "NeuroSentinel_MultiSubject_EEG_Features.csv"
EIT_FILE = ROOT / "eit" / "features" / "EIT_Feature_Summary.csv"
OUTPUT_DIR = ROOT / "fusion"
REPORT_FILE = OUTPUT_DIR / "NeuroSentinel_Fusion_Readiness.json"


def main():
    report = {
        "project": "NeuroSentinel",
        "purpose": "EEG-EIT fusion readiness audit",
        "eeg_file_exists": EEG_FILE.is_file(),
        "eit_file_exists": EIT_FILE.is_file(),
        "eeg_windows": 0,
        "eeg_participants": 0,
        "eeg_task_labels": {},
        "eit_feature_rows": 0,
        "paired_sample_ids_verified": False,
        "verified_tumour_labels_available": False,
        "longitudinal_recurrence_labels_available": False,
        "clinical_fusion_training_ready": False,
        "status": "NOT_READY",
        "warnings": [],
    }

    if EEG_FILE.is_file():
        eeg = pd.read_csv(EEG_FILE)

        required_eeg = {
            "Participant", "Run", "Task", "Window",
            "Delta", "Theta", "Alpha", "Beta"
        }

        missing = sorted(required_eeg - set(eeg.columns))
        if missing:
            report["warnings"].append(
                f"EEG file is missing required columns: {missing}"
            )
        else:
            report["eeg_windows"] = int(len(eeg))
            report["eeg_participants"] = int(
                eeg["Participant"].nunique()
            )
            report["eeg_task_labels"] = {
                str(k): int(v)
                for k, v in eeg["Task"].value_counts().items()
            }
            report["warnings"].append(
                "EEG task labels describe motor tasks, not tumour status."
            )
    else:
        report["warnings"].append("EEG feature file was not found.")

    if EIT_FILE.is_file():
        eit = pd.read_csv(EIT_FILE)
        report["eit_feature_rows"] = int(len(eit))
        report["warnings"].append(
            "EIT features describe a simulation, not a verified patient."
        )
    else:
        report["warnings"].append("EIT feature summary was not found.")

    report["warnings"].append(
        "Paired EEG-EIT sample identifiers have not been established."
    )
    report["warnings"].append(
        "Verified tumour labels and longitudinal recurrence outcomes "
        "are not available in the current inputs."
    )
    report["warnings"].append(
        "Do not interpret this audit as a clinical validation."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("NEUROSENTINEL FUSION READINESS AUDIT")
    print("-" * 40)
    print(json.dumps(report, indent=2))
    print("\nReport saved to:", REPORT_FILE)


if __name__ == "__main__":
    main()