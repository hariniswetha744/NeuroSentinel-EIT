from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

ROOT = Path(r"C:\NeuroSentinel")
DATA = ROOT / "data"

FEATURES_FILE = DATA / "NeuroSentinel_EEG_Filtered_3s_Features.csv"
DISTRIBUTION_FILE = DATA / "NeuroSentinel_EEG_Feature_Distribution_Audit.csv"
COMPARISON_FILE = DATA / "NeuroSentinel_EEG_Original_vs_Filtered_Comparison.csv"

OUTPUT_FILE = DATA / "NeuroSentinel_EEG_Validation_Report.txt"


def main():
    required_files = [
        FEATURES_FILE,
        DISTRIBUTION_FILE,
        COMPARISON_FILE,
    ]

    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Required report inputs are missing:\n" + "\n".join(missing)
        )

    features = pd.read_csv(FEATURES_FILE)
    distribution = pd.read_csv(DISTRIBUTION_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)

    required_columns = [
        "Participant", "Task", "Delta", "Theta", "Alpha", "Beta"
    ]

    if not set(required_columns).issubset(features.columns):
        raise ValueError("EEG feature file has missing required columns.")

    values = features[["Delta", "Theta", "Alpha", "Beta"]].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError("EEG feature file contains non-finite values.")

    if (values < 0).any():
        raise ValueError("EEG feature file contains negative band powers.")

    participants = sorted(features["Participant"].astype(str).unique())
    task_counts = features["Task"].value_counts().sort_index()

    total_flags = int(distribution["IQR_Outlier_Count"].sum())
    checks = len(distribution)

    lines = []

    def add(value=""):
        lines.append(str(value))

    add("NEUROSENTINEL — EEG VALIDATION REPORT")
    add("=" * 72)
    add(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add()

    add("1. DATASET SUMMARY")
    add("-" * 72)
    add("Dataset: Public EEG motor-imagery recordings (EEGBCI)")
    add("Recording used: R04")
    add(f"Total feature windows: {len(features)}")
    add(f"Unique participants: {len(participants)}")
    add(f"Participant IDs: {', '.join(participants)}")

    add("Task counts:")
    for task, count in task_counts.items():
        add(f"  {task}: {count}")

    add()
    add("2. FEATURE INTEGRITY")
    add("-" * 72)
    add("Required feature columns: Delta, Theta, Alpha, Beta")
    add(f"Missing values in required columns: {int(features[required_columns].isna().sum().sum())}")
    add(f"All feature values finite: {bool(np.isfinite(values).all())}")
    add(f"All band powers non-negative: {bool((values >= 0).all())}")
    add("No observations were removed by the distribution audit.")

    add()
    add("3. FEATURE DISTRIBUTION AUDIT")
    add("-" * 72)
    add(f"Participant-feature checks: {checks}")
    add(f"Total IQR statistical flags: {total_flags}")
    add("IQR flags are indicators for review, not proof of erroneous data.")
    add("No outlier observations were automatically deleted.")

    add()
    add("4. ORIGINAL VS FILTERED MODEL COMPARISON")
    add("-" * 72)
    add(comparison.to_string(index=False))

    add()
    add("5. INTERPRETATION")
    add("-" * 72)
    add(
        "The filtered model showed small improvements over the original "
        "model in the recorded comparison."
    )
    add(
        "The improvements do not establish reliable participant-independent "
        "motor-imagery classification."
    )
    add(
        "Previously recorded majority-baseline results were higher than "
        "the original and filtered Logistic Regression accuracy results."
    )

    add()
    add("6. IMPORTANT RESEARCH LIMITATIONS")
    add("-" * 72)
    add(
        "The EEG recordings are public motor-imagery data, not brain-tumour "
        "patient recordings."
    )
    add(
        "The task labels T1 and T2 are motor-imagery task labels, not tumour "
        "or recurrence labels."
    )
    add(
        "These EEG results do not demonstrate brain-tumour detection, "
        "tumour classification, or recurrence prediction."
    )
    add(
        "Clinical validation requires appropriately labelled patient data "
        "and independent clinical evaluation."
    )

    add()
    add("7. OUTPUT FILES")
    add("-" * 72)
    add(f"Feature data: {FEATURES_FILE}")
    add(f"Distribution audit: {DISTRIBUTION_FILE}")
    add(f"Original vs filtered comparison: {COMPARISON_FILE}")
    add(f"Consolidated report: {OUTPUT_FILE}")

    add()
    add("FINAL STATUS")
    add("-" * 72)
    add("EEG feature integrity checks: PASSED")
    add("Feature distribution audit: COMPLETED")
    add("Original vs filtered comparison: RECORDED")
    add("Clinical tumour validation: NOT ESTABLISHED")
    add("Clinical recurrence prediction: NOT ESTABLISHED")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("EEG VALIDATION REPORT CREATED")
    print("=" * 55)
    print("Windows:", len(features))
    print("Participants:", len(participants))
    print("Distribution checks:", checks)
    print("IQR flags:", total_flags)
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()