
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\NeuroSentinel\data")

INPUT_FILE = (
    DATA_DIR / "NeuroSentinel_EEG_Filtered_3s_Features.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "NeuroSentinel_EEG_Feature_Distribution_Audit.csv"
)

FEATURES = ["Delta", "Theta", "Alpha", "Beta"]


def main():
    df = pd.read_csv(INPUT_FILE)

    required = ["Participant", "Task"] + FEATURES

    if not set(required).issubset(df.columns):
        raise ValueError("Required columns are missing.")

    if df[required].isna().any().any():
        raise ValueError("Missing values detected.")

    values = df[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite values detected.")

    if (values < 0).any():
        raise ValueError("Negative band-power values detected.")

    print("EEG FEATURE DISTRIBUTION AUDIT")
    print("=" * 65)

    print("Rows:", len(df))
    print("Participants:", df["Participant"].nunique())

    print("\nOVERALL FEATURE STATISTICS")

    overall = df[FEATURES].describe().T
    print(overall.to_string())

    audit_rows = []

    print("\nPARTICIPANT-LEVEL FEATURE AUDIT")

    for participant, group in df.groupby("Participant"):

        print(f"\nParticipant: {participant}")

        for feature in FEATURES:

            series = group[feature]

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = int(
                ((series < lower) | (series > upper)).sum()
            )

            audit_rows.append({
                "Participant": participant,
                "Feature": feature,
                "Window_Count": len(series),
                "Minimum": series.min(),
                "Maximum": series.max(),
                "Mean": series.mean(),
                "Median": series.median(),
                "Standard_Deviation": series.std(),
                "IQR_Lower_Bound": lower,
                "IQR_Upper_Bound": upper,
                "IQR_Outlier_Count": outliers,
            })

            print(
                f"{feature}: "
                f"min={series.min():.5e}, "
                f"max={series.max():.5e}, "
                f"mean={series.mean():.5e}, "
                f"IQR outliers={outliers}"
            )

    report = pd.DataFrame(audit_rows)

    report.to_csv(OUTPUT_FILE, index=False)

    print("\nAUDIT COMPLETE")
    print("Total participant-feature checks:", len(report))
    print(
        "Total IQR flags:",
        int(report["IQR_Outlier_Count"].sum())
    )
    print("No observations were removed or modified.")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
