
from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\NeuroSentinel\data")

ORIGINAL = DATA_DIR / "NeuroSentinel_MultiSubject_EEG_3s_Features.csv"
FILTERED = DATA_DIR / "NeuroSentinel_EEG_Filtered_3s_Features.csv"

KEYS = ["Participant", "Run", "Task", "Window"]


def main():
    original = pd.read_csv(ORIGINAL)
    filtered = pd.read_csv(FILTERED)

    print("EEG ORIGINAL VS FILTERED COMPARISON AUDIT")
    print("=" * 60)

    print("Original rows:", len(original))
    print("Filtered rows:", len(filtered))

    for name, df in [
        ("Original", original),
        ("Filtered", filtered),
    ]:
        missing = [col for col in KEYS if col not in df.columns]

        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

        if df.duplicated(KEYS).any():
            raise ValueError(f"{name} contains duplicate window IDs.")

    original_ids = set(map(tuple, original[KEYS].to_numpy()))
    filtered_ids = set(map(tuple, filtered[KEYS].to_numpy()))

    print("\nIDENTIFIER CHECK")
    print("Original unique windows:", len(original_ids))
    print("Filtered unique windows:", len(filtered_ids))
    print("Matching identifiers:", original_ids == filtered_ids)
    print("Only in original:", len(original_ids - filtered_ids))
    print("Only in filtered:", len(filtered_ids - original_ids))

    if original_ids != filtered_ids:
        print("\nSTATUS: WINDOW MATCHING FAILED")
        return

    merged = original.merge(
        filtered,
        on=KEYS,
        how="inner",
        suffixes=("_Original", "_Filtered"),
        validate="one_to_one",
    )

    print("\nMATCHED DATA CHECK")
    print("Matched rows:", len(merged))
    print(
        "Participants:",
        merged["Participant"].nunique()
    )

    print("\nPARTICIPANT WINDOW COUNTS")
    print(
        merged.groupby("Participant").size().to_string()
    )

    print("\nTASK COUNTS")
    print(
        merged["Task"].value_counts().to_string()
    )

    print("\nFEATURE VALUE COMPARISON")

    for feature in ["Delta", "Theta", "Alpha", "Beta"]:
        old = merged[f"{feature}_Original"]
        new = merged[f"{feature}_Filtered"]

        print(
            f"{feature}: "
            f"Original mean={old.mean():.8e} | "
            f"Filtered mean={new.mean():.8e}"
        )

    print("\nSTATUS: EXACT WINDOW IDENTIFIERS MATCHED")
    print("No source files were modified.")


if __name__ == "__main__":
    main()
