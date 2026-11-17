from pathlib import Path

import numpy as np
import pandas as pd


class FusionReadinessError(ValueError):
    """Raised when EEG and EIT samples cannot be safely paired."""


def _validate_frame(frame, name, sample_id):
    if not isinstance(frame, pd.DataFrame):
        raise FusionReadinessError(
            f"{name} input must be a pandas DataFrame."
        )

    if not frame.columns.is_unique:
        duplicates = frame.columns[
            frame.columns.duplicated()
        ].tolist()
        raise FusionReadinessError(
            f"{name} contains duplicate column names: {duplicates}"
        )

    if sample_id not in frame.columns:
        raise FusionReadinessError(
            f"{name} data has no '{sample_id}' column."
        )

    if frame.empty:
        raise FusionReadinessError(f"{name} data is empty.")

    ids = frame[sample_id]

    if ids.isna().any():
        raise FusionReadinessError(
            f"{name} contains missing sample IDs."
        )

    if ids.astype(str).str.strip().eq("").any():
        raise FusionReadinessError(
            f"{name} contains blank sample IDs."
        )

    if ids.duplicated().any():
        raise FusionReadinessError(
            f"{name} contains duplicate sample IDs. "
            "Define the intended sample/session level first."
        )

    feature_columns = [
        column for column in frame.columns
        if column != sample_id
    ]

    if frame[feature_columns].isna().any().any():
        raise FusionReadinessError(
            f"{name} contains missing feature values."
        )

    numeric_columns = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]

    for column in numeric_columns:
        values = frame[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise FusionReadinessError(
                f"{name} contains non-finite values "
                f"in feature '{column}'."
            )


def prepare_fusion_table(
    eeg: pd.DataFrame,
    eit: pd.DataFrame,
    sample_id: str = "Sample_ID",
) -> pd.DataFrame:
    """
    Assemble a fusion table only when both modalities have
    explicit, non-missing, unique, matching sample identifiers
    and valid feature values.

    This function does not establish clinical validity.
    """

    _validate_frame(eeg, "EEG", sample_id)
    _validate_frame(eit, "EIT", sample_id)

    eeg_ids = set(eeg[sample_id])
    eit_ids = set(eit[sample_id])

    if eeg_ids != eit_ids:
        raise FusionReadinessError(
            "EEG and EIT sample IDs do not match. "
            f"EEG-only IDs: {len(eeg_ids - eit_ids)}; "
            f"EIT-only IDs: {len(eit_ids - eeg_ids)}."
        )

    try:
        result = eeg.merge(
            eit,
            on=sample_id,
            how="inner",
            suffixes=("_EEG", "_EIT"),
            validate="one_to_one",
        )
    except ValueError as exc:
        raise FusionReadinessError(
            "Merge produced ambiguous duplicate column names "
            "or an invalid merge result."
        ) from exc

    if not result.columns.is_unique:
        duplicates = result.columns[
            result.columns.duplicated()
        ].tolist()
        raise FusionReadinessError(
            "Merge produced ambiguous duplicate column names: "
            f"{duplicates}"
        )

    return result


def main():
    root = Path(r"C:\NeuroSentinel")
    eeg_path = (
        root / "data" /
        "NeuroSentinel_MultiSubject_EEG_Features.csv"
    )
    eit_path = (
        root / "eit" / "features" /
        "EIT_Feature_Summary.csv"
    )

    eeg = pd.read_csv(eeg_path)
    eit = pd.read_csv(eit_path)

    print("NEUROSENTINEL FUSION INTERFACE TEST")
    print("EEG rows:", len(eeg))
    print("EIT rows:", len(eit))

    try:
        prepare_fusion_table(eeg, eit)
    except FusionReadinessError as exc:
        print("Fusion correctly blocked:", exc)
        print("Status: SAFE BLOCK — no unmatched data combined.")
    else:
        print("Matching IDs were found.")
        print(
            "This alone does not establish tumour labels "
            "or clinical validity."
        )


if __name__ == "__main__":
    main()