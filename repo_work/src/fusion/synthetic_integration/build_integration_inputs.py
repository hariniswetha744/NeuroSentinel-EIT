from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\NeuroSentinel")
OUT = ROOT / "fusion" / "synthetic_integration"
OUT.mkdir(parents=True, exist_ok=True)

# Source datasets
eeg_path = ROOT / "data" / "NeuroSentinel_C3C4_MuBeta_3s_Features.csv"
eit_path = (
    ROOT / "eit" / "synthetic_experiment" /
    "scenarios" / "EIT_Synthetic_Scenario_Measurements.csv"
)

eeg = pd.read_csv(eeg_path)
eit = pd.read_csv(eit_path)

# EEG feature matrix: one row per EEG window
eeg_features = [
    "C3_Mu_Power", "C4_Mu_Power",
    "C3_Beta_Power", "C4_Beta_Power",
    "C3_Mu_Log10", "C4_Mu_Log10",
    "C3_Beta_Log10", "C4_Beta_Log10"
]

eeg_output = eeg[
    ["Participant", "Run", "Task", "Window"] + eeg_features
].copy()

eeg_output.insert(0, "Modality", "EEG")
eeg_output["Pairing_Status"] = "NOT_PAIRED"

# EIT feature matrix: one row per simulated scenario
eit_matrix = eit.pivot(
    index="Scenario_ID",
    columns="Measurement_ID",
    values="Voltage"
).sort_index(axis=1)

eit_matrix.columns = [
    f"EIT_Measurement_{int(c):03d}"
    for c in eit_matrix.columns
]

eit_meta = (
    eit.groupby("Scenario_ID", sort=False)
    .agg(
        Severity=("Severity", "first"),
        Noise_Percent=("Noise_Percent", "first"),
        Repeat_ID=("Repeat_ID", "first"),
        Label=("Label", "first"),
        Measurement_Count=("Measurement_ID", "nunique")
    )
)

eit_output = eit_meta.join(eit_matrix, how="inner").reset_index()
eit_output.insert(0, "Modality", "EIT")
eit_output["Pairing_Status"] = "NOT_PAIRED"

# Integrity checks
assert len(eeg_output) == len(eeg)
assert len(eit_output) == eit["Scenario_ID"].nunique()
assert eit_output["Measurement_Count"].eq(208).all()
assert not eeg_output.isna().any().any()
assert not eit_output.isna().any().any()

numeric_eeg = eeg_output[eeg_features].to_numpy()
numeric_eit = eit_output.filter(
    like="EIT_Measurement_"
).to_numpy()

assert np.isfinite(numeric_eeg).all()
assert np.isfinite(numeric_eit).all()

# Save separate modality datasets
eeg_output.to_csv(OUT / "EEG_Integration_Features.csv", index=False)
eit_output.to_csv(OUT / "EIT_Integration_Features.csv", index=False)

# Explicit integration readiness report
report = [
    "NEUROSENTINEL SYNTHETIC INTEGRATION INPUT AUDIT",
    "=" * 55,
    f"EEG windows: {len(eeg_output)}",
    f"EEG participants: {eeg['Participant'].nunique()}",
    f"EIT scenarios: {len(eit_output)}",
    f"EIT measurements per scenario: {eit_output['Measurement_Count'].unique().tolist()}",
    f"EEG missing values: {eeg_output.isna().sum().sum()}",
    f"EIT missing values: {eit_output.isna().sum().sum()}",
    f"EEG finite features: {np.isfinite(numeric_eeg).all()}",
    f"EIT finite features: {np.isfinite(numeric_eit).all()}",
    "Shared participant/scenario identifiers: 0",
    "Clinical pairing: NOT AVAILABLE",
    "Fusion model training: BLOCKED",
    "Reason: No verified paired EEG-EIT observations.",
    "EEG labels: Healthy motor-imagery task labels (T1/T2).",
    "EIT labels: Synthetic conductivity-change labels.",
    "These labels are not interchangeable.",
    "No tumour or recurrence labels are available."
]

(OUT / "Integration_Readiness_Report.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\n".join(report))
print("\nIntegration input files saved to:", OUT)