
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\NeuroSentinel")
OUT = ROOT / "fusion" / "synthetic_integration"

EEG_FILE = OUT / "EEG_Integration_Features.csv"
EIT_FILE = OUT / "EIT_Integration_Features.csv"

eeg = pd.read_csv(EEG_FILE)
eit = pd.read_csv(EIT_FILE)

checks = []

def record(name, passed, detail):
    checks.append((name, passed, detail))

# Check whether a verified common key exists
has_eeg_id = "Sample_ID" in eeg.columns
has_eit_id = "Sample_ID" in eit.columns

record(
    "Common Sample_ID field",
    has_eeg_id and has_eit_id,
    f"EEG={has_eeg_id}, EIT={has_eit_id}"
)

if has_eeg_id and has_eit_id:
    eeg_ids = eeg["Sample_ID"]
    eit_ids = eit["Sample_ID"]

    record(
        "No missing IDs",
        not eeg_ids.isna().any() and not eit_ids.isna().any(),
        "Missing-ID validation"
    )

    record(
        "Unique IDs",
        eeg_ids.is_unique and eit_ids.is_unique,
        "One record per modality per Sample_ID"
    )

    record(
        "Exact ID correspondence",
        set(eeg_ids) == set(eit_ids),
        "EEG and EIT identifier sets"
    )
else:
    record(
        "Pairing validation",
        False,
        "No verified shared Sample_ID; fusion is blocked."
    )

eligible = all(passed for _, passed, _ in checks)

lines = [
    "NEUROSENTINEL EEG-EIT FUSION GATE",
    "=" * 48,
    f"EEG records: {len(eeg)}",
    f"EIT records: {len(eit)}",
    ""
]

for name, passed, detail in checks:
    lines.append(
        f"{'PASS' if passed else 'BLOCKED'} | {name} | {detail}"
    )

lines.extend([
    "",
    f"FUSION ELIGIBLE: {eligible}",
    "CURRENT DECISION: " +
    ("ELIGIBLE FOR PAIRING REVIEW" if eligible else "BLOCKED"),
    "Reason: verified paired EEG-EIT observations are required.",
    "No arbitrary or index-based pairing is permitted.",
    "No clinical inference is supported by unpaired datasets."
])

report = "\n".join(lines) + "\n"

(OUT / "Fusion_Pairing_Gate_Report.txt").write_text(
    report, encoding="utf-8"
)

print(report)