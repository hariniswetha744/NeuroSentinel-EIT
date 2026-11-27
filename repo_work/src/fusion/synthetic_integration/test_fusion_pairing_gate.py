import pandas as pd


def validate_pairing(eeg, eit):
    if "Sample_ID" not in eeg.columns:
        return False, "EEG Sample_ID missing"

    if "Sample_ID" not in eit.columns:
        return False, "EIT Sample_ID missing"

    eeg_ids = eeg["Sample_ID"]
    eit_ids = eit["Sample_ID"]

    if eeg_ids.isna().any() or eit_ids.isna().any():
        return False, "Missing IDs"

    if eeg_ids.astype(str).str.strip().eq("").any():
        return False, "Empty EEG IDs"

    if eit_ids.astype(str).str.strip().eq("").any():
        return False, "Empty EIT IDs"

    if not eeg_ids.is_unique or not eit_ids.is_unique:
        return False, "Duplicate IDs"

    if set(eeg_ids) != set(eit_ids):
        return False, "Mismatched IDs"

    return True, "Verified identifier correspondence"


tests = [
    (
        "Valid matching IDs",
        pd.DataFrame({"Sample_ID": ["A", "B"]}),
        pd.DataFrame({"Sample_ID": ["B", "A"]}),
        True,
    ),
    (
        "Missing ID",
        pd.DataFrame({"Other_ID": ["A"]}),
        pd.DataFrame({"Sample_ID": ["A"]}),
        False,
    ),
    (
        "Duplicate IDs",
        pd.DataFrame({"Sample_ID": ["A", "A"]}),
        pd.DataFrame({"Sample_ID": ["A", "B"]}),
        False,
    ),
    (
        "Mismatched IDs",
        pd.DataFrame({"Sample_ID": ["A", "B"]}),
        pd.DataFrame({"Sample_ID": ["A", "C"]}),
        False,
    ),
    (
        "Empty ID",
        pd.DataFrame({"Sample_ID": ["A", ""]}),
        pd.DataFrame({"Sample_ID": ["A", "B"]}),
        False,
    ),
    (
        "Missing value",
        pd.DataFrame({"Sample_ID": ["A", None]}),
        pd.DataFrame({"Sample_ID": ["A", "B"]}),
        False,
    ),
]

passed = 0

print("NEUROSENTINEL PAIRING GATE TESTS")
print("=" * 48)

for name, eeg, eit, expected in tests:
    actual, reason = validate_pairing(eeg, eit)
    success = actual == expected

    if success:
        passed += 1

    print(
        f"{'PASS' if success else 'FAIL'} | "
        f"{name} | {reason}"
    )

print("=" * 48)
print(f"TESTS PASSED: {passed}/{len(tests)}")
print("OVERALL:", "PASS" if passed == len(tests) else "FAIL")