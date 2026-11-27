import pandas as pd
from fusion_interface import (
    prepare_fusion_table,
    FusionReadinessError,
)


passed = 0
failed = 0


def test_blocked(name, eeg, eit, expected_text):
    global passed, failed

    try:
        prepare_fusion_table(eeg, eit)
    except FusionReadinessError as exc:
        if expected_text.lower() in str(exc).lower():
            passed += 1
            print(f"PASS — {name}: correctly blocked")
        else:
            failed += 1
            print(f"FAIL — {name}: unexpected reason: {exc}")
    else:
        failed += 1
        print(f"FAIL — {name}: unsafe data was accepted")


def test_valid_matching_ids():
    global passed, failed

    eeg = pd.DataFrame({
        "Sample_ID": ["A", "B"],
        "EEG_Feature": [0.1, 0.2],
    })

    eit = pd.DataFrame({
        "Sample_ID": ["B", "A"],
        "EIT_Feature": [2.0, 1.0],
    })

    try:
        result = prepare_fusion_table(eeg, eit)

        valid = (
            len(result) == 2
            and set(result["Sample_ID"]) == {"A", "B"}
            and "EEG_Feature" in result.columns
            and "EIT_Feature" in result.columns
        )

        if valid:
            passed += 1
            print("PASS — Valid matching IDs: one-to-one merge verified")
        else:
            failed += 1
            print("FAIL — Valid matching IDs: unexpected merge output")

    except Exception as exc:
        failed += 1
        print(f"FAIL — Valid matching IDs: {exc}")


# 1. Missing sample ID column
test_blocked(
    "Missing IDs",
    pd.DataFrame({"EEG_Feature": [0.1]}),
    pd.DataFrame({"Sample_ID": ["A"]}),
    "no 'Sample_ID' column",
)

# 2. Empty EEG data
test_blocked(
    "Empty EEG",
    pd.DataFrame(columns=["Sample_ID", "EEG_Feature"]),
    pd.DataFrame({"Sample_ID": ["A"], "EIT_Feature": [1.0]}),
    "EEG data is empty",
)

# 3. Empty EIT data
test_blocked(
    "Empty EIT",
    pd.DataFrame({"Sample_ID": ["A"], "EEG_Feature": [0.1]}),
    pd.DataFrame(columns=["Sample_ID", "EIT_Feature"]),
    "EIT data is empty",
)

# 4. Missing sample ID value
test_blocked(
    "Missing sample ID value",
    pd.DataFrame({
        "Sample_ID": [None],
        "EEG_Feature": [0.1],
    }),
    pd.DataFrame({
        "Sample_ID": ["A"],
        "EIT_Feature": [1.0],
    }),
    "missing sample IDs",
)

# 5. Duplicate sample IDs
test_blocked(
    "Duplicate IDs",
    pd.DataFrame({
        "Sample_ID": ["A", "A"],
        "EEG_Feature": [0.1, 0.2],
    }),
    pd.DataFrame({
        "Sample_ID": ["A"],
        "EIT_Feature": [1.0],
    }),
    "duplicate sample IDs",
)

# 6. Mismatched sample IDs
test_blocked(
    "Mismatched IDs",
    pd.DataFrame({
        "Sample_ID": ["A"],
        "EEG_Feature": [0.1],
    }),
    pd.DataFrame({
        "Sample_ID": ["B"],
        "EIT_Feature": [1.0],
    }),
    "sample IDs do not match",
)

# 7. Valid matching sample IDs
test_valid_matching_ids()

print("\nFusion safety test summary")
print("Passed:", passed)
print("Failed:", failed)
print("Total:", passed + failed)

if failed:
    raise SystemExit("Fusion safety tests FAILED.")
else:
    print("STATUS: ALL FUSION SAFETY TESTS PASSED.")