import pandas as pd

from fusion_interface import (
    prepare_fusion_table,
    FusionReadinessError,
)


passed = 0
failed = 0


def blocked(name, eeg, eit, expected):
    global passed, failed

    try:
        prepare_fusion_table(eeg, eit)
    except FusionReadinessError as exc:
        if expected.lower() in str(exc).lower():
            passed += 1
            print(f"PASS — {name}")
        else:
            failed += 1
            print(f"FAIL — {name}: unexpected error: {exc}")
    except Exception as exc:
        failed += 1
        print(f"FAIL — {name}: unexpected exception: {exc}")
    else:
        failed += 1
        print(f"FAIL — {name}: invalid data was accepted")


def test_valid_association():
    global passed, failed

    eeg = pd.DataFrame({
        "Sample_ID": ["A", "B"],
        "Shared_Feature": [10.0, 20.0],
    })

    eit = pd.DataFrame({
        "Sample_ID": ["B", "A"],
        "Shared_Feature": [200.0, 100.0],
    })

    try:
        result = prepare_fusion_table(eeg, eit)
        result = result.set_index("Sample_ID")

        valid = (
            result.loc["A", "Shared_Feature_EEG"] == 10.0
            and result.loc["A", "Shared_Feature_EIT"] == 100.0
            and result.loc["B", "Shared_Feature_EEG"] == 20.0
            and result.loc["B", "Shared_Feature_EIT"] == 200.0
            and result.columns.is_unique
        )

        if valid:
            passed += 1
            print("PASS — Correct sample association and suffixes")
        else:
            failed += 1
            print("FAIL — Incorrect sample association or columns")
    except Exception as exc:
        failed += 1
        print(f"FAIL — Valid association: {exc}")


# 1. Missing EEG feature value
blocked(
    "Missing EEG feature value",
    pd.DataFrame({"Sample_ID": ["A"], "Feature": [None]}),
    pd.DataFrame({"Sample_ID": ["A"], "EIT_Feature": [2.0]}),
    "missing feature values",
)

# 2. Missing EIT feature value
blocked(
    "Missing EIT feature value",
    pd.DataFrame({"Sample_ID": ["A"], "Feature": [1.0]}),
    pd.DataFrame({"Sample_ID": ["A"], "EIT_Feature": [None]}),
    "missing feature values",
)

# 3. Infinite EEG feature
blocked(
    "Infinite EEG feature",
    pd.DataFrame({"Sample_ID": ["A"], "Feature": [float("inf")]}),
    pd.DataFrame({"Sample_ID": ["A"], "EIT_Feature": [2.0]}),
    "non-finite values",
)

# 4. Negative infinite EIT feature
blocked(
    "Negative infinite EIT feature",
    pd.DataFrame({"Sample_ID": ["A"], "Feature": [1.0]}),
    pd.DataFrame({"Sample_ID": ["A"], "EIT_Feature": [float("-inf")]}),
    "non-finite values",
)

# 5. Blank sample ID
blocked(
    "Blank sample ID",
    pd.DataFrame({"Sample_ID": ["  "], "Feature": [1.0]}),
    pd.DataFrame({"Sample_ID": ["  "], "EIT_Feature": [2.0]}),
    "blank sample IDs",
)

# 6. Duplicate column names
duplicate_columns = pd.DataFrame(
    [["A", 1.0, 2.0]],
    columns=["Sample_ID", "Feature", "Feature"],
)

blocked(
    "Duplicate EEG column names",
    duplicate_columns,
    pd.DataFrame({"Sample_ID": ["A"], "EIT_Feature": [2.0]}),
    "duplicate column names",
)

# 7. Merge suffix collision
blocked(
    "Ambiguous merge suffix collision",
    pd.DataFrame({
        "Sample_ID": ["A"],
        "Feature": [1.0],
        "Feature_EEG": [3.0],
    }),
    pd.DataFrame({
        "Sample_ID": ["A"],
        "Feature": [2.0],
    }),
    "ambiguous duplicate column names",
)

# 8. Correct matching and association
test_valid_association()


print("\nExpanded fusion feature validation summary")
print("Passed:", passed)
print("Failed:", failed)
print("Total:", passed + failed)

if failed:
    raise SystemExit("EXPANDED FUSION TESTS FAILED.")

print("STATUS: ALL EXPANDED TESTS PASSED.")