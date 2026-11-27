
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"C:\NeuroSentinel")
DATA = ROOT / "data"

FEATURE_FILE = DATA / "NeuroSentinel_C3C4_MuBeta_3s_Features.csv"
OUTPUT_DIR = DATA / "C3C4_MuBeta_Statistical_Audit_20261109"

FEATURES = [
    "C3_Mu_Log10",
    "C4_Mu_Log10",
    "C3_Beta_Log10",
    "C4_Beta_Log10",
]

LABELS = ["T1", "T2"]
N_PERMUTATIONS = 1000
N_BOOTSTRAPS = 10000
SEED = 20261109


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )


def score_predictions(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced_Accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "Macro_F1": f1_score(
            actual,
            predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }


def loso_predictions(X, y, groups):
    splitter = LeaveOneGroupOut()

    model_predictions = np.empty(len(y), dtype=object)
    baseline_predictions = np.empty(len(y), dtype=object)

    for train_idx, test_idx in splitter.split(X, y, groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Participant leakage detected.")

        if len(np.unique(y[train_idx])) != 2:
            raise RuntimeError("Training fold lacks a task class.")

        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        model_predictions[test_idx] = model.predict(X[test_idx])

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X[train_idx], y[train_idx])
        baseline_predictions[test_idx] = baseline.predict(
            X[test_idx]
        )

    return model_predictions, baseline_predictions


def participant_metrics(df, predictions, model_name):
    rows = []

    for participant in sorted(df["Participant"].unique()):
        mask = df["Participant"].to_numpy() == participant

        scores = score_predictions(
            df["Task"].to_numpy()[mask],
            predictions[mask],
        )

        rows.append({
            "Participant": participant,
            "Model": model_name,
            "Windows": int(mask.sum()),
            **scores,
        })

    return rows


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    boot_means = np.empty(N_BOOTSTRAPS)

    for i in range(N_BOOTSTRAPS):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        boot_means[i] = sample.mean()

    return (
        float(values.mean()),
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )


def main():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    # Never overwrite an existing audit.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output folder already exists; nothing overwritten:\n"
            f"{OUTPUT_DIR}"
        )

    df = pd.read_csv(FEATURE_FILE)

    required = ["Participant", "Task"] + FEATURES
    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found.")

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite feature values found.")

    if set(df["Task"].unique()) != set(LABELS):
        raise ValueError("Unexpected task labels.")

    counts = df.groupby(["Participant", "Task"]).size().unstack(
        fill_value=0
    )

    if not (counts.sum(axis=1) > 0).all():
        raise ValueError("Empty participant group.")

    if not (counts > 0).all().all():
        raise ValueError("A participant is missing a task class.")

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["Task"].to_numpy(dtype=object)
    groups = df["Participant"].to_numpy()

    participants = sorted(df["Participant"].unique())
    rng = np.random.default_rng(SEED)

    print("DATASET:", df.shape)
    print("PARTICIPANTS:", len(participants))
    print("TASK COUNTS:")
    print(counts.to_string())

    # Observed participant-independent predictions.
    observed_model, observed_baseline = loso_predictions(
        X, y, groups
    )

    observed_scores = score_predictions(y, observed_model)
    baseline_scores = score_predictions(y, observed_baseline)

    print("\nOBSERVED LOSO MODEL:", observed_scores)
    print("BASELINE:", baseline_scores)

    participant_rows = []
    participant_rows.extend(
        participant_metrics(
            df, observed_model, "C3/C4 Mu-Beta Logistic Regression"
        )
    )
    participant_rows.extend(
        participant_metrics(
            df, observed_baseline, "Majority-class baseline"
        )
    )

    participant_df = pd.DataFrame(participant_rows)

    model_subject = participant_df[
        participant_df["Model"]
        == "C3/C4 Mu-Beta Logistic Regression"
    ].sort_values("Participant")

    baseline_subject = participant_df[
        participant_df["Model"]
        == "Majority-class baseline"
    ].sort_values("Participant")

    paired = model_subject[
        ["Participant", "Macro_F1", "Balanced_Accuracy"]
    ].merge(
        baseline_subject[
            ["Participant", "Macro_F1", "Balanced_Accuracy"]
        ],
        on="Participant",
        suffixes=("_Model", "_Baseline"),
    )

    paired["Macro_F1_Difference"] = (
        paired["Macro_F1_Model"]
        - paired["Macro_F1_Baseline"]
    )

    # Bootstrap across participants, not individual windows.
    summary_rows = []

    for metric in [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_F1",
    ]:
        values = model_subject[metric].to_numpy(dtype=float)
        mean, lower, upper = bootstrap_mean_ci(values, rng)

        summary_rows.append({
            "Model": "C3/C4 Mu-Beta Logistic Regression",
            "Metric": metric,
            "Mean_Participant_Score": mean,
            "Bootstrap_95CI_Lower": lower,
            "Bootstrap_95CI_Upper": upper,
            "Participants": len(values),
        })

    differences = paired["Macro_F1_Difference"].to_numpy()
    mean, lower, upper = bootstrap_mean_ci(differences, rng)

    summary_rows.append({
        "Model": "Paired difference: model minus baseline",
        "Metric": "Macro_F1",
        "Mean_Participant_Score": mean,
        "Bootstrap_95CI_Lower": lower,
        "Bootstrap_95CI_Upper": upper,
        "Participants": len(differences),
    })

    subject_summary = pd.DataFrame(summary_rows)

    # Within-participant label permutation.
    # This preserves each participant's T1/T2 counts.
    observed_f1 = observed_scores["Macro_F1"]
    null_f1 = np.empty(N_PERMUTATIONS, dtype=float)

    participant_indices = {
        participant: np.flatnonzero(groups == participant)
        for participant in participants
    }

    print(
        f"\nRUNNING {N_PERMUTATIONS} "
        "WITHIN-PARTICIPANT PERMUTATIONS"
    )

    for permutation in range(N_PERMUTATIONS):
        permuted_y = y.copy()

        for participant, indices in participant_indices.items():
            permuted_y[indices] = rng.permutation(y[indices])

        permuted_predictions, _ = loso_predictions(
            X, permuted_y, groups
        )

        null_f1[permutation] = f1_score(
            permuted_y,
            permuted_predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        if (permutation + 1) % 100 == 0:
            print(
                f"Completed {permutation + 1}/"
                f"{N_PERMUTATIONS}"
            )

    # One-sided test: whether observed macro-F1 exceeds the null.
    exceedances = int(np.sum(null_f1 >= observed_f1))
    p_value = (exceedances + 1) / (N_PERMUTATIONS + 1)

    permutation_summary = pd.DataFrame([{
        "Observed_Macro_F1": observed_f1,
        "Permutation_Count": N_PERMUTATIONS,
        "Null_Mean_Macro_F1": float(null_f1.mean()),
        "Null_SD_Macro_F1": float(null_f1.std(ddof=1)),
        "Null_95th_Percentile": float(
            np.quantile(null_f1, 0.95)
        ),
        "Null_99th_Percentile": float(
            np.quantile(null_f1, 0.99)
        ),
        "Exceedances": exceedances,
        "One_Sided_Permutation_P": p_value,
        "Random_Seed": SEED,
    }])

    permutation_distribution = pd.DataFrame({
        "Permutation": np.arange(1, N_PERMUTATIONS + 1),
        "Null_Macro_F1": null_f1,
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    participant_df.to_csv(
        OUTPUT_DIR / "Participant_Level_LOSO_Metrics.csv",
        index=False,
    )
    paired.to_csv(
        OUTPUT_DIR / "Participant_Paired_Model_Baseline.csv",
        index=False,
    )
    subject_summary.to_csv(
        OUTPUT_DIR / "Participant_Bootstrap_Summary.csv",
        index=False,
    )
    permutation_summary.to_csv(
        OUTPUT_DIR / "Permutation_Test_Summary.csv",
        index=False,
    )
    permutation_distribution.to_csv(
        OUTPUT_DIR / "Permutation_Null_Distribution.csv",
        index=False,
    )

    metadata = [
        "C3/C4 MU-BETA EEG STATISTICAL AUDIT",
        f"Feature file: {FEATURE_FILE}",
        f"Rows: {len(df)}",
        f"Participants: {len(participants)}",
        f"Features: {', '.join(FEATURES)}",
        "Evaluation: Leave-One-Participant-Out",
        "Classifier: StandardScaler + LogisticRegression",
        "Permutation: labels shuffled within each participant",
        "Participant task counts preserved in every permutation",
        f"Permutation count: {N_PERMUTATIONS}",
        f"Random seed: {SEED}",
        "Bootstrap unit: participant",
        "Bootstrap resamples: 10000",
        "Permutation statistic: pooled LOSO macro-F1",
        "P-value: one-sided, finite-sample correction",
        "",
        "LIMITATIONS:",
        "Only 10 participants and one recording run per participant.",
        "Window-level label shuffling may not preserve temporal dependence.",
        "Bootstrap intervals are uncertain with only 10 participants.",
        "This is healthy motor-imagery task classification, not clinical prediction.",
    ]

    (OUTPUT_DIR / "Audit_Methodology.txt").write_text(
        "\n".join(metadata),
        encoding="utf-8",
    )

    print("\nPERMUTATION TEST:")
    print(permutation_summary.to_string(index=False))

    print("\nPARTICIPANT BOOTSTRAP SUMMARY:")
    print(subject_summary.round(4).to_string(index=False))

    print("\nAUDIT COMPLETE.")
    print("OUTPUT FOLDER:", OUTPUT_DIR)
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name, "|", path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
