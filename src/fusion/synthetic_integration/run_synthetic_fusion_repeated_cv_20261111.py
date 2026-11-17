from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "fusion" / "synthetic_integration"
    / "synthetic_benchmark"
    / "Synthetic_Fusion_Benchmark_Data.csv"
)

OUT = (
    ROOT / "fusion" / "synthetic_integration"
    / "synthetic_repeated_cv_20261111"
)

# Never overwrite previous experiment outputs.
if OUT.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUT}"
    )

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

data = pd.read_csv(SOURCE)

if not data["Data_Status"].eq("SYNTHETIC_ONLY").all():
    raise ValueError("Non-synthetic records detected.")

if data.isna().any().any():
    raise ValueError("Missing values detected.")

eeg_cols = [
    f"Synthetic_EEG_Feature_{i}"
    for i in range(1, 5)
]

eit_cols = [
    f"Synthetic_EIT_Measurement_{i:02d}"
    for i in range(1, 9)
]

required = eeg_cols + eit_cols + ["Synthetic_State"]

if not set(required).issubset(data.columns):
    raise ValueError("Required feature or target columns missing.")

X_eeg = data[eeg_cols].to_numpy(dtype=float)
X_eit = data[eit_cols].to_numpy(dtype=float)
X_both = data[eeg_cols + eit_cols].to_numpy(dtype=float)
y = data["Synthetic_State"].to_numpy(dtype=int)

if not (
    np.isfinite(X_eeg).all()
    and np.isfinite(X_eit).all()
    and np.isfinite(X_both).all()
):
    raise ValueError("Non-finite feature values detected.")

if set(np.unique(y)) != {0, 1}:
    raise ValueError("Expected binary target labels 0 and 1.")

def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=20261111
        ))
    ])

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=20261111
)

records = []

for fold_number, (train_idx, test_idx) in enumerate(
    cv.split(X_both, y), start=1
):
    y_train = y[train_idx]
    y_test = y[test_idx]

    eeg_model = make_model()
    eit_model = make_model()
    feature_model = make_model()

    eeg_model.fit(X_eeg[train_idx], y_train)
    eit_model.fit(X_eit[train_idx], y_train)
    feature_model.fit(X_both[train_idx], y_train)

    eeg_prob = eeg_model.predict_proba(
        X_eeg[test_idx]
    )[:, 1]

    eit_prob = eit_model.predict_proba(
        X_eit[test_idx]
    )[:, 1]

    feature_prob = feature_model.predict_proba(
        X_both[test_idx]
    )[:, 1]

    decision_prob = (eeg_prob + eit_prob) / 2.0

    baseline = DummyClassifier(
        strategy="most_frequent"
    )
    baseline.fit(
        np.zeros((len(train_idx), 1)),
        y_train
    )

    outputs = {
        "Majority_baseline": (
            baseline.predict(
                np.zeros((len(test_idx), 1))
            ),
            None
        ),
        "EEG_only": (
            (eeg_prob >= 0.5).astype(int),
            eeg_prob
        ),
        "EIT_only": (
            (eit_prob >= 0.5).astype(int),
            eit_prob
        ),
        "Feature_level_fusion": (
            (feature_prob >= 0.5).astype(int),
            feature_prob
        ),
        "Decision_level_fusion": (
            (decision_prob >= 0.5).astype(int),
            decision_prob
        )
    }

    for model_name, (pred, prob) in outputs.items():
        records.append({
            "Fold": fold_number,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_test, pred
            ),
            "Macro_F1": f1_score(
                y_test, pred, average="macro"
            ),
            "ROC_AUC": (
                roc_auc_score(y_test, prob)
                if prob is not None else np.nan
            ),
            "Train_Samples": len(train_idx),
            "Test_Samples": len(test_idx)
        })

fold_results = pd.DataFrame(records)

summary = (
    fold_results.groupby("Model")
    .agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_SD=("Accuracy", "std"),
        Balanced_Accuracy_Mean=("Balanced_Accuracy", "mean"),
        Balanced_Accuracy_SD=("Balanced_Accuracy", "std"),
        Macro_F1_Mean=("Macro_F1", "mean"),
        Macro_F1_SD=("Macro_F1", "std"),
        ROC_AUC_Mean=("ROC_AUC", "mean"),
        ROC_AUC_SD=("ROC_AUC", "std"),
        Fold_Count=("Fold", "nunique")
    )
    .reset_index()
)

OUT.mkdir(parents=True)

fold_results.to_csv(
    OUT / "Repeated_CV_Fold_Metrics.csv",
    index=False
)

summary.to_csv(
    OUT / "Repeated_CV_Summary_Metrics.csv",
    index=False
)

report = [
    "NEUROSENTINEL SYNTHETIC FUSION REPEATED EVALUATION",
    "=" * 60,
    "Dataset: Synthetic_Fusion_Benchmark_Data.csv",
    f"Observations: {len(data)}",
    f"Class counts: {data['Synthetic_State'].value_counts().to_dict()}",
    "Evaluation: 5-fold stratified CV, repeated 5 times",
    "Total evaluation folds: 25",
    "Scaling: fitted separately within each training fold",
    "Decision fusion: arithmetic mean of EEG and EIT probabilities",
    "Decision threshold: 0.5",
    "Target: artificial shared latent state",
    "Data status: SYNTHETIC_ONLY",
    "",
    "MEAN AND STANDARD DEVIATION ACROSS FOLDS",
    summary.to_string(index=False),
    "",
    "LIMITATIONS",
    "- Repeated folds reuse the same synthetic observations.",
    "- Fold variability is not independent replication.",
    "- The shared target is generated from an artificial latent variable.",
    "- This is a software-level architecture comparison only.",
    "- No real EEG-EIT pairing or patient observations are used.",
    "- No clinical screening or recurrence prediction is demonstrated."
]

(OUT / "Repeated_CV_Report.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("REPEATED SYNTHETIC FUSION EVALUATION COMPLETED")
print("Output folder:", OUT)
print("\nSUMMARY:")
print(summary.round(4).to_string(index=False))
print("\nFolds evaluated:", fold_results["Fold"].nunique())
print("Models evaluated:", fold_results["Model"].nunique())
print("Existing benchmark files were not modified.")