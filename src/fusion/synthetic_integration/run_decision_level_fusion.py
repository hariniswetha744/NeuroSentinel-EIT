from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT / "fusion" / "synthetic_integration"
    / "synthetic_benchmark"
)
OUT = (
    ROOT / "fusion" / "synthetic_integration"
    / "synthetic_decision_level_comparison"
)
OUT.mkdir(parents=True, exist_ok=True)

# Read the already validated synthetic dataset.
data = pd.read_csv(SOURCE / "Synthetic_Fusion_Benchmark_Data.csv")

if not data["Data_Status"].eq("SYNTHETIC_ONLY").all():
    raise ValueError("Dataset contains records not marked SYNTHETIC_ONLY.")

if data.isna().any().any():
    raise ValueError("Missing values found. Experiment stopped.")

eeg_cols = [
    f"Synthetic_EEG_Feature_{i}" for i in range(1, 5)
]
eit_cols = [
    f"Synthetic_EIT_Measurement_{i:02d}" for i in range(1, 9)
]

y = data["Synthetic_State"].to_numpy()
indices = np.arange(len(data))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])

# Separate modality models.
eeg_model = make_model()
eit_model = make_model()

# Feature-level model for a same-split comparison.
feature_model = make_model()

eeg_model.fit(data.loc[train_idx, eeg_cols], y[train_idx])
eit_model.fit(data.loc[train_idx, eit_cols], y[train_idx])

feature_model.fit(
    data.loc[train_idx, eeg_cols + eit_cols],
    y[train_idx],
)

# Decision-level fusion: average independent model probabilities.
eeg_probability = eeg_model.predict_proba(
    data.loc[test_idx, eeg_cols]
)[:, 1]

eit_probability = eit_model.predict_proba(
    data.loc[test_idx, eit_cols]
)[:, 1]

decision_probability = (
    eeg_probability + eit_probability
) / 2.0

# Predictions from all approaches.
eeg_pred = eeg_model.predict(data.loc[test_idx, eeg_cols])
eit_pred = eit_model.predict(data.loc[test_idx, eit_cols])

feature_probability = feature_model.predict_proba(
    data.loc[test_idx, eeg_cols + eit_cols]
)[:, 1]

feature_pred = feature_model.predict(
    data.loc[test_idx, eeg_cols + eit_cols]
)

decision_pred = (decision_probability >= 0.5).astype(int)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(np.zeros((len(train_idx), 1)), y[train_idx])
baseline_pred = baseline.predict(
    np.zeros((len(test_idx), 1))
)
baseline_probability = np.full(len(test_idx), 0.5)

model_outputs = {
    "Majority_baseline": (baseline_pred, baseline_probability),
    "EEG_only": (eeg_pred, eeg_probability),
    "EIT_only": (eit_pred, eit_probability),
    "EEG_EIT_feature_level": (
        feature_pred, feature_probability
    ),
    "EEG_EIT_decision_level": (
        decision_pred, decision_probability
    ),
}

results = []
predictions = pd.DataFrame({
    "Synthetic_ID": data.iloc[test_idx]["Synthetic_ID"].values,
    "Actual_State": y[test_idx],
})

for name, (predicted, probability) in model_outputs.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y[test_idx], predicted),
        "Balanced_Accuracy": balanced_accuracy_score(
            y[test_idx], predicted
        ),
        "Macro_F1": f1_score(
            y[test_idx], predicted, average="macro"
        ),
        "ROC_AUC": roc_auc_score(
            y[test_idx], probability
        ),
        "Train_Samples": len(train_idx),
        "Test_Samples": len(test_idx),
    })

    predictions[f"{name}_Prediction"] = predicted
    predictions[f"{name}_Probability"] = probability

    pd.DataFrame(
        confusion_matrix(y[test_idx], predicted, labels=[0, 1]),
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(OUT / f"{name}_Confusion_Matrix.csv")

metrics = pd.DataFrame(results)
metrics.to_csv(OUT / "Fusion_Architecture_Comparison_Metrics.csv",
               index=False)
predictions.to_csv(
    OUT / "Fusion_Architecture_Comparison_Predictions.csv",
    index=False,
)

report = [
    "NEUROSENTINEL SYNTHETIC FUSION ARCHITECTURE COMPARISON",
    "=" * 58,
    f"Random seed: {SEED}",
    f"Total observations: {len(data)}",
    f"Training observations: {len(train_idx)}",
    f"Testing observations: {len(test_idx)}",
    "Split: stratified 80/20 holdout",
    "Feature-level fusion: concatenated EEG and EIT features",
    "Decision-level fusion: average of EEG and EIT probabilities",
    "Decision threshold: 0.5",
    "Scaling: fitted on training data through Pipeline",
    "Pairing: synthetic observations only",
    "",
    "MODEL METRICS",
    metrics.to_string(index=False),
    "",
    "SCIENTIFIC LIMITATIONS",
    "- All observations and shared targets are synthetic.",
    "- No real EEG-EIT patient records were paired.",
    "- This evaluates software-level fusion architectures only.",
    "- It does not establish tumour detection or recurrence prediction.",
    "",
]

(OUT / "Fusion_Architecture_Comparison_Report.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

print("SYNTHETIC FUSION ARCHITECTURE COMPARISON COMPLETED")
print("Output folder:", OUT)
print("\nMODEL COMPARISON")
print(metrics.round(4).to_string(index=False))
print("\nAll results are SYNTHETIC_ONLY.")