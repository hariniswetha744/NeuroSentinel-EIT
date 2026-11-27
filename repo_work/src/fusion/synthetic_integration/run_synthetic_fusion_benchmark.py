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
    classification_report,
)

SEED = 42
N = 1000

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fusion" / "synthetic_integration" / "synthetic_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# One explicitly synthetic latent state shared by both modalities.
latent = rng.normal(0, 1, N)
y = (latent > 0).astype(int)

# Four independently noisy EEG-like features.
eeg_loadings = np.array([1.0, 0.85, -0.75, 0.65])
eeg = (
    latent[:, None] * eeg_loadings[None, :]
    + rng.normal(0, 1.15, (N, 4))
)

# Eight independently noisy EIT-like voltage measurements.
eit_loadings = np.array([
    0.9, 0.8, 0.7, 1.0,
    -0.8, -0.7, 0.85, 0.75
])
eit = (
    latent[:, None] * eit_loadings[None, :]
    + rng.normal(0, 1.25, (N, 8))
)

eeg_cols = [
    "Synthetic_EEG_Feature_1",
    "Synthetic_EEG_Feature_2",
    "Synthetic_EEG_Feature_3",
    "Synthetic_EEG_Feature_4",
]
eit_cols = [
    f"Synthetic_EIT_Measurement_{i:02d}"
    for i in range(1, 9)
]

data = pd.DataFrame(
    np.column_stack([eeg, eit]),
    columns=eeg_cols + eit_cols,
)

data.insert(
    0, "Synthetic_ID",
    [f"SYN_{i:04d}" for i in range(1, N + 1)]
)
data["Synthetic_State"] = y
data["Data_Status"] = "SYNTHETIC_ONLY"

data_path = OUT / "Synthetic_Fusion_Benchmark_Data.csv"
data.to_csv(data_path, index=False)

# One stratified holdout shared by all model comparisons.
indices = np.arange(N)
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

feature_sets = {
    "EEG_only": eeg_cols,
    "EIT_only": eit_cols,
    "EEG_EIT_fusion": eeg_cols + eit_cols,
}

models = {
    "Majority_baseline": DummyClassifier(strategy="most_frequent")
}

for name in feature_sets:
    models[name] = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])

results = []
predictions = pd.DataFrame({
    "Synthetic_ID": data.iloc[test_idx]["Synthetic_ID"].values,
    "Actual_State": y[test_idx],
})

for name, model in models.items():
    if name == "Majority_baseline":
        X = np.zeros((N, 1))
    else:
        X = data[feature_sets[name]].to_numpy()

    model.fit(X[train_idx], y[train_idx])
    predicted = model.predict(X[test_idx])

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], probability)
    else:
        probability = np.zeros(len(test_idx))
        auc = np.nan

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y[test_idx], predicted),
        "Balanced_Accuracy": balanced_accuracy_score(
            y[test_idx], predicted
        ),
        "Macro_F1": f1_score(
            y[test_idx], predicted, average="macro"
        ),
        "ROC_AUC": auc,
        "Train_Samples": len(train_idx),
        "Test_Samples": len(test_idx),
    })

    predictions[f"{name}_Prediction"] = predicted
    predictions[f"{name}_Probability"] = probability

    pd.DataFrame(
        confusion_matrix(y[test_idx], predicted)
    ).to_csv(
        OUT / f"{name}_Confusion_Matrix.csv",
        index=False,
    )

metrics = pd.DataFrame(results)
metrics_path = OUT / "Synthetic_Fusion_Model_Metrics.csv"
metrics.to_csv(metrics_path, index=False)

predictions.to_csv(
    OUT / "Synthetic_Fusion_Test_Predictions.csv",
    index=False,
)

report = [
    "NEUROSENTINEL SYNTHETIC EEG-EIT FUSION BENCHMARK",
    "=" * 55,
    f"Random seed: {SEED}",
    f"Total synthetic observations: {N}",
    f"Training observations: {len(train_idx)}",
    f"Testing observations: {len(test_idx)}",
    f"EEG-like features: {len(eeg_cols)}",
    f"EIT-like measurements: {len(eit_cols)}",
    "Target: controlled synthetic latent state (0/1)",
    "Pairing: generated synthetic observations only",
    "Scaling: fitted on training data through Pipeline",
    "Split: stratified 80/20 holdout",
    "",
    "MODEL METRICS",
    metrics.to_string(index=False),
    "",
    "SCIENTIFIC LIMITATIONS",
    "- No real EEG and EIT records were paired.",
    "- No patient or clinical observations were generated.",
    "- The shared target is an artificial simulation label.",
    "- This benchmark tests software-level fusion only.",
    "- It does not demonstrate tumour detection or recurrence prediction.",
    "",
]

report_path = OUT / "Synthetic_Fusion_Benchmark_Report.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("SYNTHETIC FUSION BENCHMARK COMPLETED")
print("Dataset:", data_path)
print("Metrics:", metrics_path)
print("Report:", report_path)
print("\nMODEL COMPARISON")
print(metrics.round(4).to_string(index=False))
print("\nAll generated observations are marked SYNTHETIC_ONLY.")