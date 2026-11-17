
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(r"C:\NeuroSentinel")
DATA = ROOT / "eit" / "synthetic_experiment" / "scenarios"
OUT = ROOT / "eit" / "synthetic_experiment" / "ai_results"
OUT.mkdir(parents=True, exist_ok=True)

measurements = pd.read_csv(
    DATA / "EIT_Synthetic_Scenario_Measurements.csv"
)
summary = pd.read_csv(
    DATA / "EIT_Synthetic_Scenario_Summary.csv"
)

# Convert measurement records into one feature row per scenario.
X = measurements.pivot(
    index="Scenario_ID",
    columns="Measurement_ID",
    values="Voltage"
).sort_index(axis=1)

y = summary.set_index("Scenario_ID")["Severity"]
meta = summary.set_index("Scenario_ID")

X = X.loc[y.index]

# Predefined severity interpolation split.
train_mask = y.isin([0.0, 0.5, 1.0])
test_mask = y.isin([0.25, 0.75])

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

predictions = pd.DataFrame({
    "Scenario_ID": y_test.index,
    "True_Severity": y_test.values,
    "Predicted_Severity": pred,
    "Noise_Percent": meta.loc[y_test.index, "Noise_Percent"].values
})

predictions.to_csv(
    OUT / "EIT_Severity_Predictions.csv", index=False
)

with open(OUT / "EIT_Severity_Model_Metrics.txt", "w") as f:
    f.write("NeuroSentinel synthetic EIT severity regression\n")
    f.write("Training severities: 0.0, 0.5, 1.0\n")
    f.write("Testing severities: 0.25, 0.75\n")
    f.write(f"Training scenarios: {len(X_train)}\n")
    f.write(f"Testing scenarios: {len(X_test)}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"R2: {r2:.6f}\n")
    f.write("Synthetic experiments only; not clinical validation.\n")

print("EIT severity model completed.")
print("Training scenarios:", len(X_train))
print("Testing scenarios:", len(X_test))
print("MAE:", round(mae, 6))
print("RMSE:", round(rmse, 6))
print("R2:", round(r2, 6))
print("Output:", OUT)
