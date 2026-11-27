from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(r"C:\NeuroSentinel")
OUT = ROOT / "eit" / "synthetic_experiment" / "ai_results"

source = OUT / "EIT_Severity_Predictions.csv"
data = pd.read_csv(source)

data["Clipped_Prediction"] = data["Predicted_Severity"].clip(0, 1)
data["Absolute_Error"] = (
    data["True_Severity"] - data["Predicted_Severity"]
).abs()
data["Clipped_Absolute_Error"] = (
    data["True_Severity"] - data["Clipped_Prediction"]
).abs()

rows = []
for noise, group in data.groupby("Noise_Percent"):
    rows.append({
        "Noise_Percent": noise,
        "Scenarios": len(group),
        "Raw_MAE": mean_absolute_error(
            group["True_Severity"], group["Predicted_Severity"]
        ),
        "Raw_RMSE": np.sqrt(mean_squared_error(
            group["True_Severity"], group["Predicted_Severity"]
        )),
        "Clipped_MAE": mean_absolute_error(
            group["True_Severity"], group["Clipped_Prediction"]
        ),
        "Clipped_RMSE": np.sqrt(mean_squared_error(
            group["True_Severity"], group["Clipped_Prediction"]
        ))
    })

report = pd.DataFrame(rows).round(6)
report.to_csv(OUT / "EIT_Noise_Wise_AI_Evaluation.csv", index=False)

data.to_csv(
    OUT / "EIT_Severity_Predictions_With_Clipping.csv",
    index=False
)

with open(OUT / "EIT_AI_Final_Evaluation_Report.txt", "w") as f:
    f.write("NEUROSENTINEL - SYNTHETIC EIT AI EVALUATION\n")
    f.write("=" * 55 + "\n")
    f.write("Model: StandardScaler + Ridge Regression\n")
    f.write("Training severity levels: 0.0, 0.5, 1.0\n")
    f.write("Held-out severity levels: 0.25, 0.75\n")
    f.write("Test scenarios: 62\n")
    f.write("Out-of-range predictions: 2\n\n")
    f.write(report.to_string(index=False))
    f.write("\n\nSCIENTIFIC LIMITATIONS\n")
    f.write("- Synthetic EIT measurements only.\n")
    f.write("- One underlying mesh and anomaly configuration.\n")
    f.write("- Repeated noise trials are not independent subjects.\n")
    f.write("- Severity represents simulated conductivity change.\n")
    f.write("- No tumour or recurrence labels are available.\n")
    f.write("- No clinical validation or recurrence prediction.\n")

print("EIT AI evaluation report created.")
print(report.to_string(index=False))
print("Output:", OUT)