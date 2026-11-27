
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(r"C:\NeuroSentinel")
PILOT = ROOT / "eit/results/unknown_size_localization"
AUDIT = ROOT / "eit/results/unknown_size_localization_threshold_audit_v2"
OUT = ROOT / "eit/results/unknown_size_localization_rethreshold_v2"

if OUT.exists():
    raise FileExistsError(f"Output exists; refusing to overwrite: {OUT}")

files = {
    "masks": PILOT / "Element_Level_Detection_Masks.csv",
    "metadata": PILOT / "EIT_Unknown_Size_Localization_Results.csv",
    "calibration": AUDIT / "Expanded_Null_Calibration.csv",
    "controls": AUDIT / "Expanded_Independent_Null_Controls.csv",
}

for name, path in files.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")

masks = pd.read_csv(files["masks"])
metadata = pd.read_csv(files["metadata"])
calibration = pd.read_csv(files["calibration"])
controls = pd.read_csv(files["controls"])

required = {
    "Scenario_ID", "Element_ID", "Centroid_X", "Centroid_Y",
    "Element_Area", "Truth_Mask", "Anomaly_Score",
    "Noise_Percent", "Repeat_ID"
}
if not required.issubset(masks.columns):
    raise ValueError("Element-level source columns are incomplete.")

thresholds = (
    calibration.groupby("Noise_Percent")["Detection_Threshold"]
    .first().to_dict()
)

# Verify the independent controls used the matching thresholds.
for noise, group in controls.groupby("Noise_Percent"):
    expected = float(thresholds[noise])
    if not np.allclose(group["Detection_Threshold"], expected):
        raise ValueError(f"Threshold mismatch at noise {noise}%.")

radii = [0.15, 0.22]
noise_values = [0, 1, 5, 10]

rows = []
mask_outputs = []

for scenario_id, group in masks.groupby("Scenario_ID", sort=True):

    match = re.fullmatch(
        r"P(\d+)_R(\d+)_N(\d+)_T(\d+)", scenario_id
    )
    if match is None:
        raise ValueError(f"Unexpected scenario ID: {scenario_id}")

    position_id, radius_id, noise_id, repeat_id = map(
        int, match.groups()
    )

    if radius_id not in (1, 2) or noise_id not in (1, 2, 3, 4):
        raise ValueError(f"Invalid scenario indices: {scenario_id}")

    radius = radii[radius_id - 1]
    noise = noise_values[noise_id - 1]

    if len(group) != 576:
        raise ValueError(f"{scenario_id}: expected 576 elements.")

    if group["Element_ID"].nunique() != 576:
        raise ValueError(f"{scenario_id}: duplicate element IDs.")

    if not np.all(group["Noise_Percent"].to_numpy() == noise):
        raise ValueError(f"{scenario_id}: noise label mismatch.")

    # Match the original geometry metadata.
    md = metadata[
        (metadata["Position_ID"] == position_id)
        & np.isclose(metadata["Nominal_Radius"], radius)
        & (metadata["Noise_Percent"] == noise)
        & (metadata["Repeat_ID"] == repeat_id)
    ]

    if len(md) != 1:
        raise ValueError(f"Could not uniquely match metadata: {scenario_id}")

    md = md.iloc[0]

    truth = group["Truth_Mask"].to_numpy(dtype=int).astype(bool)
    score = group["Anomaly_Score"].to_numpy(dtype=float)
    area = group["Element_Area"].to_numpy(dtype=float)
    coords = group[["Centroid_X", "Centroid_Y"]].to_numpy(dtype=float)

    if (not np.isfinite(score).all()
            or not np.isfinite(area).all()
            or not np.isfinite(coords).all()
            or (area <= 0).any()):
        raise ValueError(f"Invalid numeric values: {scenario_id}")

    peak = float(score.max())
    threshold = float(thresholds[noise])

    detected = peak > threshold + 1e-12

    # Segmentation is based on reconstruction only.
    predicted = (
        score >= 0.5 * peak
        if detected
        else np.zeros(len(group), dtype=bool)
    )

    intersection = predicted & truth
    union = predicted | truth

    tp = int(intersection.sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())

    dice_elements = 2 * tp / (int(predicted.sum()) + int(truth.sum()))
    iou_elements = tp / int(union.sum())

    intersection_area = float(area[intersection].sum())
    predicted_area = float(area[predicted].sum())
    truth_area = float(area[truth].sum())
    union_area = float(area[union].sum())

    dice_area = 2 * intersection_area / (
        predicted_area + truth_area
    )
    iou_area = intersection_area / union_area

    true_centroid = np.average(
        coords[truth], axis=0, weights=area[truth]
    )

    centroid_error = np.nan
    if predicted.any():
        predicted_centroid = np.average(
            coords[predicted], axis=0, weights=area[predicted]
        )
        centroid_error = float(
            np.linalg.norm(predicted_centroid - true_centroid)
        )

    rows.append({
        "Scenario_ID": scenario_id,
        "Geometry_ID": int(md["Geometry_ID"]),
        "Position_ID": position_id,
        "Center_X": float(md["Center_X"]),
        "Center_Y": float(md["Center_Y"]),
        "Nominal_Radius": radius,
        "Noise_Percent": noise,
        "Repeat_ID": repeat_id,
        "True_Elements": int(truth.sum()),
        "Peak_Score": peak,
        "Detection_Threshold": threshold,
        "Detected": int(detected),
        "Predicted_Elements": int(predicted.sum()),
        "Dice_Elements": dice_elements,
        "IoU_Elements": iou_elements,
        "Dice_Area": dice_area,
        "IoU_Area": iou_area,
        "Centroid_Error_Mesh_Units": centroid_error,
        "False_Positive_Elements": fp,
        "False_Negative_Elements": fn,
    })

    saved = group.copy()
    saved["Predicted_Mask_Rethresholded"] = predicted.astype(int)
    saved["Applied_Threshold"] = threshold
    saved["Detected_Rethresholded"] = int(detected)
    mask_outputs.append(saved)

result = pd.DataFrame(rows)

if len(result) != 310:
    raise ValueError(f"Expected 310 scenarios, got {len(result)}.")

# Wilson interval for proportions.
def wilson(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denominator
    margin = (
        z * np.sqrt(p*(1-p)/total + z*z/(4*total*total))
        / denominator
    )
    return max(0.0, centre-margin), min(1.0, centre+margin)

summary_rows = []

for noise, block in result.groupby("Noise_Percent", sort=True):
    total = len(block)
    detections = int(block["Detected"].sum())
    low, high = wilson(detections, total)

    null = controls[controls["Noise_Percent"] == noise]
    false_alarms = int(null["False_Alarm"].sum())
    fa_low, fa_high = wilson(false_alarms, len(null))

    centroid = block.loc[
        block["Detected"] == 1, "Centroid_Error_Mesh_Units"
    ].dropna()

    summary_rows.append({
        "Noise_Percent": int(noise),
        "Trials": total,
        "Detections": detections,
        "Detection_Rate_Percent": 100*detections/total,
        "Detection_CI95_Lower_Percent": 100*low,
        "Detection_CI95_Upper_Percent": 100*high,
        "Mean_Dice_Area_All_Trials": block["Dice_Area"].mean(),
        "Mean_IoU_Area_All_Trials": block["IoU_Area"].mean(),
        "Mean_Centroid_Error_Detected": (
            centroid.mean() if len(centroid) else np.nan
        ),
        "Mean_FP_Elements": block["False_Positive_Elements"].mean(),
        "Mean_FN_Elements": block["False_Negative_Elements"].mean(),
        "Null_Controls": len(null),
        "Null_False_Alarms": false_alarms,
        "Null_False_Alarm_Rate_Percent": (
            100*false_alarms/len(null)
        ),
        "False_Alarm_CI95_Lower_Percent": 100*fa_low,
        "False_Alarm_CI95_Upper_Percent": 100*fa_high,
        "Applied_Threshold": float(thresholds[noise]),
    })

summary = pd.DataFrame(summary_rows)

position_summary = (
    result.groupby(
        ["Noise_Percent", "Position_ID", "Nominal_Radius"],
        as_index=False
    )
    .agg(
        Trials=("Detected", "size"),
        Detections=("Detected", "sum"),
        Mean_Dice_Area=("Dice_Area", "mean"),
        Mean_IoU_Area=("IoU_Area", "mean"),
        Mean_Centroid_Error=("Centroid_Error_Mesh_Units", "mean"),
    )
)

OUT.mkdir(parents=True)

result.to_csv(OUT / "EIT_Rethresholded_Localization_Results.csv", index=False)
summary.to_csv(OUT / "EIT_Rethresholded_Localization_Summary.csv", index=False)
position_summary.to_csv(
    OUT / "EIT_Rethresholded_Position_Size_Summary.csv", index=False
)
pd.concat(mask_outputs, ignore_index=True).to_csv(
    OUT / "Rethresholded_Element_Detection_Masks.csv", index=False
)

(OUT / "Rethresholded_Evaluation_Metadata.txt").write_text(
    "NEUROSENTINEL EIT RE-THRESHOLD EVALUATION\n"
    "========================================\n"
    "Source reconstruction maps: original 310-trial pilot.\n"
    "Threshold source: independent expanded null calibration audit.\n"
    "No new reconstructions were performed.\n"
    "The detector does not receive the true anomaly mask or size.\n"
    "Segmentation rule: 50 percent of reconstruction peak after detection gate.\n"
    "Missed detections have zero Dice and IoU; centroid error is undefined.\n"
    "Centroid error is reported only for detected cases.\n"
    "Confidence intervals: 95 percent Wilson intervals for detection and false-alarm rates.\n"
    "All results are synthetic EIDORS simulation results, not clinical validation.\n",
    encoding="utf-8"
)

print("RETHRESHOLD EVALUATION COMPLETED")
print("Scenarios:", len(result))
print("Detected:", int(result["Detected"].sum()))
print("Output:", OUT)
print("\nSUMMARY")
print(summary.to_string(index=False))
