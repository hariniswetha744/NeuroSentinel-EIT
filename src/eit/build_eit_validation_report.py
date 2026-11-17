from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\NeuroSentinel\eit\results")

METRICS = ROOT / "EIT_Reproducible_Validation_Metrics.csv"
TRIALS = ROOT / "EIT_Repeated_Noise_Trials.csv"
NOISE_SUMMARY = ROOT / "EIT_Recalculated_Noise_Summary_Corrected.csv"

OUTPUT = ROOT / "NeuroSentinel_EIT_Validation_Report.txt"


def main():
    metrics = pd.read_csv(METRICS)
    trials = pd.read_csv(TRIALS)
    noise = pd.read_csv(NOISE_SUMMARY)

    m = dict(zip(metrics["metric"], metrics["value"]))

    if len(trials) != 30:
        raise ValueError("Expected exactly 30 noise trials.")

    if trials.isna().any().any():
        raise ValueError("Missing values in noise trials.")

    if len(noise) != 3:
        raise ValueError("Expected three noise summary levels.")

    lines = [
        "NEUROSENTINEL - EIT VALIDATION REPORT",
        "=" * 55,
        "",
        "1. SIMULATION CONFIGURATION",
        f"Mesh nodes: {int(m['mesh_nodes'])}",
        f"Mesh elements: {int(m['mesh_elements'])}",
        f"Baseline measurements: {int(m['baseline_measurements'])}",
        f"Anomaly measurements: {int(m['anomaly_measurements'])}",
        f"True simulated anomaly elements: {int(m['true_anomaly_elements'])}",
        "",
        "2. MEASUREMENT CONSISTENCY",
        f"Consistency error: {m['measurement_consistency_error']:.12g}",
        "",
        "3. RECONSTRUCTION METRICS",
        f"RMSE: {m['RMSE']:.8f}",
        f"MAE: {m['MAE']:.8f}",
        f"Dice: {m['Dice']:.4f}",
        f"IoU: {m['IoU']:.4f}",
        f"False positives: {int(m['false_positives'])}",
        f"False negatives: {int(m['false_negatives'])}",
        "",
        "4. NOISE ROBUSTNESS",
    ]

    for _, row in noise.iterrows():
        lines.append(
            f"Noise: {row['noise_fraction']:.0%} | "
            f"Trials: {int(row['trials'])} | "
            f"Successes: {int(row['successes'])} | "
            f"Success rate: {row['success_rate_percent']:.1f}% | "
            f"Mean localization error: "
            f"{row['mean_localization_error']:.9f}"
        )

    lines.extend([
        "",
        "5. DATA INTEGRITY",
        f"Noise trial records: {len(trials)}",
        f"Missing trial values: {int(trials.isna().sum().sum())}",
        f"Noise levels: {trials['Noise_Level'].nunique()}",
        "Trial records: 10 per noise level",
        "",
        "6. LIMITATIONS",
        "These results are from simulated EIT data.",
        "They do not establish clinical brain-tumour detection.",
        "Perfect Dice and IoU apply only to this known-size",
        "simulated localization setup.",
        "The noise-generation source script was not located.",
        "No patient data or clinical validation was used.",
        "",
        "END OF REPORT",
    ])

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print("EIT VALIDATION REPORT CREATED")
    print("Noise trials:", len(trials))
    print("Noise levels:", trials["Noise_Level"].nunique())
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()