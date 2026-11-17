from pathlib import Path
from datetime import datetime
import subprocess
import sys
import shutil
import json
import traceback
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FUSION = ROOT / "fusion"
INTEGRATION = FUSION / "synthetic_integration"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = INTEGRATION / "run_backups" / STAMP
LOG_DIR = INTEGRATION / "end_to_end_logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Preserve existing outputs before regenerating them.
PROTECTED = [
    INTEGRATION / "EEG_Integration_Features.csv",
    INTEGRATION / "EIT_Integration_Features.csv",
    INTEGRATION / "Integration_Readiness_Report.txt",
    INTEGRATION / "Fusion_Pairing_Gate_Report.txt",
    INTEGRATION / "synthetic_benchmark",
    INTEGRATION / "synthetic_decision_level_comparison",
]

for source in PROTECTED:
    if source.is_file():
        destination = BACKUP / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    elif source.is_dir():
        destination = BACKUP / source.name
        shutil.copytree(source, destination, dirs_exist_ok=True)

stages = []

def run_stage(name, script, cwd=ROOT):
    path = ROOT / script

    if not path.is_file():
        raise FileNotFoundError(f"Missing script: {path}")

    print("\n" + "=" * 65)
    print("STAGE:", name)
    print("=" * 65)

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stages.append({
        "stage": name,
        "script": str(path),
        "return_code": result.returncode,
        "status": "PASSED" if result.returncode == 0 else "FAILED",
        "stdout": result.stdout,
        "stderr": result.stderr,
    })

    print(result.stdout)

    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{name} failed.")

def main():
    status = "RUNNING"

    print("NEUROSENTINEL END-TO-END INTEGRATION")
    print("Project:", ROOT)
    print("Backup:", BACKUP)

    try:
        # 1. Build separate modality integration inputs.
        run_stage(
            "EEG-EIT Integration Input Preparation",
            "fusion/synthetic_integration/build_integration_inputs.py",
        )

        # 2. Confirm real-data pairing remains safely blocked.
        run_stage(
            "Real-Data Pairing Gate",
            "fusion/synthetic_integration/fusion_pairing_gate.py",
        )

        gate_path = INTEGRATION / "Fusion_Pairing_Gate_Report.txt"
        gate_text = gate_path.read_text(encoding="utf-8")

        if "FUSION ELIGIBLE: False" not in gate_text:
            raise RuntimeError(
                "Pairing gate did not confirm the expected blocked status."
            )

        print("PAIRING GATE: SAFE BLOCK CONFIRMED")

        # 3. Run pairing and interface safety tests.
        run_stage(
            "Pairing Gate Tests",
            "fusion/synthetic_integration/test_fusion_pairing_gate.py",
        )

        run_stage(
            "Fusion Interface Safety Tests",
            "fusion/test_fusion_interface.py",
            cwd=FUSION,
        )

        # 4. Run synthetic feature-level fusion.
        run_stage(
            "Synthetic Feature-Level Fusion",
            "fusion/synthetic_integration/run_synthetic_fusion_benchmark.py",
        )

        # 5. Run synthetic decision-level comparison.
        run_stage(
            "Synthetic Decision-Level Fusion",
            "fusion/synthetic_integration/run_decision_level_fusion.py",
        )

        # Validate final modality inputs and synthetic outputs.
        eeg_path = INTEGRATION / "EEG_Integration_Features.csv"
        eit_path = INTEGRATION / "EIT_Integration_Features.csv"

        eeg = pd.read_csv(eeg_path)
        eit = pd.read_csv(eit_path)

        if not eeg["Pairing_Status"].eq("NOT_PAIRED").all():
            raise RuntimeError("EEG pairing status validation failed.")

        if not eit["Pairing_Status"].eq("NOT_PAIRED").all():
            raise RuntimeError("EIT pairing status validation failed.")

        benchmark = (
            INTEGRATION / "synthetic_benchmark"
            / "Synthetic_Fusion_Model_Metrics.csv"
        )

        comparison = (
            INTEGRATION / "synthetic_decision_level_comparison"
            / "Fusion_Architecture_Comparison_Metrics.csv"
        )

        if not benchmark.is_file() or not comparison.is_file():
            raise FileNotFoundError(
                "One or more synthetic fusion metrics files are missing."
            )

        benchmark_metrics = pd.read_csv(benchmark)
        comparison_metrics = pd.read_csv(comparison)

        # Consolidated evidence report.
        report = [
            "NEUROSENTINEL END-TO-END INTEGRATION REPORT",
            "=" * 60,
            f"Execution time: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "MODALITY INPUTS",
            f"EEG windows: {len(eeg)}",
            f"EEG participants: {eeg['Participant'].nunique()}",
            f"EIT scenarios: {len(eit)}",
            "EEG/EIT real-data pairing: NOT AVAILABLE",
            "",
            "PAIRING SAFETY",
            "Real-data pairing gate: BLOCKED as expected",
            "No arbitrary row or index pairing performed.",
            "",
            "SYNTHETIC FEATURE-LEVEL FUSION",
            benchmark_metrics.to_string(index=False),
            "",
            "SYNTHETIC ARCHITECTURE COMPARISON",
            comparison_metrics.to_string(index=False),
            "",
            "SCIENTIFIC LIMITATIONS",
            "All fusion benchmark observations and shared targets are synthetic.",
            "The actual EEG and EIT datasets remain separate and unpaired.",
            "EEG labels represent healthy motor-imagery tasks.",
            "EIT labels represent simulated conductivity changes.",
            "No verified patient EEG-EIT pairs or recurrence labels are available.",
            "No clinical tumour detection or recurrence prediction is established.",
            "",
            "FINAL STATUS",
            "SOFTWARE INTEGRATION: COMPLETED",
            "REAL-DATA CLINICAL FUSION: BLOCKED",
            "RECURRENCE PREDICTION: NOT ESTABLISHED",
        ]

        report_path = (
            INTEGRATION / "NeuroSentinel_End_to_End_Integration_Report.txt"
        )

        report_path.write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        status = "COMPLETED_WITH_CLINICAL_FUSION_BLOCKED"

        print("\n" + "=" * 65)
        print("END-TO-END SOFTWARE INTEGRATION COMPLETED")
        print("=" * 65)
        print("EEG windows:", len(eeg))
        print("EIT scenarios:", len(eit))
        print("Stages passed:", len(stages))
        print("Clinical fusion: BLOCKED")
        print("Report:", report_path)
        print("Backup:", BACKUP)

    except Exception as exc:
        status = "FAILED"
        print("\nPIPELINE FAILED:", exc)
        traceback.print_exc()

    finally:
        execution_log = {
            "project": "NeuroSentinel",
            "status": status,
            "started_or_logged_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "stages": stages,
            "clinical_fusion_ready": False,
            "clinical_recurrence_prediction_established": False,
            "backup_directory": str(BACKUP),
        }

        log_path = (
            LOG_DIR / f"NeuroSentinel_End_to_End_{STAMP}.json"
        )

        log_path.write_text(
            json.dumps(execution_log, indent=4),
            encoding="utf-8",
        )

        print("\nExecution log:", log_path)
        print("Final status:", status)

    if status != "COMPLETED_WITH_CLINICAL_FUSION_BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()