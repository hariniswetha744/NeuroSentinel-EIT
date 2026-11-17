
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import json
import traceback


ROOT = Path(r"C:\NeuroSentinel")
FUSION_DIR = ROOT / "fusion"
LOG_DIR = FUSION_DIR / "pipeline_logs"

PIPELINE_STAGES = [
    ("Feature Integrity Audit", "fusion_feature_audit.py"),
    ("Fusion Readiness Audit", "check_fusion_readiness.py"),
    ("Recurrence Readiness Audit", "recurrence_readiness_audit.py"),
    ("Feature Preparation", "fusion_feature_preparation.py"),
    ("Feature Normalization", "fusion_feature_normalization.py"),
    ("Fusion Architecture Audit", "fusion_architecture_audit.py"),
]

TEST_SCRIPT = "test_fusion_interface.py"
MASTER_SCRIPT = "build_master_readiness_report.py"


def run_stage(name, script, log):
    script_path = FUSION_DIR / script

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required pipeline script not found: {script_path}"
        )

    print("\n" + "=" * 65)
    print("STAGE:", name)
    print("=" * 65)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(FUSION_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log.append({
        "stage": name,
        "script": script,
        "return_code": result.returncode,
        "status": "PASSED" if result.returncode == 0 else "FAILED",
        "stdout": result.stdout,
        "stderr": result.stderr,
    })

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("ERROR OUTPUT:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped: {name} failed "
            f"with return code {result.returncode}."
        )

    print("STAGE RESULT: PASSED")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"NeuroSentinel_Pipeline_{timestamp}.json"

    execution_log = {
        "project": "NeuroSentinel",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "stages": [],
        "pipeline_status": "RUNNING",
        "clinical_fusion_ready": False,
        "clinical_recurrence_prediction_established": False,
    }

    print("\nNEUROSENTINEL AUTOMATED FUSION PIPELINE")
    print("=" * 65)
    print("Python:", sys.executable)
    print("Project:", ROOT)
    print("Log:", log_file)

    try:
        for name, script in PIPELINE_STAGES:
            run_stage(name, script, execution_log["stages"])

        run_stage(
            "Fusion Safety Tests",
            TEST_SCRIPT,
            execution_log["stages"],
        )

        run_stage(
            "Master Readiness Report",
            MASTER_SCRIPT,
            execution_log["stages"],
        )

        master_file = FUSION_DIR / "NeuroSentinel_Master_Readiness.json"

        if not master_file.is_file():
            raise FileNotFoundError(
                "Master readiness report was not generated."
            )

        with master_file.open("r", encoding="utf-8") as file:
            master = json.load(file)

        summary = master.get("summary", {})

        execution_log["clinical_fusion_ready"] = bool(
            summary.get("paired_clinical_fusion_ready", False)
        )

        execution_log["clinical_recurrence_prediction_established"] = bool(
            summary.get(
                "clinical_recurrence_prediction_established",
                False,
            )
        )

        execution_log["master_report"] = str(master_file)
        execution_log["pipeline_status"] = "COMPLETED"

        print("\n" + "=" * 65)
        print("NEUROSENTINEL PIPELINE COMPLETED")
        print("=" * 65)

        print("Stages passed:", len(execution_log["stages"]))
        print("Clinical fusion ready:",
              execution_log["clinical_fusion_ready"])
        print("Clinical recurrence prediction established:",
              execution_log[
                  "clinical_recurrence_prediction_established"
              ])

        print("\nMaster report:", master_file)

    except Exception as exc:
        execution_log["pipeline_status"] = "FAILED"
        execution_log["error"] = str(exc)

        print("\nPIPELINE FAILED:", exc)
        traceback.print_exc()

    finally:
        execution_log["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )

        log_file.write_text(
            json.dumps(execution_log, indent=4),
            encoding="utf-8",
        )

        print("\nExecution log saved:", log_file)

    if execution_log["pipeline_status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
