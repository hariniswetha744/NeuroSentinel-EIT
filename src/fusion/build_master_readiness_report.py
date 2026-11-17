
from pathlib import Path
import json
from datetime import datetime

ROOT = Path(r"C:\NeuroSentinel")
FUSION_DIR = ROOT / "fusion"

REPORTS = {
    "fusion_readiness": FUSION_DIR / "NeuroSentinel_Fusion_Readiness.json",
    "feature_audit": FUSION_DIR / "NeuroSentinel_Feature_Audit.json",
    "recurrence_readiness": FUSION_DIR / "NeuroSentinel_Recurrence_Readiness.json",
    "feature_preparation": (
        FUSION_DIR / "prepared_features"
        / "NeuroSentinel_Feature_Preparation_Report.json"
    ),
    "normalization": (
        FUSION_DIR / "normalized_features"
        / "NeuroSentinel_Normalization_Report.json"
    ),
    "architecture_audit": (
        FUSION_DIR / "NeuroSentinel_Fusion_Architecture_Audit.json"
    ),
}

OUTPUT_FILE = FUSION_DIR / "NeuroSentinel_Master_Readiness.json"


def load_report(name, path):
    if not path.is_file():
        return {
            "available": False,
            "error": "Report file not found",
        }

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "available": True,
            "data": data,
        }

    except (json.JSONDecodeError, OSError) as exc:
        return {
            "available": False,
            "error": str(exc),
        }


def main():
    reports = {
        name: load_report(name, path)
        for name, path in REPORTS.items()
    }

    feature_data = reports["feature_audit"].get("data", {})
    fusion_data = reports["fusion_readiness"].get("data", {})
    recurrence_data = reports["recurrence_readiness"].get("data", {})
    preparation_data = reports["feature_preparation"].get("data", {})
    normalization_data = reports["normalization"].get("data", {})
    architecture_data = reports["architecture_audit"].get("data", {})

    feature_passed = (
        feature_data.get("EEG_audit", {}).get("passed", False)
        and feature_data.get("EIT_audit", {}).get("passed", False)
    )

    preparation_passed = (
        preparation_data.get("status")
        == "FEATURE_PREPARATION_PASSED_PAIRING_NOT_READY"
    )

    normalization_passed = (
        normalization_data.get("status")
        == "NORMALIZATION_PASSED_PAIRING_NOT_READY"
    )

    architecture_ready = architecture_data.get(
        "fusion_architecture_data_ready", False
    )

    fusion_ready = (
        fusion_data.get("paired_sample_ids_verified", False)
        and fusion_data.get("clinical_fusion_training_ready", False)
        and architecture_ready
        and preparation_passed
        and normalization_passed
    )

    recurrence_ready = recurrence_data.get(
        "recurrence_model_evaluation_ready", False
    )

    master = {
        "project": "NeuroSentinel",
        "report_purpose": "Consolidated research readiness assessment",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "individual_reports": reports,

        "summary": {
            "feature_integrity_passed": bool(feature_passed),
            "feature_preparation_passed": bool(preparation_passed),
            "normalization_pipeline_passed": bool(normalization_passed),
            "fusion_architecture_data_ready": bool(architecture_ready),
            "paired_clinical_fusion_ready": bool(fusion_ready),
            "recurrence_evaluation_ready": bool(recurrence_ready),
            "clinical_tumour_detection_established": False,
            "clinical_recurrence_prediction_established": False,
        },

        "project_status": (
            "RESEARCH_COMPONENTS_AUDITED_CLINICAL_DATA_REQUIRED"
            if feature_passed
            else "RESEARCH_READINESS_REQUIRES_ATTENTION"
        ),

        "completed_work": [
            "EEG feature integrity audit",
            "EIT feature integrity audit",
            "Fusion safety checks",
            "Independent EEG/EIT feature preparation",
            "Separate EEG/EIT feature normalization",
            "Fusion architecture readiness audit",
            "Longitudinal recurrence-readiness audit",
        ],

        "outstanding_requirements": [
            "Verified and appropriately authorised paired EEG-EIT observations",
            "Reliable de-identified subject and session identifiers",
            "Verified tumour-status outcomes",
            "Longitudinal follow-up observations",
            "Independently verified recurrence outcomes",
            "Clinical and methodological validation",
        ],

        "research_limitations": [
            "Current EEG task labels represent motor imagery, not tumour status.",
            "Current EIT features represent a simulation, not a verified patient.",
            "The EEG and EIT datasets do not have verified shared sample IDs.",
            "EIT currently contains only one feature row.",
            "Single-row EIT normalization is not statistically meaningful for model training.",
            "No clinical tumour diagnosis has been established.",
            "No clinical recurrence prediction has been established.",
            "A passed data-integrity audit does not establish clinical validity.",
        ],
    }

    FUSION_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(master, indent=2),
        encoding="utf-8",
    )

    print("NEUROSENTINEL MASTER READINESS REPORT")
    print("-" * 45)
    print("Feature integrity passed:", feature_passed)
    print("Feature preparation passed:", preparation_passed)
    print("Normalization pipeline passed:", normalization_passed)
    print("Fusion architecture data ready:", architecture_ready)
    print("Clinical fusion ready:", fusion_ready)
    print("Recurrence evaluation ready:", recurrence_ready)

    print("Project status:", master["project_status"])

    available_count = sum(
        item["available"] for item in reports.values()
    )

    print(
        "Individual reports available:",
        available_count,
        "/",
        len(reports),
    )

    print("Master report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
