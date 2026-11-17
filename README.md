# NeuroSentinel-EIT

A software-only research prototype exploring EEG features, synthetic electrical impedance tomography (EIT), and research-level integration readiness.

## Dashboard

The Streamlit dashboard contains six sections: Overview, EEG Analysis, EIT Validation, Data Tables, Fusion Readiness, and EEG Window Comparison.

## Run locally

Use a compatible Python version. From the repository root, install the dashboard dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
streamlit run NeuroSentinel_dashboard_v2.py
```

## Repository contents

- `NeuroSentinel_dashboard_v2.py`: dashboard entrypoint.
- `data/`: derived EEG features, evaluation tables, and plots.
- `eit/`: synthetic EIT validation evidence and plots.
- `fusion/`: readiness reports and synthetic integration benchmark evidence.
- `src/`: EEG, EIT, and fusion research source code.
- `requirements.txt`: dashboard dependencies.
- `requirements-research.txt`: additional research-script dependencies.

## Data and limitations

EEG analysis uses publicly available healthy-participant motor-imagery data from the PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI). This repository contains derived outputs, not raw EEG recordings. Dataset: https://physionet.org/content/eegmmidb/1.0.0/

EIT results are synthetic EIDORS simulations, not patient measurements. EIDORS is an external dependency and is not bundled here. EIT workflows require a compatible GNU Octave/MATLAB environment and an independent EIDORS installation. https://eidors.org/

The fusion integration benchmark uses an artificial latent variable for a software-level architecture experiment. It does not demonstrate real EEG-EIT pairing or biological complementarity.

No patient data, verified bleeding or recurrence labels, longitudinal clinical records, or clinically paired EEG/EIT observations are included. Clinical validation has not been performed. This project does not demonstrate diagnosis, clinical screening, bleeding detection, tumour detection, or recurrence prediction.

Metrics apply only to the documented dataset and simulation conditions. Research scripts may require external datasets and tools; inspect their paths before running them and preserve existing evidence files.

## Deployment

The dashboard is designed for repository-relative paths. See the Streamlit Community Cloud deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
