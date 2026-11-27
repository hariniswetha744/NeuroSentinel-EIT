# NeuroSentinel-EIT

A software-only research prototype exploring EEG features, synthetic electrical impedance tomography (EIT), and research-level integration readiness.

## Dashboard

The Streamlit dashboard contains seven sections: Overview, **Paper Results (IEEE v8)**, EEG Analysis, EIT Validation, Data Tables, Fusion Readiness, and EEG Window Comparison. The Paper Results tab shows the results reported in the manuscript; the other tabs show earlier archived project results.

## Paper: Silent mispairing in EEG–EIT fusion

The manuscript "Silent Mispairing in EEG–EIT Fusion: An In-Silico Evaluation of a Pre-Fusion Validity Gate with a Statistical Pairing-Dependence Test" is backed by:

- `ns_gate_suite/` – code for every experiment in the paper: pairing-validity gate (metadata predicates P1–P8 and pairing-dependence predicate P9 with CCA/HSIC), fault injection, learned matched-pair baseline, pyEIT GLRT detector, held-out EEG evaluation, and figure scripts. One entry point: `python run_all.py` (about 10 minutes); unit tests: `python -m pytest -q tests`.
- `paper_results/` – the saved outputs the paper's numbers are copied from (`results.json`, `results_for_paper.md`, CSV tables, figures) and `table3_eeg_heldout.json` (held-out EEG test on S011–S030).

Headline results (synthetic pairing study, N = 200 records per batch): with identifiers intact, mispairing every record lowered fusion accuracy from 0.872 to 0.669 while metadata checks blocked 0% of batches; the calibrated CCA predicate blocked 96.0% of fully mispaired batches at a 6.5% clean false-block rate. The frozen EEG model reached 0.537 (95% CI 0.490–0.583) on unseen participants, so the development result was not confirmed.

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
- `src/`: EEG, EIT, and fusion research source code (earlier project stages).
- `ns_gate_suite/`: code for the IEEE paper experiments.
- `paper_results/`: saved outputs and figures for the IEEE paper.
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

## Update (Nov 2026): extensions and real-data check
- `ns_gate_suite/run_extensions.py` → E1 record-level flagging, E2 sequential testing, E3 combined CCA + learned rule (outputs in `paper_results/extensions/`).
- `ns_gate_suite/run_eeg_nirs.py` → Experiment 7, the gate on real simultaneous EEG + NIRS (TU Berlin hybrid BCI dataset A, 29 subjects); see `ns_gate_suite/RUN_EXP7.md`. Copy `outputs/exp7_eeg_nirs.json` into `paper_results/`.
- Dashboard tab "Extensions & Real-Data Check" shows all of these, read from the files.

## Update (v14): power rule and repair
- `ns_gate_suite/ns_gate/theory.py`, `run_theory.py` → power and sample-size rule, Eq. (7)–(8); validation in `outputs/novelty/theory_sim.json`.
- `run_theory_real.py` → pre-declared test of the rule on real EEG–NIRS (`paper_results/exp8_theory_real.json`).
- `ns_gate_suite/ns_gate/repair.py`, `run_repair.py` → repair of shift faults (`outputs/novelty/repair.json`).
