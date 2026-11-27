# NeuroSentinel validity-gate experiment suite (in-silico)

Regenerates every figure and table of the rewritten IEEE paper.

    pip install -r requirements.txt
    python run_all.py                     # Figures A, B, C; Tables I, II, IV (~10 min, 2 CPUs)
    python run_all.py --eeg-root "C:\Users\harin\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0" --eeg-download
                                          # also Table III (needs PhysioNet access)
    python run_all.py --quick             # smoke test, NOT for the paper
    python -m pytest -q tests

Outputs (./outputs): figure PNG/PDF, one CSV per experiment, results.json, and
results_for_paper.md (paste-ready tables). Anything not run is written as NOT_RUN.

| Paper item | Code |
|---|---|
| Fig. 1 gate diagram | ns_gate/fig1.py |
| Figure A (Fig. 2) corruption sweep | experiments.figure_a |
| Figure B (Fig. 3) detection power vs q, N, r, statistic | experiments.figure_b |
| Tables I, II gate vs baselines, held-out faults | experiments.tables_i_ii, faults.py |
| Figure C (Fig. 4) GLRT + conformal EIT detector | eit_glrt.py (pyEIT, fine->coarse mesh) |
| Table III EEG held-out participants | eeg_heldout.py (same features as build_c3c4_mu_beta_eeg.py) |
| Table IV runtime / memory | experiments.table_iv |
| Reference-construction check | experiments.reference_check |
| Learned matched-pair baseline (Tables I–II, "Learn. cal.") | experiments.learned_baseline |
| Print-size Figs. 4–6 (from saved results) | plots.figure_a_print / figure_b_print / figure_c_print |

Scope: synthetic generator + simulated EIT + public EEG. No clinical claim.

## Explanatory figures (for the paper and project report)

    python -m ns_gate.fig_acq        # Fig. 2: EEG montage, simulated EIT, GLRT map, dashboard screenshot
    python -m ns_gate.fig_explain    # figure_concept.png, figure_eit_steps.png, figure_mispairing.png (Fig. 3)
    python -m ns_gate.fig_eeg_trace --edf "...\S001\S001R04.edf"   # real C3/C4 trace + spectrum (your machine)
