# NeuroSentinel validity-gate outputs (auto-generated; in-silico)

## Figure A — corruption sweep (r=0.5, N=200 per batch, 500 batches per q)

| q | Fusion no gate | Gate calib-CCA | Gate perm-CCA | EEG-only | Block calib | Block perm |
|---|---|---|---|---|---|---|
| 0.0 | 0.872 ± 0.002 | 0.868 | 0.866 | 0.769 | 4.0% | 5.4% |
| 0.1 | 0.851 ± 0.002 | 0.836 | 0.832 | 0.769 | 15.6% | 19.2% |
| 0.2 | 0.830 ± 0.002 | 0.809 | 0.807 | 0.768 | 32.0% | 34.6% |
| 0.3 | 0.811 ± 0.002 | 0.792 | 0.790 | 0.768 | 43.2% | 47.6% |
| 0.4 | 0.792 ± 0.002 | 0.778 | 0.777 | 0.770 | 61.8% | 65.4% |
| 0.5 | 0.773 ± 0.002 | 0.768 | 0.768 | 0.768 | 73.6% | 76.4% |
| 0.6 | 0.751 ± 0.003 | 0.768 | 0.769 | 0.770 | 83.0% | 85.0% |
| 0.7 | 0.733 ± 0.003 | 0.766 | 0.766 | 0.769 | 89.4% | 90.6% |
| 0.8 | 0.709 ± 0.003 | 0.763 | 0.764 | 0.768 | 92.0% | 93.4% |
| 0.9 | 0.689 ± 0.003 | 0.764 | 0.765 | 0.768 | 94.8% | 95.4% |
| 1.0 | 0.669 ± 0.003 | 0.764 | 0.765 | 0.769 | 95.4% | 96.0% |

## Figure B — detection power (block rate), 400 batches/cell

### Panel N: calib-CCA (perm-CCA in brackets)

| N | q=0 | q=0.05 | q=0.1 | q=0.2 | q=0.3 | q=0.5 | q=0.75 | q=1 |
|---|---|---|---|---|---|---|---|---|
| 50 | 4.0 (77.8) | 5.2 (84.0) | 6.2 (82.5) | 6.8 (87.2) | 13.0 (88.8) | 13.5 (90.5) | 17.8 (95.2) | 17.2 (97.0) |
| 100 | 4.5 (50.0) | 5.5 (54.2) | 5.8 (60.2) | 15.5 (70.2) | 14.5 (75.5) | 33.5 (88.8) | 44.8 (93.8) | 47.8 (95.0) |
| 200 | 4.5 (6.5) | 8.8 (13.0) | 12.5 (20.0) | 25.8 (36.0) | 45.5 (54.8) | 68.5 (76.2) | 88.8 (91.8) | 94.5 (97.2) |
| 500 | 2.8 (0.0) | 12.0 (0.0) | 25.8 (0.0) | 60.0 (0.5) | 84.5 (5.0) | 99.2 (33.2) | 100.0 (80.0) | 100.0 (95.8) |
| 1000 | 8.0 (0.0) | 26.5 (0.0) | 48.5 (0.0) | 89.2 (0.0) | 99.8 (0.0) | 100.0 (4.5) | 100.0 (68.8) | 100.0 (96.5) |

### Panel r: calib-CCA (perm-CCA in brackets)

| r | q=0 | q=0.05 | q=0.1 | q=0.2 | q=0.3 | q=0.5 | q=0.75 | q=1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 5.2 (94.2) | 7.0 (94.2) | 6.5 (97.0) | 4.5 (96.8) | 4.2 (94.8) | 6.5 (96.8) | 8.5 (97.0) | 4.5 (94.8) |
| 0.25 | 4.2 (77.5) | 6.2 (78.2) | 8.2 (82.0) | 10.5 (85.8) | 11.2 (87.8) | 15.0 (90.5) | 20.2 (95.5) | 20.5 (94.5) |
| 0.5 | 5.0 (7.0) | 9.5 (12.2) | 15.2 (20.0) | 26.2 (34.0) | 44.2 (50.5) | 72.0 (76.8) | 89.0 (90.2) | 93.2 (94.8) |
| 0.75 | 6.0 (0.0) | 12.2 (0.0) | 23.0 (0.0) | 64.0 (0.5) | 86.0 (5.5) | 98.8 (43.8) | 100.0 (88.8) | 100.0 (95.2) |
| 1 | 4.5 (0.0) | 29.0 (0.0) | 63.0 (0.0) | 96.5 (0.0) | 100.0 (0.0) | 100.0 (9.8) | 100.0 (74.2) | 100.0 (92.8) |

### Panel stat (200 batches/cell): calib-CCA / calib-HSIC / perm-CCA / perm-HSIC

| gen | q=0 | q=0.05 | q=0.1 | q=0.2 | q=0.3 | q=0.5 | q=0.75 | q=1 |
|---|---|---|---|---|---|---|---|---|
| A | 5.0/2.0/9.0/0.0 | 7.5/6.5/12.5/1.0 | 12.0/10.0/15.0/4.0 | 30.0/24.5/36.0/12.0 | 45.5/42.5/49.5/23.0 | 78.0/82.0/80.5/59.0 | 92.0/99.0/92.5/93.0 | 95.5/99.5/95.5/96.5 |
| B | 3.0/3.5/12.0/0.0 | 9.0/11.0/21.5/0.5 | 11.0/10.0/24.0/1.0 | 21.5/22.0/40.5/2.5 | 38.5/37.0/58.0/12.5 | 64.0/73.5/85.0/37.5 | 82.0/96.0/90.0/77.5 | 87.0/99.5/96.5/94.5 |

## Tables I & II — block rate per fault (%, Wilson 95% CI), N=200, 300 trials/fault, 1000 clean trials

| Set | Fault | Sev. | Fusion acc. (ungated) | ID-only (P1-P4) | Metadata (P1-P8) | Full: +P9 calib-CCA | Full: +P9 perm-CCA | Full: +P9 calib-HSIC |
|---|---|---|---|---|---|---|---|---|
| clean | Clean paired data (false-block rate) | 0 | 0.874 | 0.0 [0.0–0.4] | 0.0 [0.0–0.4] | 6.5 [5.1–8.2] | 8.9 [7.3–10.8] | 4.1 [3.0–5.5] |
| rule-written | F1 Missing/empty identifier (1 record) | 0.005 | 0.873 | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] |
| rule-written | F2 Duplicate identifier (1 record) | 0.005 | 0.873 | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] |
| rule-written | F3 Identifier mismatch (1 record) | 0.005 | 0.874 | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] |
| rule-written | F4 Rate/channel/unit error (1 record) | 0.005 | 0.873 | 0.0 [0.0–1.3] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] |
| rule-written | F5 Timestamp offset > tolerance (1 record) | 0.005 | 0.874 | 0.0 [0.0–1.3] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] |
| rule-written | F6 Missing consent/source (1 record) | 0.005 | 0.873 | 0.0 [0.0–1.3] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] | 100.0 [98.7–100.0] |
| rule-written | F7 Silent ID-consistent mispairing | 0.1 | 0.856 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 12.7 [9.4–16.9] | 16.3 [12.6–20.9] | 11.7 [8.5–15.8] |
| rule-written | F7 Silent ID-consistent mispairing | 0.25 | 0.823 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 35.3 [30.1–40.9] | 40.3 [34.9–46.0] | 33.7 [28.6–39.2] |
| rule-written | F7 Silent ID-consistent mispairing | 0.5 | 0.772 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 70.7 [65.3–75.5] | 76.7 [71.6–81.1] | 75.0 [69.8–79.6] |
| rule-written | F7 Silent ID-consistent mispairing | 1 | 0.669 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 96.0 [93.1–97.7] | 96.3 [93.6–97.9] | 99.7 [98.1–99.9] |
| held-out | H1 Dropped EIT frame (off-by-one tail) | 0.5 | 0.772 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 70.3 [64.9–75.2] | 75.7 [70.5–80.2] | 76.7 [71.6–81.1] |
| held-out | H3 Stale buffer (repeated EIT payload) | 0.5 | 0.768 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 55.3 [49.7–60.9] | 64.7 [59.1–69.9] | 93.0 [89.5–95.4] |
| held-out | H4 Electrode/column permutation (out of scope) | 1 | 0.731 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 7.0 [4.6–10.5] | 8.0 [5.4–11.6] | 5.0 [3.1–8.1] |
| held-out | H5 Cross-cohort substitution | 0.5 | 0.772 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 73.0 [67.7–77.7] | 76.0 [70.9–80.5] | 77.7 [72.6–82.0] |
| held-out | H6 Class-conditional swap | 0.5 | 0.871 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 4.0 [2.3–6.9] | 5.3 [3.3–8.5] | 2.3 [1.1–4.7] |
| held-out-sweep | H1 Dropped EIT frame (off-by-one tail) | 0.1 | 0.853 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 15.3 [11.7–19.8] | 21.0 [16.8–26.0] | 11.0 [7.9–15.0] |
| held-out-sweep | H6 Class-conditional swap | 0.1 | 0.875 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 5.3 [3.3–8.5] | 6.7 [4.4–10.1] | 3.7 [2.1–6.4] |
| held-out-sweep | H1 Dropped EIT frame (off-by-one tail) | 0.25 | 0.822 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 36.3 [31.1–41.9] | 41.7 [36.2–47.3] | 33.3 [28.2–38.8] |
| held-out-sweep | H6 Class-conditional swap | 0.25 | 0.874 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 6.7 [4.4–10.1] | 8.0 [5.4–11.6] | 4.7 [2.8–7.7] |
| held-out-sweep | H1 Dropped EIT frame (off-by-one tail) | 1 | 0.668 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 92.7 [89.1–95.1] | 96.0 [93.1–97.7] | 99.0 [97.1–99.7] |
| held-out-sweep | H6 Class-conditional swap | 1 | 0.866 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 9.7 [6.8–13.5] | 13.0 [9.7–17.3] | 2.0 [0.9–4.3] |
| held-out | Clean subject-structured data (false-block rate) | 0 | 0.916 | 0.0 [0.0–0.4] | 0.0 [0.0–0.4] | 4.7 [3.6–6.2] | 0.0 [0.0–0.4] | 5.5 [4.2–7.1] |
| held-out | H2 Within-subject swap | 0.5 | 0.853 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 41.3 [35.9–47.0] | 4.0 [2.3–6.9] | 33.0 [27.9–38.5] |
| held-out | H2 Within-subject swap | 1 | 0.793 | 0.0 [0.0–1.3] | 0.0 [0.0–1.3] | 75.3 [70.2–79.9] | 19.3 [15.3–24.2] | 58.3 [52.7–63.8] |

## Figure C — EIT GLRT vs peak (pyEIT; fine {'nodes': 1476, 'elements': 2821} -> coarse {'nodes': 312, 'elements': 560}, 500 anomalies, 200 calibration nulls, 1000 test nulls, alpha=0.05, peak lambda_rel=10.0)

| SNR (dB) | GLRT AUC | Peak AUC | GLRT det. | Peak det. | GLRT FAR (one cal.) | Peak FAR (one cal.) | GLRT FAR marginal | Peak FAR marginal | GLRT loc. err | Peak loc. err |
|---|---|---|---|---|---|---|---|---|---|---|
| -10 | 0.881 | 0.839 | 69.0% | 56.2% | 3.1% | 4.0% | 4.4% | 4.6% | 0.074 | 0.071 |
| -5 | 0.957 | 0.939 | 87.6% | 82.8% | 4.7% | 5.0% | 4.7% | 5.5% | 0.063 | 0.061 |
| 0 | 0.987 | 0.982 | 97.0% | 94.4% | 4.5% | 3.2% | 5.5% | 4.3% | 0.051 | 0.047 |
| 5 | 1.000 | 0.999 | 100.0% | 99.6% | 5.0% | 2.6% | 6.2% | 3.4% | 0.043 | 0.043 |
| 10 | 1.000 | 1.000 | 100.0% | 100.0% | 3.4% | 3.3% | 4.9% | 4.6% | 0.041 | 0.038 |
| 15 | 1.000 | 1.000 | 100.0% | 100.0% | 4.9% | 5.0% | 5.5% | 5.5% | 0.040 | 0.038 |
| 20 | 1.000 | 1.000 | 100.0% | 100.0% | 5.8% | 6.9% | 5.4% | 5.3% | 0.038 | 0.036 |

Noise-percentage mapping (this pyEIT model only):

- 1% of std(v_h) -> SNR_dv 5.8 dB (reference anomaly), 9.9 dB (population median)
- 5% of std(v_h) -> SNR_dv -8.2 dB (reference anomaly), -4.1 dB (population median)
- 10% of std(v_h) -> SNR_dv -14.2 dB (reference anomaly), -10.1 dB (population median)

## Table III — EEG held-out participants

NOT_RUN. python run_all.py --eeg-root <EEGBCI 1.0.0 folder> --eeg-download


## Learned matched-pair baseline (separate run, seed 7; record AUC at q=0.5: 0.60)

| Set | Fault | Sev. | Block % | 95% CI |
|---|---|---|---|---|
| clean | Clean paired data (false-block rate) | 0 | 4.2 | 3.1–5.6 |
| rule-written | F7 Silent ID-consistent mispairing | 0.1 | 11.3 | 8.2–15.4 |
| rule-written | F7 Silent ID-consistent mispairing | 0.25 | 47.7 | 42.1–53.3 |
| rule-written | F7 Silent ID-consistent mispairing | 0.5 | 81.0 | 76.2–85.0 |
| rule-written | F7 Silent ID-consistent mispairing | 1 | 100.0 | 98.7–100.0 |
| held-out | H1 Dropped EIT frame (off-by-one tail) | 0.5 | 89.7 | 85.7–92.6 |
| held-out | H3 Stale buffer (repeated EIT payload) | 0.5 | 87.0 | 82.7–90.3 |
| held-out | H4 Electrode/column permutation (out of scope) | 1 | 91.3 | 87.6–94.0 |
| held-out | H5 Cross-cohort substitution | 0.5 | 80.7 | 75.8–84.7 |
| held-out | H6 Class-conditional swap | 0.5 | 2.3 | 1.1–4.7 |
| held-out | Clean subject-structured data (false-block rate) | 0 | 4.1 | 3.0–5.5 |
| held-out | H2 Within-subject swap | 0.5 | 17.7 | 13.8–22.4 |
| held-out | H2 Within-subject swap | 1 | 29.7 | 24.8–35.1 |

## Reference-construction check (calib-CCA clean false-block, nominal 5%)

| N | Reference | False-block | 95% CI |
|---|---|---|---|
| 200 | independent batches | 4.3% | 3.5–5.3% |
| 200 | subsampled 20000-record pool | 5.1% | 4.3–6.2% |
| 1000 | independent batches | 4.0% | 3.2–4.9% |
| 1000 | subsampled 20000-record pool | 5.1% | 4.2–6.2% |

## Table IV — overhead ({'python': '3.13.18', 'platform': 'Linux-6.18.44-fc-v880-x86_64-with-glibc2.39', 'processor': 'x86_64', 'numpy': '2.5.3', 'cpu_count': 2})

| Component | Batch N | Median ms | µs/record | Peak MB |
|---|---|---|---|---|
| Metadata predicates P1-P8 | 200 | 5.279 | 26.40 | 0.031 |
| P9 calib-CCA statistic | 200 | 0.069 | 0.35 | 0.071 |
| P9 perm-CCA (199 perms) | 200 | 2.601 | 13.01 | 0.071 |
| Fusion inference (LR predict) | 200 | 0.298 | 1.49 | 0.057 |
| P9 calib-HSIC statistic | 200 | 1.640 | 8.20 | 1.235 |
| P9 perm-HSIC (199 perms) | 200 | 12.088 | 60.44 | 1.235 |
| Metadata predicates P1-P8 | 2000 | 13.682 | 6.84 | 0.284 |
| P9 calib-CCA statistic | 2000 | 0.456 | 0.23 | 0.675 |
| P9 perm-CCA (199 perms) | 2000 | 11.857 | 5.93 | 0.675 |
| Fusion inference (LR predict) | 2000 | 0.421 | 0.21 | 0.431 |
| P9 calib-HSIC statistic | 2000 | 173.158 | 86.58 | 122.208 |
| Metadata predicates P1-P8 | 10000 | 27.218 | 2.72 | 1.128 |
| P9 calib-CCA statistic | 10000 | 1.735 | 0.17 | 3.361 |
| P9 perm-CCA (199 perms) | 10000 | 77.949 | 7.79 | 3.361 |
| Fusion inference (LR predict) | 10000 | 1.059 | 0.11 | 2.062 |

Total runtime: 583 s