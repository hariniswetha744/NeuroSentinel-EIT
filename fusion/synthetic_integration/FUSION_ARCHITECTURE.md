
# NeuroSentinel EEG-EIT AI Fusion Architecture

## 1. System Objective

NeuroSentinel is a reproducible software research framework
for EEG and Electrical Impedance Tomography (EIT) analysis.

Its long-term research objective is multimodal neural
monitoring using EEG, EIT and artificial intelligence.

## 2. Architecture

### Layer 1: EEG Processing
- Public EEGBCI motor-imagery dataset
- EEG preprocessing and filtering
- C3/C4 Mu and Beta feature extraction
- Logistic Regression classification
- Leave-One-Subject-Out evaluation

### Layer 2: EIT Processing
- EIDORS-based forward simulation
- Electrical measurement generation
- Image reconstruction
- Noise robustness evaluation
- Synthetic conductivity severity estimation

### Layer 3: Data Integration
- Separate modality-specific feature matrices
- Common Sample_ID requirement
- Subject and session verification
- Missing-value and duplicate checks
- Pairing eligibility gate

### Layer 4: Multimodal AI
Future implementation after verified paired data:
- EEG feature encoder
- EIT feature encoder
- Multimodal feature integration
- Subject-independent evaluation
- Performance and robustness analysis

### Layer 5: Monitoring Dashboard
- EEG analysis
- EIT reconstruction and validation
- Synthetic EIT AI evaluation
- Data tables
- Fusion readiness
- Integration limitations

## 3. Current Implementation Status

| Component | Status |
|---|---|
| EEG processing | Implemented |
| EEG exploratory AI | Implemented |
| EIT simulation | Implemented |
| EIT reconstruction | Implemented |
| EIT synthetic AI | Implemented |
| Fusion data contract | Implemented |
| Pairing gate | Implemented |
| Pairing gate tests | 6/6 passed |
| Clinical multimodal AI | Blocked |
| Recurrence prediction | Future work |

## 4. Integration Safety

The current EEG and EIT datasets are independent.

No arbitrary pairing, row-index matching or fabricated
clinical relationships are permitted.

The fusion gate must remain blocked until verified
paired observations are available.

## 5. Recurrence Research

Longitudinal recurrence research requires appropriately
governed clinical observations, verified outcomes and
longitudinal follow-up.

The present synthetic EIT severity model does not
represent tumour severity or recurrence.

NeuroSentinel does not currently predict or prevent
tumour recurrence.

## 6. Research Reproducibility

Preserve:
- Original datasets
- Preprocessing scripts
- Feature extraction scripts
- Model configurations
- Evaluation metrics
- Validation reports
- Software environment
- Dashboard source code

## 7. Conclusion

NeuroSentinel currently provides separate EEG and EIT
research pipelines, synthetic EIT AI evaluation,
visualization and a tested fusion-readiness architecture.

Verified multimodal clinical integration remains future
work requiring paired observations.