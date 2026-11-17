# NeuroSentinel EEG-EIT Fusion Data Contract

## 1. Purpose
Define the data requirements for reproducible EEG-EIT
multimodal research integration.

## 2. Current Data Sources

### EEG
- Source: Public EEGBCI motor-imagery dataset
- Participants: 10
- Windows: 150
- Features: C3/C4 Mu and Beta band power
- Labels: T1 and T2 motor-imagery tasks

### EIT
- Source: EIDORS synthetic simulations
- Scenarios: 155
- Measurements per scenario: 208
- Severity: Synthetic conductivity-change levels
- Noise: 0%, 1%, 5%, and 10%

## 3. Required Pairing Fields

A future paired dataset must contain:

- Sample_ID
- Subject_ID
- Session_ID
- Observation_Date
- EEG_Record_ID
- EIT_Record_ID
- EEG_Source
- EIT_Source
- Pairing_Verification_Status

## 4. Pairing Requirements

1. EEG and EIT must originate from verified corresponding
   observations.
2. Sample_ID must be present and non-empty in both modalities.
3. Identifiers must be unique at the defined observation level.
4. Pairing must be verified using source records.
5. Arbitrary row matching and index-based pairing are prohibited.
6. Train/test splitting must prevent subject-level data leakage.

## 5. Model Input Requirements

EEG and EIT features must retain their modality-specific
definitions, units, preprocessing and provenance.

A multimodal model may be evaluated only after verified
paired observations are available.

## 6. Current Integration Decision

Status: BLOCKED

Reason:
The current public EEG dataset and synthetic EIT dataset
do not contain verified shared participant or sample IDs.

The datasets must remain separate.

## 7. Scientific Limitations

- EEG labels represent healthy motor-imagery tasks.
- EIT severity represents synthetic conductivity change.
- Neither dataset provides verified tumour labels.
- Neither dataset provides longitudinal recurrence labels.
- Synthetic conductivity severity is not tumour severity.
- No recurrence prediction or prevention claim is supported.

## 8. Future Work

Obtain appropriately governed, paired EEG-EIT observations
with verified longitudinal clinical annotations.

Only then evaluate multimodal AI and recurrence-related
research hypotheses.

## 9. Reproducibility

Preserve source datasets, preprocessing scripts, model
configurations, evaluation reports and software versions.