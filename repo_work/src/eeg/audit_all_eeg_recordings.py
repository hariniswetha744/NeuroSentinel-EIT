
from pathlib import Path
import mne

EDF_ROOT = Path(
    r"C:\Users\harin\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0"
)

recordings = sorted(EDF_ROOT.rglob("*R04.edf"))

print("NEUROSENTINEL: ALL PARTICIPANT EEG AUDIT")
print("=" * 90)
print("Recordings found:", len(recordings))

for edf_path in recordings:
    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=False,
        verbose="ERROR"
    )

    annotations = raw.annotations

    t0 = sum(a["description"] == "T0" for a in annotations)
    t1 = sum(a["description"] == "T1" for a in annotations)
    t2 = sum(a["description"] == "T2" for a in annotations)

    t1_durations = [
        a["duration"] for a in annotations
        if a["description"] == "T1"
    ]

    t2_durations = [
        a["duration"] for a in annotations
        if a["description"] == "T2"
    ]

    mean_t1 = (
        sum(t1_durations) / len(t1_durations)
        if t1_durations else 0
    )

    mean_t2 = (
        sum(t2_durations) / len(t2_durations)
        if t2_durations else 0
    )

    print("\nParticipant:", edf_path.parent.name)
    print("Channels:", len(raw.ch_names))
    print("Sampling frequency:", raw.info["sfreq"], "Hz")
    print("Duration:", round(raw.times[-1], 2), "seconds")
    print("High-pass:", raw.info["highpass"], "Hz")
    print("Low-pass:", raw.info["lowpass"], "Hz")
    print("T0 events:", t0)
    print("T1 events:", t1)
    print("T2 events:", t2)
    print("Mean T1 event duration:", round(mean_t1, 3))
    print("Mean T2 event duration:", round(mean_t2, 3))

    raw.close()

print("\nAUDIT COMPLETE")
