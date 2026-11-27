
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

# Input data and separate output files
EDF_ROOT = Path(
    r"C:\Users\harin\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0"
)
OUTPUT_DIR = Path(r"C:\NeuroSentinel\data")

OUTPUT_CSV = OUTPUT_DIR / "NeuroSentinel_MultiSubject_EEG_3s_Features.csv"
SUMMARY_CSV = OUTPUT_DIR / "NeuroSentinel_MultiSubject_EEG_3s_Summary.csv"

# EEG frequency bands, in Hz
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
}

# Analyse only R04, the same motor-imagery run for each participant.
# Use a consistent 1-second buffer at the beginning and end of each task.
BUFFER_SECONDS = 0.5

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recordings = sorted(EDF_ROOT.rglob("*R04.edf"))

    if not recordings:
        raise FileNotFoundError(f"No R04 EDF recordings found in {EDF_ROOT}")

    rows = []

    for edf_path in recordings:
        participant = edf_path.parent.name

        print(f"Processing {participant}: {edf_path.name}")
        raw = mne.io.read_raw_edf(
            str(edf_path), preload=True, verbose="ERROR"
        )

        sfreq = float(raw.info["sfreq"])
        events, event_ids = mne.events_from_annotations(raw, verbose="ERROR")

        for task in ("T1", "T2"):
            if task not in event_ids:
                print(f"  Skipping missing event {task}")
                continue

            task_code = event_ids[task]
            task_events = events[events[:, 2] == task_code]

            for window_number, event in enumerate(task_events, start=1):
                start = event[0] / sfreq + BUFFER_SECONDS
                stop = event[0] / sfreq + 3.5

                start_sample = int(round(start * sfreq))
                stop_sample = int(round(stop * sfreq))

                if stop_sample <= start_sample:
                    continue

                segment = raw.get_data(
                    start=start_sample, stop=stop_sample
                )

                if segment.shape[1] < int(sfreq):
                    continue

                # Average power spectral density across EEG channels.
                freqs, psd = welch(
                    segment,
                    fs=sfreq,
                    nperseg=min(segment.shape[1], int(2 * sfreq)),
                    axis=1,
                )

                row = {
                    "Participant": participant,
                    "Run": "R04",
                    "Task": task,
                    "Window": window_number,
                    "Onset_seconds": round(event[0] / sfreq, 3),
                    "Duration_seconds": round(
                        segment.shape[1] / sfreq, 3
                    ),
                }

                for band_name, (low, high) in BANDS.items():
                    mask = (freqs >= low) & (freqs < high)
                    if not np.any(mask):
                        row[band_name] = np.nan
                    else:
                        channel_powers = np.trapezoid(
                            psd[:, mask], freqs[mask], axis=1
                        )
                        row[band_name] = float(
                            np.mean(channel_powers)
                        )

                rows.append(row)

        raw.close()

    features = pd.DataFrame(rows)

    if features.empty:
        raise RuntimeError("No task features extracted; outputs not written.")

    if features[BANDS.keys()].isna().any().any():
        raise RuntimeError("Missing band-power values; outputs not written.")

    features.to_csv(OUTPUT_CSV, index=False)

    summary = (
        features.groupby(["Task"])[list(BANDS.keys())]
        .agg(["count", "mean", "std"])
    )
    summary.to_csv(SUMMARY_CSV)

    print("\nFEATURE EXTRACTION COMPLETE")
    print("Recordings found:", len(recordings))
    print("Feature rows:", len(features))
    print("Participants:", features["Participant"].nunique())
    print("Task counts:\n", features["Task"].value_counts().to_string())
    print("Rows per participant:\n",
          features.groupby("Participant").size().to_string())
    print("\nSaved feature dataset:", OUTPUT_CSV)
    print("Saved task summary:", SUMMARY_CSV)
    print("\nPreview:\n", features.head().to_string(index=False))


if __name__ == "__main__":
    main()
