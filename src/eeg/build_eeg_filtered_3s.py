
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

EDF_ROOT = Path(
    r"C:\Users\harin\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0"
)

OUTPUT_DIR = Path(r"C:\NeuroSentinel\data")

FEATURES_FILE = OUTPUT_DIR / "NeuroSentinel_EEG_Filtered_3s_Features.csv"
SUMMARY_FILE = OUTPUT_DIR / "NeuroSentinel_EEG_Filtered_3s_Summary.csv"
QUALITY_FILE = OUTPUT_DIR / "NeuroSentinel_EEG_Filtered_3s_Quality_Audit.csv"

LOW_CUT = 1.0
HIGH_CUT = 40.0
BUFFER_SECONDS = 0.5
WINDOW_END_SECONDS = 3.5

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recordings = sorted(EDF_ROOT.rglob("*R04.edf"))

    if not recordings:
        raise FileNotFoundError("No R04 EDF recordings found.")

    feature_rows = []
    quality_rows = []

    print("NEUROSENTINEL FILTERED 3-SECOND EEG PIPELINE")
    print("=" * 65)

    for edf_path in recordings:
        participant = edf_path.parent.name

        print("Processing:", participant)

        raw = mne.io.read_raw_edf(
            str(edf_path),
            preload=True,
            verbose="ERROR"
        )

        raw.pick("eeg")

        sfreq = float(raw.info["sfreq"])

        if HIGH_CUT >= sfreq / 2:
            raise ValueError(
                f"High cutoff must be below Nyquist frequency: {participant}"
            )

        raw.filter(
            l_freq=LOW_CUT,
            h_freq=HIGH_CUT,
            method="fir",
            phase="zero",
            verbose="ERROR"
        )

        events, event_ids = mne.events_from_annotations(
            raw,
            verbose="ERROR"
        )

        for task in ("T1", "T2"):
            if task not in event_ids:
                continue

            task_events = events[
                events[:, 2] == event_ids[task]
            ]

            for window_number, event in enumerate(
                task_events, start=1
            ):
                event_onset = event[0] / sfreq

                start = event_onset + BUFFER_SECONDS
                stop = event_onset + WINDOW_END_SECONDS

                start_sample = int(round(start * sfreq))
                stop_sample = int(round(stop * sfreq))

                if (
                    start_sample < 0
                    or stop_sample > raw.n_times
                    or stop_sample <= start_sample
                ):
                    continue

                segment = raw.get_data(
                    start=start_sample,
                    stop=stop_sample
                )

                finite = bool(np.isfinite(segment).all())

                if not finite:
                    quality_rows.append({
                        "Participant": participant,
                        "Task": task,
                        "Window": window_number,
                        "Finite": False,
                        "Accepted": False,
                        "Reason": "Non-finite signal values"
                    })
                    continue

                channel_std = np.std(segment, axis=1)
                peak_to_peak = np.ptp(segment, axis=1)

                flat_channels = int(
                    np.sum(channel_std == 0)
                )

                freqs, psd = welch(
                    segment,
                    fs=sfreq,
                    nperseg=min(
                        segment.shape[1],
                        int(2 * sfreq)
                    ),
                    axis=1
                )

                row = {
                    "Participant": participant,
                    "Run": "R04",
                    "Task": task,
                    "Window": window_number,
                    "Onset_seconds": round(event_onset, 3),
                    "Duration_seconds": round(
                        segment.shape[1] / sfreq, 3
                    ),
                    "Channel_Count": segment.shape[0],
                }

                for band_name, (low, high) in BANDS.items():
                    mask = (
                        (freqs >= low)
                        & (freqs < high)
                    )

                    if not np.any(mask):
                        raise RuntimeError(
                            f"No PSD bins for {band_name}"
                        )

                    channel_powers = np.trapezoid(
                        psd[:, mask],
                        freqs[mask],
                        axis=1
                    )

                    row[band_name] = float(
                        np.mean(channel_powers)
                    )

                feature_rows.append(row)

                quality_rows.append({
                    "Participant": participant,
                    "Task": task,
                    "Window": window_number,
                    "Finite": finite,
                    "Flat_Channel_Count": flat_channels,
                    "Min_Channel_SD_V": float(
                        channel_std.min()
                    ),
                    "Max_Channel_SD_V": float(
                        channel_std.max()
                    ),
                    "Max_Peak_to_Peak_V": float(
                        peak_to_peak.max()
                    ),
                    "Accepted": True,
                    "Reason": "Passed basic signal integrity checks"
                })

        raw.close()

    features = pd.DataFrame(feature_rows)
    quality = pd.DataFrame(quality_rows)

    if features.empty:
        raise RuntimeError("No EEG feature windows extracted.")

    if features[list(BANDS)].isna().any().any():
        raise RuntimeError("Missing band-power values detected.")

    if not np.isfinite(
        features[list(BANDS)].to_numpy()
    ).all():
        raise RuntimeError("Non-finite feature values detected.")

    if (features[list(BANDS)] < 0).any().any():
        raise RuntimeError("Negative band-power values detected.")

    if quality.empty or not quality["Accepted"].all():
        raise RuntimeError("Quality audit contains rejected windows.")

    summary = (
        features.groupby("Task")[list(BANDS)]
        .agg(["count", "mean", "std"])
    )

    features.to_csv(FEATURES_FILE, index=False)
    summary.to_csv(SUMMARY_FILE)
    quality.to_csv(QUALITY_FILE, index=False)

    print("\nPIPELINE COMPLETE")
    print("Participants:", features["Participant"].nunique())
    print("Feature windows:", len(features))
    print("Task counts:")
    print(features["Task"].value_counts().to_string())
    print("Rejected windows:", int((~quality["Accepted"]).sum()))

    print("\nSaved files:")
    print(FEATURES_FILE)
    print(SUMMARY_FILE)
    print(QUALITY_FILE)


if __name__ == "__main__":
    main()
