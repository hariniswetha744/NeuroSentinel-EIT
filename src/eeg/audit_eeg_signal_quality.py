
from pathlib import Path
import numpy as np
import mne

EDF_ROOT = Path(
    r"C:\Users\harin\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0"
)

recordings = sorted(EDF_ROOT.rglob("*R04.edf"))

print("NEUROSENTINEL: EEG SIGNAL QUALITY AUDIT")
print("=" * 75)
print("Recordings found:", len(recordings))

for edf_path in recordings:
    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=True,
        verbose="ERROR"
    )

    data = raw.get_data(picks="eeg")

    finite_values = np.isfinite(data)
    channel_std = np.std(data, axis=1)
    channel_range = np.ptp(data, axis=1)

    flat_channels = [
        raw.ch_names[i]
        for i, value in enumerate(channel_std)
        if value == 0
    ]

    print("\nParticipant:", edf_path.parent.name)
    print("Signal shape:", data.shape)
    print("All values finite:", bool(finite_values.all()))
    print("Non-finite values:", int((~finite_values).sum()))
    print("Completely flat channels:", flat_channels)
    print("Minimum channel SD:", float(channel_std.min()))
    print("Maximum channel SD:", float(channel_std.max()))
    print("Minimum peak-to-peak:", float(channel_range.min()))
    print("Maximum peak-to-peak:", float(channel_range.max()))

    raw.close()

print("\nSIGNAL QUALITY AUDIT COMPLETE")
