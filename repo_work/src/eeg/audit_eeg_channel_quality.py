
from pathlib import Path
import mne

EDF_ROOT = Path(
    r"C:\Users\harin\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0"
)

recordings = sorted(EDF_ROOT.rglob("*R04.edf"))

print("NEUROSENTINEL: EEG CHANNEL QUALITY AUDIT")
print("=" * 75)
print("Recordings found:", len(recordings))

for edf_path in recordings:
    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=False,
        verbose="ERROR"
    )

    channel_types = raw.get_channel_types()
    eeg_count = channel_types.count("eeg")
    non_eeg_count = len(channel_types) - eeg_count
    bad_channels = list(raw.info["bads"])

    print("\nParticipant:", edf_path.parent.name)
    print("Total channels:", len(raw.ch_names))
    print("EEG channels:", eeg_count)
    print("Non-EEG channels:", non_eeg_count)
    print("Marked bad channels:", bad_channels)
    print("Unique channel names:", len(set(raw.ch_names)))

    raw.close()

print("\nCHANNEL QUALITY AUDIT COMPLETE")
