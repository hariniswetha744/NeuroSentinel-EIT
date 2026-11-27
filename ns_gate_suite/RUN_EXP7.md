# Experiment 7: gate on real simultaneous EEG + NIRS

1. Get the data: open https://doc.ml.tu-berlin.de/hBCI/ , go to "Downloading the data",
   fill in the short form, and download dataset A:
     - all EEG zips (EEG_01-05 ... EEG_26-29)
     - the NIRS zip
   Unzip so that you have, e.g.
     D:/hbci/EEG/.../subject 01/.../cnt.mat, mrk.mat, mnt.mat
     D:/hbci/NIRS/.../subject 01/cnt.mat, mrk.mat, mnt.mat
   (exact sub-folder names do not matter; the script searches for "subject NN" folders)

2. From the ns_gate_suite folder:
     pip install pymatreader
     python run_eeg_nirs.py --inspect --eeg-root "D:/hbci/EEG" --nirs-root "D:/hbci/NIRS"
   This only prints the file structure. If it shows 29 subjects for both, run:
     python run_eeg_nirs.py --eeg-root "D:/hbci/EEG" --nirs-root "D:/hbci/NIRS"

3. Send back outputs/exp7_eeg_nirs.json (and the console output if anything fails).
