"""Experiment 7 runner: pairing-validity gate on real simultaneous EEG + NIRS.

Usage (from the ns_gate_suite folder):
    pip install pymatreader          # reads both old and new .mat formats
    python run_eeg_nirs.py --eeg-root "D:/hbci/EEG" --nirs-root "D:/hbci/NIRS"
    python run_eeg_nirs.py --inspect --eeg-root ... --nirs-root ...   # print file structure only

Writes outputs/exp7_eeg_nirs.json  (send this file back; every number in the paper comes from it).
"""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
from ns_gate import eeg_nirs_real as E

ap = argparse.ArgumentParser()
ap.add_argument("--eeg-root", required=True); ap.add_argument("--nirs-root", required=True)
ap.add_argument("--inspect", action="store_true"); ap.add_argument("--trials", type=int, default=300)
a = ap.parse_args()


def show(o, pre="    ", depth=0):
    if depth > 3:
        return
    if isinstance(o, dict):
        for k, w in o.items():
            print(pre + str(k), type(w).__name__, getattr(w, "shape", "")); show(w, pre + "  ", depth + 1)
    elif isinstance(o, list) and o:
        print(pre + f"list[{len(o)}] of {type(o[0]).__name__}"); show(o[0], pre + "  ", depth + 1)


if a.inspect:
    for kind, root in (("eeg", a.eeg_root), ("nirs", a.nirs_root)):
        d = E.find_subject_dirs(root, kind)
        print(kind.upper(), "subjects found:", list(d)[:40])
        if d:
            s = next(iter(d)); print("  example folder:", d[s])
            for f in ("cnt.mat", "mrk.mat"):
                print(" ", f); show(E._load(d[s] / f).get(f[:3]))
    sys.exit()

t0 = time.time()
xe, xn, subj, ses, y, info = E.build_trials(a.eeg_root, a.nirs_root)
print(f"trials: {len(y)}  (EEG features {xe.shape[1]}, NIRS features {xn.shape[1]})", info)
res = {"dataset": "Shin et al. hybrid EEG-NIRS BCI, dataset A (motor imagery, sessions 1/3/5)",
       "channels": info, "python": platform.python_version(), "runs": {}}
for std in ("subject", "none"):
    print(f"\n== features standardized: {std}")
    res["runs"][std] = E.run(xe, xn, subj, y, standardize=std, trials=a.trials)
res["runtime_s"] = round(time.time() - t0, 1)
out = Path(__file__).parent / "outputs" / "exp7_eeg_nirs.json"; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
print("\nsaved", out)
