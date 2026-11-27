"""Table III: EEG held-out (untouched participant) confirmation.

Runs on YOUR machine (needs the PhysioNet EEGBCI EDF files; the sandbox that
built this suite cannot reach physionet.org).

Protocol (frozen before looking at the held-out participants):
  * Features: identical to src/eeg/build_c3c4_mu_beta_eeg.py in the
    NeuroSentinel-EIT repo: run R04, T1/T2 cues, 0.5-3.5 s after cue, raw
    (unfiltered) C3/C4, Welch PSD (nperseg = 2 s), trapezoid band power over
    mu [8,13) and beta [13,30) Hz, log10.
  * Model: StandardScaler + LogisticRegression(class_weight="balanced",
    max_iter=2000, random_state=42), fitted ONCE on all development windows
    (default S001-S010) and applied unchanged to each held-out participant.
  * Baseline: CSP (4 components, log-variance) + LDA on 8-30 Hz band-passed
    64-channel epochs over the same 0.5-3.5 s window, also fitted once on the
    development participants. Optional: tangent-space + LR if pyriemann is
    installed.
  * Uncertainty: the participant is the unit. Mean of per-participant
    accuracy with a percentile bootstrap 95% CI over participants (10,000
    resamples), one-sided Wilcoxon signed-rank test against 0.5, and a
    paired bootstrap CI for (frozen - CSP).

Usage
  python -m ns_gate.eeg_heldout --edf-root "C:\\Users\\harin\\mne_data\\MNE-eegbci-data\\files\\eegmmidb\\1.0.0" \
         --dev 1-10 --heldout 11-30 --download --out outputs
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wilcoxon

BANDS = {"Mu": (8.0, 13.0), "Beta": (13.0, 30.0)}
LABELS = ["T1", "T2"]
T0, T1_ = 0.5, 3.5
FEATURES = ["C3_Mu_Log10", "C4_Mu_Log10", "C3_Beta_Log10", "C4_Beta_Log10"]
# participants with known recording irregularities in EEGBCI; skipped by default
KNOWN_BAD = {88, 92, 100, 104}


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def find_channel(raw, target):
    for name in raw.ch_names:
        if "".join(ch for ch in name.upper() if ch.isalnum()) == target:
            return name
    raise ValueError(f"{target} not found")


def band_power(sig, sfreq, lo, hi):
    f, p = welch(sig, fs=sfreq, nperseg=min(len(sig), int(2 * sfreq)))
    mask = (f >= lo) & (f < hi)
    return float(np.trapezoid(p[mask], f[mask]))


def windows(raw):
    """Yield (task, start_sample, stop_sample) for each T1/T2 cue."""
    import mne
    sfreq = float(raw.info["sfreq"])
    events, ids = mne.events_from_annotations(raw, verbose="ERROR")
    for task in LABELS:
        for ev in events[events[:, 2] == ids[task]]:
            onset = ev[0] / sfreq
            a, b = int(round((onset + T0) * sfreq)), int(round((onset + T1_) * sfreq))
            if b <= raw.n_times and (b - a) >= int(2 * sfreq):
                yield task, a, b


def features_from_raw(raw, participant):
    sfreq = float(raw.info["sfreq"])
    c3, c4 = find_channel(raw, "C3"), find_channel(raw, "C4")
    rows = []
    for task, a, b in windows(raw):
        seg = raw.get_data(picks=[c3, c4], start=a, stop=b)
        row = {"Participant": participant, "Task": task}
        for band, (lo, hi) in BANDS.items():
            row[f"C3_{band}_Log10"] = np.log10(band_power(seg[0], sfreq, lo, hi))
            row[f"C4_{band}_Log10"] = np.log10(band_power(seg[1], sfreq, lo, hi))
        rows.append(row)
    return rows


def epochs_from_raw(raw):
    r = raw.copy().pick("eeg").filter(8.0, 30.0, verbose="ERROR")
    X, y = [], []
    for task, a, b in windows(raw):
        X.append(r.get_data(start=a, stop=b)); y.append(task)
    n = min(x.shape[1] for x in X)
    return np.stack([x[:, :n] for x in X]), np.array(y)


def edf_path(root, subj):
    return Path(root) / f"S{subj:03d}" / f"S{subj:03d}R04.edf"


def maybe_download(subjects, root):
    import mne
    # MNE stores files under <path>/MNE-eegbci-data/files/eegmmidb/1.0.0, so <path> is 4 levels up
    base = Path(root).parents[3] if Path(root).name == "1.0.0" else None
    for s in subjects:
        if edf_path(root, s).exists():
            continue
        mne.datasets.eegbci.load_data([s], [4], path=str(base) if base else None, update_path=False)
        if not edf_path(root, s).exists():
            raise FileNotFoundError(f"download finished but {edf_path(root, s)} is missing; check --edf-root")


def load(subjects, root):
    import mne
    feats, epochs = [], {}
    for s in subjects:
        p = edf_path(root, s)
        raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
        pid = f"S{s:03d}"
        feats += features_from_raw(raw, pid)
        epochs[pid] = epochs_from_raw(raw)
        raw.close()
    return pd.DataFrame(feats), epochs


def boot_ci(v, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(dev_df, dev_ep, ho_df, ho_ep):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from mne.decoding import CSP

    assert not set(dev_df.Participant) & set(ho_df.Participant), "participant overlap"
    frozen = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced",
                                                                random_state=42))
    frozen.fit(dev_df[FEATURES].to_numpy(), dev_df.Task.to_numpy())
    major = DummyClassifier(strategy="most_frequent").fit(dev_df[FEATURES], dev_df.Task)
    Xd = np.concatenate([dev_ep[p][0] for p in dev_ep]); yd = np.concatenate([dev_ep[p][1] for p in dev_ep])
    n = min(Xd.shape[2], min(ho_ep[p][0].shape[2] for p in ho_ep))
    csp = make_pipeline(CSP(n_components=4, log=True), LDA()).fit(Xd[:, :, :n], yd)
    models = {"C3/C4 mu-beta LR (frozen)": None, "CSP+LDA": None, "Majority (dev)": None}
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        ts = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=2000)).fit(
            Xd[:, :, :n], yd)
        models["Tangent-space + LR"] = None
    except ImportError:
        ts = None

    per = []
    for pid, g in ho_df.groupby("Participant"):
        Xe, ye = ho_ep[pid][0][:, :, :n], ho_ep[pid][1]
        preds = {"C3/C4 mu-beta LR (frozen)": frozen.predict(g[FEATURES].to_numpy()),
                 "CSP+LDA": csp.predict(Xe), "Majority (dev)": major.predict(g[FEATURES])}
        if ts is not None:
            preds["Tangent-space + LR"] = ts.predict(Xe)
        truth = {"C3/C4 mu-beta LR (frozen)": g.Task.to_numpy(), "Majority (dev)": g.Task.to_numpy()}
        for name, p in preds.items():
            t = truth.get(name, ye)
            per.append({"Participant": pid, "Model": name, "windows": len(t),
                        "accuracy": float(np.mean(p == t)),
                        "balanced_accuracy": float(balanced_accuracy_score(t, p))})
    per = pd.DataFrame(per)
    summary = []
    for name, g in per.groupby("Model", sort=False):
        acc = g.accuracy.to_numpy()
        lo, hi = boot_ci(acc)
        try:
            w_p = float(wilcoxon(acc - 0.5, alternative="greater", zero_method="zsplit").pvalue)
        except ValueError:
            w_p = None
        summary.append({"Model": name, "participants": int(len(g)), "windows": int(g.windows.sum()),
                        "mean_subject_accuracy": float(acc.mean()),
                        "mean_subject_balanced_accuracy": float(g.balanced_accuracy.mean()),
                        "ci95_low": lo, "ci95_high": hi, "sd": float(acc.std(ddof=1)),
                        "n_above_0.5": int((acc > 0.5).sum()), "wilcoxon_p_vs_0.5": w_p})
    a = per[per.Model == "C3/C4 mu-beta LR (frozen)"].set_index("Participant").accuracy
    b = per[per.Model == "CSP+LDA"].set_index("Participant").accuracy
    dlo, dhi = boot_ci((a - b.loc[a.index]).to_numpy())
    paired = {"frozen_minus_csp_mean": float((a - b.loc[a.index]).mean()), "ci95": [dlo, dhi]}
    return per, pd.DataFrame(summary), paired


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-root", required=True)
    ap.add_argument("--dev", default="1-10")
    ap.add_argument("--heldout", default="11-30")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args(argv)
    dev = [s for s in parse_range(a.dev) if s not in KNOWN_BAD]
    ho = [s for s in parse_range(a.heldout) if s not in KNOWN_BAD and s not in dev]
    if a.download:
        maybe_download(dev + ho, a.edf_root)
    t0 = time.perf_counter(); tracemalloc.start()
    dev_df, dev_ep = load(dev, a.edf_root)
    ho_df, ho_ep = load(ho, a.edf_root)
    per, summ, paired = evaluate(dev_df, dev_ep, ho_df, ho_ep)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "table3_eeg_heldout_per_participant.csv", index=False)
    summ.to_csv(out / "table3_eeg_heldout_summary.csv", index=False)
    res = {"status": "RUN", "dev_participants": [f"S{s:03d}" for s in dev],
           "heldout_participants": [f"S{s:03d}" for s in ho], "run": "R04",
           "dev_windows": int(len(dev_df)), "heldout_windows": int(len(ho_df)),
           "summary": summ.to_dict(orient="records"), "paired_frozen_vs_csp": paired,
           "runtime_s": time.perf_counter() - t0, "peak_mb": peak / 2**20}
    (out / "table3_eeg_heldout.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
