"""Real EEG figure for the project report (run on your machine; needs the EEGBCI EDF files).

(a) 30 s of the actual C3 and C4 recording of one participant (run R04), with the
    left-fist (T1) and right-fist (T2) imagery cues and the 0.5-3.5 s analysis windows shaded.
(b) Welch power spectrum of C3 and C4 averaged over T1 and over T2 windows, with the
    mu (8-13 Hz) and beta (13-30 Hz) bands used as features shaded.

Usage:
  python -m ns_gate.fig_eeg_trace --edf "C:\\Users\\harin\\mne_data\\MNE-eegbci-data\\files\\eegmmidb\\1.0.0\\S001\\S001R04.edf"
Output: outputs/figure_eeg_trace.png (+ .pdf)
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from .eeg_heldout import find_channel, T0, T1_

BLUE, ORANGE = "#257de0", "#f47230"
plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "font.family": "DejaVu Sans"})


def draw(raw, title, out=Path("outputs"), t_start=None, span=30.0):
    import mne
    sf = float(raw.info["sfreq"])
    c3, c4 = find_channel(raw, "C3"), find_channel(raw, "C4")
    events, ids = mne.events_from_annotations(raw, verbose="ERROR")
    cues = [(e[0] / sf, k) for k in ("T1", "T2") for e in events[events[:, 2] == ids[k]]]
    cues.sort()
    t_start = cues[0][0] - 2.0 if t_start is None else t_start
    t_start = max(t_start, 0.0)
    disp = raw.copy().pick([c3, c4]).filter(1.0, 40.0, verbose="ERROR")
    a, b = int(t_start * sf), int(min((t_start + span) * sf, raw.n_times))
    x = disp.get_data(start=a, stop=b) * 1e6
    t = np.arange(a, b) / sf

    fig = plt.figure(figsize=(7.16, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.4)
    ax = fig.add_subplot(gs[0])
    off = 4 * np.std(x)
    ax.plot(t, x[0] + off, color="#333", lw=0.5); ax.plot(t, x[1] - off, color="#333", lw=0.5)
    ax.set_yticks([off, -off]); ax.set_yticklabels(["C3", "C4"])
    for on, k in cues:
        if t_start <= on <= t_start + span:
            ax.axvspan(on + T0, on + T1_, color=BLUE if k == "T1" else ORANGE, alpha=0.18, lw=0)
            ax.text(on + 2.0, 1.02, "left" if k == "T1" else "right", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=6, color=BLUE if k == "T1" else ORANGE)
    ax.set_xlabel("time (s)"); ax.set_xlim(t_start, t_start + span)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"(a) Recorded EEG, {title} (1-40 Hz for display); shaded = 0.5-3.5 s analysis windows",
                 fontsize=7, loc="left", pad=12)

    ax = fig.add_subplot(gs[1])
    raw_d = raw.get_data(picks=[c3, c4])
    for ch_i, (name, ls) in enumerate((("C3", "-"), ("C4", "--"))):
        for k, col in (("T1", BLUE), ("T2", ORANGE)):
            ps = []
            for on, kk in cues:
                if kk != k:
                    continue
                s0, s1 = int(round((on + T0) * sf)), int(round((on + T1_) * sf))
                if s1 <= raw.n_times:
                    f, p = welch(raw_d[ch_i, s0:s1], fs=sf, nperseg=min(s1 - s0, int(2 * sf)))
                    ps.append(p)
            if ps:
                ax.semilogy(f, np.mean(ps, 0) * 1e12, color=col, ls=ls, lw=1.0,
                            label=f"{name}, {'left' if k == 'T1' else 'right'} imagery")
    ax.axvspan(8, 13, color="#999", alpha=0.15, lw=0); ax.axvspan(13, 30, color="#999", alpha=0.07, lw=0)
    ax.text(10.5, 1.01, "mu", transform=ax.get_xaxis_transform(), ha="center", fontsize=6)
    ax.text(21.5, 1.01, "beta", transform=ax.get_xaxis_transform(), ha="center", fontsize=6)
    ax.set_xlim(1, 40); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (µV²/Hz)")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(b) Power spectrum -> 4 features", fontsize=7, loc="left", pad=12)
    out.mkdir(exist_ok=True)
    fig.savefig(out / "figure_eeg_trace.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_eeg_trace.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    import mne
    ap = argparse.ArgumentParser(); ap.add_argument("--edf", required=True); ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    raw = mne.io.read_raw_edf(a.edf, preload=True, verbose="ERROR")
    draw(raw, Path(a.edf).stem, Path(a.out))
    print("saved", Path(a.out) / "figure_eeg_trace.png")


if __name__ == "__main__":
    main()
