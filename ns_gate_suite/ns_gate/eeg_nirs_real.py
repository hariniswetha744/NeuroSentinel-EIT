"""Experiment 7: the pairing-validity gate on REAL simultaneously recorded EEG + NIRS.

Dataset: Shin et al., open-access hybrid EEG-NIRS BCI dataset (TU Berlin), dataset A,
motor imagery sessions 1, 3, 5 (left vs right hand, 20 trials per session, 29 subjects).
EEG and NIRS were recorded at the same time, so every EEG trial has exactly one true
NIRS partner (same subject, session and trial order). We then corrupt that pairing the
way a software join bug would and ask whether the gate notices.

Features (one row per trial, computed from each modality alone):
  EEG  (4): log mu (8-13 Hz) and beta (13-30 Hz) power, left and right sensorimotor
            channels, task window 0-10 s.
  NIRS (8): optical-density change at both wavelengths (a fixed linear map of HbO/HbR),
            left and right motor channels, mean change (2-12 s minus -2-0 s baseline)
            and slope over 0-10 s.

Design: subjects are split into a REFERENCE half (trains the learned scorer and builds
the clean-batch reference) and a TEST half (every reported block rate). So the clean
false-block rate is measured on people the gate has never seen.

Nothing in this file is tuned on the test half. Run:  python run_eeg_nirs.py --eeg-root ... --nirs-root ...
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy.signal import welch

from .gate import Reference, cca_pillai, perm_pvalue
from .generator import derangement_indices

MI_SESSIONS = (0, 2, 4)  # zero-based cells 1, 3, 5
LEFT_EEG = ("C3", "C5", "C1", "FCC3h", "FCC5h", "CCP3h", "CCP5h", "FC3", "CP3", "FC5", "CP5")
RIGHT_EEG = ("C4", "C6", "C2", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "FC4", "CP4", "FC6", "CP6")


# ------------------------------------------------------------------ loading

def _load(path):
    """Load a .mat file (v5 or v7.3) into plain Python dicts/lists."""
    try:
        from pymatreader import read_mat
        return read_mat(str(path))
    except ImportError:
        pass
    import scipy.io as sio
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, simplify_cells=True)
    return m


def _as_list(c):
    if isinstance(c, (list, tuple)):
        return list(c)
    if isinstance(c, np.ndarray) and c.dtype == object:
        return list(c.ravel())
    return [c]


def _clab(s):
    return [str(c) for c in _as_list(s["clab"])]


def _x(s):
    x = np.asarray(s["x"], dtype=float)
    n_ch = len(_clab(s))
    return x if x.shape[1] == n_ch else x.T  # samples x channels


def _markers(m):
    """Return (onset_ms array, class array 0=left 1=right)."""
    t = np.atleast_1d(np.asarray(m["time"], dtype=float)).ravel()
    if "y" in m and np.size(m["y"]):
        y = np.asarray(m["y"], dtype=float)
        if y.ndim == 2:
            if y.shape[1] != len(t):
                y = y.T
            cls = np.argmax(y, axis=0)
        else:
            cls = (y.ravel() > y.min()).astype(int)
    else:
        desc = np.atleast_1d(np.asarray(m["event"]["desc"]).ravel())
        cls = (desc > desc.min()).astype(int)
    order = np.argsort(t)
    return t[order], cls[order]


def find_subject_dirs(root, kind):
    """Map subject number -> folder containing cnt.mat/mrk.mat for EEG or NIRS."""
    out = {}
    for f in Path(root).rglob("cnt.mat"):
        d = f.parent
        if not (d / "mrk.mat").exists():
            continue
        m = re.search(r"(?:subject|VP)\s*0*(\d+)", str(d), re.I)
        if not m:
            continue
        s = int(m.group(1))
        path_l = str(d).lower()
        if kind == "eeg":
            # prefer the EOG-corrected copy if both are present
            if s in out and "without" in str(out[s]).lower():
                continue
        out[s] = d
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ features

def _pick(clab, wanted):
    idx = [i for i, c in enumerate(clab) if c in wanted]
    return idx


def _side(clab):
    """Split labels into left/right sensorimotor sets from the 10-5 names inside them
    (odd number = left, even = right; only C, FC, CP-type central positions)."""
    left, right = [], []
    for i, c in enumerate(clab):
        cc = re.sub(r"(low|high)WL$", "", c)
        nums = [int(n) for n in re.findall(r"(?:FC|CP|FCC|CCP|C)(\d+)", cc)]
        if not nums:
            continue
        if all(n % 2 == 1 for n in nums):
            left.append(i)
        elif all(n % 2 == 0 for n in nums):
            right.append(i)
    return left, right


def _wavelength_split(clab):
    lo = [i for i, c in enumerate(clab) if c.endswith("lowWL")]
    hi = [i for i, c in enumerate(clab) if c.endswith("highWL")]
    if lo and hi and len(lo) == len(hi):
        return lo, hi, False
    h = len(clab) // 2
    return list(range(h)), list(range(h, len(clab))), True


def eeg_features(cnt, mrk, win=(0.0, 10.0)):
    clab, x, fs = _clab(cnt), _x(cnt), float(cnt["fs"])
    li, ri = _pick(clab, LEFT_EEG), _pick(clab, RIGHT_EEG)
    if not li or not ri:
        li, ri = _side(clab)
    if not li or not ri:
        raise RuntimeError(f"EEG motor channels not found in {clab}")
    t, cls = _markers(mrk)
    feats = []
    for onset in t:
        a = int(round((onset / 1000 + win[0]) * fs)); b = int(round((onset / 1000 + win[1]) * fs))
        seg = x[a:b]
        f, p = welch(seg, fs=fs, nperseg=int(fs), axis=0)
        mu = np.log(p[(f >= 8) & (f <= 13)].mean(0)); be = np.log(p[(f >= 13) & (f <= 30)].mean(0))
        feats.append([mu[li].mean(), mu[ri].mean(), be[li].mean(), be[ri].mean()])
    return np.array(feats), cls, {"eeg_left": [clab[i] for i in li], "eeg_right": [clab[i] for i in ri]}


def nirs_features(cnt, mrk):
    """Features from raw NIRS light intensity at two wavelengths.

    For each trial, the optical-density change dOD = -log10(I / I0), with I0 the mean
    intensity in the -2..0 s baseline, is computed per channel and wavelength. By the
    modified Beer-Lambert law, HbO and HbR changes are a fixed linear map of the two
    wavelengths' dOD, so dOD features carry the same information without assuming
    extinction coefficients, path-length factors or optode distances.
    Per wavelength (low, high) and side (left, right): mean dOD 2-12 s and dOD slope 0-10 s.
    """
    clab, fs, x = _clab(cnt), float(cnt["fs"]), _x(cnt)
    lo, hi, wl_fallback = _wavelength_split(clab)
    base_lab = [re.sub(r"(low|high)WL$", "", clab[i]) for i in lo]
    li, ri = _side(base_lab)
    side_fallback = False
    if not li or not ri:
        h = len(lo) // 2; li, ri = list(range(h)), list(range(h, len(lo))); side_fallback = True
    t, cls = _markers(mrk)
    x = np.clip(x, 1e-9, None)
    out = []
    for onset in t:
        o = onset / 1000
        i0 = x[int((o - 2) * fs):int(o * fs)].mean(0)
        row = []
        for wl in (lo, hi):
            od = -np.log10(x[:, wl] / i0[wl])
            resp = od[int((o + 2) * fs):int((o + 12) * fs)].mean(0)
            seg = od[int(o * fs):int((o + 10) * fs)]
            slope = np.polyfit(np.arange(len(seg)) / fs, seg, 1)[0]
            row += [resp[li].mean(), resp[ri].mean(), slope[li].mean(), slope[ri].mean()]
        out.append(row)
    return np.array(out), cls, {"nirs_left": [base_lab[i] for i in li], "nirs_right": [base_lab[i] for i in ri],
                                "nirs_side_fallback": side_fallback, "nirs_wavelength_fallback": wl_fallback,
                                "nirs_wavelengths": np.asarray(cnt.get("wavelengths", [])).tolist(),
                                "nirs_signal": str(cnt.get("signal", "")), "nirs_clab_example": clab[:6]}


def build_trials(eeg_root, nirs_root, log=print):
    eeg_dirs, nirs_dirs = find_subject_dirs(eeg_root, "eeg"), find_subject_dirs(nirs_root, "nirs")
    subjects = sorted(set(eeg_dirs) & set(nirs_dirs))
    if not subjects:
        raise RuntimeError(f"No subject found in both roots. EEG: {list(eeg_dirs)[:5]}  NIRS: {list(nirs_dirs)[:5]}")
    log(f"subjects with both modalities: {len(subjects)}")
    XE, XN, S, SES, Y, info = [], [], [], [], [], {}
    for s in subjects:
        ec, em = _load(eeg_dirs[s] / "cnt.mat")["cnt"], _load(eeg_dirs[s] / "mrk.mat")["mrk"]
        nc, nm = _load(nirs_dirs[s] / "cnt.mat")["cnt"], _load(nirs_dirs[s] / "mrk.mat")["mrk"]
        ec, em, nm, nc = _as_list(ec), _as_list(em), _as_list(nm), _as_list(nc)
        for k in MI_SESSIONS:
            fe, ce, ie = eeg_features(ec[k], em[k])
            fn, cn, inn = nirs_features(nc[k], nm[k])
            if len(fe) != len(fn) or not np.array_equal(ce, cn):
                raise RuntimeError(f"subject {s} session {k+1}: EEG/NIRS trials do not line up "
                                   f"({len(fe)} vs {len(fn)} trials)")
            XE.append(fe); XN.append(fn); Y.append(ce)
            S.append(np.full(len(fe), s)); SES.append(np.full(len(fe), k + 1))
            info = {**ie, **inn}
    return (np.vstack(XE), np.vstack(XN), np.concatenate(S), np.concatenate(SES),
            np.concatenate(Y), info)


# ------------------------------------------------------------------ gate on real pairs

def _zscore_within(x, groups):
    x = x.copy()
    for g in np.unique(groups):
        m = groups == g
        x[m] = (x[m] - x[m].mean(0)) / (x[m].std(0) + 1e-12)
    return x


def _learned(xe, xn, rng):
    from sklearn.linear_model import LogisticRegression
    mu_e, sd_e, mu_n, sd_n = xe.mean(0), xe.std(0) + 1e-12, xn.mean(0), xn.std(0) + 1e-12
    feat = lambda a, b: (((a - mu_e) / sd_e)[:, :, None] * ((b - mu_n) / sd_n)[:, None, :]).reshape(len(a), -1)
    neg = xn[derangement_indices(np.arange(len(xn)), rng)]
    clf = LogisticRegression(max_iter=3000).fit(np.vstack([feat(xe, xn), feat(xe, neg)]),
                                                 np.r_[np.ones(len(xe)), np.zeros(len(xe))])
    return lambda a, b: float(clf.predict_proba(feat(a, b))[:, 1].mean())


def _wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run(xe, xn, subj, y, n_grid=(100, 200), q_grid=(0.0, 0.1, 0.25, 0.5, 1.0), trials=300,
        ref_batches=1000, n_perm=199, alpha=0.05, seed=7, standardize="subject", log=print):
    rng = np.random.default_rng(seed)
    if standardize == "subject":
        xe, xn = _zscore_within(xe, subj), _zscore_within(xn, subj)
    subjects = np.unique(subj)
    ref_s = subjects[: len(subjects) // 2]; test_s = subjects[len(subjects) // 2:]
    R = np.isin(subj, ref_s); T = ~R
    out = {"standardize": standardize, "n_trials": int(len(y)), "n_subjects": int(len(subjects)),
           "reference_subjects": ref_s.tolist(), "test_subjects": test_s.tolist(),
           "n_reference_trials": int(R.sum()), "n_test_trials": int(T.sum())}

    # 1. is there cross-modal dependence in correctly paired real data? (whole test half)
    t, p = perm_pvalue(xe[T], xn[T], "cca", n_perm=999, rng=rng)
    out["test_half_pillai"] = t; out["test_half_perm_p"] = p
    lstat = _learned(xe[R], xn[R], rng)

    rows = []
    for n in n_grid:
        if n > T.sum():
            continue
        refc = Reference("cca", n, np.array([cca_pillai(*(lambda i: (xe[R][i], xn[R][i]))(
            rng.choice(R.sum(), n, replace=False))) for _ in range(ref_batches)]))
        refl = Reference("learned", n, np.array([lstat(*(lambda i: (xe[R][i], xn[R][i]))(
            rng.choice(R.sum(), n, replace=False))) for _ in range(ref_batches)]))
        xt_e, xt_n, st_, yt = xe[T], xn[T], subj[T], y[T]
        cases = [("random", q) for q in q_grid] + [("within-subject", 1.0), ("class-conditional", 0.5)]
        for kind, q in cases:
            k_cc = k_cp = k_l = k_comb = 0
            for _ in range(trials):
                idx = rng.choice(len(yt), n, replace=False)
                a, b = xt_e[idx], xt_n[idx].copy()
                if kind == "random":
                    k = int(round(q * n))
                    if k >= 2:
                        sel = rng.choice(n, k, replace=False); b[sel] = b[derangement_indices(sel, rng)]
                else:
                    g = st_[idx] if kind == "within-subject" else yt[idx]
                    for v in np.unique(g):
                        mem = np.flatnonzero(g == v); k = int(round(q * len(mem)))
                        if k >= 2:
                            sel = rng.choice(mem, k, replace=False); b[sel] = b[derangement_indices(sel, rng)]
                pc = refc.pvalue(cca_pillai(a, b)); pl = refl.pvalue(lstat(a, b))
                _, pp = perm_pvalue(a, b, "cca", n_perm=n_perm, rng=rng)
                k_cc += pc <= alpha; k_l += pl <= alpha; k_cp += pp > alpha
                k_comb += min(pc, pl) <= alpha / 2
            row = {"n": n, "fault": kind, "q": q, "trials": trials}
            for name, k in (("cca_calib", k_cc), ("cca_perm", k_cp), ("learned_calib", k_l), ("combined", k_comb)):
                lo, hi = _wilson(k, trials); row[name] = k / trials; row[name + "_ci"] = [lo, hi]
            rows.append(row)
            log(f"  N={n:4d} {kind:17s} q={q:4.2f}  CCA-cal {row['cca_calib']:.3f}  CCA-perm {row['cca_perm']:.3f}"
                f"  learned {row['learned_calib']:.3f}  combined {row['combined']:.3f}")
    out["rows"] = rows
    out["settings"] = {"trials": trials, "ref_batches": ref_batches, "n_perm": n_perm, "alpha": alpha, "seed": seed}
    return out
