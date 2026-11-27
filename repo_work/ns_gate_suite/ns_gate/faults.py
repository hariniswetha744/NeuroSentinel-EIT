"""Fault injectors. Each returns corrupted copies (eeg, eit, x_eeg, x_eit).

RULE-WRITTEN faults (F1-F6) target one metadata predicate each.
F7 is the silent, identifier-consistent mispairing fault that motivates the
dependence predicate.

HELD-OUT faults (H1-H6) were specified after the gate and its thresholds were
frozen, and none is an instance of a single metadata predicate. H4 is
deliberately out of scope (a within-modality fault) and is included to show
where the dependence predicate cannot help.
"""
from __future__ import annotations

import numpy as np

from .generator import derangement_indices, generate, mispair


def _pick(n, frac, rng, min_k=1):
    k = max(min_k, int(round(frac * n)))
    return rng.choice(n, size=min(k, n), replace=False)


# ------------------------------------------------------------- rule-written

def f1_missing_id(eeg, eit, xe, xi, rng, sev=0.01):
    eit = eit.copy(); idx = _pick(len(eit), sev, rng)
    eit.loc[eit.index[idx], "sample_id"] = ""
    return eeg, eit, xe, xi


def f2_duplicate_id(eeg, eit, xe, xi, rng, sev=0.01):
    eit = eit.copy(); idx = _pick(len(eit), sev, rng)
    donor = (idx + 1) % len(eit)
    eit.loc[eit.index[idx], "sample_id"] = eit["sample_id"].values[donor]
    return eeg, eit, xe, xi


def f3_id_mismatch(eeg, eit, xe, xi, rng, sev=0.01):
    eit = eit.copy(); idx = _pick(len(eit), sev, rng)
    eit.loc[eit.index[idx], "sample_id"] = [f"XX{j:06d}" for j in range(len(idx))]
    return eeg, eit, xe, xi


def f4_rate_channel_unit(eeg, eit, xe, xi, rng, sev=0.01):
    eit = eit.copy(); idx = eit.index[_pick(len(eit), sev, rng)]
    kind = rng.integers(3)
    if kind == 0:
        eit["fs"] = eit["fs"].astype(float); eit.loc[idx, "fs"] = 1000.0
    elif kind == 1:
        eit.loc[idx, "n_channels"] = 64
    else:
        eit.loc[idx, "units"] = "mV"
    return eeg, eit, xe, xi


def f5_timestamp_offset(eeg, eit, xe, xi, rng, sev=0.01):
    eit = eit.copy(); idx = eit.index[_pick(len(eit), sev, rng)]
    eit.loc[idx, "timestamp"] = eit.loc[idx, "timestamp"] + 10.0
    return eeg, eit, xe, xi


def f6_missing_consent(eeg, eit, xe, xi, rng, sev=0.01):
    eit = eit.copy(); idx = eit.index[_pick(len(eit), sev, rng)]
    eit.loc[idx, "consent"] = ""
    return eeg, eit, xe, xi


def f7_silent_mispair(eeg, eit, xe, xi, rng, sev=0.25):
    xi2, _ = mispair(xi, sev, rng)
    return eeg, eit, xe, xi2


RULE_FAULTS = {
    "F1 Missing/empty identifier": f1_missing_id,
    "F2 Duplicate identifier": f2_duplicate_id,
    "F3 Identifier mismatch": f3_id_mismatch,
    "F4 Rate/channel/unit error": f4_rate_channel_unit,
    "F5 Timestamp offset > tolerance": f5_timestamp_offset,
    "F6 Missing consent/source": f6_missing_consent,
}

# ------------------------------------------------------------------ held-out

def h1_dropped_frame_shift(eeg, eit, xe, xi, rng, sev=0.5):
    """One EIT frame dropped at position k: every later payload is off by one."""
    n = len(xi); k = int(round(n * (1 - sev)))
    xi2 = xi.copy()
    if n - k >= 2:
        xi2[k:] = np.roll(xi[k:], -1, axis=0)
    return eeg, eit, xe, xi2


def h3_stale_buffer(eeg, eit, xe, xi, rng, sev=0.5):
    """A frozen acquisition buffer repeats one EIT payload for a fraction of records."""
    idx = _pick(len(xi), sev, rng)
    xi2 = xi.copy(); xi2[idx] = xi[idx[0]]
    return eeg, eit, xe, xi2


def h4_electrode_permutation(eeg, eit, xe, xi, rng, sev=1.0):
    """Cable/electrode-order swap: EIT feature columns permuted (within-modality)."""
    xi2 = xi[:, rng.permutation(xi.shape[1])]
    return eeg, eit, xe, xi2


def h5_cross_cohort(eeg, eit, xe, xi, rng, sev=0.5, r=0.5):
    """Fraction of EIT payloads substituted from an unrelated cohort."""
    idx = _pick(len(xi), sev, rng)
    other = generate(len(idx), r, rng)["x_eit"]
    xi2 = xi.copy(); xi2[idx] = other
    return eeg, eit, xe, xi2


def h6_class_conditional_swap(eeg, eit, xe, xi, rng, sev=0.5, y=None):
    """Mispairing restricted to records of the same class (sort-by-label bug)."""
    xi2 = xi.copy()
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        k = int(round(sev * len(members)))
        if k >= 2:
            sel = rng.choice(members, size=k, replace=False)
            xi2[sel] = xi[derangement_indices(sel, rng)]
    return eeg, eit, xe, xi2


def h2_within_subject_swap(eeg, eit, xe, xi, rng, sev=1.0, subject=None):
    """Mispairing restricted to records of the same subject."""
    xi2 = xi.copy()
    for s in np.unique(subject):
        members = np.flatnonzero(subject == s)
        k = int(round(sev * len(members)))
        if k >= 2:
            sel = rng.choice(members, size=k, replace=False)
            xi2[sel] = xi[derangement_indices(sel, rng)]
    return eeg, eit, xe, xi2
