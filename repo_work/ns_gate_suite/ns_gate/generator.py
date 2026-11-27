"""Controlled EEG-like / EIT-like paired feature generator (in-silico only).

Generator A follows Eq. (8) of the NeuroSentinel draft:
    s, e_E, e_I ~ N(0, 1)
    z_m = sqrt(r) * s + sqrt(1 - r) * e_m          (m in {E, I})
    y   = 1[z_E + z_I > 0]
    x_E = z_E * a + eps_E   (4 EEG-like features, noise SD 1.15)
    x_I = z_I * b + eps_I   (8 EIT-like features, noise SD 1.25)

r is the redundancy (corr(z_E, z_I) = r). Cross-modal statistical dependence
between x_E and x_I exists only through r, which is why the pairing-dependence
gate loses power as r -> 0 (complementary modalities).

Generator B (robustness family): Student-t noise, tanh mixing, loadings
redrawn for every dataset.

Optional subject structure (used only by the held-out within-subject swap
fault): a per-subject offset u_j is added to both modality latents, so that
records from the same subject are dependent across modalities even when
mispaired inside that subject.

Nothing here is physiological data. These are software test signals.
"""
from __future__ import annotations

import numpy as np

D_EEG, D_EIT = 4, 8
SD_EEG, SD_EIT = 1.15, 1.25
_LOADING_SEED = 20261127


def fixed_loadings():
    rng = np.random.default_rng(_LOADING_SEED)
    a = rng.normal(size=D_EEG)
    b = rng.normal(size=D_EIT)
    a /= np.linalg.norm(a) / np.sqrt(D_EEG)
    b /= np.linalg.norm(b) / np.sqrt(D_EIT)
    return a, b


def generate(n, r, rng, family="A", n_subjects=None, subject_sd=0.0, t_df=3):
    """Return dict with x_eeg (n,4), x_eit (n,8), y (n,), subject (n,)."""
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must be in [0, 1]")
    s = rng.normal(size=n)
    e_e = rng.normal(size=n)
    e_i = rng.normal(size=n)
    z_e = np.sqrt(r) * s + np.sqrt(1 - r) * e_e
    z_i = np.sqrt(r) * s + np.sqrt(1 - r) * e_i

    if n_subjects:
        subject = np.sort(rng.integers(0, n_subjects, size=n))
        u = rng.normal(scale=subject_sd, size=n_subjects)
        z_e = z_e + u[subject]
        z_i = z_i + u[subject]
    else:
        subject = np.zeros(n, dtype=int)

    y = (z_e + z_i > 0).astype(int)

    if family == "A":
        a, b = fixed_loadings()
        x_e = np.outer(z_e, a) + rng.normal(scale=SD_EEG, size=(n, D_EEG))
        x_i = np.outer(z_i, b) + rng.normal(scale=SD_EIT, size=(n, D_EIT))
    elif family == "B":
        a = rng.normal(size=D_EEG)
        b = rng.normal(size=D_EIT)
        scale_t = np.sqrt((t_df - 2) / t_df)  # unit-variance t noise
        x_e = np.tanh(np.outer(z_e, a)) * 2.0 + SD_EEG * scale_t * rng.standard_t(t_df, size=(n, D_EEG))
        x_i = np.tanh(np.outer(z_i, b)) * 2.0 + SD_EIT * scale_t * rng.standard_t(t_df, size=(n, D_EIT))
    else:
        raise ValueError("family must be 'A' or 'B'")
    return {"x_eeg": x_e, "x_eit": x_i, "y": y, "subject": subject}


def derangement_indices(idx, rng):
    """Permute idx so that no element stays in place (requires len >= 2)."""
    idx = np.asarray(idx)
    if len(idx) < 2:
        return idx.copy()
    while True:
        p = rng.permutation(len(idx))
        if not np.any(p == np.arange(len(idx))):
            return idx[p]


def mispair(x_eit, q, rng):
    """Silently mispair a fraction q of records: the EIT payload rows of the
    selected records are deranged among themselves; identifiers/metadata stay
    intact (positional-join bug). Returns (corrupted copy, mask of mispaired)."""
    n = len(x_eit)
    k = int(round(q * n))
    out = x_eit.copy()
    mask = np.zeros(n, dtype=bool)
    if k >= 2:
        sel = rng.choice(n, size=k, replace=False)
        out[sel] = x_eit[derangement_indices(sel, rng)]
        mask[sel] = True
    return out, mask
