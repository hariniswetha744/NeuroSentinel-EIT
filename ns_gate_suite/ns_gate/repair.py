"""Novelty 2: repairing shift-type mispairing instead of only blocking it.

A dropped or duplicated frame, or a header-row offset, shifts every EIT payload after
some row k by s positions. Such a batch fails P9 because the shifted rows are mispaired.
Because the corruption has only two unknowns (k, s), it can be undone: scan all candidate
(k, s), realign, and keep the candidate that restores the most cross-modal dependence.

To avoid "repairing" batches that were never shifted (the scan always finds *some*
improvement by chance), a repair is accepted only if its Pillai gain exceeds a margin
calibrated on clean batches: the (1 - alpha) quantile of the maximal chance gain. So on
clean data, at most a fraction alpha of batches receive a spurious repair.
"""
from __future__ import annotations

import numpy as np

from .gate import _orth, cca_pillai

SHIFTS = (-2, -1, 1, 2)


def realign(xi, k, s):
    """Undo a shift of the rows from k on: row j >= k takes payload j - s (circular in the tail)."""
    out = xi.copy()
    out[k:] = np.roll(xi[k:], s, axis=0)
    return out


def scan(xe, xi, shifts=SHIFTS, step=1):
    """Return (best_pillai, k, s, base_pillai) over all tail shifts."""
    qx = _orth(xe)
    base = cca_pillai(xe, xi)
    best = (base, None, 0)
    n = len(xi)
    for s in shifts:
        for k in range(0, n - abs(s), step):
            v = float(np.sum((qx.T @ _orth(realign(xi, k, s))) ** 2))
            if v > best[0]:
                best = (v, k, s)
    return best[0], best[1], best[2], base


def calibrate_margin(clean_batches, alpha=0.05, **kw):
    """(1 - alpha) quantile of the chance gain max_{k,s} P(realigned) - P(original) on clean batches."""
    gains = np.array([scan(xe, xi, **kw)[0] - scan_base(xe, xi) for xe, xi in clean_batches])
    k = int(np.ceil((1 - alpha) * (len(gains) + 1))) - 1
    return float(np.sort(gains)[min(k, len(gains) - 1)]), gains


def scan_base(xe, xi):
    return cca_pillai(xe, xi)


def try_repair(xe, xi, margin, threshold=None, **kw):
    """Return (accepted, repaired_xi, k, s, gain). If threshold is given, the repaired batch
    must also pass the calibrated test (T >= threshold)."""
    best, k, s, base = scan(xe, xi, **kw)
    gain = best - base
    ok = k is not None and gain > margin and (threshold is None or best >= threshold)
    return ok, (realign(xi, k, s) if ok else xi), k, s, gain
