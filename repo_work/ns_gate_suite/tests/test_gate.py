"""Unit tests for the validity gate. Run: python -m pytest -q tests"""
import numpy as np
import pytest

from ns_gate import faults as F
from ns_gate.gate import (GateBlocked, GateConfig, build_reference, guarded_fuse,
                          make_tables, metadata_predicates, perm_pvalue, run_gate)
from ns_gate.generator import generate, mispair


def _clean(n=200, seed=0):
    rng = np.random.default_rng(seed)
    d = generate(n, 0.5, rng)
    eeg, eit = make_tables(n, rng)
    return rng, d, eeg, eit


def test_clean_tables_pass_all_metadata():
    _, _, eeg, eit = _clean()
    assert all(metadata_predicates(eeg, eit).values())


@pytest.mark.parametrize("name,fn", list(F.RULE_FAULTS.items()))
def test_each_rule_fault_trips_metadata(name, fn):
    rng, d, eeg, eit = _clean()
    eeg2, eit2, xe, xi = fn(eeg, eit, d["x_eeg"], d["x_eit"], rng, sev=0.0)
    assert not all(metadata_predicates(eeg2, eit2).values()), name


def test_silent_mispair_passes_metadata_but_not_dependence():
    rng, d, eeg, eit = _clean(n=500)
    xi, mask = mispair(d["x_eit"], 1.0, rng)
    assert mask.sum() == 500
    assert all(metadata_predicates(eeg, eit).values())
    _, p = perm_pvalue(d["x_eeg"], xi, "cca", 199, rng)
    assert p > 0.01  # dependence destroyed
    _, p_clean = perm_pvalue(d["x_eeg"], d["x_eit"], "cca", 199, rng)
    assert p_clean <= 0.01


def test_mispair_derangement_moves_every_selected_row():
    rng = np.random.default_rng(1)
    x = np.arange(100, dtype=float)[:, None]
    x2, mask = mispair(x, 0.3, rng)
    assert np.all(x2[mask] != x[mask]) and np.all(x2[~mask] == x[~mask])


def test_no_fusion_call_when_gate_blocks():
    rng, d, eeg, eit = _clean()
    eit = eit.copy(); eit.loc[0, "sample_id"] = ""
    calls = []
    dec = run_gate(eeg, eit, d["x_eeg"], d["x_eit"], GateConfig(dependence=None), rng)
    assert not dec.passed
    with pytest.raises(GateBlocked):
        guarded_fuse(dec, lambda: calls.append(1))
    assert calls == []


def test_calibrated_reference_requires_matching_n():
    rng, d, eeg, eit = _clean()
    p = generate(5000, 0.5, rng)
    ref = build_reference(p["x_eeg"], p["x_eit"], 100, "cca", 50, rng)
    with pytest.raises(ValueError):
        run_gate(eeg, eit, d["x_eeg"], d["x_eit"], GateConfig(reference=ref), rng)
