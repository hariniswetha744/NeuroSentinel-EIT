"""Extension experiments (v11): addressing the limitations reported in Section V.

E1  Record-level flagging   - which records in a batch are mispaired?
E2  Sequential testing      - accumulate evidence over successive batches.
E3  Combined CCA + learned  - one rule that covers both statistics' blind spots.

All in-silico (same controlled generator, family A, N = 200, r = 0.5 unless
stated). Clean reference batches are drawn fresh from the generator (not
subsampled from one finite pool), so the conformal false-block bound applies
directly. Every number reported in the paper is produced by these functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score

from . import faults as F
from .experiments import train_learned_scorer, wilson
from .gate import Reference, cca_pillai
from .generator import generate, mispair

ALPHA = 0.05


def _ref(stat_fn, n, r, rng, b=2000, gen_kw=None):
    vals = np.empty(b)
    for i in range(b):
        d = generate(n, r, rng, **(gen_kw or {}))
        vals[i] = stat_fn(d["x_eeg"], d["x_eit"])
    return Reference("x", n, vals)


# ------------------------------------------------------------------ E1

def bh_reject(p, fdr):
    """Benjamini-Hochberg: boolean mask of rejected hypotheses."""
    m = len(p); o = np.argsort(p)
    ok = p[o] <= fdr * np.arange(1, m + 1) / m
    k = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    rej = np.zeros(m, bool); rej[o[:k]] = True
    return rej


def e1_record_flagging(n=200, r_grid=(0.5, 0.8, 0.95), q_grid=(0.1, 0.25, 0.5), batches=300,
                       fdr=0.1, seed=11, calib_records=200000, clean_batches=2000):
    """Per-record conformal p-values from the learned match probability,
    calibrated on clean records; BH at the given FDR inside each batch."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in r_grid:
        _, rec = train_learned_scorer(r, rng)
        d = generate(calib_records, r, rng)
        cal = np.sort(rec(d["x_eeg"], d["x_eit"]))  # clean record scores
        for q in (0.0,) + tuple(q_grid):
            aucs, prec, recall, nflag, fdp = [], [], [], [], []
            nb = clean_batches if q == 0 else batches
            for _ in range(nb):
                d = generate(n, r, rng)
                xi, m = mispair(d["x_eit"], q, rng)
                s = rec(d["x_eeg"], xi)
                p = (1 + np.searchsorted(cal, s, side="right")) / (len(cal) + 1)  # low score -> small p
                flag = bh_reject(p, fdr)
                nflag.append(flag.sum())
                if q > 0:
                    aucs.append(roc_auc_score(m.astype(int), -s))
                    tp = np.sum(flag & m)
                    recall.append(tp / m.sum())
                    if flag.sum():
                        prec.append(tp / flag.sum())
                    fdp.append((flag.sum() - tp) / max(flag.sum(), 1))
                else:
                    fdp.append(float(flag.sum() > 0))  # any false flag in a clean batch
            rows.append({"r": r, "q": q, "batches": nb,
                         "record_auc": float(np.mean(aucs)) if aucs else np.nan,
                         "recall": float(np.mean(recall)) if recall else np.nan,
                         "precision_when_flagged": float(np.mean(prec)) if prec else np.nan,
                         "batches_with_flags": float(np.mean(np.array(nflag) > 0)),
                         "mean_flagged": float(np.mean(nflag)),
                         "mean_fdp_or_clean_any_flag": float(np.mean(fdp))})
    return pd.DataFrame(rows), {"n": n, "fdr": fdr, "seed": seed, "calib_records": calib_records,
                                "score": "learned match probability (32 cross-products)"}


# ------------------------------------------------------------------ E2

def e2_sequential(n=200, r=0.5, q_grid=(0.0, 0.05, 0.1, 0.2), k_grid=(1, 2, 5, 10, 20), streams=500, seed=12):
    """Fisher combination of the conformal CCA p-values of k successive batches
    of size n (fault persistent over the stream). Block when the combined
    p-value <= alpha."""
    rng = np.random.default_rng(seed)
    ref = _ref(cca_pillai, n, r, rng)
    kmax = max(k_grid)
    rows = []
    for q in q_grid:
        P = np.empty((streams, kmax))
        for s_ in range(streams):
            for j in range(kmax):
                d = generate(n, r, rng)
                xi, _ = mispair(d["x_eit"], q, rng)
                P[s_, j] = ref.pvalue(cca_pillai(d["x_eeg"], xi))
        for k in k_grid:
            chi = -2 * np.log(P[:, :k]).sum(1)
            pc = st.chi2.sf(chi, 2 * k)
            b = int(np.sum(pc <= ALPHA)); lo, hi = wilson(b, streams)
            rows.append({"q": q, "k_batches": k, "records_seen": k * n, "block_rate": b / streams,
                         "lo": lo, "hi": hi, "streams": streams})
    return pd.DataFrame(rows), {"n": n, "r": r, "seed": seed, "reference_batches": 2000,
                                "combination": "Fisher, chi2 with 2k d.f."}


# ------------------------------------------------------------------ E3

def e3_combined(n=200, r=0.5, trials=300, clean_trials=1000, seed=13):
    """Block if min(p_CCA, p_learned) <= alpha/2 (Bonferroni; clean false-block
    <= alpha). Compared with each statistic alone at alpha."""
    rng = np.random.default_rng(seed)

    def setup(gen_kw=None):
        lstat, _ = train_learned_scorer(r, rng, gen_kw=gen_kw)
        return lstat, _ref(cca_pillai, n, r, rng, gen_kw=gen_kw), _ref(lstat, n, r, rng, gen_kw=gen_kw)

    def run(fn, trials_, lstat, rc, rl, kw_fn=None, gen_kw=None):
        c = l = b = 0
        for _ in range(trials_):
            d = generate(n, r, rng, **(gen_kw or {}))
            kw = kw_fn(d) if kw_fn else {}
            _, _, xe, xi = fn(None, None, d["x_eeg"], d["x_eit"], rng, **kw)
            pc, pl = rc.pvalue(cca_pillai(xe, xi)), rl.pvalue(lstat(xe, xi))
            c += pc <= ALPHA; l += pl <= ALPHA; b += min(pc, pl) <= ALPHA / 2
        return c / trials_, l / trials_, b / trials_

    ident = lambda e, i, xe, xi, rng, **k: (e, i, xe, xi)
    lstat, rc, rl = setup()
    cases = [("Clean (false-block rate)", ident, 0.0, None, clean_trials)]
    cases += [("F7 Silent mispairing", (lambda q: lambda *x, **k: F.f7_silent_mispair(*x, sev=q))(q), q, None, trials)
              for q in (0.1, 0.25, 0.5, 1.0)]
    cases += [("H1 Dropped EIT frame", lambda *x, **k: F.h1_dropped_frame_shift(*x, sev=0.5), 0.5, None, trials),
              ("H3 Stale buffer", lambda *x, **k: F.h3_stale_buffer(*x, sev=0.5), 0.5, None, trials),
              ("H5 Cross-cohort", lambda *x, **k: F.h5_cross_cohort(*x, sev=0.5), 0.5, None, trials),
              ("H6 Class-cond. swap", lambda *x, **k: F.h6_class_conditional_swap(*x, sev=0.5, **k), 0.5,
               lambda d: {"y": d["y"]}, trials),
              ("H4 Electrode perm. (out of scope)", lambda *x, **k: F.h4_electrode_permutation(*x), 1.0, None, trials)]
    rows = []
    for name, fn, sev, kwf, t in cases:
        c, l, b = run(fn, t, lstat, rc, rl, kw_fn=kwf)
        rows.append({"case": name, "severity": sev, "trials": t, "cca": c, "learned": l, "combined": b})
    gk = {"n_subjects": 10, "subject_sd": 1.0}
    lstat2, rc2, rl2 = setup(gk)
    for name, fn, sev, kwf, t in [
        ("Clean subject-structured", ident, 0.0, None, clean_trials),
        ("H2 Within-subject swap", lambda *x, **k: F.h2_within_subject_swap(*x, sev=0.5, **k), 0.5,
         lambda d: {"subject": d["subject"]}, trials),
        ("H2 Within-subject swap", lambda *x, **k: F.h2_within_subject_swap(*x, sev=1.0, **k), 1.0,
         lambda d: {"subject": d["subject"]}, trials)]:
        c, l, b = run(fn, t, lstat2, rc2, rl2, kw_fn=kwf, gen_kw=gk)
        rows.append({"case": name, "severity": sev, "trials": t, "cca": c, "learned": l, "combined": b})
    return pd.DataFrame(rows), {"n": n, "r": r, "seed": seed, "reference_batches": 2000,
                                "rule": "block if min(p_cca, p_learned) <= alpha/2", "alpha": ALPHA}
