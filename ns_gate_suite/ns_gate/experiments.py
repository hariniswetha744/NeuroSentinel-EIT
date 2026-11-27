"""Gate experiments: Figure A, Figure B, Table I, Table II, Table IV.

All results are in-silico (controlled generator). Every function returns a
plain dict / DataFrame and the caller saves it; nothing is typed by hand.
"""
from __future__ import annotations

import platform
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import faults as F
from .gate import (GateConfig, ID_ONLY, PREDICATE_NAMES, build_reference, cca_pillai,
                   hsic, make_tables, metadata_predicates, perm_pvalue, run_gate)
from .generator import generate, mispair


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def train_models(r, rng, n_train=2000, family="A"):
    d = generate(n_train, r, rng, family=family)
    fused = _lr().fit(np.hstack([d["x_eeg"], d["x_eit"]]), d["y"])
    eeg = _lr().fit(d["x_eeg"], d["y"])
    return fused, eeg


# ------------------------------------------------------------------ Figure A

def figure_a(r=0.5, n=200, reps=200, q_grid=None, alpha=0.05, seed=1, pool=20000):
    rng = np.random.default_rng(seed)
    q_grid = q_grid if q_grid is not None else np.round(np.linspace(0, 1, 11), 2)
    fused, eeg_m = train_models(r, rng)
    pd_ = generate(pool, r, rng)
    ref = build_reference(pd_["x_eeg"], pd_["x_eit"], n, "cca", 1000, rng)
    thr = ref.threshold(alpha)
    rows = []
    for q in q_grid:
        acc_u, acc_e, acc_gc, acc_gp, blk_c, blk_p = [], [], [], [], [], []
        acc_mis, acc_ok = [], []
        for _ in range(reps):
            d = generate(n, r, rng)
            xi, mask = mispair(d["x_eit"], q, rng)
            hit = fused.predict(np.hstack([d["x_eeg"], xi])) == d["y"]
            a_u = np.mean(hit)
            if mask.any():
                acc_mis.append(np.mean(hit[mask]))
            if (~mask).any():
                acc_ok.append(np.mean(hit[~mask]))
            a_e = np.mean(eeg_m.predict(d["x_eeg"]) == d["y"])
            bc = cca_pillai(d["x_eeg"], xi) < thr
            _, p = perm_pvalue(d["x_eeg"], xi, "cca", 199, rng)
            bp = p > alpha
            acc_u.append(a_u); acc_e.append(a_e)
            acc_gc.append(a_e if bc else a_u); acc_gp.append(a_e if bp else a_u)
            blk_c.append(bc); blk_p.append(bp)

        def ms(v):
            v = np.asarray(v, float)
            return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))

        row = {"q": float(q)}
        for name, v in [("ungated_fusion", acc_u), ("eeg_only", acc_e),
                        ("gated_calib_cca", acc_gc), ("gated_perm_cca", acc_gp)]:
            m, h = ms(v); row[f"acc_{name}"] = m; row[f"acc_{name}_ci95"] = h
        row["acc_mispaired_records"] = float(np.mean(acc_mis)) if acc_mis else None
        row["acc_correctly_paired_records"] = float(np.mean(acc_ok)) if acc_ok else None
        row["block_rate_calib_cca"] = float(np.mean(blk_c))
        row["block_rate_perm_cca"] = float(np.mean(blk_p))
        rows.append(row)
    return pd.DataFrame(rows), {"r": r, "n": n, "reps": reps, "alpha": alpha,
                                "reference_batches": 1000, "reference_pool": pool,
                                "fallback": "EEG-only model when blocked"}


# ------------------------------------------------------------------ Figure B

def _power_cell(r, n, q, reps, rng, ref_c, ref_h=None, family="A", do_perm=True,
                do_hsic=False, alpha=0.05):
    thr_c = ref_c.threshold(alpha)
    thr_h = ref_h.threshold(alpha) if ref_h is not None else None
    res = {"calib_cca": 0, "perm_cca": 0, "calib_hsic": 0, "perm_hsic": 0}
    for _ in range(reps):
        d = generate(n, r, rng, family=family)
        xi, _ = mispair(d["x_eit"], q, rng)
        res["calib_cca"] += cca_pillai(d["x_eeg"], xi) < thr_c
        if do_perm:
            res["perm_cca"] += perm_pvalue(d["x_eeg"], xi, "cca", 199, rng)[1] > alpha
        if do_hsic:
            res["calib_hsic"] += hsic(d["x_eeg"], xi) < thr_h
            res["perm_hsic"] += perm_pvalue(d["x_eeg"], xi, "hsic", 199, rng)[1] > alpha
    out = {}
    for k, v in res.items():
        lo, hi = wilson(v, reps)
        out[k] = v / reps; out[k + "_lo"] = lo; out[k + "_hi"] = hi
    return out


def figure_b(reps=200, seed=2, alpha=0.05, pool=20000,
             q_grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
             n_grid=(50, 100, 200, 500, 1000), r_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
             hsic_reps=100):
    rng = np.random.default_rng(seed)
    rows = []
    # (b1) sample size sweep at r = 0.5
    p = generate(pool, 0.5, rng)
    for n in n_grid:
        ref = build_reference(p["x_eeg"], p["x_eit"], n, "cca", 1000, rng)
        for q in q_grid:
            rows.append({"panel": "N", "r": 0.5, "n": n, "q": q, "family": "A",
                         **_power_cell(0.5, n, q, reps, rng, ref, alpha=alpha)})
    # (b2) redundancy sweep at N = 200
    for r in r_grid:
        p = generate(pool, r, rng)
        ref = build_reference(p["x_eeg"], p["x_eit"], 200, "cca", 1000, rng)
        for q in q_grid:
            rows.append({"panel": "r", "r": r, "n": 200, "q": q, "family": "A",
                         **_power_cell(r, 200, q, reps, rng, ref, alpha=alpha)})
    # (b3) CCA vs HSIC on both generator families at r = 0.5, N = 200
    for fam in ("A", "B"):
        # family B redraws loadings per dataset, so the reference pool is a
        # concatenation of many small datasets (one loading draw each)
        if fam == "A":
            p = generate(pool, 0.5, rng)
            xe, xi = p["x_eeg"], p["x_eit"]
            ref_c = build_reference(xe, xi, 200, "cca", 1000, rng)
            ref_h = build_reference(xe, xi, 200, "hsic", 1000, rng)
        else:
            vc, vh = [], []
            for _ in range(1000):
                d = generate(200, 0.5, rng, family="B")
                vc.append(cca_pillai(d["x_eeg"], d["x_eit"])); vh.append(hsic(d["x_eeg"], d["x_eit"]))
            from .gate import Reference
            ref_c = Reference("cca", 200, np.array(vc)); ref_h = Reference("hsic", 200, np.array(vh))
        for q in q_grid:
            rows.append({"panel": "stat", "r": 0.5, "n": 200, "q": q, "family": fam,
                         **_power_cell(0.5, 200, q, hsic_reps, rng, ref_c, ref_h, family=fam,
                                       do_perm=True, do_hsic=True, alpha=alpha)})
    return pd.DataFrame(rows), {"reps": reps, "hsic_reps": hsic_reps, "alpha": alpha,
                                "n_perm": 199, "reference_batches": 1000}


# --------------------------------------------------------------- Tables I/II

def _variants(ref_c, ref_h):
    return {
        "ID-only (P1-P4)": GateConfig(use_metadata=list(ID_ONLY), dependence=None),
        "Metadata (P1-P8)": GateConfig(dependence=None),
        "Full: +P9 calib-CCA": GateConfig(dependence="calib", stat="cca", reference=ref_c),
        "Full: +P9 perm-CCA": GateConfig(dependence="perm", stat="cca"),
        "Full: +P9 calib-HSIC": GateConfig(dependence="calib", stat="hsic", reference=ref_h),
    }


def _eval_fault(fn, trials, n, r, rng, variants, fused, kw_fn=None, gen_kw=None):
    blocks = {k: 0 for k in variants}
    acc = []
    for _ in range(trials):
        d = generate(n, r, rng, **(gen_kw or {}))
        eeg, eit = make_tables(n, rng)
        kw = kw_fn(d) if kw_fn else {}
        eeg2, eit2, xe, xi = fn(eeg, eit, d["x_eeg"], d["x_eit"], rng, **kw)
        acc.append(np.mean(fused.predict(np.hstack([xe, xi])) == d["y"]))
        for name, cfg in variants.items():
            blocks[name] += not run_gate(eeg2, eit2, xe, xi, cfg, rng).passed
    return blocks, float(np.mean(acc))


def _identity(eeg, eit, xe, xi, rng, **kw):
    return eeg, eit, xe, xi


def tables_i_ii(n=200, r=0.5, trials=200, clean_trials=1000, seed=3, pool=20000):
    rng = np.random.default_rng(seed)
    fused, _ = train_models(r, rng)
    p = generate(pool, r, rng)
    ref_c = build_reference(p["x_eeg"], p["x_eit"], n, "cca", 1000, rng)
    ref_h = build_reference(p["x_eeg"], p["x_eit"], n, "hsic", 1000, rng)
    V = _variants(ref_c, ref_h)

    rows = []

    def add(set_name, fault, sev, blocks, trials_, acc):
        row = {"set": set_name, "fault": fault, "severity": sev, "trials": trials_,
               "ungated_fusion_acc": acc}
        for k, v in blocks.items():
            lo, hi = wilson(v, trials_)
            row[k] = v / trials_; row[k + " lo"] = lo; row[k + " hi"] = hi
        rows.append(row)

    b, a = _eval_fault(_identity, clean_trials, n, r, rng, V, fused)
    add("clean", "Clean paired data (false-block rate)", 0.0, b, clean_trials, a)
    for name, fn in F.RULE_FAULTS.items():
        b, a = _eval_fault(lambda *x, **k: fn(*x, sev=0.0), trials, n, r, rng, V, fused)
        add("rule-written", name + " (1 record)", 1 / n, b, trials, a)
    for q in (0.1, 0.25, 0.5, 1.0):
        b, a = _eval_fault(lambda *x, q=q, **k: F.f7_silent_mispair(*x, sev=q), trials, n, r, rng, V, fused)
        add("rule-written", "F7 Silent ID-consistent mispairing", q, b, trials, a)

    # ---- held-out faults (Table II), severity 0.5 unless noted
    held = [
        ("H1 Dropped EIT frame (off-by-one tail)", F.h1_dropped_frame_shift, 0.5, None),
        ("H3 Stale buffer (repeated EIT payload)", F.h3_stale_buffer, 0.5, None),
        ("H4 Electrode/column permutation (out of scope)", F.h4_electrode_permutation, 1.0, None),
        ("H5 Cross-cohort substitution", F.h5_cross_cohort, 0.5, None),
        ("H6 Class-conditional swap", F.h6_class_conditional_swap, 0.5, lambda d: {"y": d["y"]}),
    ]
    for name, fn, sev, kwf in held:
        b, a = _eval_fault(lambda *x, fn=fn, sev=sev, **k: fn(*x, sev=sev, **k), trials, n, r, rng, V,
                           fused, kw_fn=kwf)
        add("held-out", name, sev, b, trials, a)
    for sev in (0.1, 0.25, 1.0):
        for name, fn in [("H1 Dropped EIT frame (off-by-one tail)", F.h1_dropped_frame_shift),
                         ("H6 Class-conditional swap", F.h6_class_conditional_swap)]:
            kwf = (lambda d: {"y": d["y"]}) if fn is F.h6_class_conditional_swap else None
            b, a = _eval_fault(lambda *x, fn=fn, sev=sev, **k: fn(*x, sev=sev, **k), trials, n, r, rng,
                               V, fused, kw_fn=kwf)
            add("held-out-sweep", name, sev, b, trials, a)

    # H2 within-subject swap needs subject-structured data and its own reference
    gk = {"n_subjects": 10, "subject_sd": 1.0}
    ps = generate(pool, r, rng, **gk)
    # reference batches must keep subject structure: draw whole fresh batches
    from .gate import Reference
    vc, vh = [], []
    for _ in range(1000):
        d = generate(n, r, rng, **gk)
        vc.append(cca_pillai(d["x_eeg"], d["x_eit"])); vh.append(hsic(d["x_eeg"], d["x_eit"]))
    V2 = _variants(Reference("cca", n, np.array(vc)), Reference("hsic", n, np.array(vh)))
    b, a = _eval_fault(_identity, clean_trials, n, r, rng, V2, fused, gen_kw=gk)
    add("held-out", "Clean subject-structured data (false-block rate)", 0.0, b, clean_trials, a)
    for sev in (0.5, 1.0):
        b, a = _eval_fault(lambda *x, sev=sev, **k: F.h2_within_subject_swap(*x, sev=sev, **k), trials, n,
                           r, rng, V2, fused, kw_fn=lambda d: {"subject": d["subject"]}, gen_kw=gk)
        add("held-out", "H2 Within-subject swap", sev, b, trials, a)
    del ps
    return pd.DataFrame(rows), {"n": n, "r": r, "trials": trials, "clean_trials": clean_trials,
                                "alpha": 0.05, "n_perm": 199}


# ------------------------------------------------------------------ Table IV

def _profile(fn, runs=20):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    tracemalloc.start(); fn(); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return float(np.median(times) * 1000), float(peak / 2**20)


def table_iv(seed=4, runs=20):
    rng = np.random.default_rng(seed)
    rows = []
    fused, _ = train_models(0.5, rng)
    for n in (200, 2000, 10000):
        d = generate(n, 0.5, rng); eeg, eit = make_tables(n, rng)
        xe, xi = d["x_eeg"], d["x_eit"]
        ref = build_reference(xe, xi, min(n, 200), "cca", 50, rng)
        items = [
            ("Metadata predicates P1-P8", lambda: metadata_predicates(eeg, eit)),
            ("P9 calib-CCA statistic", lambda: cca_pillai(xe, xi)),
            ("P9 perm-CCA (199 perms)", lambda: perm_pvalue(xe, xi, "cca", 199, rng)),
            ("Fusion inference (LR predict)", lambda: fused.predict(np.hstack([xe, xi]))),
        ]
        if n <= 2000:
            items.append(("P9 calib-HSIC statistic", lambda: hsic(xe, xi)))
        if n <= 200:
            items.append(("P9 perm-HSIC (199 perms)", lambda: perm_pvalue(xe, xi, "hsic", 199, rng)))
        for name, fn in items:
            r_ = runs if "perm" not in name or n <= 2000 else 5
            ms, mb = _profile(fn, r_)
            rows.append({"component": name, "batch_n": n, "median_ms": ms,
                         "per_record_us": ms * 1000 / n, "peak_mb": mb, "runs": r_})
        del ref
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor() or "unknown", "numpy": np.__version__}
    try:
        import os
        env["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return pd.DataFrame(rows), env


# ------------------------------------------------ reference-construction check

def reference_check(seed=99, n_grid=(200, 1000), n_test=2000, pool=20000, alpha=0.05):
    """Clean false-block rate of calib-CCA when the reference is (i) 1000
    independently generated clean batches or (ii) 1000 subsamples of one
    finite pool. Explains why Figure B/Table I sit slightly above alpha."""
    from .gate import Reference
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_grid:
        def stat():
            d = generate(n, 0.5, rng); return cca_pillai(d["x_eeg"], d["x_eit"])
        fresh = Reference("cca", n, np.array([stat() for _ in range(1000)]))
        p = generate(pool, 0.5, rng)
        sub = build_reference(p["x_eeg"], p["x_eit"], n, "cca", 1000, rng)
        test = np.array([stat() for _ in range(n_test)])
        for name, ref in (("independent batches", fresh), (f"subsampled {pool}-record pool", sub)):
            k = int(np.sum(test < ref.threshold(alpha)))
            lo, hi = wilson(k, n_test)
            rows.append({"n": n, "reference": name, "false_block": k / n_test, "lo": lo, "hi": hi,
                         "test_batches": n_test})
    return pd.DataFrame(rows)


# ------------------------------------------- learned-correspondence baseline
# A supervised alternative to P9: a logistic regression trained to separate
# matched pairs from randomly re-paired ones (negatives made by derangement),
# using the 32 cross-products x_E (x) x_I of standardized features. Each record
# gets a match probability; the batch statistic is their mean, thresholded with
# the same conformal rule (4) on 1,000 clean reference batches.

def _pair_features(xe, xi, mu_e, sd_e, mu_i, sd_i):
    ze, zi = (xe - mu_e) / sd_e, (xi - mu_i) / sd_i
    return (ze[:, :, None] * zi[:, None, :]).reshape(len(ze), -1)


def train_learned_scorer(r, rng, n_train=20000, family="A", gen_kw=None):
    from .generator import derangement_indices
    d = generate(n_train, r, rng, family=family, **(gen_kw or {}))
    xe, xi = d["x_eeg"], d["x_eit"]
    mu_e, sd_e, mu_i, sd_i = xe.mean(0), xe.std(0), xi.mean(0), xi.std(0)
    neg = xi[derangement_indices(np.arange(n_train), rng)]
    X = np.vstack([_pair_features(xe, xi, mu_e, sd_e, mu_i, sd_i), _pair_features(xe, neg, mu_e, sd_e, mu_i, sd_i)])
    y = np.r_[np.ones(n_train), np.zeros(n_train)]
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X, y)

    def record_scores(a, b):
        return clf.predict_proba(_pair_features(a, b, mu_e, sd_e, mu_i, sd_i))[:, 1]

    def batch_stat(a, b):
        return float(record_scores(a, b).mean())
    return batch_stat, record_scores


def learned_baseline(n=200, r=0.5, trials=300, clean_trials=1000, seed=7, ref_batches=1000):
    from . import gate as G
    from .gate import Reference
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    fused, _ = train_models(r, rng)
    stat, rec = train_learned_scorer(r, rng)
    G.STATS["learned"] = stat
    vals = []
    for _ in range(ref_batches):
        d = generate(n, r, rng); vals.append(stat(d["x_eeg"], d["x_eit"]))
    V = {"Full: +P9 learned-calib": GateConfig(dependence="calib", stat="learned",
                                               reference=Reference("learned", n, np.array(vals)))}
    rows = []

    def add(set_name, fault, sev, blocks, trials_, acc):
        k = blocks["Full: +P9 learned-calib"]; lo, hi = wilson(k, trials_)
        rows.append({"set": set_name, "fault": fault, "severity": sev, "trials": trials_,
                     "ungated_fusion_acc": acc, "learned_block": k / trials_, "lo": lo, "hi": hi})

    b, a = _eval_fault(_identity, clean_trials, n, r, rng, V, fused)
    add("clean", "Clean paired data (false-block rate)", 0.0, b, clean_trials, a)
    for q in (0.1, 0.25, 0.5, 1.0):
        b, a = _eval_fault(lambda *x, q=q, **k: F.f7_silent_mispair(*x, sev=q), trials, n, r, rng, V, fused)
        add("rule-written", "F7 Silent ID-consistent mispairing", q, b, trials, a)
    held = [("H1 Dropped EIT frame (off-by-one tail)", F.h1_dropped_frame_shift, 0.5, None),
            ("H3 Stale buffer (repeated EIT payload)", F.h3_stale_buffer, 0.5, None),
            ("H4 Electrode/column permutation (out of scope)", F.h4_electrode_permutation, 1.0, None),
            ("H5 Cross-cohort substitution", F.h5_cross_cohort, 0.5, None),
            ("H6 Class-conditional swap", F.h6_class_conditional_swap, 0.5, lambda d: {"y": d["y"]})]
    for name, fn, sev, kwf in held:
        b, a = _eval_fault(lambda *x, fn=fn, sev=sev, **k: fn(*x, sev=sev, **k), trials, n, r, rng, V, fused, kw_fn=kwf)
        add("held-out", name, sev, b, trials, a)
    # H2 on the subject-structured generator: scorer and reference trained on that generator
    gk = {"n_subjects": 10, "subject_sd": 1.0}
    stat2, _ = train_learned_scorer(r, rng, gen_kw=gk)
    G.STATS["learned"] = stat2
    vals = [stat2(*(lambda d: (d["x_eeg"], d["x_eit"]))(generate(n, r, rng, **gk))) for _ in range(ref_batches)]
    V2 = {"Full: +P9 learned-calib": GateConfig(dependence="calib", stat="learned",
                                                reference=Reference("learned", n, np.array(vals)))}
    b, a = _eval_fault(_identity, clean_trials, n, r, rng, V2, fused, gen_kw=gk)
    add("held-out", "Clean subject-structured data (false-block rate)", 0.0, b, clean_trials, a)
    for sev in (0.5, 1.0):
        b, a = _eval_fault(lambda *x, sev=sev, **k: F.h2_within_subject_swap(*x, sev=sev, **k), trials, n, r, rng, V2,
                           fused, kw_fn=lambda d: {"subject": d["subject"]}, gen_kw=gk)
        add("held-out", "H2 Within-subject swap", sev, b, trials, a)
    G.STATS["learned"] = stat
    # per-record localisation: AUC of record scores, mispaired vs correct, at q = 0.5
    aucs = []
    for _ in range(200):
        d = generate(n, r, rng); xi, m = mispair(d["x_eit"], 0.5, rng)
        aucs.append(roc_auc_score(m.astype(int), -rec(d["x_eeg"], xi)))
    # cost of the learned statistic at N = 200
    d = generate(n, r, rng)
    t0 = time.perf_counter()
    for _ in range(50):
        stat(d["x_eeg"], d["x_eit"])
    ms = (time.perf_counter() - t0) / 50 * 1000
    meta = {"n": n, "r": r, "trials": trials, "clean_trials": clean_trials, "seed": seed,
            "train_records": 20000, "features": "32 cross-products of standardized x_E and x_I",
            "record_auc_q0.5_mean": float(np.mean(aucs)), "record_auc_q0.5_sd": float(np.std(aucs, ddof=1)),
            "record_auc_batches": 200, "batch_stat_ms_n200": ms}
    return pd.DataFrame(rows), meta
