"""Novelty 1 experiment: predicted vs simulated power of the calibrated Pillai gate.
Writes outputs/novelty/theory_sim.json. Population spectra are estimated once per setting
from 400,000 clean records; empirical power uses a fresh 1,000-batch conformal reference
and 400 test batches per cell."""
import json, time
from pathlib import Path
import numpy as np
from ns_gate.generator import generate, mispair
from ns_gate.gate import cca_pillai
from ns_gate import theory as T

SEED, ALPHA = 41, 0.05
rng = np.random.default_rng(SEED); t0 = time.time(); rows = []
for fam in ("A", "B"):
    for r in (0.1, 0.25, 0.5, 0.75):
        big = generate(400000, r, rng, family=fam); rho2 = T.spectrum(big["x_eeg"], big["x_eit"])
        for N in (50, 100, 200, 500, 1000):
            ref = np.sort([cca_pillai(*(lambda g: (g["x_eeg"], g["x_eit"]))(generate(N, r, rng, family=fam)))
                           for _ in range(1000)])
            thr = ref[int(np.floor(ALPHA * 1001)) - 1]
            for q in (0.1, 0.25, 0.5, 1.0):
                V = []
                for _ in range(400):
                    g = generate(N, r, rng, family=fam); xi, _ = mispair(g["x_eit"], q, rng)
                    V.append(cca_pillai(g["x_eeg"], xi))
                rows.append({"family": fam, "r": r, "N": N, "q": q, "theta": float(rho2.sum()),
                             "empirical": float(np.mean(np.array(V) < thr)),
                             "predicted": T.power(rho2, N, q, ALPHA)})
err = np.array([abs(r_["predicted"] - r_["empirical"]) for r_ in rows])
summ = {fam: {"mae": float(np.mean([abs(r_["predicted"] - r_["empirical"]) for r_ in rows if r_["family"] == fam])),
              "max": float(np.max([abs(r_["predicted"] - r_["empirical"]) for r_ in rows if r_["family"] == fam])),
              "within_10pts": float(np.mean([abs(r_["predicted"] - r_["empirical"]) <= 0.10 for r_ in rows if r_["family"] == fam])),
              "cells": sum(r_["family"] == fam for r_ in rows)} for fam in ("A", "B")}
# sample-size rule check: N* predicted for 80% power vs simulated power at N*
checks = []
for r in (0.25, 0.5):
    big = generate(400000, r, rng); rho2 = T.spectrum(big["x_eeg"], big["x_eit"])
    for q in (0.2, 0.5, 1.0):
        n_star = T.sample_size(rho2, q, ALPHA, 0.2)
        if n_star is None or n_star > 3000:
            continue
        ref = np.sort([cca_pillai(*(lambda g: (g["x_eeg"], g["x_eit"]))(generate(n_star, r, rng))) for _ in range(1000)])
        thr = ref[int(np.floor(ALPHA * 1001)) - 1]
        V = [cca_pillai(*(lambda g: (g["x_eeg"], mispair(g["x_eit"], q, rng)[0]))(generate(n_star, r, rng))) for _ in range(400)]
        checks.append({"r": r, "q": q, "theta": float(rho2.sum()), "N_star": n_star,
                       "N_star_weak_formula": float(T.sample_size_weak(rho2.sum(), q, ALPHA, 0.2)),
                       "simulated_power_at_N_star": float(np.mean(np.array(V) < thr))})
out = Path(__file__).parent / "outputs" / "novelty"; out.mkdir(parents=True, exist_ok=True)
(out / "theory_sim.json").write_text(json.dumps({"meta": {"seed": SEED, "alpha": ALPHA, "runtime_s": round(time.time() - t0, 1)},
                                                 "summary": summ, "sample_size_checks": checks, "rows": rows}, indent=2))
print(json.dumps(summ, indent=1)); print(json.dumps(checks, indent=1))
