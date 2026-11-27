"""Pilot check for the finite-sample factor in the null variance of eq. (7).

The paper states that the factor (1 - (p + d + 1)/N)^2 in v(q) is an empirical correction set on a
pilot simulation separate from the validation run (run_theory.py, seed 41). The original pilot
output was not saved; this script re-runs a pilot on a separate seed (0) and reports the
prediction error of eq. (8) with and without the factor, so the choice can be reproduced.

Writes outputs/novelty/theory_pilot.json
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from ns_gate.generator import generate, mispair
from ns_gate.gate import cca_pillai
from ns_gate import theory as T

SEED, ALPHA, P, D = 0, 0.05, 4, 8
rng = np.random.default_rng(SEED); t0 = time.time()


def power_nofactor(rho2, N, q):
    def mom(qq):
        rq = np.asarray(rho2) * (1 - qq) ** 2
        return rq.sum() + P * D / N, 4 * np.sum(rq * (1 - rq) ** 2) / N + 2 * P * D / N ** 2
    m0, v0 = mom(0.0); mq, vq = mom(q)
    t = m0 - norm.ppf(1 - ALPHA) * np.sqrt(v0)
    return float(norm.cdf((t - mq) / np.sqrt(vq)))


rows = []
for r in (0.1, 0.25, 0.5, 0.75):
    big = generate(400000, r, rng); rho2 = T.spectrum(big["x_eeg"], big["x_eit"])
    for N in (50, 100, 200, 500):
        ref = np.sort([cca_pillai(*(lambda g: (g["x_eeg"], g["x_eit"]))(generate(N, r, rng))) for _ in range(1000)])
        thr = ref[int(np.floor(ALPHA * 1001)) - 1]
        for q in (0.0, 0.25, 0.5, 1.0):
            V = []
            for _ in range(400):
                g = generate(N, r, rng); xi, _ = mispair(g["x_eit"], q, rng)
                V.append(cca_pillai(g["x_eeg"], xi))
            rows.append({"r": r, "N": N, "q": q, "empirical": float(np.mean(np.array(V) < thr)),
                         "pred_with_factor": T.power(rho2, N, q, ALPHA),
                         "pred_without_factor": power_nofactor(rho2, N, q)})

summ = {}
for key in ("pred_with_factor", "pred_without_factor"):
    e = np.array([abs(x[key] - x["empirical"]) for x in rows])
    summ[key] = {"mae": float(e.mean()), "max": float(e.max()), "within_10pts": float(np.mean(e <= 0.10))}
out = Path(__file__).parent / "outputs" / "novelty"; out.mkdir(parents=True, exist_ok=True)
(out / "theory_pilot.json").write_text(json.dumps({"meta": {"seed": SEED, "alpha": ALPHA, "family": "A",
    "note": "re-run of the pilot; the original pilot output was not saved",
    "runtime_s": round(time.time() - t0, 1)}, "summary": summ, "rows": rows}, indent=2))
for k, v in summ.items():
    print(f"{k:22s} MAE {100*v['mae']:.1f} pts  max {100*v['max']:.1f}  within 10 pts {100*v['within_10pts']:.1f}%")
