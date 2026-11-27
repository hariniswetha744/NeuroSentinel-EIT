"""Recompute the model side of Experiment 8 with the current eq. (7) (ns_gate/theory.py).

Experiment 8 (run_theory_real.py) needs the raw EEG-NIRS files. Its predictions depend on the
data only through theta_ref, theta_test and the pool sizes, which it saved in
outputs/exp8_theory_real.json together with the empirical clean-batch moments. This script keeps
every measured value unchanged and recomputes predictions A and B and the model moments, writing
outputs/exp8_theory_real_v20.json. Rerunning run_theory_real.py on the raw data gives the same
predictions.
"""
import json
from pathlib import Path
from ns_gate import theory as T

out = Path(__file__).parent / "outputs"
e = json.loads((out / "exp8_theory_real.json").read_text())
ref, tst, p, d = e["reference_half"], e["test_half"], e["p"], e["d"]
for r in e["rows"]:
    N, q = r["N"], r["q"]
    r["pred_A_practitioner"] = T.power([ref["theta"]], N, q, p=p, d=d)
    r["pred_B_two_population"] = T.power([tst["theta"]], N, q, p=p, d=d, rho2_ref=[ref["theta"]],
                                         pool_ref=ref["n"], pool_test=tst["n"])
e["model_moments_check"] = {str(N): {
    "ref_model_mean": T.moments([ref["theta"]], N, 0, p, d)[0],
    "ref_model_sd_pool": float(T.moments([ref["theta"]], N, 0, p, d, ref["n"])[1] ** 0.5),
    "test_model_mean": T.moments([tst["theta"]], N, 0, p, d)[0],
    "test_model_sd_pool": float(T.moments([tst["theta"]], N, 0, p, d, tst["n"])[1] ** 0.5)}
    for N in (100, 200, 300, 450)}
e["recomputed"] = {"script": "run_theory_real_recompute.py", "finite_factor": T.FINITE_FACTOR,
                   "note": "measured values unchanged; model predictions recomputed"}
(out / "exp8_theory_real_v20.json").write_text(json.dumps(e, indent=2))
for r in e["rows"]:
    print(f"N={r['N']:3d} q={r['q']:.1f}  A {r['pred_A_practitioner']:.3f}  B {r['pred_B_two_population']:.3f}  measured {r['measured']:.3f}")
