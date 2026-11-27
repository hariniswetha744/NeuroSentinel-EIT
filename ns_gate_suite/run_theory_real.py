"""Experiment 8: does the power formula (ns_gate/theory.py) predict the REAL EEG-NIRS block rates?

PRE-DECLARED (written before running on the real data):
  * theta for each half (reference subjects 1-14, test subjects 15-29) = Pillai of the whole
    half minus p*d/n (the bias term of the model), features standardized within subject.
  * Prediction A ("practitioner"): only the reference half is used (theta_ref for both
    reference and test), no finite-pool correction. This is what a user would compute
    before deployment.
  * Prediction B ("two-population"): theta_ref for the threshold, theta_test for test batches,
    and the finite-pool factor (1 - N / n_pool) on each variance, because Table V batches are
    drawn without replacement from 840 reference and 900 test trials.
  * Both are compared with the measured CCA-calibrated block rates in exp7_eeg_nirs.json and
    exp7b_large_batches.json (N = 100, 200, 300, 450; clean, q = 0.5, q = 1). Reported as is.

Usage:  python run_theory_real.py --eeg-root "C:\\hbci\\EEG" --nirs-root "C:\\hbci\\NIRS"
Writes outputs/exp8_theory_real.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from ns_gate import eeg_nirs_real as E
from ns_gate import theory as T
from ns_gate.gate import cca_pillai

ap = argparse.ArgumentParser(); ap.add_argument("--eeg-root", required=True); ap.add_argument("--nirs-root", required=True)
a = ap.parse_args()
OUT = Path(__file__).parent / "outputs"
xe, xn, subj, ses, y, info = E.build_trials(a.eeg_root, a.nirs_root)
xe, xn = E._zscore_within(xe, subj), E._zscore_within(xn, subj)
subjects = np.unique(subj); ref_s = subjects[: len(subjects) // 2]
R = np.isin(subj, ref_s); Tm = ~R
p, d = xe.shape[1], xn.shape[1]
rng = np.random.default_rng(8)


def half(mask):
    a_, b_ = xe[mask], xn[mask]; n = len(a_)
    V = cca_pillai(a_, b_); theta = max(V - p * d / n, 1e-6)
    emp = {}
    for N in (100, 200, 300, 450):
        if N >= n:
            continue
        vals = [cca_pillai(a_[i], b_[i]) for i in (rng.choice(n, N, replace=False) for _ in range(1000))]
        emp[N] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
    return {"n": int(n), "pillai": float(V), "theta": float(theta), "clean_batch_moments": emp}


ref, tst = half(R), half(Tm)
measured = {}
for f in ("exp7_eeg_nirs.json", "exp7b_large_batches.json"):
    fp = OUT / f
    if fp.exists():
        for r_ in json.loads(fp.read_text())["runs"]["subject"]["rows"]:
            if r_["fault"] == "random" and r_["q"] in (0.0, 0.5, 1.0):
                measured[(r_["n"], r_["q"])] = r_
rows = []
for N in (100, 200, 300, 450):
    for q in (0.0, 0.5, 1.0):
        pa = T.power([ref["theta"]], N, q, p=p, d=d)
        pb = T.power([tst["theta"]], N, q, p=p, d=d, rho2_ref=[ref["theta"]], pool_ref=ref["n"], pool_test=tst["n"])
        m = measured.get((N, q))
        rows.append({"N": N, "q": q, "pred_A_practitioner": pa, "pred_B_two_population": pb,
                     "measured": m["cca_calib"] if m else None, "measured_ci": m["cca_calib_ci"] if m else None})
        print(f"N={N:3d} q={q:.1f}  A {pa:.3f}  B {pb:.3f}  measured {m['cca_calib'] if m else float('nan'):.3f}")
res = {"reference_half": ref, "test_half": tst, "p": p, "d": d, "rows": rows,
       "model_moments_check": {str(N): {"ref_model_mean": T.moments([ref["theta"]], N, 0, p, d)[0],
                                        "ref_model_sd_pool": float(np.sqrt(T.moments([ref["theta"]], N, 0, p, d, ref["n"])[1])),
                                        "test_model_mean": T.moments([tst["theta"]], N, 0, p, d)[0],
                                        "test_model_sd_pool": float(np.sqrt(T.moments([tst["theta"]], N, 0, p, d, tst["n"])[1]))}
                               for N in (100, 200, 300, 450)}}
(OUT / "exp8_theory_real.json").write_text(json.dumps(res, indent=2))
print("theta ref %.4f  test %.4f" % (ref["theta"], tst["theta"])); print("saved", OUT / "exp8_theory_real.json")
