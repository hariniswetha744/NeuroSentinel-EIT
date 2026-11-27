"""Novelty 2 experiment: gate + repair for shift-type mispairing. Writes outputs/novelty/repair.json."""
import json, time
from pathlib import Path
import numpy as np
from ns_gate.experiments import train_models, wilson
from ns_gate.gate import Reference, cca_pillai
from ns_gate.generator import generate, mispair
from ns_gate import repair as RP

N, R, ALPHA, TRIALS, SEED = 200, 0.5, 0.05, 300, 21
rng = np.random.default_rng(SEED)
t0 = time.time()
fused, eeg = train_models(R, rng)
ref = Reference("cca", N, np.array([cca_pillai(*(lambda g: (g["x_eeg"], g["x_eit"]))(generate(N, R, rng))) for _ in range(1000)]))
thr = ref.threshold(ALPHA)
# Acceptance margin: the repair must beat the chance gain seen on batches that are corrupted
# but NOT shift-structured. These are built from trusted clean pairs by random derangement
# (q = 0.25, 0.5, 1), so no knowledge of the test faults is used. Margin = max over q of the
# (1 - alpha) quantile of the maximal chance gain; the clean-batch gain is reported too.
null_q = (0.25, 0.5, 1.0)
margins = {}
for qq in null_q:
    nb = [(lambda g: (g["x_eeg"], mispair(g["x_eit"], qq, rng)[0]))(generate(N, R, rng)) for _ in range(400)]
    margins[qq], _ = RP.calibrate_margin(nb, ALPHA)
clean = [(lambda g: (g["x_eeg"], g["x_eit"]))(generate(N, R, rng)) for _ in range(400)]
margins[0.0], gains = RP.calibrate_margin(clean, ALPHA)
margin = max(margins.values())
print("margins", margins)


def tail_shift(xi, k, s):  # corruption: rows from k on take payload j + s (s=+1: dropped frame)
    out = xi.copy(); out[k:] = np.roll(xi[k:], -s, axis=0); return out


faults = [("Clean", None)]
for sev in (0.1, 0.25, 0.5, 0.75):
    faults.append((f"Dropped frame (tail {int(sev*100)}%)", ("tail", sev, 1)))
faults.append(("Duplicated frame (tail 50%)", ("tail", 0.5, -1)))
faults.append(("Whole-batch lag of 2", ("tail", 1.0, 2)))
faults.append(("Random mispairing q = 0.5 (control)", ("random", 0.5, None)))

rows = []
for name, spec in faults:
    blk = acc_rep = exact = near = 0
    a_ungated, a_gate, a_repair, a_true = [], [], [], []
    kerr = []
    for _ in range(TRIALS):
        d = generate(N, R, rng); xe, xi, y = d["x_eeg"], d["x_eit"], d["y"]
        k_true = s_true = None
        if spec and spec[0] == "tail":
            k_true = int(round(N * (1 - spec[1]))); s_true = spec[2]
            xc = tail_shift(xi, k_true, s_true)
        elif spec:
            xc, _ = mispair(xi, spec[1], rng)
        else:
            xc = xi
        f = lambda b: float(np.mean(fused.predict(np.hstack([xe, b])) == y))
        a_ungated.append(f(xc)); a_true.append(f(xi))
        e_only = float(np.mean(eeg.predict(xe) == y))
        if cca_pillai(xe, xc) >= thr:
            a_gate.append(f(xc)); a_repair.append(f(xc)); continue
        blk += 1; a_gate.append(e_only)
        ok, xr, k, s, g = RP.try_repair(xe, xc, margin, thr)
        if ok:
            acc_rep += 1; a_repair.append(f(xr))
            if k_true is not None:
                kerr.append(abs(k - k_true) if s == s_true else None)
                if s == s_true:
                    exact += (k == k_true); near += (abs(k - k_true) <= 3)
        else:
            a_repair.append(e_only)
    row = {"fault": name, "trials": TRIALS, "blocked": blk / TRIALS,
           "repaired_of_blocked": acc_rep / blk if blk else None,
           "repaired_of_all": acc_rep / TRIALS,
           "exact_k_s_of_repaired": exact / acc_rep if acc_rep and k_true is not None else None,
           "near_k_s_of_repaired": near / acc_rep if acc_rep and k_true is not None else None,
           "shift_correct_of_repaired": (sum(e is not None for e in kerr) / len(kerr)) if kerr else None,
           "median_k_error_when_shift_correct": (float(np.median([e for e in kerr if e is not None])) if any(e is not None for e in kerr) else None),
           "acc_true_pairing": float(np.mean(a_true)), "acc_ungated": float(np.mean(a_ungated)),
           "acc_gate_fallback": float(np.mean(a_gate)), "acc_gate_repair": float(np.mean(a_repair))}
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

out = Path(__file__).parent / "outputs" / "novelty"; out.mkdir(parents=True, exist_ok=True)
(out / "repair.json").write_text(json.dumps({"meta": {"N": N, "r": R, "alpha": ALPHA, "trials": TRIALS, "seed": SEED,
    "shifts": list(RP.SHIFTS), "margin": margin, "margins_by_null_q": {str(k): v for k, v in margins.items()}, "threshold": thr, "clean_gain_mean": float(np.mean(gains)),
    "runtime_s": round(time.time() - t0, 1)}, "rows": rows}, indent=2))
print("margin", margin, "thr", thr, "runtime", round(time.time() - t0, 1))
