"""Single entry point: regenerates every figure/table output for the paper.

  python run_all.py                       # full run (gate + EIT); EEG Table III skipped
  python run_all.py --eeg-root <EDF dir>  # also runs Table III (needs PhysioNet EEGBCI)
  python run_all.py --quick               # smoke test with few repetitions

Outputs go to ./outputs: CSV/JSON per experiment, figure PDFs/PNGs,
results.json (everything) and results_for_paper.md (tables ready to paste).
Anything that was not run is written as NOT_RUN, never as a number.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ns_gate import experiments as X
from ns_gate import plots


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-eit", action="store_true")
    ap.add_argument("--eeg-root", default=None)
    ap.add_argument("--eeg-download", action="store_true")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    R = {}
    k = 0.1 if a.quick else 1.0
    t_all = time.perf_counter()

    t = time.perf_counter()
    df, meta = X.figure_a(reps=int(500 * k))
    df.to_csv(out / "figure_A_corruption.csv", index=False); plots.figure_a(df, out)
    R["figure_A"] = {"meta": meta, "rows": df.to_dict(orient="records"), "runtime_s": time.perf_counter() - t}
    print("Figure A done", round(time.perf_counter() - t, 1), "s")

    t = time.perf_counter()
    df, meta = X.figure_b(reps=int(400 * k), hsic_reps=int(200 * k))
    df.to_csv(out / "figure_B_power.csv", index=False); plots.figure_b(df, out)
    R["figure_B"] = {"meta": meta, "rows": df.to_dict(orient="records"), "runtime_s": time.perf_counter() - t}
    print("Figure B done", round(time.perf_counter() - t, 1), "s")

    t = time.perf_counter()
    df, meta = X.tables_i_ii(trials=int(300 * k), clean_trials=int(1000 * k))
    df.to_csv(out / "tables_I_II_gate.csv", index=False)
    R["tables_I_II"] = {"meta": meta, "rows": df.to_dict(orient="records"), "runtime_s": time.perf_counter() - t}
    print("Tables I/II done", round(time.perf_counter() - t, 1), "s")

    if not a.skip_eit:
        from ns_gate.eit_glrt import run as eit_run
        t = time.perf_counter()
        levels, rocs, meta = eit_run(n_anom=int(500 * k) or 50, n_null_test=int(1000 * k) or 100,
                                     n_tune=int(100 * k) or 20, roc_snrs=(-10, -5, 0))
        pd.DataFrame(levels).to_csv(out / "figure_C_eit_glrt.csv", index=False)
        plots.figure_c(levels, rocs, out)
        (out / "figure_C_roc_curves.json").write_text(json.dumps(rocs))
        R["figure_C"] = {"meta": meta, "levels": levels, "runtime_s": time.perf_counter() - t}
        print("Figure C done", round(time.perf_counter() - t, 1), "s")
    else:
        R["figure_C"] = {"status": "NOT_RUN"}

    if a.eeg_root:
        from ns_gate import eeg_heldout
        argv = ["--edf-root", a.eeg_root, "--out", str(out)] + (["--download"] if a.eeg_download else [])
        eeg_heldout.main(argv)
        R["table_III"] = json.loads((out / "table3_eeg_heldout.json").read_text())
    else:
        R["table_III"] = {"status": "NOT_RUN",
                          "how": "python run_all.py --eeg-root <EEGBCI 1.0.0 folder> --eeg-download"}

    lb, lmeta = X.learned_baseline(trials=int(300 * k), clean_trials=int(1000 * k))
    lb.to_csv(out / "learned_baseline.csv", index=False)
    R["learned_baseline"] = {"meta": lmeta, "rows": lb.to_dict(orient="records")}
    from ns_gate import plots as P2
    P2.figure_a_print(pd.read_csv(out / "figure_A_corruption.csv"), out)
    P2.figure_b_print(pd.read_csv(out / "figure_B_power.csv"), out)
    if R["figure_C"].get("levels"):
        P2.figure_c_print(R["figure_C"]["levels"], json.loads((out / "figure_C_roc_curves.json").read_text()), out)
    rc = X.reference_check(); rc.to_csv(out / "reference_check.csv", index=False)
    R["reference_check"] = rc.to_dict(orient="records")
    df, env = X.table_iv()
    df.to_csv(out / "table_IV_overhead.csv", index=False)
    R["table_IV"] = {"environment": env, "rows": df.to_dict(orient="records")}
    R["total_runtime_s"] = time.perf_counter() - t_all
    R["quick_mode"] = a.quick
    (out / "results.json").write_text(json.dumps(R, indent=2, default=_json))
    from ns_gate.report import write_markdown
    write_markdown(R, out / "results_for_paper.md")
    print("All outputs in", out.resolve())


if __name__ == "__main__":
    main()
