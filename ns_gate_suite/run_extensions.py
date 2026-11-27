"""Run the v11 extension experiments and save every result to outputs/extensions/."""
import json, sys, time
from pathlib import Path
from ns_gate import extensions as X

out = Path(__file__).parent / "outputs" / "extensions"; out.mkdir(parents=True, exist_ok=True)
quick = "--quick" in sys.argv
for name, fn, kw in [("e1_record_flagging", X.e1_record_flagging, {"batches": 30} if quick else {}),
                     ("e2_sequential", X.e2_sequential, {"streams": 40} if quick else {}),
                     ("e3_combined", X.e3_combined, {"trials": 40, "clean_trials": 100} if quick else {})]:
    t0 = time.time(); df, meta = fn(**kw); meta["runtime_s"] = round(time.time() - t0, 1)
    if not quick:
        df.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.json").write_text(json.dumps({"meta": meta, "rows": df.to_dict("records")}, indent=2))
    print(f"== {name} ({meta['runtime_s']} s)"); print(df.to_string(index=False))
