"""Sample sizes the power rule (8) requires for the measured real EEG-NIRS dependence.
Reads outputs/exp8_theory_real.json, writes outputs/novelty/real_sample_size.json.
These are model predictions, not measurements."""
import json
from pathlib import Path
from ns_gate import theory as T
out = Path(__file__).parent / "outputs"
e = json.loads((out / "exp8_theory_real.json").read_text())
rows = []
for half in ("test_half", "reference_half"):
    th = e[half]["theta"]
    for q in (1.0, 0.5, 0.25):
        rows.append({"half": half, "theta": th, "q": q, "N_star_power80": T.sample_size([th], q, 0.05, 0.2),
                     "N_star_power90": T.sample_size([th], q, 0.05, 0.1)})
(out / "novelty" / "real_sample_size.json").write_text(json.dumps({"alpha": 0.05, "rows": rows}, indent=2))
print(json.dumps(rows, indent=1))
