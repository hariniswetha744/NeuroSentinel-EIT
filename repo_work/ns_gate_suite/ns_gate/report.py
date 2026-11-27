"""results.json -> results_for_paper.md (tables to paste; no hand-typed numbers)."""
from __future__ import annotations


def _f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def write_markdown(R, path):
    L = ["# NeuroSentinel validity-gate outputs (auto-generated; in-silico)\n"]
    if R.get("quick_mode"):
        L.append("**QUICK MODE — not for the paper. Re-run without --quick.**\n")

    fa = R["figure_A"]; m = fa["meta"]
    L.append(f"## Figure A — corruption sweep (r={m['r']}, N={m['n']} per batch, {m['reps']} batches per q)\n")
    L.append("| q | Fusion no gate | Gate calib-CCA | Gate perm-CCA | EEG-only | Block calib | Block perm |")
    L.append("|---|---|---|---|---|---|---|")
    for r in fa["rows"]:
        L.append(f"| {r['q']:.1f} | {_f(r['acc_ungated_fusion'])} ± {_f(r['acc_ungated_fusion_ci95'])} | "
                 f"{_f(r['acc_gated_calib_cca'])} | {_f(r['acc_gated_perm_cca'])} | {_f(r['acc_eeg_only'])} | "
                 f"{_pct(r['block_rate_calib_cca'])}% | {_pct(r['block_rate_perm_cca'])}% |")

    fb = R["figure_B"]
    L.append(f"\n## Figure B — detection power (block rate), {fb['meta']['reps']} batches/cell\n")
    rows = fb["rows"]
    for panel, key, lab in (("N", "n", "N"), ("r", "r", "r")):
        qs = sorted({r["q"] for r in rows if r["panel"] == panel})
        L.append(f"### Panel {panel}: calib-CCA (perm-CCA in brackets)\n")
        L.append(f"| {lab} | " + " | ".join(f"q={q:g}" for q in qs) + " |")
        L.append("|---|" + "---|" * len(qs))
        for v in sorted({r[key] for r in rows if r["panel"] == panel}):
            cells = []
            for q in qs:
                rr = next(r for r in rows if r["panel"] == panel and r[key] == v and r["q"] == q)
                cells.append(f"{_pct(rr['calib_cca'])} ({_pct(rr['perm_cca'])})")
            L.append(f"| {v:g} | " + " | ".join(cells) + " |")
        L.append("")
    qs = sorted({r["q"] for r in rows if r["panel"] == "stat"})
    L.append(f"### Panel stat ({fb['meta']['hsic_reps']} batches/cell): calib-CCA / calib-HSIC / perm-CCA / perm-HSIC\n")
    L.append("| gen | " + " | ".join(f"q={q:g}" for q in qs) + " |")
    L.append("|---|" + "---|" * len(qs))
    for fam in ("A", "B"):
        cells = []
        for q in qs:
            rr = next(r for r in rows if r["panel"] == "stat" and r["family"] == fam and r["q"] == q)
            cells.append("/".join(_pct(rr[k]) for k in ("calib_cca", "calib_hsic", "perm_cca", "perm_hsic")))
        L.append(f"| {fam} | " + " | ".join(cells) + " |")

    t = R["tables_I_II"]
    V = ["ID-only (P1-P4)", "Metadata (P1-P8)", "Full: +P9 calib-CCA", "Full: +P9 perm-CCA", "Full: +P9 calib-HSIC"]
    L.append(f"\n## Tables I & II — block rate per fault (%, Wilson 95% CI), N={t['meta']['n']}, "
             f"{t['meta']['trials']} trials/fault, {t['meta']['clean_trials']} clean trials\n")
    L.append("| Set | Fault | Sev. | Fusion acc. (ungated) | " + " | ".join(V) + " |")
    L.append("|---|---|---|---|" + "---|" * len(V))
    for r in t["rows"]:
        cells = [f"{_pct(r[v])} [{_pct(r[v + ' lo'])}–{_pct(r[v + ' hi'])}]" for v in V]
        L.append(f"| {r['set']} | {r['fault']} | {r['severity']:.3g} | {_f(r['ungated_fusion_acc'])} | "
                 + " | ".join(cells) + " |")

    c = R["figure_C"]
    if c.get("status") == "NOT_RUN":
        L.append("\n## Figure C — NOT_RUN\n")
    else:
        m = c["meta"]
        L.append(f"\n## Figure C — EIT GLRT vs peak (pyEIT; fine {m['fine_mesh']} -> coarse {m['coarse_mesh']}, "
                 f"{m['n_anomalies']} anomalies, {m['n_cal_null']} calibration nulls, {m['n_test_null']} test nulls, "
                 f"alpha={m['alpha']}, peak lambda_rel={m['peak_lambda_rel']})\n")
        L.append("| SNR (dB) | GLRT AUC | Peak AUC | GLRT det. | Peak det. | GLRT FAR (one cal.) | Peak FAR (one cal.) "
                 "| GLRT FAR marginal | Peak FAR marginal | GLRT loc. err | Peak loc. err |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in c["levels"]:
            L.append(f"| {r['snr_db']} | {_f(r['glrt_auc'])} | {_f(r['peak_auc'])} | {_pct(r['glrt_detection'])}% | "
                     f"{_pct(r['peak_detection'])}% | {_pct(r['glrt_far'])}% | {_pct(r['peak_far'])}% | "
                     f"{_pct(r['glrt_far_marginal_mean'])}% | {_pct(r['peak_far_marginal_mean'])}% | "
                     f"{_f(r['glrt_loc_err_median_detected'])} | {_f(r['peak_loc_err_median_detected'])} |")
        L.append("\nNoise-percentage mapping (this pyEIT model only):\n")
        for mm in m["noise_mapping"]:
            L.append(f"- {mm['noise_pct_of_std_vh']}% of std(v_h) -> SNR_dv {mm['snr_dv_reference_anomaly_db']:.1f} dB "
                     f"(reference anomaly), {mm['snr_dv_population_median_db']:.1f} dB (population median)")

    e = R["table_III"]
    L.append("\n## Table III — EEG held-out participants\n")
    if e.get("status") != "RUN":
        L.append(f"NOT_RUN. {e.get('how', '')}\n")
    else:
        L.append(f"Dev: {', '.join(e['dev_participants'])}; held-out: {', '.join(e['heldout_participants'])}\n")
        L.append("| Model | Participants | Mean subj. acc. | 95% CI | Bal. acc. | >0.5 | Wilcoxon p |")
        L.append("|---|---|---|---|---|---|---|")
        for s in e["summary"]:
            L.append(f"| {s['Model']} | {s['participants']} | {_f(s['mean_subject_accuracy'])} | "
                     f"{_f(s['ci95_low'])}–{_f(s['ci95_high'])} | {_f(s['mean_subject_balanced_accuracy'])} | "
                     f"{s['n_above_0.5']} | {_f(s['wilcoxon_p_vs_0.5'], 4)} |")
        p = e["paired_frozen_vs_csp"]
        L.append(f"\nFrozen − CSP: {_f(p['frozen_minus_csp_mean'])} (95% CI {_f(p['ci95'][0])}–{_f(p['ci95'][1])})")

    if "learned_baseline" in R:
        m = R["learned_baseline"]["meta"]
        L.append(f"\n## Learned matched-pair baseline (separate run, seed {m['seed']}; record AUC at q=0.5: {m['record_auc_q0.5_mean']:.2f})\n")
        L.append("| Set | Fault | Sev. | Block % | 95% CI |")
        L.append("|---|---|---|---|---|")
        for r in R["learned_baseline"]["rows"]:
            L.append(f"| {r['set']} | {r['fault']} | {r['severity']:.3g} | {_pct(r['learned_block'])} | {_pct(r['lo'])}–{_pct(r['hi'])} |")

    if "reference_check" in R:
        L.append("\n## Reference-construction check (calib-CCA clean false-block, nominal 5%)\n")
        L.append("| N | Reference | False-block | 95% CI |")
        L.append("|---|---|---|---|")
        for r in R["reference_check"]:
            L.append(f"| {r['n']} | {r['reference']} | {_pct(r['false_block'])}% | {_pct(r['lo'])}–{_pct(r['hi'])}% |")

    T = R["table_IV"]
    L.append(f"\n## Table IV — overhead ({T['environment']})\n")
    L.append("| Component | Batch N | Median ms | µs/record | Peak MB |")
    L.append("|---|---|---|---|---|")
    for r in T["rows"]:
        L.append(f"| {r['component']} | {r['batch_n']} | {r['median_ms']:.3f} | {r['per_record_us']:.2f} | {r['peak_mb']:.3f} |")
    L.append(f"\nTotal runtime: {R['total_runtime_s']:.0f} s")
    path.write_text("\n".join(L), encoding="utf-8")
