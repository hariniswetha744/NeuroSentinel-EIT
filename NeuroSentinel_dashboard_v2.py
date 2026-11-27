
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EIT = ROOT / "eit" / "results"
EIT_AI = ROOT / "eit" / "synthetic_experiment" / "ai_results"
st.set_page_config(
    page_title="NeuroSentinel Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroSentinel")
st.subheader("EEG Analysis and EIT Validation")
st.caption(
    "Research prototype | Displays saved project results. "
    "Simulation results are not clinical diagnoses."
)

def load_csv(filename):
    path = DATA / filename
    if not path.exists():
        path = EIT / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as error:
            st.error(f"Could not read {filename}: {error}")
    return None

def show_image(filename, folder):
    path = folder / filename
    if path.exists():
        st.image(str(path), caption=filename, width="stretch")
    else:
        st.warning(f"Image not found: {path}")

def show_table(filename):
    df = load_csv(filename)
    st.markdown(f"**{filename}**")
    if df is None:
        st.info("File not found in the expected project folders.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            f"Download {filename}",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}"
        )

overview, paper, extra, eeg, eit, files, fusion, window_comparison = st.tabs([
    "Overview", "Paper Results (IEEE v11)", "Extensions & Real-Data Check", "EEG Analysis", "EIT Validation",
    "Data Tables", "Fusion Readiness", "EEG Window Comparison"
])

PAPER = ROOT / "paper_results"
PAPER_FIG = PAPER / "figures"


def paper_csv(name):
    path = PAPER / name
    return pd.read_csv(path) if path.exists() else None


def paper_json(name):
    import json
    path = PAPER / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def paper_figure(name, caption):
    path = PAPER_FIG / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Figure not found: {path}")


with paper:
    st.header("Silent Mispairing in EEG–EIT Fusion: Paper Results")
    st.info(
        "Results behind the IEEE manuscript (v11). Pairing experiments use a synthetic "
        "generator, the EIT detector uses simulated pyEIT frames, and the EEG test uses "
        "public PhysioNet data. Software-only; no clinical claim. Every number below is "
        "read from the files in paper_results/, produced by ns_gate_suite/run_all.py."
    )
    fig_a = paper_csv("figure_A_corruption.csv")
    gate = paper_csv("tables_I_II_gate.csv")
    eeg_ho = paper_json("table3_eeg_heldout.json")
    if fig_a is None or gate is None:
        st.error("paper_results/ is missing. Run ns_gate_suite/run_all.py and copy its outputs.")
    else:
        q0 = fig_a.loc[fig_a.q == 0.0].iloc[0]
        q1 = fig_a.loc[fig_a.q == 1.0].iloc[0]
        f7 = gate[gate.fault.str.startswith("F7")].set_index("severity")
        clean = gate[gate.set == "clean"].iloc[0]
        m = st.columns(4)
        m[0].metric("Fusion accuracy, q = 0 → 1", f"{q0.acc_ungated_fusion:.3f} → {q1.acc_ungated_fusion:.3f}")
        m[1].metric("Metadata checks: mispaired batches blocked", f"{100 * f7.loc[1.0, 'Metadata (P1-P8)']:.1f}%")
        m[2].metric("Pairing gate (calib-CCA) at q = 1", f"{100 * f7.loc[1.0, 'Full: +P9 calib-CCA']:.1f}%")
        m[3].metric("Clean batches wrongly blocked", f"{100 * clean['Full: +P9 calib-CCA']:.1f}%")

        st.subheader("1. What mispairing is")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            paper_figure("figure_mispairing.png", "Fig. 3: IDs match but payloads are swapped; the gate sees the lost cross-modal link")
        with c2:
            paper_figure("figure_1_gate.png", "Fig. 1: pre-fusion validity gate (P1–P8 metadata, P9 pairing dependence)")
        paper_figure("figure_concept.png", "Measurement concept and what the study did (conceptual drawing)")

        st.subheader("2. Fusion fails silently (Fig. 4)")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            paper_figure("figure_A_corruption_print.png", "Accuracy and block rate vs mispaired fraction q")
        with c2:
            cols = {"q": "q", "acc_ungated_fusion": "Fusion, no gate", "acc_gated_calib_cca": "With gate",
                    "acc_eeg_only": "EEG only", "acc_mispaired_records": "Mispaired records",
                    "block_rate_calib_cca": "Blocked (calib-CCA)"}
            st.dataframe(fig_a[[c for c in cols if c in fig_a]].rename(columns=cols).round(3),
                         hide_index=True, width="stretch")

        st.subheader("3. Gate versus baselines (Tables I–II)")
        variants = ["ID-only (P1-P4)", "Metadata (P1-P8)", "Full: +P9 calib-CCA", "Full: +P9 perm-CCA",
                    "Full: +P9 calib-HSIC"]
        shown = gate[["set", "fault", "severity", "ungated_fusion_acc"] + variants].copy()
        for v in variants:
            shown[v] = (100 * shown[v]).round(1)
        learned = paper_csv("learned_baseline.csv")
        if learned is not None:
            key = learned.set_index(["fault", "severity"])["learned_block"]
            shown["Learned matched-pair (%)"] = [
                round(100 * key[(f, s)], 1) if (f, s) in key.index else None
                for f, s in zip(shown.fault, shown.severity)]
        st.caption("Block rate (%) per fault; N = 200 records per batch. The learned baseline is a separate seeded run.")
        shown["ungated_fusion_acc"] = shown["ungated_fusion_acc"].round(3)
        st.dataframe(shown.rename(columns={"ungated_fusion_acc": "Fusion acc. (no gate)"}),
                     hide_index=True, width="stretch")

        st.subheader("4. Detection power (Fig. 5)")
        paper_figure("figure_B_power_print.png", "Power vs q by batch size, redundancy and statistic")
        st.caption("At redundancy r = 0 the two modalities share no information, so no dependence test can detect mispairing.")

        st.subheader("5. EIT detector: GLRT vs peak (Fig. 6)")
        c1, c2 = st.columns([1, 1.4])
        with c1:
            paper_figure("figure_C_eit_glrt_print.png", "ROC and detection at a 5% false-alarm threshold")
        with c2:
            eit_tab = paper_csv("figure_C_eit_glrt.csv")
            if eit_tab is not None:
                keep = ["snr_db", "glrt_auc", "peak_auc", "glrt_detection", "peak_detection",
                        "glrt_far_marginal_mean", "peak_far_marginal_mean"]
                st.dataframe(eit_tab[keep].round(3), hide_index=True, width="stretch")
        paper_figure("figure_eit_steps.png", "EIT step by step: true change → electrode data → image → detection map (simulation)")

        st.subheader("6. EEG on unseen participants (Table III)")
        if eeg_ho is None:
            st.info("table3_eeg_heldout.json not found.")
        else:
            summ = pd.DataFrame(eeg_ho["summary"])
            st.dataframe(summ[["Model", "participants", "mean_subject_accuracy", "ci95_low", "ci95_high",
                               "wilcoxon_p_vs_0.5"]].round(3), hide_index=True, width="stretch")
            frozen = summ.iloc[0]
            confirmed = frozen.ci95_low > 0.5
            pc = eeg_ho["paired_frozen_vs_csp"]
            st.markdown(
                f"Held-out participants: {eeg_ho['heldout_participants'][0]}–{eeg_ho['heldout_participants'][-1]} "
                f"({len(eeg_ho['heldout_participants'])}). The 95% CI {'excludes' if confirmed else 'includes'} 0.5, "
                f"so the development accuracy of 0.573 was {'confirmed' if confirmed else 'not confirmed'}. "
                f"Frozen minus CSP+LDA: {pc['frozen_minus_csp_mean']:.3f} "
                f"(95% CI {pc['ci95'][0]:.3f} to {pc['ci95'][1]:.3f})."
            )
            paper_figure("figure_2_acquisition.png", "Fig. 2: EEG electrode layout, simulated EIT anomaly, GLRT map, dashboard")

        st.subheader("7. Cost (Table IV)")
        ov = paper_csv("table_IV_overhead.csv")
        if ov is not None:
            st.dataframe(ov.round(3), hide_index=True, width="stretch")

        md = PAPER / "results_for_paper.md"
        if md.exists():
            st.download_button("Download all paper tables (Markdown)", md.read_bytes(),
                               file_name="results_for_paper.md", mime="text/markdown")


EXT = PAPER / "extensions"


def ext_json(name):
    import json
    path = EXT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


with extra:
    st.header("Extensions after v10: addressing the stated limitations")
    st.info(
        "Four follow-up experiments. E1–E3 use the same synthetic generator as the paper "
        "(N = 200 records per batch, r = 0.5); Experiment 7 uses real, simultaneously recorded "
        "EEG + NIRS. Every number is read from files in paper_results/extensions/ and "
        "paper_results/exp7_eeg_nirs.json. E2, E3 and Experiment 7 are summarized in manuscript v11 (Section IV-I, Table V, Section V)."
    )

    st.subheader("E2. Accumulating evidence over successive batches")
    e2 = ext_json("e2_sequential.json")
    if e2 is None:
        st.info("e2_sequential.json not found. Run ns_gate_suite/run_extensions.py.")
    else:
        d = pd.DataFrame(e2["rows"])
        piv = d.pivot(index="records_seen", columns="q", values="block_rate")
        piv.columns = [("clean (false block)" if q == 0 else f"{int(round(100 * q))}% mispaired") for q in piv.columns]
        st.line_chart(100 * piv, x_label="Records seen (batches of 200)", y_label="Batches blocked (%)")
        last = d[d.k_batches == d.k_batches.max()].set_index("q")
        m = st.columns(3)
        m[0].metric(f"20% mispaired, after {int(last.loc[0.2, 'records_seen'])} records",
                    f"{100 * last.loc[0.2, 'block_rate']:.1f}%")
        m[1].metric("10% mispaired, same", f"{100 * last.loc[0.1, 'block_rate']:.1f}%")
        m[2].metric("Clean data wrongly blocked", f"{100 * last.loc[0.0, 'block_rate']:.1f}%")
        st.caption("Fisher combination of the conformal CCA p-values of k batches, with k fixed in advance. "
                   "Repeatedly testing after every batch would raise the false-alarm rate.")

    st.subheader("E3. One rule combining CCA and the learned scorer")
    e3 = ext_json("e3_combined.json")
    if e3 is None:
        st.info("e3_combined.json not found.")
    else:
        d = pd.DataFrame(e3["rows"])
        for c in ("cca", "learned", "combined"):
            d[c] = (100 * d[c]).round(1)
        st.dataframe(d.rename(columns={"case": "Fault", "severity": "Severity", "trials": "Batches",
                                       "cca": "CCA (%)", "learned": "Learned (%)", "combined": "Combined (%)"}),
                     hide_index=True, width="stretch")
        st.caption("Block if min(p_CCA, p_learned) ≤ α/2. The combined rule removes CCA's stale-buffer and "
                   "electrode blind spots and the learned scorer's within-subject blind spot. "
                   "Class-conditional swaps stay undetected by every rule.")

    st.subheader("E1. Which records are mispaired?")
    e1 = ext_json("e1_record_flagging.json")
    if e1 is None:
        st.info("e1_record_flagging.json not found.")
    else:
        d = pd.DataFrame(e1["rows"])
        d = d[d.q > 0][["r", "q", "record_auc", "recall", "precision_when_flagged", "batches_with_flags"]]
        st.dataframe((d.assign(recall=100 * d.recall, precision_when_flagged=100 * d.precision_when_flagged,
                               batches_with_flags=100 * d.batches_with_flags).round(3)
                      .rename(columns={"r": "Redundancy r", "q": "Mispaired fraction", "record_auc": "Record AUC",
                                       "recall": "Mispaired records found (%)",
                                       "precision_when_flagged": "Flags that are correct (%)",
                                       "batches_with_flags": "Batches with any flag (%)"})),
                     hide_index=True, width="stretch")
        st.caption(f"Per-record conformal p-values with Benjamini–Hochberg at FDR {e1['meta']['fdr']}. "
                   "Flags are mostly correct, but most mispaired records are not found: a single record "
                   "carries too little information. The gate decides per batch.")

    st.subheader("Experiment 7. Real simultaneous EEG + NIRS (29 subjects)")
    e7 = paper_json("exp7_eeg_nirs.json")
    if e7 is None:
        st.warning("exp7_eeg_nirs.json not found. Run ns_gate_suite/run_eeg_nirs.py (see RUN_EXP7.md) "
                   "and copy outputs/exp7_eeg_nirs.json into paper_results/.")
    else:
        run = e7["runs"]["subject"]
        e7b = paper_json("exp7b_large_batches.json")
        rows = list(run["rows"]) + (list(e7b["runs"]["subject"]["rows"]) if e7b else [])
        st.markdown(
            f"**Data:** {e7['dataset']}. {run['n_trials']} trials from {run['n_subjects']} subjects; "
            f"subjects {run['reference_subjects'][0]}–{run['reference_subjects'][-1]} set the reference, "
            f"subjects {run['test_subjects'][0]}–{run['test_subjects'][-1]} are the test set (all rates below)."
        )
        m = st.columns(2)
        m[0].metric("EEG–NIRS dependence on correct pairs (Pillai trace, test half)", f"{run['test_half_pillai']:.3f}")
        m[1].metric("Permutation p-value for that dependence", f"{run['test_half_perm_p']:.3f}")
        d = pd.DataFrame(rows)
        st.markdown("**Calibrated CCA block rate (%) by batch size (Table V of the paper)**")
        tv = pd.DataFrame({
            "Batch size N": sorted(d.n.unique()),
        })
        def cell(n, fault, q, col="cca_calib"):
            r = d[(d.n == n) & (d.fault == fault) & (d.q == q)]
            return round(100 * float(r[col].iloc[0]), 1) if len(r) else None
        tv["Clean (false block)"] = [cell(n, "random", 0.0) for n in tv["Batch size N"]]
        tv["50% mispaired"] = [cell(n, "random", 0.5) for n in tv["Batch size N"]]
        tv["100% mispaired"] = [cell(n, "random", 1.0) for n in tv["Batch size N"]]
        tv["Within-subject swap"] = [cell(n, "within-subject", 1.0) for n in tv["Batch size N"]]
        st.dataframe(tv, hide_index=True, width="stretch")
        st.line_chart(tv.set_index("Batch size N")[["Clean (false block)", "100% mispaired", "Within-subject swap"]],
                      x_label="Batch size N", y_label="Batches blocked (%)")
        st.markdown("**All statistics, all faults (features standardized within subject)**")
        full = d.copy()
        for c in ("cca_calib", "cca_perm", "learned_calib", "combined"):
            full[c] = (100 * full[c]).round(1)
        st.dataframe(full[["n", "fault", "q", "cca_calib", "cca_perm", "learned_calib", "combined"]]
                     .rename(columns={"n": "N", "fault": "Fault", "cca_calib": "CCA calib.",
                                      "cca_perm": "CCA perm. (no dependence found)", "learned_calib": "Learned calib.",
                                      "combined": "Combined"}),
                     hide_index=True, width="stretch")
        lc = d[(d.fault == "random") & (d.q == 0.0)].learned_calib
        st.warning(
            f"Finding: real EEG–NIRS trial features share only weak information, so detection rises with batch size "
            f"but stays modest (up to {100 * d[(d.fault == 'random') & (d.q == 1.0)].cca_calib.max():.1f}% of fully "
            f"mispaired batches). The calibrated CCA rule keeps clean false blocks low on unseen subjects, while the "
            f"learned scorer does not transfer across people (clean false blocks {100 * lc.min():.0f}–{100 * lc.max():.0f}%). "
            "Before using the gate, measure the dependence on a trusted paired subset from the same population."
        )

with overview:
    st.header("Project Overview")
    st.info(
        "The 'Paper Results' tab holds the results reported in the IEEE manuscript (v10); the "
        "'Extensions & Real-Data Check' tab holds the follow-up experiments. "
        "The other tabs show earlier archived project results (EIDORS EIT, initial "
        "fusion benchmark), which the manuscript does not use."
    )

    st.subheader("Project status")
    st.markdown(
        "**Done**\n"
        "- IEEE manuscript v11 (regular and double-blind), 6 pages, IEEE A4 conference template; "
        "29 references, all after 2010 except the two PhysioNet dataset citations, numbered in order of first citation.\n"
        "- Pairing-validity gate (P1–P9), learned matched-pair baseline, EIT GLRT detector, held-out EEG test (Table III).\n"
        "- Extensions: sequential testing (E2), combined CCA + learned rule (E3), record-level flagging (E1).\n"
        "- Experiment 7: gate run on real simultaneous EEG + NIRS from 29 subjects (batch sizes 100–450), reported in v11 Section IV-I and Table V.\n\n"
        "**Open**\n"
        "- Review v11 with the guide.\n"
        "- IEEE requires an AI-use statement in the Acknowledgment; v11 has none (removed on request)."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("EEG")
        features = load_csv("NeuroSentinel_MultiSubject_EEG_Features.csv")
        summary = load_csv("NeuroSentinel_EEG_Summary.csv")

        st.metric(
            "Feature rows",
            len(features) if features is not None else "Unavailable"
        )
        st.metric(
            "Summary file",
            "Available" if summary is not None else "Unavailable"
        )

    with c2:
        st.subheader("Electrical Impedance Tomography")
        metrics = load_csv("EIT_Reproducible_Validation_Metrics.csv")
        noise = load_csv("EIT_Recalculated_Noise_Summary_Corrected.csv")

        st.metric(
            "Validation metrics",
            "Available" if metrics is not None else "Unavailable"
        )
        st.metric(
            "Corrected noise summary",
            "Available" if noise is not None else "Unavailable"
        )

    st.divider()
    st.subheader("EEG Band Power")
    show_image("NeuroSentinel_EEG_Band_Power.png", DATA)

    st.subheader("EIT Reconstruction")
    show_image("EIT_Reconstruction.png", EIT)


with eeg:
    st.header("EEG Feature Analysis")

    st.subheader("Original EEG Band Power")
    show_image("NeuroSentinel_EEG_Band_Power.png", DATA)
    show_table("NeuroSentinel_EEG_Summary.csv")
    show_table("NeuroSentinel_MultiSubject_EEG_Features.csv")

    st.divider()
    st.subheader("Task-Specific EEG Analysis")
    st.caption(
        "S001R04 motor-task recording. "
        "T1/T2 are task labels, not tumour diagnoses."
    )

    show_image(
        "NeuroSentinel_EEG_TaskBandPower_Comparison.png",
        DATA
    )

    show_table(
        "NeuroSentinel_EEG_TaskComparison_Statistics.csv"
    )

    show_table(
        "NeuroSentinel_EEG_TaskBandPower_S001R04.csv"
    )
    st.divider()
    st.subheader("Exploratory EEG Task Classification")

    st.warning(
        "Exploratory evaluation only: 15 task windows from one "
        "participant. T1/T2 are motor-task labels, not tumour labels. "
        "These results do not demonstrate tumour detection or recurrence prediction."
    )

    show_table("NeuroSentinel_EEG_TaskModel_Metrics.csv")
    show_table("NeuroSentinel_EEG_TaskModel_Predictions.csv")

    st.divider()
    st.subheader("Multi-Participant EEG Task Classification")
    st.warning(
        "Exploratory motor-task classification using 10 participants and 150 windows. "
        "T1/T2 are motor-task labels, not tumour diagnoses. These results do not "
        "demonstrate tumour detection or recurrence prediction."
    )

    multi_metrics = load_csv("NeuroSentinel_MultiSubject_EEG_Grouped_Metrics.csv")
    if multi_metrics is not None and not multi_metrics.empty:
        st.write("Grouped evaluation: each participant was held out in turn.")
        st.dataframe(multi_metrics, width="stretch")
        metric_cols = [c for c in ["Accuracy", "Balanced_Accuracy", "Macro_F1"] if c in multi_metrics.columns]
        if "Model" in multi_metrics.columns and metric_cols:
            chart_data = multi_metrics.set_index("Model")[metric_cols].T
            st.bar_chart(chart_data)
    else:
        st.info("Grouped EEG metrics CSV is not available yet.")

    participant_metrics = load_csv("NeuroSentinel_MultiSubject_EEG_Participant_Metrics.csv")
    if participant_metrics is not None and not participant_metrics.empty:
        st.subheader("Participant-Level Metrics")
        st.dataframe(participant_metrics, width="stretch")
    else:
        st.info("Participant-level metrics CSV is not available yet.")

    show_table("NeuroSentinel_MultiSubject_EEG_Grouped_Confusion.csv")
    show_table("NeuroSentinel_MultiSubject_EEG_Grouped_Predictions.csv")
    st.divider()
    st.subheader("C3/C4 Mu-Beta EEG Classification")

    st.warning(
        "Exploratory leave-one-participant-out motor-task classification "
        "using C3/C4 Mu (8–13 Hz) and Beta (13–30 Hz) band-power features. "
        "T1/T2 are motor-task labels, not tumour diagnoses. "
        "These results do not demonstrate clinical detection or recurrence prediction."
    )

    mu_beta_metrics = load_csv(
        "NeuroSentinel_C3C4_MuBeta_LOSO_Metrics.csv"
    )

    if mu_beta_metrics is not None and not mu_beta_metrics.empty:
        st.write(
            "Evaluation: 150 windows from 10 participants, "
            "with each participant held out in turn."
        )
        st.dataframe(
            mu_beta_metrics,
            width="stretch",
            hide_index=True
        )

        metric_columns = [
            column for column in
            ["Accuracy", "Balanced_Accuracy", "Macro_F1"]
            if column in mu_beta_metrics.columns
        ]

        if "Model" in mu_beta_metrics.columns and metric_columns:
            st.subheader("C3/C4 Model Performance")
            st.bar_chart(
                mu_beta_metrics.set_index("Model")[metric_columns]
            )
    else:
        st.info("C3/C4 Mu-Beta metrics file is unavailable.")

    st.subheader("C3/C4 Confusion Matrix")
    show_table("NeuroSentinel_C3C4_MuBeta_LOSO_Confusion.csv")

    st.subheader("C3/C4 Feature Dataset")
    show_table("NeuroSentinel_C3C4_MuBeta_3s_Features.csv")

    st.subheader("C3/C4 LOSO Predictions")
    show_table("NeuroSentinel_C3C4_MuBeta_LOSO_Predictions.csv")
with eit:
    st.header("EIT Reconstruction and Validation")
    st.warning(
        "These are saved simulation/phantom validation outputs. "
        "They do not establish clinical tumour detection or recurrence performance."
    )

    c1, c2 = st.columns(2)

    with c1:
        show_image("EIT_True_Anomaly.png", EIT)
        show_image("EIT_Reconstruction.png", EIT)
        show_image("EIT_Noise_Localization_Error.png", EIT)

    with c2:
        show_image("EIT_Noisy_Reconstruction.png", EIT)
        show_image("EIT_Noise_Success_Rate.png", EIT)

    st.subheader("Corrected Noise Robustness")
    show_table("EIT_Recalculated_Noise_Summary_Corrected.csv")

    st.subheader("Validation Metrics")
    show_table("EIT_Reproducible_Validation_Metrics.csv")

    st.subheader("Repeated Noise Trials")
    show_table("EIT_Repeated_Noise_Trials.csv")
    
    st.divider()
    st.header("Synthetic EIT AI Severity Estimation")

    st.warning(
        "Synthetic simulation experiment only. Severity represents "
        "simulated conductivity change, not tumour severity or recurrence."
    )

    metrics_file = EIT_AI / "EIT_Severity_Model_Metrics.txt"
    noise_file = EIT_AI / "EIT_Noise_Wise_AI_Evaluation.csv"
    predictions_file = EIT_AI / "EIT_Severity_Predictions_With_Clipping.csv"

    if metrics_file.exists():
        st.subheader("Model Configuration and Overall Metrics")
        st.code(metrics_file.read_text(encoding="utf-8"))

    if noise_file.exists():
        st.subheader("Noise-wise AI Performance")
        noise_df = pd.read_csv(noise_file)
        st.dataframe(noise_df, width="stretch", hide_index=True)

        st.download_button(
            "Download Noise-wise Evaluation",
            data=noise_df.to_csv(index=False).encode("utf-8"),
            file_name="EIT_Noise_Wise_AI_Evaluation.csv",
            mime="text/csv",
            key="download_eit_ai_noise"
        )

    if predictions_file.exists():
        st.subheader("True vs Predicted Severity")
        pred_df = pd.read_csv(predictions_file)

        st.scatter_chart(
            pred_df,
            x="True_Severity",
            y="Clipped_Prediction",
            color="Noise_Percent"
        )

        st.dataframe(pred_df, width="stretch", hide_index=True)

    if not all(p.exists() for p in [
        metrics_file, noise_file, predictions_file
    ]):
        st.info("One or more synthetic EIT AI result files are unavailable.")

    st.caption(
        "Evaluation uses held-out synthetic severity levels. "
        "Scenarios derive from one underlying EIT configuration and "
        "do not represent independent subjects or clinical validation."
    )
with files:
    st.header("Project Data Tables")

    csv_files = sorted(DATA.glob("*.csv")) + sorted(EIT.glob("*.csv"))

    if csv_files:
        selected = st.selectbox(
            "Choose a CSV file",
            csv_files,
            format_func=lambda path: str(path.relative_to(ROOT))
        )

        try:
            df = pd.read_csv(selected)
            st.dataframe(df, width="stretch", hide_index=True)

            st.download_button(
                "Download selected CSV",
                data=selected.read_bytes(),
                file_name=selected.name,
                mime="text/csv"
            )
        except Exception as error:
            st.error(f"Could not load file: {error}")
    else:
        st.info("No CSV files found.")



with fusion:
    import json

    st.header("EEG–EIT Fusion Readiness")
    st.caption(
        "Software readiness checks only — not a clinical diagnostic tool."
    )

    report_path = (
        ROOT / "fusion" / "NeuroSentinel_Fusion_Readiness.json"
    )

    if report_path.exists():
        try:
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )

            status = report.get("status", "UNKNOWN")
            st.metric("Clinical fusion readiness", status)

            c1, c2, c3 = st.columns(3)
            c1.metric("EEG windows", report.get("eeg_windows", "N/A"))
            c2.metric("EEG participants", report.get("eeg_participants", "N/A"))
            c3.metric("EIT feature rows", report.get("eit_feature_rows", "N/A"))

            st.subheader("Readiness checks")
            checks = {
                "Paired sample IDs verified":
                    report.get("paired_sample_ids_verified", False),
                "Verified tumour labels available":
                    report.get("verified_tumour_labels_available", False),
                "Longitudinal recurrence labels available":
                    report.get("longitudinal_recurrence_labels_available", False),
                "Clinical fusion training ready":
                    report.get("clinical_fusion_training_ready", False),
            }
            st.table([
                {"Check": name, "Result": "YES" if value else "NO"}
                for name, value in checks.items()
            ])

            st.subheader("Warnings")
            for warning in report.get("warnings", []):
                st.warning(warning)

        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not read the readiness report: {error}")
    else:
        st.warning("Fusion readiness report not found.")

    st.subheader("Fusion safety tests")
    st.write(
        "The interface and synthetic matching tests have been run "
        "separately in Command Prompt. This dashboard does not rerun "
        "those tests automatically."
    )
    st.code(
        "python src\\fusion\\fusion_interface.py\n"
        "python src\\fusion\\test_fusion_interface.py",
        language="bat"
    )

    st.info(
        "Synthetic matching tests verify software behaviour only. "
        "Current motor-task EEG and simulated EIT data are not paired "
        "clinical samples and cannot establish tumour detection "
        "or recurrence prediction."
    )

    st.subheader('Verified EEG-EIT Pairing Gate')
    pairing_report_path = ROOT / 'fusion' / 'synthetic_integration' / 'Fusion_Pairing_Gate_Report.txt'
    if pairing_report_path.is_file():
        pairing_text = pairing_report_path.read_text(encoding='utf-8', errors='replace')
        decision_line = next((line for line in pairing_text.splitlines() if line.startswith('CURRENT DECISION:')), '')
        decision = decision_line.partition(':')[2].strip().upper() if decision_line else 'UNKNOWN'
        st.metric('Verified pairing decision', decision)
        if decision == 'BLOCKED':
            st.error('BLOCKED: No verified shared Sample_ID exists. EEG-EIT fusion training is not permitted.')
        elif decision == 'ELIGIBLE':
            st.success('Pairing gate reports ELIGIBLE. Review the underlying validation evidence before proceeding.')
        else:
            st.warning('Pairing decision could not be determined from the report.')
        st.code(pairing_text, language='text')
    else:
        st.warning('Fusion pairing gate report not found: ' + str(pairing_report_path))
        st.divider()

    st.header("Synthetic EEG-EIT Fusion: Repeated-CV Results")

    st.warning(
        "SYNTHETIC DATA ONLY: All fusion observations, modality features "
        "and targets in this benchmark are artificial. These results "
        "are not evidence of biological or clinical EEG-EIT integration."
    )

    repeated_cv_dir = (
        ROOT / "fusion" / "synthetic_integration"
        / "synthetic_repeated_cv_20261111"
    )
    repeated_summary_path = (
        repeated_cv_dir / "Repeated_CV_Summary_Metrics.csv"
    )
    repeated_report_path = (
        repeated_cv_dir / "Repeated_CV_Report.txt"
    )

    if repeated_summary_path.is_file():
        try:
            repeated_metrics = pd.read_csv(repeated_summary_path)

            required_columns = [
                "Model",
                "Accuracy_Mean",
                "Accuracy_SD",
                "Balanced_Accuracy_Mean",
                "Balanced_Accuracy_SD",
                "Macro_F1_Mean",
                "Macro_F1_SD",
                "ROC_AUC_Mean",
                "ROC_AUC_SD",
                "Fold_Count",
            ]

            missing_columns = [
                col for col in required_columns
                if col not in repeated_metrics.columns
            ]

            if missing_columns:
                st.error(
                    "Repeated-CV summary is missing columns: "
                    + ", ".join(missing_columns)
                )
            else:
                expected_models = {
                    "Majority_baseline",
                    "EEG_only",
                    "EIT_only",
                    "Feature_level_fusion",
                    "Decision_level_fusion",
                }
                actual_models = set(repeated_metrics["Model"])

                if actual_models != expected_models:
                    st.error(
                        "Model names do not match the validated "
                        "five-model benchmark."
                    )
                else:
                    st.caption(
                        "Evaluation: stratified 5-fold cross-validation "
                        "repeated 5 times (25 folds). The same 1,000 "
                        "synthetic observations are reused across folds."
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Synthetic observations", "1,000")
                    c2.metric("Evaluation folds", "25")
                    c3.metric("CV design", "5 folds x 5 repeats")

                    display_columns = [
                        "Model",
                        "Accuracy_Mean",
                        "Accuracy_SD",
                        "Balanced_Accuracy_Mean",
                        "Macro_F1_Mean",
                        "Macro_F1_SD",
                        "ROC_AUC_Mean",
                        "ROC_AUC_SD",
                        "Fold_Count",
                    ]

                    st.subheader("Repeated-CV model comparison")
                    st.dataframe(
                        repeated_metrics[display_columns],
                        width="stretch",
                        hide_index=True,
                    )

                    chart_data = repeated_metrics[
                        ["Model", "Accuracy_Mean", "Macro_F1_Mean"]
                    ].set_index("Model")

                    st.subheader("Mean accuracy and macro-F1")
                    st.bar_chart(chart_data)

                    feature_row = repeated_metrics[
                        repeated_metrics["Model"]
                        == "Feature_level_fusion"
                    ]
                    decision_row = repeated_metrics[
                        repeated_metrics["Model"]
                        == "Decision_level_fusion"
                    ]

                    if not feature_row.empty and not decision_row.empty:
                        feature = feature_row.iloc[0]
                        decision = decision_row.iloc[0]

                        a, b = st.columns(2)
                        a.metric(
                            "Feature-level mean accuracy",
                            f"{feature['Accuracy_Mean']:.2%}",
                            delta=(
                                f"SD {feature['Accuracy_SD']:.4f}"
                            ),
                        )
                        b.metric(
                            "Decision-level mean accuracy",
                            f"{decision['Accuracy_Mean']:.2%}",
                            delta=(
                                f"SD {decision['Accuracy_SD']:.4f}"
                            ),
                        )

                    st.info(
                        "Interpretation: feature-level fusion has a "
                        "small numerical lead in this synthetic benchmark. "
                        "The observed difference does not establish "
                        "statistical superiority."
                    )

                    st.warning(
                        "The 25 folds reuse the same synthetic dataset. "
                        "Fold standard deviations describe variation "
                        "across partitions, not independent experimental "
                        "replications."
                    )

                    if repeated_report_path.is_file():
                        with st.expander(
                            "View repeated-CV experiment report"
                        ):
                            st.text(
                                repeated_report_path.read_text(
                                    encoding="utf-8",
                                    errors="replace",
                                )
                            )

        except Exception as error:
            st.error(
                f"Could not load repeated-CV results: {error}"
            )
    else:
        st.error(
            "Repeated-CV summary not found: "
            + str(repeated_summary_path)
        )

    st.divider()
    with st.expander(
        "Historical synthetic 80/20 holdout results"
    ):
        st.caption(
            "Supplementary historical results only. The repeated-CV "
            "comparison above is the primary synthetic benchmark."
        )

        old_synthetic_dir = (
            ROOT / "fusion" / "synthetic_integration"
            / "synthetic_benchmark"
        )
        old_synthetic_metrics = (
            old_synthetic_dir / "Synthetic_Fusion_Model_Metrics.csv"
        )

        if old_synthetic_metrics.is_file():
            try:
                old_metrics = pd.read_csv(old_synthetic_metrics)
                st.subheader("Historical synthetic fusion benchmark")
                st.dataframe(
                    old_metrics,
                    width="stretch",
                    hide_index=True,
                )
            except Exception as error:
                st.error(
                    f"Could not load historical benchmark: {error}"
                )
        else:
            st.caption(
                "Historical synthetic benchmark metrics are unavailable."
            )

        old_architecture_path = (
            ROOT / "fusion" / "synthetic_integration"
            / "synthetic_decision_level_comparison"
            / "Fusion_Architecture_Comparison_Metrics.csv"
        )

        if old_architecture_path.is_file():
            try:
                old_architecture = pd.read_csv(old_architecture_path)
                st.subheader(
                    "Historical feature-level and decision-level comparison"
                )
                st.dataframe(
                    old_architecture,
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "This is the earlier stratified 80/20 holdout "
                    "experiment, not the repeated-CV evaluation."
                )
            except Exception as error:
                st.error(
                    f"Could not load historical architecture results: {error}"
                )

        st.info(
            "Historical holdout results are retained for traceability "
            "and should not be combined with repeated-CV metrics."
        )

    st.info(
        "Scientific limitation: current EEG and EIT datasets have no "
        "verified shared participant or sample identifiers. Real-data "
        "fusion remains BLOCKED. No tumour detection, clinical screening "
        "or recurrence prediction capability is established."
    )

    st.divider()
    st.header("End-to-End Integration Execution")

    integration_dir = ROOT / "fusion" / "synthetic_integration"
    integration_report = integration_dir / "NeuroSentinel_End_to_End_Integration_Report.txt"
    log_dir = integration_dir / "end_to_end_logs"
    log_files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if integration_report.is_file() and log_files:
        try:
            execution = json.loads(log_files[0].read_text(encoding="utf-8"))
            stages = execution.get("stages", [])
            passed = sum(stage.get("status") == "PASSED" for stage in stages)

            c1, c2, c3 = st.columns(3)
            c1.metric("Pipeline status", execution.get("status", "UNKNOWN"))
            c2.metric("Stages passed", f"{passed}/{len(stages)}")
            c3.metric("Clinical fusion", "READY" if execution.get("clinical_fusion_ready", False) else "NOT READY")

            st.dataframe(pd.DataFrame(stages), width="stretch", hide_index=True)

            st.warning(
                "Real clinical EEG-EIT fusion remains blocked until verified "
                "paired patient observations are available."
            )
            st.info(
                "Clinical recurrence prediction has not been established. "
                "It remains future work requiring longitudinal clinical data "
                "and appropriate validation."
            )

            with st.expander("View complete end-to-end integration report"):
                st.code(integration_report.read_text(encoding="utf-8", errors="replace"))

        except Exception as error:
            st.error(f"Could not load end-to-end integration status: {error}")
    else:
        st.warning("End-to-end integration report or execution log is missing.")

    st.divider()
    st.header("NeuroSentinel Research Audit Reports")

    audit_reports = {
        "Feature Integrity Audit":
            ROOT / "fusion" / "NeuroSentinel_Feature_Audit.json",
        "Recurrence Readiness Audit":
            ROOT / "fusion" / "NeuroSentinel_Recurrence_Readiness.json",
        "Master Project Readiness":
            ROOT / "fusion" / "NeuroSentinel_Master_Readiness.json",
        "Feature Preparation Audit":
            ROOT / "fusion" / "prepared_features" / "NeuroSentinel_Feature_Preparation_Report.json",
        "Normalization Audit":
            ROOT / "fusion" / "normalized_features" / "NeuroSentinel_Normalization_Report.json",
        "Fusion Architecture Audit":
            ROOT / "fusion" / "NeuroSentinel_Fusion_Architecture_Audit.json",
    }

    for title, audit_path in audit_reports.items():
        st.subheader(title)

        if audit_path.is_file():
            try:
                audit_data = json.loads(
                    audit_path.read_text(encoding="utf-8")
                )

                if title == "Feature Integrity Audit":
                    eeg_passed = audit_data.get(
                        "EEG_audit", {}
                    ).get("passed", False)

                    eit_passed = audit_data.get(
                        "EIT_audit", {}
                    ).get("passed", False)

                    a, b = st.columns(2)
                    a.metric(
                        "EEG feature audit",
                        "PASSED" if eeg_passed else "REVIEW"
                    )
                    b.metric(
                        "EIT feature audit",
                        "PASSED" if eit_passed else "REVIEW"
                    )

                    st.write(
                        "EEG participants:",
                        len(audit_data.get("EEG_participants", []))
                    )
                    st.write(
                        "EEG task counts:",
                        audit_data.get("EEG_task_counts", {})
                    )

                elif title == "Recurrence Readiness Audit":
                    st.metric(
                        "Recurrence readiness",
                        audit_data.get("status", "UNKNOWN")
                    )
                    st.write(
                        "Longitudinal records:",
                        audit_data.get("records", 0)
                    )
                    st.write(
                        "Verified outcomes available:",
                        audit_data.get(
                            "verified_outcomes_available", False
                        )
                    )

                elif title == "Master Project Readiness":
                    summary = audit_data.get("summary", {})
                    st.metric(
                        "Master project status",
                        audit_data.get("project_status", "UNKNOWN")
                    )
                    st.json(summary)

                    with st.expander("Outstanding requirements"):
                        for item in audit_data.get(
                            "outstanding_requirements", []
                        ):
                            st.write("- " + item)

                else:
                    st.json(audit_data)

                with st.expander("View complete JSON report"): 
                    st.json(audit_data)

            except (OSError, json.JSONDecodeError) as error:
                st.error(f"Could not load {title}: {error}")
        else:
            st.warning(f"{title} report not found: {audit_path.name}")

    st.caption(
        "These reports describe software and research-data readiness. "
        "They do not establish clinical tumour detection or recurrence prediction."
    )



with window_comparison:
    st.header("EEG Window Comparison")
    st.caption(
        "Research evaluation of 2.1-second versus 3-second EEG windows. "
        "These results concern motor-task classification, not tumour diagnosis."
    )

    original_name = "NeuroSentinel_MultiSubject_EEG_Grouped_Metrics.csv"
    three_sec_name = "NeuroSentinel_MultiSubject_EEG_3s_Grouped_Metrics.csv"
    participant_name = "NeuroSentinel_Window_Comparison.csv"

    original = load_csv(original_name)
    three_sec = load_csv(three_sec_name)
    participant = load_csv(participant_name)

    if original is None or three_sec is None or participant is None:
        st.warning(
            "One or more comparison CSV files could not be loaded. "
            "Check the data folder and filenames."
        )
    else:
        required_metrics = {
            "Model", "Windows", "Participants",
            "Accuracy", "Balanced_Accuracy", "Macro_F1"
        }
        valid_metrics = (
            required_metrics.issubset(original.columns)
            and required_metrics.issubset(three_sec.columns)
        )
        required_participant = {
            "Participant", "Accuracy_2_1s",
            "Accuracy_3s", "Difference_percentage_points"
        }
        valid_participant = required_participant.issubset(participant.columns)

        if not valid_metrics or not valid_participant:
            st.error(
                "CSV column validation failed. "
                "No comparison was displayed; check the saved file schemas."
            )
        else:
            st.subheader("Overall evaluation metrics")

            original_lr = original[
                original["Model"].astype(str).str.contains(
                    "Logistic Regression", case=False, regex=False
                )
            ]
            three_sec_lr = three_sec[
                three_sec["Model"].astype(str).str.contains(
                    "Logistic Regression", case=False, regex=False
                )
            ]

            if len(original_lr) != 1 or len(three_sec_lr) != 1:
                st.warning(
                    "Could not uniquely identify Logistic Regression "
                    "in both metric files."
                )
            else:
                a = original_lr.iloc[0]
                b = three_sec_lr.iloc[0]

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Accuracy",
                    f"{a['Accuracy']:.2%}",
                    delta=f"{(b['Accuracy'] - a['Accuracy']) * 100:+.2f} pp",
                    delta_color="off"
                )
                c2.metric(
                    "Balanced accuracy",
                    f"{a['Balanced_Accuracy']:.2%}",
                    delta=(
                        f"{(b['Balanced_Accuracy'] - a['Balanced_Accuracy']) * 100:+.2f} pp"
                    ),
                    delta_color="off"
                )
                c3.metric(
                    "Macro-F1",
                    f"{a['Macro_F1']:.2%}",
                    delta=f"{(b['Macro_F1'] - a['Macro_F1']) * 100:+.2f} pp",
                    delta_color="off"
                )

                summary = pd.DataFrame([
                    {
                        "Window": "Original (2.1 s)",
                        "Model": a["Model"],
                        "Windows": int(a["Windows"]),
                        "Participants": int(a["Participants"]),
                        "Accuracy": a["Accuracy"],
                        "Balanced accuracy": a["Balanced_Accuracy"],
                        "Macro-F1": a["Macro_F1"]
                    },
                    {
                        "Window": "Experimental (3 s)",
                        "Model": b["Model"],
                        "Windows": int(b["Windows"]),
                        "Participants": int(b["Participants"]),
                        "Accuracy": b["Accuracy"],
                        "Balanced accuracy": b["Balanced_Accuracy"],
                        "Macro-F1": b["Macro_F1"]
                    }
                ])

                st.dataframe(
                    summary.style.format({
                        "Accuracy": "{:.2%}",
                        "Balanced accuracy": "{:.2%}",
                        "Macro-F1": "{:.2%}"
                    }),
                    width="stretch",
                    hide_index=True
                )

                st.subheader("All saved model metrics")
                all_metrics = pd.concat([
                    original.assign(Window="Original (2.1 s)"),
                    three_sec.assign(Window="Experimental (3 s)")
                ], ignore_index=True)

                st.dataframe(
                    all_metrics,
                    width="stretch",
                    hide_index=True
                )
                st.download_button(
                    "Download combined evaluation metrics",
                    data=all_metrics.to_csv(index=False).encode("utf-8"),
                    file_name="NeuroSentinel_Window_Comparison_Metrics.csv",
                    mime="text/csv",
                    key="download_window_comparison_metrics"
                )

            st.divider()
            st.subheader("Participant-level accuracy")

            participant_display = participant.copy()
            participant_display = participant_display.rename(columns={
                "Accuracy_2_1s": "Accuracy (2.1 s)",
                "Accuracy_3s": "Accuracy (3 s)",
                "Difference_percentage_points": "Difference (percentage points)"
            })

            st.dataframe(
                participant_display.style.format({
                    "Accuracy (2.1 s)": "{:.2%}",
                    "Accuracy (3 s)": "{:.2%}",
                    "Difference (percentage points)": "{:+.2f}"
                }),
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "Download participant comparison",
                data=participant.to_csv(index=False).encode("utf-8"),
                file_name=participant_name,
                mime="text/csv",
                key="download_window_comparison_participants"
            )

            improved = int(
                (participant["Difference_percentage_points"] > 0).sum()
            )
            unchanged = int(
                (participant["Difference_percentage_points"] == 0).sum()
            )
            declined = int(
                (participant["Difference_percentage_points"] < 0).sum()
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Participants improved", improved)
            c2.metric("Unchanged", unchanged)
            c3.metric("Participants declined", declined)

            st.info(
                "Interpretation: compare the results descriptively. "
                "Participant-level results vary, and neither window setup "
                "demonstrates reliable tumour detection. The current EEG "
                "labels represent motor tasks, not tumour status."
            )


st.sidebar.title("NeuroSentinel")
st.sidebar.write("EEG + EIT research prototype")
st.sidebar.caption("Using existing local project outputs.")
