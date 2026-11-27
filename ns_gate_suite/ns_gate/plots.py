"""Figures A, B, C for an IEEE column (3.5 in). Colour + marker/linestyle, so
the figures stay readable in grayscale print and for colour-vision deficiency."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, GRAY = "#257de0", "#f47230", "#00b894", "#7a7a7a"
SEQ = ["#b7d3f5", "#80b2ec", "#4a8fe0", "#2265b5", "#123d75"]  # one hue, light->dark

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.4, "lines.markersize": 3.5, "pdf.fonttype": 42,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png")
    plt.close(fig)


def figure_a(df, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.6, 1]})
    q = df.q
    for col, lab, c, mk, ls in [("ungated_fusion", "Fusion, no gate", ORANGE, "o", "-"),
                                ("gated_calib_cca", "Fusion + gate (calib-CCA)", BLUE, "s", "-"),
                                ("gated_perm_cca", "Fusion + gate (perm-CCA)", AQUA, "^", "--"),
                                ("eeg_only", "EEG-only fallback", GRAY, None, ":")]:
        a1.plot(q, df[f"acc_{col}"], color=c, marker=mk, ls=ls, label=lab)
        a1.fill_between(q, df[f"acc_{col}"] - df[f"acc_{col}_ci95"], df[f"acc_{col}"] + df[f"acc_{col}_ci95"],
                        color=c, alpha=0.15, lw=0)
    a1.set_ylabel("Accuracy"); a1.legend(frameon=False, loc="lower left")
    a1.set_title("(a) Classification accuracy vs. mispaired fraction", loc="left")
    a2.plot(q, df.block_rate_calib_cca, color=BLUE, marker="s", label="calib-CCA")
    a2.plot(q, df.block_rate_perm_cca, color=AQUA, marker="^", ls="--", label="perm-CCA")
    a2.set_ylabel("Gate block rate"); a2.set_xlabel("Mispaired fraction q"); a2.set_ylim(-0.03, 1.03)
    a2.legend(frameon=False, loc="lower right")
    a2.set_title("(b) Fraction of batches blocked", loc="left")
    _save(fig, out, "figure_A_corruption")


def figure_b(df, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.75), sharey=True)
    sub = df[df.panel == "N"]
    for c, (n, g) in zip(SEQ, sub.groupby("n")):
        axs[0].plot(g.q, g.calib_cca, color=c, marker="o", label=f"N={n}")
    axs[0].set_title("(a) Sample size (r = 0.5)", loc="left")
    sub = df[df.panel == "r"]
    for c, (r, g) in zip(SEQ, sub.groupby("r")):
        axs[1].plot(g.q, g.calib_cca, color=c, marker="o", label=f"r={r:g}")
    axs[1].set_title("(b) Redundancy r (N = 200)", loc="left")
    sub = df[df.panel == "stat"]
    for fam, ls in (("A", "-"), ("B", "--")):
        g = sub[sub.family == fam]
        axs[2].plot(g.q, g.calib_cca, color=BLUE, ls=ls, marker="s", label=f"CCA, gen. {fam}")
        axs[2].plot(g.q, g.calib_hsic, color=ORANGE, ls=ls, marker="o", label=f"HSIC, gen. {fam}")
    axs[2].set_title("(c) Statistic x generator (N = 200)", loc="left")
    for a in axs:
        a.axhline(0.05, color=GRAY, lw=0.8, ls=":")
        a.set_xlabel("Mispaired fraction q"); a.set_ylim(-0.03, 1.03)
        a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, columnspacing=0.8, handlelength=1.6)
    axs[0].set_ylabel("Detection power (block rate)")
    _save(fig, out, "figure_B_power")


def figure_c(levels, rocs, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.4))
    styles = {}
    snrs = sorted({int(k.split("_")[1][:-2]) for k in rocs})
    lss = ["-", "--", ":"]
    for i, s in enumerate(snrs):
        for det, c, lab in (("glrt", BLUE, "GLRT"), ("peak", ORANGE, "Peak")):
            r = rocs[f"{det}_{s}dB"]
            a1.plot(r["fpr"], r["tpr"], color=c, ls=lss[i % 3], lw=1.2, label=f"{lab}, {s} dB")
            a1.plot([r["op_far"]], [r["op_tpr"]], marker="o", color=c, ms=4, mec="white", mew=0.8)
    a1.plot([0, 1], [0, 1], color=GRAY, lw=0.7, ls=":")
    a1.set_xlabel("False-alarm rate"); a1.set_ylabel("Detection rate")
    a1.set_title("(a) ROC; dots = conformal operating point (α=0.05)", loc="left")
    a1.legend(frameon=False, loc="lower right", ncol=1)
    snr = [l["snr_db"] for l in levels]
    a2.plot(snr, [l["glrt_detection"] for l in levels], color=BLUE, marker="s", label="GLRT detection")
    a2.plot(snr, [l["peak_detection"] for l in levels], color=ORANGE, marker="o", label="Peak detection")
    a2.plot(snr, [l["glrt_far_marginal_mean"] for l in levels], color=BLUE, ls=":", marker="s", mfc="white",
            label="GLRT false alarm")
    a2.plot(snr, [l["peak_far_marginal_mean"] for l in levels], color=ORANGE, ls=":", marker="o",
            mfc="white", label="Peak false alarm")
    a2.axhline(0.05, color=GRAY, lw=0.7, ls="--")
    a2.set_xlabel("SNR of difference signal (dB)"); a2.set_ylabel("Rate"); a2.set_ylim(-0.03, 1.03)
    a2.set_title("(b) Detection and false alarm at α = 0.05", loc="left")
    a2.legend(frameon=False, loc="center right")
    _save(fig, out, "figure_C_eit_glrt")


# ---------------------------------------------------------------- print-size versions
# Same plotted data as figure_b / figure_c, drawn at the width used in the paper so
# that every text element prints at >= 7 pt (IEEE legibility). Values are read from
# the saved outputs; no experiment is rerun.
_PRINT = {"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
          "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.markersize": 3.2, "lines.linewidth": 1.2}


def figure_b_print(df, out, width=5.4, theory=None):
    """theory: rows of outputs/novelty/theory_sim.json -> adds panel (d), predicted vs simulated."""
    with plt.rc_context(_PRINT):
        k = 4 if theory else 3
        fig, axs = plt.subplots(1, k, figsize=(width, 2.3), sharey=True)
        sub = df[df.panel == "N"]
        for c, (n, g) in zip(SEQ, sub.groupby("n")):
            axs[0].plot(g.q, g.calib_cca, color=c, marker="o", label=f"{n}")
        axs[0].set_title("(a) Batch size N", loc="left")
        sub = df[df.panel == "r"]
        for c, (r, g) in zip(SEQ, sub.groupby("r")):
            axs[1].plot(g.q, g.calib_cca, color=c, marker="o", label=f"{r:g}")
        axs[1].set_title("(b) Redundancy r", loc="left")
        sub = df[df.panel == "stat"]
        for fam, ls in (("A", "-"), ("B", "--")):
            g = sub[sub.family == fam]
            axs[2].plot(g.q, g.calib_cca, color=BLUE, ls=ls, marker="s", label=f"CCA {fam}")
            axs[2].plot(g.q, g.calib_hsic, color=ORANGE, ls=ls, marker="o", label=f"HSIC {fam}")
        axs[2].set_title("(c) Statistic, gen.", loc="left")
        titles = ("N (r = 0.5)", "r (N = 200)", "N = 200, r = 0.5")
        for a, t in zip(axs, titles):
            a.axhline(0.05, color=GRAY, lw=0.8, ls=":")
            a.set_xlabel("Mispaired fraction q"); a.set_ylim(-0.03, 1.03); a.set_xticks([0, 0.5, 1])
            a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3,
                     columnspacing=0.5, handlelength=1.2, handletextpad=0.3, title=t, title_fontsize=7,
                     borderaxespad=0, labelspacing=0.2)
        axs[0].set_ylabel("Detection power")
        if theory:
            a = axs[3]
            for fam, c, mk in (("A", BLUE, "o"), ("B", ORANGE, "s")):
                rr = [t for t in theory if t["family"] == fam]
                a.scatter([t["predicted"] for t in rr], [t["empirical"] for t in rr], s=7, color=c, marker=mk,
                          lw=0, alpha=0.8, label=f"gen. {fam}")
            a.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls=":")
            a.set_xlim(-0.03, 1.03); a.set_xticks([0, 0.5, 1]); a.set_xlabel("Predicted power, (8)")
            a.set_title("(d) Theory", loc="left")
            a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2, handletextpad=0.2,
                     title="160 settings; y-axis: simulated", title_fontsize=7, borderaxespad=0)
        fig.tight_layout(w_pad=0.4)
        name = "figure_B_power_theory_print" if theory else "figure_B_power_print"
        fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def figure_c_print(levels, rocs, out, width=2.85):
    with plt.rc_context(_PRINT):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(width, 3.0))
        snrs = sorted({int(k.split("_")[1][:-2]) for k in rocs})
        lss = ["-", "--", ":"]
        for i, s in enumerate(snrs):
            for det, c, lab in (("glrt", BLUE, "GLRT"), ("peak", ORANGE, "Peak")):
                r = rocs[f"{det}_{s}dB"]
                a1.plot(r["fpr"], r["tpr"], color=c, ls=lss[i % 3], lw=1.1, label=f"{lab} {s} dB")
                a1.plot([r["op_far"]], [r["op_tpr"]], marker="o", color=c, ms=3.5, mec="white", mew=0.6)
        a1.plot([0, 1], [0, 1], color=GRAY, lw=0.6, ls=":")
        a1.set_xlabel("False-alarm rate"); a1.set_ylabel("Detection rate"); a1.set_title("(a) ROC", loc="left")
        a1.legend(frameon=False, loc="lower right", fontsize=6.5, handlelength=1.4, labelspacing=0.1, ncol=2, columnspacing=0.4, handletextpad=0.3)
        snr = [l["snr_db"] for l in levels]
        a2.plot(snr, [l["glrt_detection"] for l in levels], color=BLUE, marker="s", label="GLRT det.")
        a2.plot(snr, [l["peak_detection"] for l in levels], color=ORANGE, marker="o", label="Peak det.")
        a2.plot(snr, [l["glrt_far_marginal_mean"] for l in levels], color=BLUE, ls=":", marker="s", mfc="white",
                label="GLRT FA")
        a2.plot(snr, [l["peak_far_marginal_mean"] for l in levels], color=ORANGE, ls=":", marker="o", mfc="white",
                label="Peak FA")
        a2.axhline(0.05, color=GRAY, lw=0.6, ls="--")
        a2.set_xlabel("SNR$_{δv}$ (dB)"); a2.set_ylabel("Rate"); a2.set_ylim(-0.03, 1.03)
        a2.set_title("(b) At α = 0.05", loc="left")
        a2.legend(frameon=False, loc="center right", fontsize=6.5, handlelength=1.6, labelspacing=0.25)
        fig.tight_layout(h_pad=0.6)
        fig.savefig(out / "figure_C_eit_glrt_print.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out / "figure_C_eit_glrt_print.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def figure_a_print(df, out, width=2.75):
    with plt.rc_context(_PRINT):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(width, 2.55), sharex=True,
                                     gridspec_kw={"height_ratios": [1.5, 1]})
        q = df.q
        for col, lab, c, mk, ls in [("ungated_fusion", "No gate", ORANGE, "o", "-"),
                                    ("gated_calib_cca", "Gate, calib-CCA", BLUE, "s", "-"),
                                    ("gated_perm_cca", "Gate, perm-CCA", AQUA, "^", "--"),
                                    ("eeg_only", "EEG only", GRAY, None, ":")]:
            a1.plot(q, df[f"acc_{col}"], color=c, marker=mk, ls=ls, label=lab, ms=2.8)
            a1.fill_between(q, df[f"acc_{col}"] - df[f"acc_{col}_ci95"], df[f"acc_{col}"] + df[f"acc_{col}_ci95"],
                            color=c, alpha=0.15, lw=0)
        a1.set_ylim(0.58, 0.89)
        a1.set_ylabel("Accuracy"); a1.legend(frameon=False, loc="lower left", fontsize=6.5, labelspacing=0.15,
                                             handlelength=1.5)
        a1.set_title("(a) Accuracy", loc="left")
        a2.plot(q, df.block_rate_calib_cca, color=BLUE, marker="s", ms=2.8, label="calib-CCA")
        a2.plot(q, df.block_rate_perm_cca, color=AQUA, marker="^", ms=2.8, ls="--", label="perm-CCA")
        a2.set_ylabel("Blocked"); a2.set_xlabel("Mispaired fraction q"); a2.set_ylim(-0.03, 1.03)
        a2.legend(frameon=False, loc="lower right", fontsize=6.5, labelspacing=0.15, handlelength=1.5)
        a2.set_title("(b) Batches blocked", loc="left")
        fig.tight_layout(h_pad=0.4)
        fig.savefig(out / "figure_A_corruption_print.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out / "figure_A_corruption_print.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
