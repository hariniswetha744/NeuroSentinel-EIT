"""Explanatory figures (all original, generated from this project's own code).

figure_concept.png   : what the project is, in one picture (measurement concept + what we did)
figure_eit_steps.png : the EIT pipeline step by step (true change -> electrode data -> image -> detection map)
figure_mispairing.png: what mispairing does and how the gate sees it (results evidence)
"""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

from .fig_acq import CH
from .generator import generate, mispair
from .gate import build_reference, cca_pillai, _center

BLUE, ORANGE, AQUA, GRAY, RED, GREEN = "#257de0", "#f47230", "#00a888", "#7a7a7a", "#d64545", "#0a8a3a"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
OUT = Path("outputs")


def _eeg_xy():
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    P = np.array([pos[c] for c in CH]) - np.array([0, -0.017, 0.0])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    th = np.arccos(np.clip(P[:, 2], -1, 1)); ph = np.arctan2(P[:, 1], P[:, 0])
    r = th / (np.pi / 2)
    return np.c_[r * np.cos(ph), r * np.sin(ph)]


def _box(ax, x, y, w, h, title, body, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="white", ec=ec, lw=1.0, transform=ax.transAxes))
    ax.text(x + 0.015, y + h - 0.03, title, transform=ax.transAxes, va="top", fontsize=7.4, weight="bold", color=ec)
    ax.text(x + 0.015, y + h - 0.105, body, transform=ax.transAxes, va="top", fontsize=6.4, color="#222",
            linespacing=1.35)


def _arrow(ax, p, q, c="#555"):
    ax.add_patch(FancyArrowPatch(p, q, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=c))


# ------------------------------------------------------------- concept figure
def concept(results):
    fig = plt.figure(figsize=(7.16, 3.3))
    ax = fig.add_axes([0.0, 0.02, 0.36, 0.86])
    th = np.linspace(0, 2 * np.pi, 300)
    ax.fill(np.cos(th), np.sin(th), color="#f6efe9", zorder=0)
    ax.plot(np.cos(th), np.sin(th), color="#444", lw=1.0)
    ax.add_patch(Polygon([[-0.12, 0.99], [0, 1.15], [0.12, 0.99]], closed=False, fill=False, ec="#444", lw=1.0))
    xy = _eeg_xy() * 0.84
    sel = [CH.index("C3"), CH.index("C4")]
    oth = [i for i in range(len(CH)) if i not in sel]
    ax.scatter(xy[oth, 0], xy[oth, 1], s=7, color="#9a9a9a", zorder=3)
    ax.scatter(xy[sel, 0], xy[sel, 1], s=45, color=BLUE, edgecolor="white", lw=0.8, zorder=4)
    for i in sel:
        ax.text(xy[i, 0], xy[i, 1] + 0.1, CH[i], ha="center", fontsize=7, weight="bold", color="#123d75")
    ang = np.pi / 2 - np.arange(16) * 2 * np.pi / 16
    ex, ey = np.cos(ang), np.sin(ang)
    ax.scatter(ex, ey, marker="s", s=28, color=AQUA, edgecolor="white", lw=0.6, zorder=5)
    for k in range(16):
        ax.text(1.17 * ex[k], 1.17 * ey[k], str(k + 1), ha="center", va="center", fontsize=5.5, color=AQUA)
    # current injection between electrodes 1-2, voltage on 4-5
    def meter(a, b, lab, c, rad):
        ma = (ang[a] + ang[b]) / 2
        mx, my = rad * np.cos(ma), rad * np.sin(ma)
        for k in (a, b):
            ax.plot([ex[k] * 1.03, mx], [ey[k] * 1.03, my], color=c, lw=0.9)
        ax.add_patch(Circle((mx, my), 0.1, fc="white", ec=c, lw=1.0, zorder=6))
        ax.text(mx, my, lab, ha="center", va="center", fontsize=7.5, weight="bold", color=c, zorder=7)
    meter(0, 1, "I", ORANGE, 1.38)
    meter(3, 4, "V", GREEN, 1.38)
    ax.set_xlim(-1.55, 1.6); ax.set_ylim(-1.95, 1.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) Measurement concept (top view)", fontsize=7.6, loc="left")
    ax.text(-1.5, -1.92, "Grey/blue dots: 64 EEG electrodes (C3, C4 used).\n"
            "Green squares: 16-electrode EIT ring. I: current injected\n"
            "between a pair; V: voltage read on another pair.\n"
            "Conceptual drawing: no hardware recording in this study.", fontsize=5.6, color="#444", va="bottom")

    # right: what we did, with measured headline numbers
    bx = fig.add_axes([0.37, 0.0, 0.63, 0.97]); bx.axis("off")
    bx.text(0.0, 0.985, "(b) What this study did", fontsize=7.6, va="top")
    _box(bx, 0.0, 0.62, 0.31, 0.32, "EEG (real, public)",
         "PhysioNet motor imagery.\nC3/C4 mu + beta power\n= 4 features. Model frozen\non S001-S010, tested on\nunseen participants.", BLUE)
    _box(bx, 0.0, 0.25, 0.31, 0.32, "EIT (simulated)",
         "16-electrode difference\nframes, fine -> coarse mesh.\nGLRT detector with a 5%\nfalse-alarm threshold.", AQUA)
    _box(bx, 0.35, 0.40, 0.30, 0.40, "Pairing study",
         "Synthetic data.\nEEG-like (4) + EIT-like (8)\nfeatures with known pairing.\nA fraction q of EIT rows is\nswapped; IDs stay correct.\n\nGate: P1-P8 metadata checks\n+ P9 pairing-dependence test.", ORANGE)
    _box(bx, 0.69, 0.55, 0.30, 0.39, "Finding 1: silent failure",
         f"Fusion accuracy fell from\n{results['acc0']:.3f} to {results['acc1']:.3f} (q = 0 -> 1).\n"
         f"ID + metadata checks\nblocked {results['meta_block']:.0f}% of these batches.", RED)
    _box(bx, 0.69, 0.10, 0.30, 0.40, "Finding 2: gate sees it",
         f"P9 blocked {results['p9_q1']:.1f}% at q = 1\nand {results['p9_q05']:.1f}% at q = 0.5\n"
         f"(N = 200), wrongly blocking\n{results['fb']:.1f}% of clean batches.\nNo power when the two\nmodalities share nothing.", GREEN)
    _arrow(bx, (0.31, 0.78), (0.35, 0.68)); _arrow(bx, (0.31, 0.41), (0.35, 0.52))
    _arrow(bx, (0.65, 0.66), (0.69, 0.74)); _arrow(bx, (0.65, 0.54), (0.69, 0.32))
    bx.text(0.0, 0.03, "In-silico methodological study. No patient data; no clinical claim.",
            fontsize=6.2, style="italic", color="#444")
    fig.savefig(OUT / "figure_concept.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------- EIT step-by-step
def eit_steps(snr_db=0.0, seed=5):
    import pyeit.eit.protocol as protocol
    from pyeit.eit.fem import EITForward
    from .eit_glrt import _meshes, _centroids, _laplacian
    fine, coarse = _meshes()
    prot = protocol.create(16, dist_exc=1, step_meas=1, parser_meas="std")
    ff, fc = EITForward(fine, prot), EITForward(coarse, prot)
    cf, cc = _centroids(fine), _centroids(coarse)
    centre, rad = np.array([0.35, 0.25]), 0.18
    perm = np.ones(len(fine.element)); perm[np.linalg.norm(cf - centre, axis=1) <= rad] = 0.5
    vh = np.real(ff.solve_eit(perm=np.ones(len(fine.element))))
    dv = np.real(ff.solve_eit(perm=perm)) - vh
    m = len(dv)
    s = np.linalg.norm(dv) / (math.sqrt(2 * m) * 10 ** (snr_db / 20))
    dvn = dv + np.random.default_rng(seed).normal(0, s * math.sqrt(2), m)
    J, _ = fc.compute_jac(perm=np.ones(len(coarse.element))); J = np.real(J)
    R = _laplacian(coarse); JtJ = J.T @ J
    u = np.linalg.solve(JtJ + 10.0 * np.trace(JtJ) / np.trace(R) * R, J.T @ dvn)
    u = u * np.sign(u[np.argmax(np.abs(u))])
    T = (dvn @ J) ** 2 / np.sum(J * J, 0)
    el = fine.node[fine.el_pos][:, :2]

    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.1), gridspec_kw={"width_ratios": [1, 1.35, 1, 1]})
    axs[0].tripcolor(fine.node[:, 0], fine.node[:, 1], fine.element, facecolors=perm, cmap="Blues_r", vmin=0.3,
                     vmax=1, edgecolors="#dddddd", linewidth=0.12)
    axs[0].set_title("1. True change\n(conductivity drop, simulated)", fontsize=6.8)
    x = np.arange(m)
    axs[1].plot(x, dvn, ".", ms=1.6, color="#9a9a9a", label="measured (0 dB noise)")
    axs[1].plot(x, dv, color=AQUA, lw=0.9, label="noise-free")
    for k in range(1, 16):
        axs[1].axvline(13 * k, color="#eeeeee", lw=0.5, zorder=0)
    axs[1].set_title("2. What the electrodes record\n(208 voltage changes δv)", fontsize=6.8)
    axs[1].set_xlabel("measurement index (13 per current pair)", fontsize=6)
    axs[1].tick_params(labelsize=5.5); axs[1].set_ylim(dvn.min() * 1.15, dvn.max() * 1.45)
    axs[1].legend(frameon=False, fontsize=5.5, loc="upper center", ncol=2, handletextpad=0.3, columnspacing=0.8)
    for a in ("top", "right"):
        axs[1].spines[a].set_visible(False)
    axs[2].tripcolor(coarse.node[:, 0], coarse.node[:, 1], coarse.element, facecolors=u, cmap="RdBu_r",
                     vmin=-np.abs(u).max(), vmax=np.abs(u).max(), edgecolors="#dddddd", linewidth=0.15)
    axs[2].set_title("3. Reconstructed image\n(coarse mesh, Laplacian prior)", fontsize=6.8)
    axs[3].tripcolor(coarse.node[:, 0], coarse.node[:, 1], coarse.element, facecolors=T / T.max(), cmap="Blues",
                     edgecolors="#dddddd", linewidth=0.15)
    k = int(np.argmax(T)); axs[3].scatter(*cc[k], marker="x", s=30, color=ORANGE, zorder=4)
    axs[3].set_title("4. GLRT detection map\n(× = detected location)", fontsize=6.8)
    for a in (axs[0], axs[2], axs[3]):
        a.scatter(el[:, 0], el[:, 1], s=7, color=AQUA, zorder=3)
        a.add_patch(Circle(centre, rad, fill=False, ls="--", lw=0.7, color="#333", zorder=5))
        a.set_aspect("equal"); a.axis("off")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "figure_eit_steps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"loc_error": float(np.linalg.norm(cc[k] - centre))}


# --------------------------------------------------------- mispairing figure
def mispairing(r=0.5, n=200, seed=31, alpha=0.05):
    rng = np.random.default_rng(seed)
    pool = generate(20000, r, rng)
    ref = build_reference(pool["x_eeg"], pool["x_eit"], n, "cca", 1000, rng)
    thr = ref.threshold(alpha)
    # first canonical directions learned on the clean pool (fixed for display)
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=1).fit(pool["x_eeg"][:5000], pool["x_eit"][:5000])
    d = generate(n, r, rng)
    xi_half, mask_half = mispair(d["x_eit"], 0.5, rng)
    xi_full, _ = mispair(d["x_eit"], 1.0, rng)
    stats = {"clean": cca_pillai(d["x_eeg"], d["x_eit"]), "half": cca_pillai(d["x_eeg"], xi_half),
             "full": cca_pillai(d["x_eeg"], xi_full), "thr": thr}
    null = np.array([cca_pillai(*(lambda g: (g["x_eeg"], mispair(g["x_eit"], 1.0, rng)[0]))(generate(n, r, rng)))
                     for _ in range(1000)])

    fig = plt.figure(figsize=(7.16, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1, 1, 1.25], wspace=0.5)
    # (a) schematic
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(-0.04, 1.04); ax.set_ylim(0, 1)
    rows = 7; ys = np.linspace(0.82, 0.12, rows)
    swap = {1: 4, 4: 1, 5: 6, 6: 5}
    for i, y in enumerate(ys):
        ax.add_patch(FancyBboxPatch((0.0, y - 0.04), 0.36, 0.08, boxstyle="round,pad=0.005", fc="#e8f1fc", ec=BLUE, lw=0.7))
        ax.text(0.18, y, f"ID {101 + i} EEG", ha="center", va="center", fontsize=5.6)
        ax.add_patch(FancyBboxPatch((0.64, y - 0.04), 0.36, 0.08, boxstyle="round,pad=0.005", fc="#e6f6f1", ec=AQUA, lw=0.7))
        ax.text(0.82, y, f"ID {101 + i} EIT", ha="center", va="center", fontsize=5.6)
        j = swap.get(i, i)
        ax.annotate("", xy=(0.64, ys[j]), xytext=(0.36, y),
                    arrowprops=dict(arrowstyle="-", color=RED if j != i else GREEN, lw=1.0 if j != i else 0.8))
    ax.text(0.5, 1.02, "(a) IDs match,\npayloads swapped", ha="center", fontsize=6.8)
    ax.text(0.5, 0.0, "green: correct pair   red: wrong partner", ha="center", fontsize=5.6, color="#444")

    def scat(axx, xi, title, stat, col):
        u, v = cca.transform(d["x_eeg"], xi)
        cc_ = np.corrcoef(u[:, 0], v[:, 0])[0, 1]
        axx.scatter(u[:, 0], v[:, 0], s=4, color=col, alpha=0.6, lw=0)
        axx.set_xlabel("EEG canonical score", fontsize=6); axx.set_ylabel("EIT canonical score", fontsize=6)
        axx.tick_params(labelsize=5.5)
        for s_ in ("top", "right"):
            axx.spines[s_].set_visible(False)
        dec = "PASS" if stat >= thr else "BLOCK"
        axx.set_title(f"{title}\ncorr = {cc_:.2f}, Pillai = {stat:.3f}\ngate: {dec}", fontsize=6.4)
        return cc_

    c1 = scat(fig.add_subplot(gs[1]), d["x_eit"], "(b) Correct pairing", stats["clean"], BLUE)
    c2 = scat(fig.add_subplot(gs[2]), xi_full, "(c) All records mispaired", stats["full"], RED)
    ax = fig.add_subplot(gs[3])
    bins = np.linspace(min(null.min(), ref.values.min()), max(null.max(), ref.values.max()), 45)
    ax.hist(ref.values, bins=bins, color=BLUE, alpha=0.65, label="clean batches")
    ax.hist(null, bins=bins, color=RED, alpha=0.55, label="fully mispaired")
    ax.axvline(thr, color="#222", ls="--", lw=0.9)
    ax.text(thr, ax.get_ylim()[1] * 0.62, " threshold:\n block below", fontsize=5.6, va="top")
    ax.set_xlabel("Pillai trace of a 200-record batch", fontsize=6); ax.set_ylabel("batches", fontsize=6)
    ax.tick_params(labelsize=5.5); ax.legend(frameon=False, fontsize=5.6, loc="upper left", bbox_to_anchor=(-0.02, 1.0))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title("(d) The gate's decision rule\n(1,000 batches each)\n", fontsize=6.4)
    fig.savefig(OUT / "figure_mispairing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    stats.update({"corr_clean": float(c1), "corr_full": float(c2),
                  "null_below_thr": float(np.mean(null < thr)), "ref_below_thr": float(np.mean(ref.values < thr))})
    return stats


def headline_numbers(results_json="outputs/results.json"):
    import json
    R = json.load(open(results_json))
    fa = {round(r["q"], 2): r for r in R["figure_A"]["rows"]}
    t = {(r["fault"], round(r["severity"], 3)): r for r in R["tables_I_II"]["rows"]}
    f7 = "F7 Silent ID-consistent mispairing"
    clean = next(r for r in R["tables_I_II"]["rows"] if r["set"] == "clean")
    return {"acc0": fa[0.0]["acc_ungated_fusion"], "acc1": fa[1.0]["acc_ungated_fusion"],
            "meta_block": 100 * t[(f7, 1.0)]["Metadata (P1-P8)"],
            "p9_q1": 100 * t[(f7, 1.0)]["Full: +P9 calib-CCA"], "p9_q05": 100 * t[(f7, 0.5)]["Full: +P9 calib-CCA"],
            "fb": 100 * clean["Full: +P9 calib-CCA"]}


if __name__ == "__main__":
    import json
    h = headline_numbers(); concept(h)
    e = eit_steps(); s = mispairing()
    (OUT / "explain_figures_values.json").write_text(json.dumps({"headline": h, "eit_steps": e, "mispairing": s}, indent=2))
    print(json.dumps({"headline": h, "eit_steps": e, "mispairing": s}, indent=2))
