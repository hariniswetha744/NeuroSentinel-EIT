"""Figure 2: how the data were obtained.
(a) EEG: the 64-electrode 10-10 montage of the PhysioNet EEGBCI recordings,
    with the two channels used by the model (C3, C4) highlighted.
(b) EIT: one simulated anomaly on the fine forward mesh with its 16 electrodes.
(c) EIT: GLRT statistic T_k on the coarse detection mesh for that anomaly at 0 dB.
(d) Project dashboard (screenshot) showing the archived EIDORS simulation.
Panels (a)-(c) are generated; (d) is a screenshot. None is a patient recording."""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CH = ("FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6 Fp1 Fpz Fp2 AF7 AF3 AFz "
      "AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10 TP7 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz "
      "PO4 PO8 O1 Oz O2 Iz").split()


def eeg_panel(ax, fs=6.6, short=False):
    import mne
    mont = mne.channels.make_standard_montage("standard_1005")  # 10-05 positions
    pos = mont.get_positions()["ch_pos"]
    P = np.array([pos[c] for c in CH]); P = P - np.array([0, -0.017, 0.0])  # rough head centre
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    theta = np.arccos(np.clip(P[:, 2], -1, 1)); phi = np.arctan2(P[:, 1], P[:, 0])
    r = theta / (np.pi / 2)  # azimuthal equidistant: equator (Fpz-T7-Oz-T8 line) on the unit circle
    xy = np.c_[r * np.cos(phi), r * np.sin(phi)]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(1.1 * np.cos(th), 1.1 * np.sin(th), color="#444", lw=0.8)
    ax.plot([-0.12, 0, 0.12], [1.09, 1.24, 1.09], color="#444", lw=0.8)
    sel = [CH.index("C3"), CH.index("C4")]
    other = [i for i in range(len(CH)) if i not in sel]
    ax.scatter(xy[other, 0], xy[other, 1], s=5, color="#b0b0b0", zorder=3)
    ax.scatter(xy[sel, 0], xy[sel, 1], s=40, color="#257de0", zorder=4, edgecolor="white", lw=0.8)
    for i in sel:
        ax.text(xy[i, 0], xy[i, 1] + 0.13, CH[i], ha="center", fontsize=fs, color="#123d75", weight="bold")
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.35, 1.35)
    ax.set_title("(a) EEG, 64 ch;\nC3, C4 used" if short else "(a) EEG: 64-ch 10–10 montage\n(EEGBCI); C3, C4 used", fontsize=fs)


def eit_panels(ax_b, ax_c, seed=5, fs=6.6, short=False):
    import pyeit.eit.protocol as protocol
    from pyeit.eit.fem import EITForward
    from ns_gate.eit_glrt import _meshes, _centroids
    fine, coarse = _meshes()
    prot = protocol.create(16, dist_exc=1, step_meas=1, parser_meas="std")
    ff, fc = EITForward(fine, prot), EITForward(coarse, prot)
    cf, cc = _centroids(fine), _centroids(coarse)
    centre, rad = np.array([0.35, 0.25]), 0.18
    perm = np.ones(len(fine.element)); perm[np.linalg.norm(cf - centre, axis=1) <= rad] = 0.5
    vh = np.real(ff.solve_eit(perm=np.ones(len(fine.element))))
    dv = np.real(ff.solve_eit(perm=perm)) - vh
    m = len(dv)
    s = np.linalg.norm(dv) / (math.sqrt(2 * m) * 10 ** (0 / 20))  # 0 dB, Eq. (5)
    rng = np.random.default_rng(seed)
    dvn = dv + rng.normal(0, s * math.sqrt(2), m)
    J, _ = fc.compute_jac(perm=np.ones(len(coarse.element))); J = np.real(J)
    T = (dvn @ J) ** 2 / np.sum(J * J, 0)
    el_f = fine.node[fine.el_pos][:, :2]
    ax_b.tripcolor(fine.node[:, 0], fine.node[:, 1], fine.element, facecolors=perm, cmap="Blues_r",
                   vmin=0.3, vmax=1.0, edgecolors="#d0d0d0", linewidth=0.15)
    ax_b.scatter(el_f[:, 0], el_f[:, 1], s=10, color="#00a888", zorder=3)
    ax_b.set_title("(b) EIT anomaly,\nfine mesh" if short else "(b) EIT: simulated anomaly\n(fine mesh, 16 electrodes)", fontsize=fs)
    ax_c.tripcolor(coarse.node[:, 0], coarse.node[:, 1], coarse.element, facecolors=T / T.max(), cmap="Blues",
                   edgecolors="#d0d0d0", linewidth=0.2)
    k = int(np.argmax(T))
    ax_c.scatter(*cc[k], marker="x", s=25, color="#e76624", zorder=4)
    ax_c.add_patch(plt.Circle(centre, rad, fill=False, ls="--", lw=0.8, color="#444"))
    ax_c.scatter(el_f[:, 0], el_f[:, 1], s=10, color="#00a888", zorder=3)
    ax_c.set_title("(c) GLRT map\nat 0 dB" if short else "(c) GLRT map T$_k$ at 0 dB\n(coarse mesh; × = max)", fontsize=fs)
    for a in (ax_b, ax_c):
        a.set_aspect("equal"); a.axis("off")


def draw(out=Path("outputs"), blind=False):
    if blind:  # the version used in the paper: drawn at its printed width (one column), 8 pt text
        fig = plt.figure(figsize=(3.35, 1.55))
        gs = fig.add_gridspec(1, 3, wspace=0.06)
        eeg_panel(fig.add_subplot(gs[0]), fs=8, short=True)
        eit_panels(fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fs=8, short=True)
        fig.savefig(out / "figure_2_acquisition_blind.png", dpi=300, bbox_inches="tight", pad_inches=0.02); return
    fig = plt.figure(figsize=(7.16, 2.15))
    gs = fig.add_gridspec(1, 3 if blind else 4, width_ratios=[1, 1, 1] if blind else [1, 1, 1, 1.45], wspace=0.08)
    eeg_panel(fig.add_subplot(gs[0]))
    eit_panels(fig.add_subplot(gs[1]), fig.add_subplot(gs[2]))
    if blind:
        fig.savefig(out / "figure_2_acquisition_blind.png", dpi=300, bbox_inches="tight"); return
    ax = fig.add_subplot(gs[3]); ax.imshow(mpimg.imread(out / "dashboard_eit_tab.png")); ax.axis("off")
    ax.set_title("(d) Project dashboard: archived EIDORS\nsimulation (screenshot)", fontsize=6.6)
    fig.savefig(out / "figure_2_acquisition.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_2_acquisition.pdf", bbox_inches="tight")


if __name__ == "__main__":
    draw(); draw(blind=True)
