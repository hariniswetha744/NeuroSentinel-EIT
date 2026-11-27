"""Figure 1: validity-gate architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw(out):
    fig, ax = plt.subplots(figsize=(3.5, 2.25)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    def box(x, y, w, h, t, fc="#ffffff", ec="#444444", fs=6.3, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")
    def arr(x1, y1, x2, y2, c="#444444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=c))
    box(0.1, 4.6, 2.0, 1.0, "EEG table\n+ features $x_E$")
    box(0.1, 2.6, 2.0, 1.0, "EIT table\n+ features $x_I$")
    ax.add_patch(FancyBboxPatch((2.7, 1.55), 4.15, 4.6, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#eef4fc", ec="#257de0", lw=1.0))
    ax.text(4.78, 5.85, "Pre-fusion validity gate  $\\Gamma$", ha="center", fontsize=6.6, weight="bold", color="#123d75")
    box(2.95, 4.3, 3.65, 1.15, "P1–P4 identifiers\nP5–P8 rate, channels, units,\ntimestamps, consent/source", fs=5.9)
    box(2.95, 1.85, 3.65, 2.1, "P9 pairing dependence\nPillai trace (CCA) or HSIC\nvs. clean-batch reference\n(calib) or permutation null", fs=5.9, fc="#ffffff", ec="#257de0")
    for y in (5.1, 3.1): arr(2.1, y, 2.95, y if y > 4 else 2.9)
    arr(4.78, 4.3, 4.78, 3.95)
    box(7.45, 4.25, 2.45, 1.2, "PASS\nfeature-level\nfusion", fc="#e6f6f1", ec="#00a888", fs=6.1)
    box(7.45, 1.85, 2.45, 1.4, "BLOCK + reason\ncode; fallback to\nanchor modality", fc="#fdeee6", ec="#e76624", fs=6.1)
    arr(6.85, 4.4, 7.45, 4.85); arr(6.85, 2.6, 7.45, 2.55)
    ax.text(5.0, 0.75, "Silent fault: IDs and metadata intact, payload rows mispaired.\nOnly P9 can see it.", ha="center", fontsize=5.7, style="italic", color="#444444")
    fig.tight_layout(pad=0.1); fig.savefig(out / "figure_1_gate.png", dpi=300); fig.savefig(out / "figure_1_gate.pdf")
if __name__ == "__main__":
    from pathlib import Path; draw(Path("outputs"))
