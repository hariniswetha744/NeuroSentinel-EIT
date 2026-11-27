"""Figure 1: validity-gate architecture diagram, drawn at its printed size
(one IEEE column, 3.35 in) so every label prints at 8 pt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FS = 8  # points at printed size


def draw(out):
    fig, ax = plt.subplots(figsize=(3.35, 2.75)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, t, fc="#ffffff", ec="#444444", bold=False, color="#111"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=FS, weight="bold" if bold else "normal",
                color=color, linespacing=1.15)

    def arr(x1, y1, x2, y2, c="#444444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=c))

    box(0.1, 8.55, 4.4, 1.35, "EEG table + $x_E$")
    box(5.5, 8.55, 4.4, 1.35, "EIT table + $x_I$")
    ax.add_patch(FancyBboxPatch((0.1, 2.75), 9.8, 4.95, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc="#eef4fc", ec="#257de0", lw=1.1))
    ax.text(5.0, 7.2, "Pre-fusion validity gate $\\Gamma$", ha="center", va="center", fontsize=FS,
            weight="bold", color="#123d75")
    box(0.45, 5.2, 9.1, 1.45, "P1–P8: identifiers, rate, channels,\nunits, timestamps, consent")
    box(0.45, 3.05, 9.1, 1.85, "P9: pairing dependence (CCA or HSIC)\nvs. clean reference or permutation",
        ec="#257de0")
    arr(2.3, 8.55, 2.3, 7.7); arr(7.7, 8.55, 7.7, 7.7)
    box(0.1, 0.1, 4.4, 2.0, "PASS:\nfuse", fc="#e6f6f1", ec="#00a888", bold=True, color="#00684f")
    box(5.5, 0.1, 4.4, 2.0, "BLOCK: reason\ncode, EEG only", fc="#fdeee6", ec="#e76624", bold=True, color="#a83e0c")
    arr(2.3, 2.75, 2.3, 2.1, c="#00a888"); arr(7.7, 2.75, 7.7, 2.1, c="#e76624")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out / "figure_1_gate.png", dpi=300); fig.savefig(out / "figure_1_gate.pdf")
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path; draw(Path("outputs"))
