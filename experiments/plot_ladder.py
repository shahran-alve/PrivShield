"""Fig. 2 from results/ladder.csv.

Run:  python experiments/plot_ladder.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import result_path

ORDER = ["undefended", "empty-suppression", "patient-k", "sym. containment", "PrivShield-Lite"]
LABELS = ["Undefended", "Empty refusal", "Patient-$k$", "Sym. containment", "Lite"]

if __name__ == "__main__":
    plt.rcParams.update({"font.size": 13, "font.family": "serif", "hatch.linewidth": 1.6})
    df = pd.read_csv(result_path("ladder.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, name in zip(axes, ["Hospital", "Adult", "NHANES"]):
        sub = df[df.dataset == name].set_index("defense").reindex(ORDER)
        x = np.arange(len(ORDER))
        w = 0.4
        rec = sub["reconstructed"].values.astype(float)
        disc = sub["disclosed"].values.astype(float)
        b1 = ax.bar(x - w / 2, rec, w, color="#2E86AB", label="Cells reconstructed")
        b2 = ax.bar(x + w / 2, disc, w, color="#1F4E79", hatch="//", edgecolor="white",
                    label="Individuals disclosed")
        ax.set_title(name, fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=12, rotation=35, ha="right")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, 8000)
        ax.set_yticks([0, 1, 10, 100, 1000])
        ax.set_yticklabels(["0", "1", "10", "100", "1000"], fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for xi, (a, b) in enumerate(zip(rec, disc)):
            ax.text(xi - w / 2, a * 1.25 if a > 0 else 0.12, int(a), ha="center", va="bottom", fontsize=11)
            ax.text(xi + w / 2, b * 1.25 if b > 0 else 0.12, int(b), ha="center", va="bottom", fontsize=11)
    axes[0].set_ylabel("Count (symlog)", fontsize=14)
    fig.legend(handles=[b1, b2], loc="upper center", ncol=2, fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = sys.argv[1] if len(sys.argv) > 1 else result_path("ladder.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print("wrote", out)
