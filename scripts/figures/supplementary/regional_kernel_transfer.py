"""Figure: correlation transfer between brain regions (data/results/panel/regions.json).

One violin per source of the whitening pattern: another region of the same cut, the whole
brain, or none (Standard MPPCA). Every cut axis, region and scan pooled, the median as a bar.

    python scripts/figures/supplementary/regional_kernel_transfer.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from camppca_paper.paths import RESULTS
from camppca_paper.style import CMP, COU

OUT = RESULTS / "supplementary/regional_kernel_transfer.pdf"
SOURCES = (("region", "Other\nregion", COU), ("whole_brain", "Whole\nbrain", COU),
           ("none", "None\n(Standard MPPCA)", CMP))


def main():
    rows = json.loads((RESULTS / "panel/regions.json").read_text())
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.axhline(1, color="0.4", lw=0.8, ls="--", zorder=1)
    for i, (source, _, col) in enumerate(SOURCES):
        v = np.array([r["value"] for r in rows if r["source"] == source])
        vp = ax.violinplot(v, positions=[i], widths=0.7, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_alpha(0.45)
        ax.hlines(np.median(v), i - 0.22, i + 0.22, color="k", lw=2, zorder=4)
        print(f"  {source:12s} n {len(v):4d}  median {np.median(v):.3f}  range {v.min():.3f}-{v.max():.3f}")
    ax.set_xticks(range(len(SOURCES)), [s[1] for s in SOURCES], fontsize=8)
    ax.set_xlim(-0.6, len(SOURCES) - 0.4)
    ax.set_xlabel("Source of correlation pattern", fontsize=9)
    ax.set_ylabel(r"$\hat{\sigma}\,/\,\sigma$ on simulated noise", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
