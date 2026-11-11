"""Figure: how many volumes and patches a correlation pattern needs (data/results/panel/convergence.json).

a   patterns estimated on fewer volumes, b   the loop fitted on fewer patches. One violin per
subset size, every scan and subset pooled, the median as a bar. The last violin is no
whitening (Standard MPPCA). Each value is Standard's noise level on pure noise with the whole
series' pattern, whitened with the subset's pattern, over that of white noise.

    python scripts/figures/supp/correlation_convergence.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from camppca_paper.paths import RESULTS
from camppca_paper.style import CMP, COU

OUT = RESULTS / "supp/correlation_convergence.pdf"
KINDS = (("volumes", "Number of volumes"), ("patches", "Number of patches"))


def violin(ax, i, v, col):
    v = v[np.isfinite(v)]
    vp = ax.violinplot(v, positions=[i], widths=0.7, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.45)
    ax.hlines(np.median(v), i - 0.22, i + 0.22, color="k", lw=2, zorder=4)


def main():
    rows = json.loads((RESULTS / "panel/convergence.json").read_text())
    none = np.array([r["value"] for r in rows if r["kind"] == "none"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, (kind, xlabel), letter in zip(axes, KINDS, "ab", strict=True):
        sizes = sorted({r["size"] for r in rows if r["kind"] == kind})
        ax.axhline(1, color="0.4", lw=0.8, ls="--", zorder=1)
        for i, n in enumerate(sizes):
            v = np.array([r["value"] for r in rows if r["kind"] == kind and r["size"] == n])
            violin(ax, i, v, COU)
            print(f"  {kind:8s} {n:4d}  median {np.nanmedian(v):.3f}  range {np.nanmin(v):.3f}-{np.nanmax(v):.3f}")
        violin(ax, len(sizes), none, CMP)
        ax.set_xticks(range(len(sizes) + 1), [str(n) for n in sizes] + ["Standard\nMPPCA"], fontsize=8)
        ax.set_xlim(-0.6, len(sizes) + 0.6)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.tick_params(labelsize=8, labelleft=True)
        ax.set_ylabel(r"$\hat{\sigma}\,/\,\sigma$ on simulated noise", fontsize=9)
        ax.text(-0.14, 0.99, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
