"""Figure: the ex vivo DTI and DKI maps of the twenty-average reference and each arm, on the slice
of the main ex vivo figure (`camppca_paper.exvivo_mouse.shown`). One row per map, the arms in
columns, each row on one grey scale from the 1st to the 99th percentile of its panels, MD to the
90th so the fluid does not set it. The white matter of the metrics, the reference's FA above WM_FA,
is outlined.

Reads data/analyses/exvivo_mouse/maps/.

    python scripts/figures/supp/exvivo_maps.py
"""

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.ticker import MaxNLocator

from camppca_paper.exvivo_mouse import MAPS, NATIVE, shown
from camppca_paper.paths import RESULTS
from camppca_paper.style import MAP_SCALE, MAP_UNITS

OUT = RESULTS / "supp/exvivo_maps.pdf"
NAMES = ("FA", "MD", "MK", "AK", "RK")
ARMS = (("reference", "Reference (20-average)"), ("noisy", "Undenoised"), ("standard", "Standard MPPCA"),
        ("proposed", "Proposed"))
WM_FA = 0.25
CONTOUR = "#39FF14"
WIDTH, LEFT, RIGHT, GAP = 9.0, 0.07, 0.9, 0.02     # inches, and the grid's margins and gaps as fractions


def main():
    mask = np.asarray(nib.load(NATIVE / "mask.nii.gz").dataobj) > 0
    k, aspect, cut = shown(mask)
    wm = cut(((np.asarray(nib.load(MAPS / "reference_FA.nii.gz").dataobj) > WM_FA) & mask)[:, :, k])
    cell = WIDTH * (RIGHT - LEFT) / (len(ARMS) + (len(ARMS) - 1) * GAP) * aspect
    h = cell * (len(NAMES) + (len(NAMES) - 1) * GAP) + 0.4
    fig, axes = plt.subplots(len(NAMES), len(ARMS), figsize=(WIDTH, h),
                             gridspec_kw=dict(wspace=GAP, hspace=GAP, left=LEFT, right=RIGHT, top=1 - 0.35 / h,
                                              bottom=0.05 / h))
    for r, m in enumerate(NAMES):
        panels = {arm: MAP_SCALE.get(m, 1.0) * cut(np.asarray(nib.load(MAPS / f"{arm}_{m}.nii.gz").dataobj,
                                                              float)[:, :, k]) for arm, _ in ARMS}
        vals = np.concatenate([v[np.isfinite(v)] for v in panels.values()])
        lo, hi = np.percentile(vals, (1, 90 if m == "MD" else 99))
        for c, (arm, title) in enumerate(ARMS):
            ax = axes[r, c]
            im = ax.imshow(panels[arm], origin="lower", cmap="gray", vmin=lo, vmax=hi)
            ax.contour(wm.astype(float), levels=[0.5], colors=CONTOUR, linewidths=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
        axes[r, 0].set_ylabel(m, fontsize=11, fontweight="bold")
        cb = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.02, pad=0.01)
        cb.ax.tick_params(labelsize=8)
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        cb.set_label(MAP_UNITS[m], fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
