"""Figures: the shown participant's FA and MK maps in every session of a set, on the slice of the
set's main figure (`camppca_paper.invivo`), in white matter: the T1's posterior above WM_PV, as
in the metrics. Sessions across; for each map Noisy, Standard MPPCA and Proposed one above the
other, on one grey scale from the 5th to the 99th percentile of all its panels. A last column
gives each arm's CoV across sessions per voxel, on one scale per map up to the 95th percentile of
Noisy's. Its label is the median over the whole white matter, from the set's metrics
(data/results/on_harmony/<set>.json).

Reads data/analyses/on_harmony/<subject>/maps/<set>/.

    python scripts/figures/supplementary/invivo_maps.py
"""

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec

from camppca_paper.invivo import ARMS, SHOW, SLICE_AT, VENDOR, metrics, shown_sessions
from camppca_paper.on_harmony import SITES, WORK
from camppca_paper.paths import RESULTS
from camppca_paper.style import MAP_SCALE, MAP_UNITS

OUT = {"within": RESULTS / "supplementary/invivo_within_scanner_maps.pdf",
       "cross": RESULTS / "supplementary/invivo_across_scanner_maps.pdf"}
MAPS = ("FA", "MK")
WM_PV = 0.5
MIN_MEAN = {"FA": 0.0, "MK": 0.1}
COL_W, GAP = 1.25, 0.18     # column width in inches, gap between maps in rows


def figure(which):
    ses = shown_sessions(which)
    frame, maps = WORK / SHOW[which] / "frames" / which, WORK / SHOW[which] / "maps" / which
    mask = np.asarray(nib.load(frame / "mask.nii.gz").dataobj) > 0
    zs = np.where(mask.any((0, 1)))[0]
    k = int(zs[0] + SLICE_AT * (zs[-1] - zs[0]))
    m2 = np.rot90(mask[:, :, k])
    wm = np.rot90((mask & (np.asarray(nib.load(frame / "wm_prob.nii.gz").dataobj) > WM_PV))[:, :, k])
    rows, cols = np.where(m2)
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    whole = metrics(which)[SHOW[which]]["cov"]
    label = (lambda s: f"Session {s.run}") if which == "within" else (lambda s: f"{VENDOR[s.site]}\n{SITES[s.site]}")

    def sl(s, arm, m):
        v = np.rot90(np.asarray(nib.load(maps / f"{s.name}_{arm}_{m}.nii.gz").dataobj, float)[:, :, k])
        return MAP_SCALE.get(m, 1.0) * np.where(wm, v, np.nan)[box]

    data = {(s.name, arm, m): sl(s, arm, m) for s in ses for arm, *_ in ARMS for m in MAPS}
    cov = {}
    for arm, *_ in ARMS:
        for m in MAPS:
            v = np.stack([data[s.name, arm, m] for s in ses])
            mean = v.mean(0)
            c = 100 * v.std(0, ddof=1) / np.where(mean > MAP_SCALE.get(m, 1.0) * MIN_MEAN[m], mean, np.nan)
            cov[arm, m] = c
    h, w = data[ses[0].name, "raw", MAPS[0]].shape
    ratios = sum(([1, 1, 1] + ([GAP] if i < len(MAPS) - 1 else []) for i in range(len(MAPS))), [])
    fig_h = COL_W * h / w * sum(ratios) + 0.45
    widths = [1] * len(ses) + [0.55, 1]
    fig_w = sum(widths) * COL_W + 0.9 + 0.75
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(len(ratios), len(widths), figure=fig, left=0.9 / fig_w, right=1 - 0.75 / fig_w,
                  top=1 - 0.35 / fig_h, bottom=0.05 / fig_h, wspace=0.03, hspace=0.03, height_ratios=ratios,
                  width_ratios=widths)
    for j, m in enumerate(MAPS):
        vals = np.concatenate([v[np.isfinite(v)] for (_, _, mm), v in data.items() if mm == m])
        lo, hi = np.percentile(vals, (5, 99))
        top = np.nanpercentile(cov["raw", m], 95)
        block = []
        for a, (arm, title, _) in enumerate(ARMS):
            r = 4 * j + a
            for c, s in enumerate(ses):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(data[s.name, arm, m], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(label(s), fontsize=9.5)
                if c == 0:
                    ax.set_ylabel(title, fontsize=9.5)
                    if a == 1:
                        ax.text(-0.42, 0.5, m, transform=ax.transAxes, rotation=90, ha="center", va="center",
                                fontsize=12, fontweight="bold")
                block.append(ax)
            ax = fig.add_subplot(gs[r, len(ses) + 1])
            ax.imshow(cov[arm, m], cmap="magma", vmin=0, vmax=top, interpolation="nearest")
            ax.text(0.5, 0.02, f"median {100 * whole[m][arm]:.1f} %", transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title("CoV across\nsessions", fontsize=9.5)
        y0, y1 = block[-1].get_position().y0, block[0].get_position().y1
        x = block[len(ses) - 1].get_position().x1
        cax = fig.add_axes([x + 0.12 / fig_w, y0 + 0.15 * (y1 - y0), 0.12 / fig_w, 0.7 * (y1 - y0)])
        cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(lo, hi), cmap="gray"), cax=cax)
        cb.set_label(MAP_UNITS[m], fontsize=10.5)
        cb.ax.tick_params(labelsize=9.5)
        cax = fig.add_axes([1 - 0.65 / fig_w, y0 + 0.15 * (y1 - y0), 0.12 / fig_w, 0.7 * (y1 - y0)])
        cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, top), cmap="magma"), cax=cax)
        cb.set_label(f"{m} CoV [%]", fontsize=10.5)
        cb.ax.tick_params(labelsize=9.5)
    return fig


def main():
    for which, out in OUT.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        figure(which).savefig(out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
