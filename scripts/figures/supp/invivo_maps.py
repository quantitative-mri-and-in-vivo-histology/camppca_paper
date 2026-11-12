"""Figures: FA and MK maps in every session of a set, on the slice of the set's main figure, over
the brain, with the white matter outlined: the T1's posterior above WM_PV, as in the metrics. The
sets are ON-Harmony's shown participant within and across scanners (`camppca_paper.harmony_figure`)
and the EDDEN runs, magnitude and complex. Sessions across; for each map Undenoised, MPPCA
and Proposed one above the other, with NORDIC for the complex runs, on one grey scale from the 5th
to the 99th percentile of the denoised panels in white matter. A last column gives each arm's CoV
across repeats or scanners per voxel, on one scale per map up to the 95th percentile of MPPCA's in
white matter. The colour bars sit below each map. The CoV label is the median over the whole white
matter, from the set's metrics (data/results/on_harmony/<set>.json, data/results/edden/<part>.json).

Reads each set's frames (mask, wm_prob) and maps.

    python scripts/figures/supp/invivo_maps.py
"""

import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec

from camppca_paper import edden
from camppca_paper.edden_figure import NORDIC
from camppca_paper.harmony_figure import ARMS, SHOW, SLICE_AT, VENDOR, metrics, shown_sessions
from camppca_paper.on_harmony import SITES, WORK
from camppca_paper.paths import RESULTS
from camppca_paper.style import MAP_SCALE, MAP_UNITS

MAPS = ("FA", "MK")
WM_PV = 0.5
MIN_MEAN = {"FA": 0.0, "MK": 0.1}
COL_W, CB_IN = 1.25, 0.85     # column width, room for a colour bar below each map block, inches
CONTOUR = "#39FF14"


def sets():
    """(output, frame folder, maps folder, [(session name, column title)], white-matter CoV, arms, CoV across) per set."""
    out = []
    for which, name in (("within", "invivo_repeat_harmony_maps"), ("cross", "invivo_reprod_harmony_maps")):
        label = ((lambda s: f"Repeat {s.run}") if which == "within"
                 else (lambda s: f"{VENDOR[s.site]}\n{SITES[s.site]}"))
        base = WORK / SHOW[which]
        out.append((RESULTS / f"supp/{name}.pdf", base / "frames" / which, base / "maps" / which,
                    [(s.name, label(s)) for s in shown_sessions(which)], metrics(which)[SHOW[which]]["cov"], ARMS,
                    "repeats" if which == "within" else "scanners"))
    for part, name, arms in (("magnitude", "invivo_repeat_edden_maps", ARMS),
                             ("complex", "invivo_repeat_edden_complex_maps", ARMS[:2] + (NORDIC,) + ARMS[2:])):
        base = edden.work(part)
        out.append((RESULTS / f"supp/{name}.pdf", base / "frames", base / "maps",
                    [(r, f"Repeat {i}") for i, r in enumerate(edden.RUNS, start=1)],
                    json.loads((RESULTS / f"edden/{part}.json").read_text())["cov"], arms, "repeats"))
    return out


def figure(frame, maps, ses, whole, arms, across):
    mask = np.asarray(nib.load(frame / "mask.nii.gz").dataobj) > 0
    zs = np.where(mask.any((0, 1)))[0]
    k = int(zs[0] + SLICE_AT * (zs[-1] - zs[0]))
    m2 = np.rot90(mask[:, :, k])
    wm = np.rot90((mask & (np.asarray(nib.load(frame / "wm_prob.nii.gz").dataobj) > WM_PV))[:, :, k])
    rows, cols = np.where(m2)
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))

    def sl(s, arm, m):
        v = np.rot90(np.asarray(nib.load(maps / f"{s}_{arm}_{m}.nii.gz").dataobj, float)[:, :, k])
        return MAP_SCALE.get(m, 1.0) * np.where(m2, v, np.nan)[box]

    wmb = wm[box]
    data = {(s, arm, m): sl(s, arm, m) for s, _ in ses for arm, *_ in arms for m in MAPS}
    cov = {}
    for arm, *_ in arms:
        for m in MAPS:
            v = np.stack([data[s, arm, m] for s, _ in ses])
            mean = v.mean(0)
            c = 100 * v.std(0, ddof=1) / np.where(mean > MAP_SCALE.get(m, 1.0) * MIN_MEAN[m], mean, np.nan)
            cov[arm, m] = c
    h, w = data[ses[0][0], "raw", MAPS[0]].shape
    n = len(arms)
    map_h = COL_W * h / w
    cb_row = CB_IN / map_h
    ratios = sum(([1] * n + [cb_row] for _ in MAPS), [])
    fig_h = map_h * sum(ratios) + 0.4
    widths = [1] * len(ses) + [0.12, 1]
    fig_w = sum(widths) * COL_W + 0.45 + 0.1
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(len(ratios), len(widths), figure=fig, left=0.45 / fig_w, right=1 - 0.1 / fig_w,
                  top=1 - 0.35 / fig_h, bottom=0.05 / fig_h, wspace=0.03, hspace=0.03, height_ratios=ratios,
                  width_ratios=widths)
    for j, m in enumerate(MAPS):
        vals = np.concatenate([v[wmb & np.isfinite(v)] for (_, arm, mm), v in data.items() if mm == m and arm != "raw"])
        lo, hi = np.percentile(vals, (5, 99))
        top = np.nanpercentile(cov["standard", m][wmb], 95)
        block, covs = [], []
        for a, (arm, title, _) in enumerate(arms):
            r = (n + 1) * j + a
            for c, (s, title_s) in enumerate(ses):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(data[s, arm, m], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
                ax.contour(wmb, levels=[0.5], colors=CONTOUR, linewidths=0.4)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(title_s, fontsize=9.5)
                if c == 0:
                    ax.set_ylabel(title, fontsize=9.5)
                block.append(ax)
            ax = fig.add_subplot(gs[r, len(ses) + 1])
            ax.imshow(cov[arm, m], cmap="magma", vmin=0, vmax=top, interpolation="nearest")
            ax.contour(wmb, levels=[0.5], colors=CONTOUR, linewidths=0.7)
            ax.text(0.5, 0.02, f"WM median {100 * whole[m][arm]:.1f} %", transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"CoV across\n{across}", fontsize=9.5)
            covs.append(ax)
        top_left = block[0].get_position()
        fig.text(top_left.x0 - 0.24 / fig_w, top_left.y1 - 0.02 / fig_h, "ab"[j], fontsize=14, fontweight="bold",
                 va="bottom")
        y = block[-1].get_position().y0 - (0.1 + 0.1) / fig_h
        x0, x1 = block[0].get_position().x0, block[len(ses) - 1].get_position().x1
        cax = fig.add_axes([x0 + 0.2 * (x1 - x0), y, 0.6 * (x1 - x0), 0.1 / fig_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(lo, hi), cmap="gray"), cax=cax, orientation="horizontal")
        cb.set_label(MAP_UNITS[m], fontsize=10.5)
        cb.ax.tick_params(labelsize=9.5)
        c0, c1 = covs[-1].get_position().x0, covs[-1].get_position().x1
        cax = fig.add_axes([c0 + 0.08 * (c1 - c0), y, 0.84 * (c1 - c0), 0.1 / fig_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, top), cmap="magma"), cax=cax, orientation="horizontal")
        cb.set_label(f"{m} CoV [%]", fontsize=10.5)
        cb.ax.tick_params(labelsize=9.5)
    return fig


def main():
    for out, *args in sets():
        out.parent.mkdir(parents=True, exist_ok=True)
        figure(*args).savefig(out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
