"""Figure: in vivo repeatability at 0.9 mm (EDDEN, four repeat runs of one subject on one scanner).

a       one axial slice of run-01 with a region boxed, then the region in every run, one row per
        arm. Each run shows the b SHELL volume whose direction is closest to run-01's VOLUME,
        divided by that run's mean Noisy b0 on the slice in the mask. One grey scale
b       noise level of Standard MPPCA and Proposed over the mean b0, one point per run
c       signal error per shell, variance (light) and bias squared (dark), as % of Noisy's
d, e    DTI and DKI coefficient of variation across the runs, median over white matter

Written twice: without NORDIC (main) and with it (supplementary).

Reads data/analyses/edden/frames and data/results/edden/within.json.

    python scripts/figures/main/edden_within.py
"""

import json
from multiprocessing import Pool

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle

from camppca_paper.edden import RUNS, WORK
from camppca_paper.invivo import SHELL, SLICE_AT, VOLUME, Key, geometry
from camppca_paper.paths import RESULTS
from camppca_paper.style import CMP, CND, CNO, COU

OUT = {"main": RESULTS / "main/edden_within.pdf", "nordic": RESULTS / "supplementary/edden_within_nordic.pdf"}
FRAME = WORK / "frames"
ARMS = (("raw", "Noisy", CNO), ("standard", "Standard MPPCA", CMP), ("proposed", "Proposed", COU))
NORDIC = ("nordic2", "NORDIC", CND)


def volumes():
    """The shown volume of every run and its b0 volumes."""
    out, target = {}, None
    for r in RUNS:
        bval = np.loadtxt(FRAME / f"{r}.bval").ravel()
        bvec = np.loadtxt(FRAME / f"{r}.bvec").reshape(3, -1)
        shell = np.where(np.abs(bval - SHELL) < 100)[0]
        if target is None:
            target = bvec[:, shell[VOLUME]]
        out[r] = int(shell[np.argmax(np.abs(target @ bvec[:, shell]))]), np.where(bval < 100)[0]
    return out


def read(task):
    r, arm, k, v, b0 = task
    img = nib.load(FRAME / f"{r}_{arm}.nii.gz").dataobj
    return (r, arm), np.asarray(img[:, :, k, v], np.float32), (np.asarray(img[:, :, k, :], np.float32)[..., b0]
                                                                if arm == "raw" else None)


def images(arms):
    """The shown slice of every run and arm, and the mask on it."""
    mask = np.asarray(nib.load(FRAME / "mask.nii.gz").dataobj) > 0
    zs = np.where(mask.any((0, 1)))[0]
    k = int(zs[0] + SLICE_AT * (zs[-1] - zs[0]))
    vols = volumes()
    with Pool(8) as pool:
        got = pool.map(read, [(r, arm, k, *vols[r]) for r in RUNS for arm, *_ in arms])
    level = {r: float(b0[mask[:, :, k]].mean()) for (r, _), _, b0 in got if b0 is not None}
    return {key: np.rot90(x / level[key[0]]) for key, x, _ in got}, np.rot90(mask[:, :, k])


def draw_images(fig, spec, ims, m, arms):
    (r0, r1, c0, c1), (y0, y1, x0, x1), widths = geometry(m, len(RUNS))
    first = RUNS[0]
    lo, hi = np.percentile(np.concatenate([ims[first, arm][m] for arm, *_ in arms]), (0.5, 99.5))
    grid = GridSpecFromSubplotSpec(len(arms), len(widths), subplot_spec=spec, width_ratios=widths, wspace=0.02,
                                   hspace=0.02)
    for i, (arm, title, _) in enumerate(arms):
        ax = fig.add_subplot(grid[i, 0])
        ax.imshow(ims[first, arm][r0:r1 + 1, c0:c1 + 1], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.add_patch(Rectangle((x0 - c0 - 0.5, y0 - r0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="#E69F00", lw=1.0))
        ax.set_ylabel(title, fontsize=9)
        axes = [ax]
        if i == 0:
            ax.set_title("Session 1", fontsize=8)
            ax.text(-0.13, 1.03, "a", transform=ax.transAxes, fontsize=14, fontweight="bold")
        for j, r in enumerate(RUNS, start=1):
            bx = fig.add_subplot(grid[i, j])
            bx.imshow(ims[r, arm][y0:y1, x0:x1], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            if i == 0:
                bx.set_title(f"Session {j}", fontsize=8)
            axes.append(bx)
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])
            a.set_aspect("auto")


def noise_points(ax, noise):
    pos = (-0.2, 0.2)
    v = np.array([[noise[r][arm] for arm in ("standard", "proposed")] for r in RUNS])
    for row in v:
        ax.plot(pos, row, "-", color="0.6", lw=0.6, zorder=2)
    for k, col in enumerate((CMP, COU)):
        ax.plot(np.full(len(v), pos[k]), v[:, k], "o", color=col, mec="k", mew=0.4, ms=4, zorder=3)
        ax.hlines(np.median(v[:, k]), pos[k] - 0.1, pos[k] + 0.1, color=col, lw=2.2, zorder=1)
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.8 * v.min(), 1.12 * v.max())


def signal_bars(ax, sig, arms):
    arms = [(arm, col) for arm, _, col in arms if arm != "raw"]
    w = 0.8 / len(arms)
    for i, b in enumerate((1000, 2000)):
        for k, (arm, col) in enumerate(arms):
            x = i + (k - (len(arms) - 1) / 2) * w
            row = sig[f"b{b}"][arm]
            ax.bar(x, row["var_pct"], 0.85 * w, color=col, alpha=0.3, ec=col, lw=0.6)
            ax.bar(x, row["bias2_pct"], 0.85 * w, bottom=row["var_pct"], color=col, ec="k", lw=0.4)
    ax.set_xticks([0, 1], ["b 1000", "b 2000"], fontsize=9)


def cov_bars(ax, c, maps, arms):
    w = 0.8 / len(arms)
    for i, m in enumerate(maps):
        for k, (arm, _, col) in enumerate(arms):
            ax.bar(i + (k - (len(arms) - 1) / 2) * w, 100 * c[m][arm], 0.92 * w, color=col, ec="k", lw=0.4)
    ax.set_xticks(range(len(maps)), maps, fontsize=9)


def figure(res, ims, m, arms):
    fig_w, top_h, row_h, img_left, right, left = 0.9 * 8.64, 0.3, 1.15 * 1.62, 0.028, 0.996, 0.06
    *_, widths = geometry(m, len(RUNS))
    img_h = fig_w * (right - img_left) / sum(widths) * len(arms)
    legend_h, bottom_h = 0.6, 0.27
    height = top_h + img_h + legend_h + row_h + bottom_h
    fig = plt.figure(figsize=(fig_w, height))
    y = 1 - top_h / height
    draw_images(fig, GridSpec(1, 1, figure=fig, left=img_left, right=right, top=y, bottom=y - img_h / height)[0],
                ims, m, arms)
    row = GridSpec(1, 4, figure=fig, left=left, right=right, top=(bottom_h + row_h) / height, bottom=bottom_h / height,
                   width_ratios=[1, 2, 2, 3], wspace=0.35)
    panels = [fig.add_subplot(row[0, i]) for i in range(4)]
    noise_points(panels[0], res["noise"])
    signal_bars(panels[1], res["signal"], arms)
    cov_bars(panels[2], res["cov"], ("FA", "MD"), arms)
    cov_bars(panels[3], res["cov"], ("MK", "AK", "RK"), arms)
    labels = (("Noise level", "$\\hat\\sigma$ / mean b0 [%]"), ("Signal level", "MSE relative to Noisy [%]"),
              ("DTI", "CoV across sessions [%]"), ("DKI", "CoV across sessions [%]"))
    for ax, (title, ylabel), letter in zip(panels, labels, "bcde", strict=True):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.text(-0.55 if ax is panels[0] else -0.22, 1.08, letter, transform=ax.transAxes, fontsize=14,
                fontweight="bold")
    for ax in panels[1:]:
        ax.set_ylim(bottom=0)
    inside = Key("bars", "Variance (bright)\nand bias$^2$ (dark)", tuple(c for a, _, c in arms if a != "raw"))
    panels[1].set_ylim(top=1.35 * panels[1].get_ylim()[1])
    panels[1].legend(handles=[inside], handler_map={inside: inside}, frameon=False, fontsize=7, loc="upper left",
                     handlelength=2.2, labelspacing=0.2)
    handles = [Patch(fc=c, ec="k", lw=0.4, label=t) for _, t, c in arms]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles), loc="center",
               bbox_to_anchor=(0.5, panels[0].get_position().y1 + 0.42 / height))
    return fig


def main():
    res = json.loads((RESULTS / "edden/within.json").read_text())
    ims, m = images(ARMS + (NORDIC,))
    for key, arms in (("main", ARMS), ("nordic", ARMS[:2] + (NORDIC,) + ARMS[2:])):
        fig = figure(res, ims, m, arms)
        OUT[key].parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"savefig.pad_inches": 0.02}):
            fig.savefig(OUT[key])
        print(f"  wrote {OUT[key]}")


if __name__ == "__main__":
    main()
