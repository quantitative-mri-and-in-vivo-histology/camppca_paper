"""Drawing for the in vivo repeatability figures at 0.9 mm (EDDEN, four repeat runs of one subject
on one scanner).

a       one axial slice of run-01 with a region boxed, then the region in every run, one row per
        arm. Each run shows the b SHELL volume whose direction is closest to run-01's VOLUME,
        divided by that run's mean Noisy b0 on the slice in the mask. One grey scale
b       noise level of each denoiser over the mean b0, mean over the runs
c       signal error per shell, variance (light) and bias squared (dark), as % of Noisy's,
        magnitude only
d, e    DTI and DKI coefficient of variation across the runs, median over white matter

Bar width and spacing as in `camppca_paper.invivo`.

Reads data/analyses/edden/<part>/frames and data/results/edden/<part>.json.
"""


import json
from multiprocessing import Pool

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle

from .edden import RUNS, work
from .invivo import (
    SHELL,
    SLICE_AT,
    STEP,
    VOLUME,
    Key,
    W,
    centre,
    draw_row,
    geometry,
    panel_letters,
)
from .paths import RESULTS
from .style import CMP, CND, CNO, COU

ARMS = (("raw", "Noisy", CNO), ("standard", "Standard MPPCA", CMP), ("proposed", "Proposed", COU))
NORDIC = ("nordic2", "NORDIC", CND)


def volumes(frame):
    """The shown volume of every run and its b0 volumes."""
    out, target = {}, None
    for r in RUNS:
        bval = np.loadtxt(frame / f"{r}.bval").ravel()
        bvec = np.loadtxt(frame / f"{r}.bvec").reshape(3, -1)
        shell = np.where(np.abs(bval - SHELL) < 100)[0]
        if target is None:
            target = bvec[:, shell[VOLUME]]
        out[r] = int(shell[np.argmax(np.abs(target @ bvec[:, shell]))]), np.where(bval < 100)[0]
    return out


def read(task):
    frame, r, arm, k, v, b0 = task
    img = nib.load(frame / f"{r}_{arm}.nii.gz").dataobj
    return (r, arm), np.asarray(img[:, :, k, v], np.float32), (np.asarray(img[:, :, k, :], np.float32)[..., b0]
                                                                if arm == "raw" else None)


def images(frame, arms):
    """The shown slice of every run and arm, and the mask on it."""
    mask = np.asarray(nib.load(frame / "mask.nii.gz").dataobj) > 0
    zs = np.where(mask.any((0, 1)))[0]
    k = int(zs[0] + SLICE_AT * (zs[-1] - zs[0]))
    vols = volumes(frame)
    with Pool(8) as pool:
        got = pool.map(read, [(frame, r, arm, k, *vols[r]) for r in RUNS for arm, *_ in arms])
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


def bars(ax, groups, arms):
    """One light bar per arm in each group of values."""
    n = len(arms)
    for i, vals in enumerate(groups):
        for k, (arm, col) in enumerate(arms):
            ax.bar(centre(i, n) + (k - (n - 1) / 2) * STEP, vals[arm], W, color=col, alpha=0.3, ec=col, lw=0.6)


def signal_bars(ax, sig, arms):
    n = len(arms)
    for i, b in enumerate((1000, 2000)):
        for k, (arm, col) in enumerate(arms):
            x = centre(i, n) + (k - (n - 1) / 2) * STEP
            row = sig[f"b{b}"][arm]
            ax.bar(x, row["var_pct"], W, color=col, alpha=0.3, ec=col, lw=0.6)
            ax.bar(x, row["bias2_pct"], W, bottom=row["var_pct"], color=col, ec="k", lw=0.4)


def draw(res, ims, m, arms):
    fig_w, top_h, row_h, img_left, right, left = 0.9 * 8.64, 0.3, 1.15 * 1.62, 0.028, 0.996, 0.06
    *_, widths = geometry(m, len(RUNS))
    img_h = fig_w * (right - img_left) / sum(widths) * len(arms)
    legend_h, bottom_h = 0.6, 0.27
    height = top_h + img_h + legend_h + row_h + bottom_h
    fig = plt.figure(figsize=(fig_w, height))
    y = 1 - top_h / height
    draw_images(fig, GridSpec(1, 1, figure=fig, left=img_left, right=right, top=y, bottom=y - img_h / height)[0],
                ims, m, arms)
    every = [(a, c) for a, _, c in arms]
    denoised = [(a, c) for a, c in every if a != "raw"]

    two = [(a, c) for a, c in denoised if a in res["noise"][RUNS[0]]]
    noise = {a: np.mean([res["noise"][r][a] for r in RUNS]) for a, _ in two}

    def cv(maps):
        return lambda ax: bars(ax, [{a: 100 * res["cov"][x][a] for a, _ in every} for x in maps], every)

    panels = [("Noise level", "$\\hat\\sigma$ / mean b0 [%]", [("", None)], two, lambda ax: bars(ax, [noise], two))]
    if "signal" in res:
        panels.append(("Signal level", "MSE relative to Noisy [%]", [("b 1000", None), ("b 2000", None)], denoised,
                       lambda ax: signal_bars(ax, res["signal"], denoised)))
    panels += [("DTI", "CoV across sessions [%]", [("FA", None), ("MD", None)], every, cv(("FA", "MD"))),
              ("DKI", "CoV across sessions [%]", [("MK", None), ("AK", None), ("RK", None)], every,
               cv(("MK", "AK", "RK")))]
    row = GridSpec(1, 1, figure=fig, left=left, right=right, top=(bottom_h + row_h) / height, bottom=bottom_h / height)
    axes = draw_row(fig, row[0], ["edden"], panels)
    if "signal" in res:
        inside = Key("bars", "Variance (bright)\nand bias$^2$ (dark)", tuple(c for _, c in denoised))
        axes[1].set_ylim(top=1.35 * axes[1].get_ylim()[1])
        axes[1].legend(handles=[inside], handler_map={inside: inside}, frameon=False, fontsize=6.5, loc="upper left",
                       handlelength=2.2, labelspacing=0.2)
    handles = [Key("bar", t, (c,)) for _, t, c in arms]
    fig.legend(handles=handles, handler_map={h: h for h in handles}, frameon=False, fontsize=8, ncol=len(handles),
               loc="center", bbox_to_anchor=(0.5, axes[0].get_position().y1 + 0.42 / height))
    panel_letters(fig, axes)
    return fig


def figure(part):
    """The figure of one input: the magnitude, or the complex data with NORDIC."""
    arms = ARMS if part == "magnitude" else ARMS[:2] + (NORDIC,) + ARMS[2:]
    res = json.loads((RESULTS / f"edden/{part}.json").read_text())
    return draw(res, *images(work(part) / "frames", arms), arms)
