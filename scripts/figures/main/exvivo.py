"""Figure: the ex vivo mouse brain against its twenty-average reference.

a       the middle slice, flipped along y as Henriques et al. show it, at the middle b SHELL
        volume. Columns Reference, Undenoised, MPPCA and Proposed. Rows the image, the image
        minus the reference, and the noise level, the true one under Reference
b       sigma-hat against the true level, voxel by voxel: smoothed 2-D histograms with contours
        around 50 and 90 % of the voxels, a cross at the medians
c       signal mean squared error per shell, as % of Undenoised's
d, e    DTI and DKI RMS difference from the reference's maps in white matter

Reads data/analyses/exvivo_mouse/native and data/results/exvivo_mouse/metrics.json.

    python scripts/figures/main/exvivo.py
"""

import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from camppca_paper.exvivo_mouse import BVAL, NATIVE, shown
from camppca_paper.harmony_figure import Key, draw_row, panel_letters
from camppca_paper.paths import RESULTS
from camppca_paper.style import B_AXIS, CMP, CNO, COU

OUT = RESULTS / "main/exvivo.pdf"
SHELL = 2000
ARMS = (("noisy", "Undenoised", CNO), ("standard", "MPPCA", CMP), ("proposed", "Proposed", COU))


def load(name):
    return np.asarray(nib.load(NATIVE / f"{name}.nii.gz").dataobj, np.float64)


def draw_images(fig, spec, z):
    k, _, cut = shown(z["mask"])
    vols = np.where(np.round(np.loadtxt(BVAL).ravel(), -3) == SHELL)[0]
    v = int(vols[len(vols) // 2])
    cols = [("Reference", "reference")] + [(t, a) for a, t, _ in ARMS]
    imgs = [cut(z[a][:, :, k, v]) for _, a in cols]
    floor = 1e-3 * np.median(z["sigma_true"][z["mask"]])
    ok = ((z["sigma_standard"] > floor) & (z["sigma_proposed"] > floor) & z["mask"])[:, :, k]
    sigma = {c: cut(np.where(ok, z[c][:, :, k], np.nan)) for c in ("sigma_true", "sigma_standard", "sigma_proposed")}
    grid = spec.subgridspec(3, 4, wspace=0.02, hspace=0.02)
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(4)] for i in range(3)])
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    vmax = float(np.nanpercentile(imgs[0], 99.5))
    for j, ((title, _), im) in enumerate(zip(cols, imgs, strict=True)):
        axes[0, j].imshow(im, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        axes[0, j].set_title(title, fontsize=10)
    lim = float(np.nanpercentile(np.abs(imgs[1] - imgs[0]), 99))
    for j in (1, 2, 3):
        rim = axes[1, j].imshow(imgs[j] - imgs[0], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    cb = fig.colorbar(rim, cax=axes[1, 0].inset_axes([0.1, 0.45, 0.8, 0.08]), orientation="horizontal")
    cb.set_label("Difference to reference", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    vals = np.concatenate([s[np.isfinite(s)] for s in sigma.values()])
    lo, hi = np.percentile(vals, (1, 99))
    for j, key in ((0, "sigma_true"), (2, "sigma_standard"), (3, "sigma_proposed")):
        im = axes[2, j].imshow(sigma[key], origin="lower", cmap="magma", vmin=lo, vmax=hi)
    cb = fig.colorbar(im, cax=axes[2, 1].inset_axes([0.1, 0.45, 0.8, 0.08]), orientation="horizontal")
    cb.set_label("Noise level $\\sigma$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for a, name in zip(axes[:, 0], ("Image", "Difference", "Noise level"), strict=True):
        a.set_ylabel(name, fontsize=9)
    axes[0, 0].text(-0.12, 1.03, "a", transform=axes[0, 0].transAxes, fontsize=14, fontweight="bold")


def noise_panel(ax, z):
    x = z["sigma_true"][z["mask"]]
    lo, hi = np.percentile(np.concatenate([x, z["sigma_standard"][z["mask"]], z["sigma_proposed"][z["mask"]]]), (0.5, 99.5))
    for key, col in (("sigma_standard", CMP), ("sigma_proposed", COU)):
        y = z[key][z["mask"]]
        H, xe, ye = np.histogram2d(x, y, bins=150, range=((lo, hi), (lo, hi)))
        H = gaussian_filter(H, 1.2).T
        order = np.sort(H.ravel())[::-1]
        cum = np.cumsum(order) / order.sum()
        levels = [order[np.searchsorted(cum, q)] for q in (0.9, 0.5)]
        xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
        ax.contourf(xc, yc, H, levels=[levels[0], H.max()], colors=[col], alpha=0.15)
        ax.contour(xc, yc, H, levels=levels, colors=[col], linewidths=[0.8, 1.4])
        ax.plot(np.median(x), np.median(y), "x", color=col, ms=8, mew=2, zorder=5)
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Reference noise level $\\sigma$", fontsize=8)
    ax.set_ylabel("Estimated noise level $\\hat\\sigma$", fontsize=8)
    ax.set_title("Noise level", fontsize=10)
    ax.tick_params(labelsize=8)


def bars(ax, groups, arms):
    from camppca_paper.harmony_figure import STEP, W, centre
    n = len(arms)
    for i, vals in enumerate(groups):
        for k, (arm, col) in enumerate(arms):
            ax.bar(centre(i, n) + (k - (n - 1) / 2) * STEP, vals[arm], W, color=col, alpha=0.3, ec=col, lw=0.6)


def figure(z, res):
    _, aspect, _ = shown(z["mask"])
    fig_w, left, right = 0.9 * 8.64, 0.06, 0.996
    img_h = fig_w * (right - left) / (4 + 3 * 0.02) * aspect * (3 + 2 * 0.02)
    top_h, legend_h, row_h, bottom_h = 0.3, 0.6, 1.15 * 1.62, 0.45
    height = top_h + img_h + legend_h + row_h + bottom_h
    fig = plt.figure(figsize=(fig_w, height))
    y = 1 - top_h / height
    draw_images(fig, GridSpec(1, 1, figure=fig, left=left, right=right, top=y, bottom=y - img_h / height)[0], z)
    lo, hi = bottom_h / height, (bottom_h + row_h) / height
    b_w = row_h / fig_w
    b = fig.add_axes((left + 0.02, lo, b_w, hi - lo))
    noise_panel(b, z)
    every = [(a, c) for a, _, c in ARMS]
    two = every[1:]
    shells = list(res["signal"])

    def maps(names):
        return lambda ax: bars(ax, [res["maps"][m] for m in names], every)

    panels = [("Signal level", "MSE relative to Undenoised [%]", [(s[1:], None) for s in shells], two,
               lambda ax: bars(ax, [res["signal"][s] for s in shells], two)),
              ("DTI", "RMS difference to\nreference [%]", [("FA", None), ("MD", None)], every, maps(("FA", "MD"))),
              ("DKI", "RMS difference to\nreference [%]", [("MK", None), ("AK", None), ("RK", None)], every,
               maps(("MK", "AK", "RK")))]
    row = GridSpec(1, 1, figure=fig, left=left + 0.02 + b_w + 0.1, right=right, top=hi, bottom=lo)
    axes = draw_row(fig, row[0], ["sample"], panels)
    axes[0].set_xlabel(B_AXIS, fontsize=9)
    handles = [Key("bar", t, (c,)) for _, t, c in ARMS]
    fig.legend(handles=handles, handler_map={h: h for h in handles}, frameon=False, fontsize=8, ncol=len(handles),
               loc="center", bbox_to_anchor=(0.5, hi + 0.42 / height))
    panel_letters(fig, [b] + axes)
    return fig


if __name__ == "__main__":
    names = ("mask", "reference", "noisy", "standard", "proposed", "sigma_true", "sigma_standard", "sigma_proposed")
    z = {n: load(n) for n in names}
    z["mask"] = z["mask"] > 0
    fig = figure(z, json.loads((RESULTS / "exvivo_mouse/metrics.json").read_text()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        fig.savefig(OUT)
    print(f"  wrote {OUT}")
