"""Figure: the noise covariance across the public datasets (data/results/panel/noise.json).

a   prediction error of Standard MPPCA's noise level against the correction, one point per scan
    by vendor, medians dashed, the distributions in strips on top and on the right
b   the correction with the level only and with the correlation only, as % of the full model's
c   the correlation table along phase encode, readout and slice, lags 1 to REACH, one thin line
    per scan, the vendor medians and the mean over all scans thick
d   sigma-hat over sigma on pure noise whitened with the other subject's correlation, the mean
    over the other configurations, or none

    python scripts/figures/main/generalization_invivo.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from camppca_paper.panel import vendor
from camppca_paper.paths import RESULTS
from camppca_paper.style import VENDOR_COLOURS

OUT = RESULTS / "main/generalization_invivo.pdf"
REACH = 4
TITLE = dict(fontsize=10, fontweight="bold", pad=10)


def rows():
    out = []
    for r in json.loads((RESULTS / "panel/noise.json").read_text()).values():
        rho = np.asarray(r["rho"])
        c = rho.shape[0] // 2
        x, y, z = rho[c:c + REACH + 1, c, 0], rho[c, c:c + REACH + 1, 0], rho[c, c, :REACH + 1]
        pe, ro = (y, x) if r["pe_axis"] == 1 else (x, y)
        out.append(r | dict(vendor=vendor(r["config"]), profile={"Phase encode": pe, "Readout": ro, "Slice": z}))
    return out


def vendor_dots(ax, i, values, vendors, rng):
    for v, vend in zip(values, vendors, strict=True):
        ax.plot(i + rng.uniform(-0.18, 0.18), v, "o", color=VENDOR_COLOURS[vend], ms=4, mec="k", mew=0.3,
                alpha=0.85, zorder=3)
    ax.hlines(np.median(values), i - 0.28, i + 0.28, color="k", lw=2, zorder=4)


def panel_prediction(ax, rs):
    corr = [r["correction"] for r in rs]
    err = [r["prediction"] for r in rs]
    for v, col in VENDOR_COLOURS.items():
        sel = [r for r in rs if r["vendor"] == v]
        ax.plot([r["correction"] for r in sel], [r["prediction"] for r in sel], "o", color=col, ms=5, mec="k",
                mew=0.4, label=f"{v} ({len(sel)})")
    xmed, ymed = np.median(corr), np.median(err)
    ax.axhline(ymed, color="k", lw=1, ls="--", zorder=1)
    ax.axvline(xmed, color="k", lw=1, ls="--", zorder=1)
    ax.text(0.98, ymed, f"median {ymed:.1f} %", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8)
    ax.text(xmed + 0.3, 0.98, f"median {xmed:.1f} %", transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=8)
    ax.set_xlabel("Correction ($\\sigma_{proposed} / \\sigma_{standard} - 1$) [%]")
    ax.set_ylabel("Prediction error\n($\\sigma_{pred} / \\sigma_{standard} - 1$) [%]")
    ax.set_xlim(0, np.ceil(max(corr)) + 1.5)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    div = make_axes_locatable(ax)
    top = div.append_axes("top", size="24%", pad=0.08, sharex=ax)
    right = div.append_axes("right", size="18%", pad=0.08, sharey=ax)
    top.hist(corr, bins=np.arange(0, np.ceil(max(corr)) + 1, 1.0), color="0.75", edgecolor="k", lw=0.3)
    right.hist(err, bins=np.arange(0, np.ceil(4 * max(err)) / 4 + 0.25, 0.25), color="0.75", edgecolor="k",
               lw=0.3, orientation="horizontal")
    top.axvline(xmed, color="k", lw=1, ls="--", zorder=3)
    right.axhline(ymed, color="k", lw=1, ls="--", zorder=3)
    top.tick_params(labelbottom=False, labelsize=8)
    right.tick_params(labelleft=False, labelsize=8)
    top.set_ylabel("Datasets", fontsize=8)
    right.set_xlabel("Datasets", fontsize=8)
    for a in (top, right):
        a.spines[["top", "right"]].set_visible(False)
    return top


def panel_components(ax, rs):
    rng = np.random.default_rng(0)
    for i, arm in enumerate(("level", "correlation")):
        vendor_dots(ax, i, [100 * r["components"][arm] / r["components"]["full"] for r in rs],
                    [r["vendor"] for r in rs], rng)
    ax.axhline(100, color="0.4", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Spatially varying\nlevel only", "Correlation\nonly"], fontsize=9)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Correction\n[% of the full model's]")


def panel_profiles(axes, rs):
    lags = np.arange(1, REACH + 1)
    for ax, direction in zip(axes, ("Phase encode", "Readout", "Slice"), strict=True):
        for v, col in VENDOR_COLOURS.items():
            p = np.array([r["profile"][direction] for r in rs if r["vendor"] == v])
            for line in p:
                ax.plot(lags, line[1:], color=col, lw=0.6, alpha=0.45)
            ax.plot(lags, np.median(p, 0)[1:], color=col, lw=2.4, label=f"{v} ({len(p)})")
        ax.plot(lags, np.mean([r["profile"][direction] for r in rs], 0)[1:], color="0.35", lw=3.2, zorder=5,
                label=f"Mean ({len(rs)})")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(lags)
        ax.set_xlabel(f"{direction} lag [voxels]")
        ax.set_ylim(-0.2, 0.45)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("Noise correlation")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")


def panel_transfer(ax, rs):
    rng = np.random.default_rng(0)
    arms = (("other_subject", "Second\nsubject"), ("mean", "Across-dataset\nmean"),
            ("none", "None\n(Standard MPPCA)"))
    ax.axhline(1, color="0.4", lw=0.8, ls="--", zorder=1)
    for i, (arm, _) in enumerate(arms):
        vendor_dots(ax, i, [r["transfer"][arm] for r in rs], [r["vendor"] for r in rs], rng)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([t for _, t in arms], fontsize=8)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_xlabel("Source of correlation pattern")
    ax.set_ylabel(r"$\hat{\sigma}\,/\,\sigma$ on simulated noise")


def figure():
    rs = rows()
    fig = plt.figure(figsize=(0.95 * 0.88 * 12.0, 0.95 * 0.88 * 8.4))
    top = GridSpec(1, 2, figure=fig, left=0.063, right=0.993, top=0.961, bottom=0.545, wspace=0.25,
                   width_ratios=[1.5, 1])
    low = GridSpec(1, 2, figure=fig, left=0.051, right=0.993, top=0.415, bottom=0.059, wspace=0.18,
                   width_ratios=[3, 1.25])

    a = fig.add_subplot(top[0])
    panel_prediction(a, rs).set_title("Noise level correction versus Standard MPPCA", **TITLE)
    b = fig.add_subplot(top[1])
    panel_components(b, rs)
    b.set_title("What drives the noise level correction?", **TITLE)
    vendors = [plt.Line2D([], [], ls="", marker="o", color=col, ms=5, mec="k", mew=0.4,
                          label=f"{v} ({sum(r['vendor'] == v for r in rs)})") for v, col in VENDOR_COLOURS.items()]
    b.legend(handles=vendors, frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.08))

    axes = [fig.add_subplot(g) for g in low[0].subgridspec(1, 3, wspace=0.08)]
    panel_profiles(axes, rs)
    y1 = axes[0].get_position().y1
    mid = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
    fig.text(mid, y1 + TITLE["pad"] / 72 / fig.get_size_inches()[1], "Correlation along the image axes",
             ha="center", va="baseline", fontsize=TITLE["fontsize"], fontweight=TITLE["fontweight"])

    d = fig.add_subplot(low[1])
    panel_transfer(d, rs)
    d.set_title("Correlation transfer", **TITLE)
    d.legend(handles=vendors, frameon=False, fontsize=8, loc="lower left")

    for x, y, s in ((-0.005, 0.956, "a"), (0.6, 0.956, "b"), (-0.005, y1 + 0.03, "c")):
        fig.text(x, y, s, fontsize=14, fontweight="bold")
    d.text(-0.25, 1.06, "d", transform=d.transAxes, fontsize=14, fontweight="bold")
    for ax in (a, b, d, *axes):
        ax.tick_params(labelsize=8)
    return fig


if __name__ == "__main__":
    fig = figure()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        fig.savefig(OUT)
    print(f"  wrote {OUT}")
