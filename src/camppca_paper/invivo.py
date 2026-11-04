"""Drawing for the in vivo repeatability figures (ON-Harmony), within and across scanners.

    a       the shown participant (SHOW): the first session's slice with a region boxed, then the
            region in every session of the set, one row per arm. Each session shows the b SHELL
            volume whose direction is closest to the first session's VOLUME, divided by that
            session's mean b0 in the mask. One grey scale within; across scanners one per scanner
            for the regions, ROI_WINDOW percentiles of its Noisy region
    below   the noise level correction per session (across), the signal error (within), and the
            DTI and DKI coefficient of variation, every participant joined across the arms, a
            bar at each median

Reads the pipeline's frames (data/analyses/on_harmony/<subject>/frames/<set>/) and metrics
(data/results/on_harmony/<set>.json).
"""

import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

from .on_harmony import SITES, WORK, sessions
from .paths import RESULTS
from .style import CMP, CNO, COU

SHOW = {"within": "sub-15320", "cross": "sub-16975"}
SHELL = 2000
VOLUME = 30                     # which b SHELL volume of the first session sets the direction
SLICE_AT = 0.6                  # axial slice, fraction of the mask's extent from the bottom
BOX = (0.30, 0.70, 0.60, 0.98)  # region: row and column fractions of the brain's bounding box
ROI_WINDOW = (1, 99)            # across scanners: each region between these Noisy percentiles
ROW_H = 1.62                    # inches per row of point panels
VENDOR = {"OXF1PRI": "Siemens", "OXF2PRI": "Siemens", "OXF3TRI": "Siemens", "NOT1ACH": "Philips",
          "NOT2ING": "Philips", "NOT4GEP": "GE", "OXF4GEP": "GE"}
CROSS_ORDER = ("OXF1PRI", "OXF2PRI", "OXF3TRI", "NOT1ACH", "NOT2ING", "OXF4GEP", "NOT4GEP")
ARMS = (("raw", "Noisy", CNO), ("standard", "Standard MPPCA", CMP), ("proposed", "Proposed", COU))


def metrics(which):
    return json.loads((RESULTS / "on_harmony" / f"{which}.json").read_text())


def shown_sessions(which):
    """SHOW's sessions in display order: within the repeats, across the scanners in CROSS_ORDER."""
    ses = sessions(SHOW[which], which)
    return ses if which == "within" else sorted(ses, key=lambda s: CROSS_ORDER.index(s.site))


def images(which):
    """The shown slice of every session and arm, and the mask on it."""
    frame = WORK / SHOW[which] / "frames" / which
    mask = np.asarray(nib.load(frame / "mask.nii.gz").dataobj) > 0
    zs = np.where(mask.any((0, 1)))[0]
    k = int(zs[0] + SLICE_AT * (zs[-1] - zs[0]))
    out, target = {}, None
    for s in shown_sessions(which):
        bval = np.loadtxt(frame / f"{s.name}.bval").ravel()
        bvec = np.loadtxt(frame / f"{s.name}.bvec").reshape(3, -1)
        shell = np.where(np.abs(bval - SHELL) < 100)[0]
        if target is None:
            target = bvec[:, shell[VOLUME]]
        v = shell[np.argmax(np.abs(target @ bvec[:, shell]))]
        b0 = np.where(bval < 100)[0]
        level = None
        for arm, *_ in ARMS:
            data = np.asarray(nib.load(frame / f"{s.name}_{arm}.nii.gz").dataobj, np.float32)
            if level is None:
                level = float(data[..., b0][mask].mean())
            out[s.name, arm] = np.rot90(data[:, :, k, v] / level)
    return out, np.rot90(mask[:, :, k])


IMAGES = {}


def geometry(m, n):
    """The brain's bounding box, the region's corners and the column width ratios (brain, then n regions)."""
    rows, cols = np.where(m)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    y0, y1 = int(r0 + BOX[0] * (r1 - r0)), int(r0 + BOX[1] * (r1 - r0))
    x0, x1 = int(c0 + BOX[2] * (c1 - c0)), int(c0 + BOX[3] * (c1 - c0))
    widths = [(c1 - c0 + 1) / (r1 - r0 + 1)] + [(x1 - x0) / (y1 - y0)] * n
    return (r0, r1, c0, c1), (y0, y1, x0, x1), widths


def draw_images(fig, spec, which, arrows=None):
    """`arrows` maps a session's run number (within) or site (cross) to (row, column, tail row offset, tail column
    offset) tuples within
    the region; each gets a red arrow in the Standard MPPCA row, pointing at texture that Standard keeps and
    Proposed removes, its tail at the offset from the tip."""
    ims, m = IMAGES[which]
    ses = shown_sessions(which)
    (r0, r1, c0, c1), (y0, y1, x0, x1), widths = geometry(m, len(ses))
    brain = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    first = ses[0].name
    lo, hi = np.percentile(np.concatenate([ims[first, arm][m] for arm, *_ in ARMS]), (0.5, 99.5))
    # cross: each scanner's region on its own grey scale, from its Noisy region (contrast differs by scanner)
    window = {s.name: (np.percentile(ims[s.name, "raw"][y0:y1, x0:x1], ROI_WINDOW) if which == "cross" else (lo, hi))
              for s in ses}
    label = (lambda s: f"Session {s.run}") if which == "within" else (lambda s: f"{VENDOR[s.site]}\n{SITES[s.site]}")
    grid = GridSpecFromSubplotSpec(3, len(widths), subplot_spec=spec, width_ratios=widths, wspace=0.02, hspace=0.02)
    for i, (arm, title, _) in enumerate(ARMS):
        ax = fig.add_subplot(grid[i, 0])
        ax.imshow(ims[first, arm][brain], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.add_patch(Rectangle((x0 - c0 - 0.5, y0 - r0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="#E69F00", lw=1.0))
        ax.set_ylabel(title, fontsize=9)
        axes = [ax]
        if i == 0:
            ax.set_title(label(ses[0]), fontsize=8)
            ax.text(-0.13, 1.03, "a", transform=ax.transAxes, fontsize=14, fontweight="bold")
        for j, s in enumerate(ses, start=1):
            bx = fig.add_subplot(grid[i, j])
            bx.imshow(ims[s.name, arm][y0:y1, x0:x1], cmap="gray", vmin=window[s.name][0], vmax=window[s.name][1],
                      interpolation="nearest")
            if i == 0:
                bx.set_title(label(s), fontsize=8)
            for y, x, dy, dx in (arrows or {}).get(s.run if which == "within" else s.site, ()):
                if arm == "standard":
                    bx.annotate("", xy=(x, y), xytext=(x + dx, y + dy), annotation_clip=False,
                                arrowprops=dict(arrowstyle="-|>", color="#D62728", lw=1.6, shrinkA=0, shrinkB=1.5,
                                                mutation_scale=9))
            axes.append(bx)
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])
            a.set_aspect("auto")


def signal_bars(ax, d, subjects):
    """The within signal error per shell: one bar per arm, the mean variance (light) with the mean bias squared (dark)
    stacked on top, so the bar is the mean squared error (means add, medians do not); every participant's squared
    error as points joined across the arms."""
    arms = (("standard", CMP), ("proposed", COU))
    for i, b in enumerate((1000, 2000)):
        pos = [i - 0.2, i + 0.2]
        for k, (arm, col) in enumerate(arms):
            var = np.mean([d[s]["signal"][f"b{b}"][arm]["var_pct"] for s in subjects])
            b2 = np.mean([d[s]["signal"][f"b{b}"][arm]["bias2_pct"] for s in subjects])
            ax.bar(pos[k], var, 0.34, color=col, alpha=0.3, ec=col, lw=0.6, zorder=1)
            ax.bar(pos[k], b2, 0.34, bottom=var, color=col, ec="k", lw=0.4, zorder=1)
        vals = np.array([[d[s]["signal"][f"b{b}"][arm]["mse_pct"] for arm, _ in arms] for s in subjects])
        for row in vals:
            ax.plot(pos, row, "-", color="0.4", lw=0.5, zorder=2)
        for k, (_, col) in enumerate(arms):
            ax.plot(np.full(len(vals), pos[k]), vals[:, k], "o", color=col, mec="k", mew=0.4, ms=3, zorder=3)


def correction_points(ax, which):
    """The applied correction of every session of the set (`metrics.py`: the median over a session's brain of
    sigma_proposed / sigma_standard - 1), one column per scanner in CROSS_ORDER, a bar at each median. Within, a
    column is one participant's six sessions; across, every participant's session on that scanner."""
    d = metrics(which)
    rng = np.random.default_rng(0)
    for i, site in enumerate(correction_sites(which)):
        v = [y for r in d.values() for session, y in r["correction"].items() if session[:7] == site]
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", color=COU, mec="k", mew=0.4, ms=4, zorder=3)
        ax.hlines(np.median(v), i - 0.18, i + 0.18, color=COU, lw=2.2, zorder=1)


def correction_sites(which):
    """The scanners the set has sessions on, in CROSS_ORDER."""
    d = metrics(which)
    return [x for x in CROSS_ORDER if any(k[:7] == x for r in d.values() for k in r["correction"])]


def rows(which, correction=False):
    """The panels below the images, one list per row: (title, y label, [(group label, value(subject, arm))], arms),
    or (title, y label, [(group label, None)], None, draw(ax)) for a panel drawn its own way."""
    d = metrics(which)
    three = (("raw", CNO), ("standard", CMP), ("proposed", COU))
    across = "sessions" if which == "within" else "scanners"

    def cv(maps):
        return [(m, lambda s, a, m=m: 100 * d[s]["cov"][m][a]) for m in maps]

    maps = [("DTI", f"CoV across {across} [%]", cv(("FA", "MD")), three),
            ("DKI", f"CoV across {across} [%]", cv(("MK", "AK", "RK")), three)]
    bars = ("Signal level", "MSE relative to Noisy [%]", [("b 1000", None), ("b 2000", None)], None,
            lambda ax: signal_bars(ax, d, list(d)))
    scanners = [(SITES[site], None) for site in correction_sites(which)]
    first = [("Noise level correction", "$\\sigma_{proposed} / \\sigma_{standard} - 1$ [%]",
              scanners, None,
              lambda ax: correction_points(ax, which))] if correction else []
    if which == "cross":
        return list(d), [first + maps]
    return list(d), [first + [bars] + maps]


def draw_row(fig, spec, subjects, panels, letters):
    """One row of point panels: every participant joined across the arms, a bar at each median."""
    # a panel of many groups (the correction by scanner) packs them closer, so it does not take the row
    grid = GridSpecFromSubplotSpec(1, len(panels), subplot_spec=spec,
                                   width_ratios=[len(p[2]) if len(p[2]) <= 3 else 0.55 * len(p[2]) for p in panels],
                                   wspace=0.5 if len(panels) > 3 else 0.3)
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(panels))]
    for ax, (title, ylabel, groups, arms, *draw), letter in zip(axes, panels, letters, strict=True):
        if draw:
            draw[0](ax)
        step = 0.4 if arms and len(arms) == 2 else 0.27
        for i, (_, get) in enumerate(groups if arms else ()):
            vals = np.array([[get(s, a) for a, _ in arms] for s in subjects])
            pos = [i + (k - (len(arms) - 1) / 2) * step for k in range(len(arms))]
            for row in vals:
                ax.plot(pos, row, "-", color="0.6", lw=0.6, zorder=2)
            for k, (_, col) in enumerate(arms):
                ax.plot(np.full(len(vals), pos[k]), vals[:, k], "o", color=col, mec="k", mew=0.4, ms=4, zorder=3)
                ax.hlines(np.median(vals[:, k]), pos[k] - 0.1, pos[k] + 0.1, color=col, lw=2.2, zorder=1)
        ax.set_xticks(range(len(groups)))
        if len(groups) > 3:
            ax.set_xticklabels([g[0] for g in groups], fontsize=8, rotation=40, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels([g[0] for g in groups], fontsize=9)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.text(-0.22 if len(groups) < 3 else -0.15, 1.08, letter, transform=ax.transAxes, fontsize=14,
                fontweight="bold")
    return axes


class Key(HandlerBase):
    """Legend key drawn from one colour per arm: "bars", small upright bars side by side, light body under a dark
    cap, as the stacked bars; "median", short thick stripes on top of each other, as the median marks. The
    instance is its own handle."""

    def __init__(self, style, label, colours):
        super().__init__()
        self.style, self.label, self.colours = style, label, colours

    def get_label(self):
        return self.label

    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        out = []
        if self.style == "bars":
            w = 0.3 * width
            for k, col in enumerate(self.colours):
                x = x0 + 0.17 * width + k * (w + 0.06 * width)
                out.append(Rectangle((x, y0 - 0.25 * height), w, 1.05 * height, transform=trans, fc=col, alpha=0.3,
                                     ec=col, lw=0.6))
                out.append(Rectangle((x, y0 + 0.8 * height), w, 0.3 * height, transform=trans, fc=col, ec="k",
                                     lw=0.4))
            return out
        n, gap = len(self.colours), 0.06 * height
        h = (1.4 * height - (n - 1) * gap) / n
        top = y0 + height / 2 + (n * h + (n - 1) * gap) / 2
        for k, col in enumerate(self.colours):
            y = top - (k + 1) * h - k * gap
            out.append(Rectangle((x0 + 0.12 * width, y + 0.05 * h), 0.76 * width, 0.9 * h, transform=trans, fc=col,
                                 ec="none"))
        return out


def figure(which, correction=True, fig_w=8.64, top_h=0.14, row_h=ROW_H, bars_key_inside=False, left=0.055,
           arrows=None):
    """The within or cross figure: a, the images; below, the point panels; the legend between. `fig_w` is the width in inches at fixed font size, `top_h` the room above the
    images and `row_h` the height of a row of point panels, in inches. `bars_key_inside` moves the variance/bias
    key of the within figure from the shared legend into the signal panel, with headroom made for it. `left` is
    the point panels' left margin as a figure fraction."""
    img_left, right = 0.028, 0.996  # the images use the full width; the point panels keep room for labels
    n = len(shown_sessions(which))
    IMAGES[which] = images(which)
    *_, widths = geometry(IMAGES[which][1], n)
    img_h = fig_w * (right - img_left) / sum(widths) * 3
    subjects, point_rows = rows(which, correction)
    k = len(point_rows)
    legend_h, gap_h, bottom_h = 0.72, 0.45, 0.27  # the legend, between rows, below
    block_h = k * row_h + (k - 1) * gap_h
    height = top_h + img_h + legend_h + block_h + bottom_h
    fig = plt.figure(figsize=(fig_w, height))
    y = 1 - top_h / height
    draw_images(fig, GridSpec(1, 1, figure=fig, left=img_left, right=right, top=y, bottom=y - img_h / height)[0],
                which, arrows)
    block = GridSpec(k, 1, figure=fig, left=left, right=right, top=bottom_h / height + block_h / height,
                     bottom=bottom_h / height, hspace=gap_h / row_h)
    letters = iter("bcdefg")
    first, bars_ax = None, None
    for r, panels in enumerate(point_rows):
        axes = draw_row(fig, block[r], subjects, panels, [next(letters) for _ in panels])
        first = first or axes[0]
        bars_ax = next((ax for ax, p in zip(axes, panels, strict=True) if p[0] == "Signal level"), bars_ax)
    handles = [plt.Line2D([], [], color=c, marker="o", mec="k", mew=0.4, ls="", ms=5, label=t) for _, t, c in ARMS]
    keys = [Key("median", f"Median across {'scanners' if which == 'within' else 'subjects'}",
                    [c for *_, c in ARMS])]
    bars_key = Key("bars", "Mean variance (bright) and bias$^2$ (dark)", (CMP, COU))
    if which == "within" and not bars_key_inside:
        keys += [bars_key]
    handles += keys
    handler = {h: h for h in keys}
    if which == "within" and bars_key_inside and bars_ax is not None:
        inside = Key("bars", "Mean variance (bright)\nand bias$^2$ (dark)", (CMP, COU))
        bars_ax.set_ylim(top=1.35 * bars_ax.get_ylim()[1])
        bars_ax.legend(handles=[inside], handler_map={inside: inside}, frameon=False, fontsize=7,
                       loc="upper left", handlelength=1.6, labelspacing=0.2)
    fig.legend(handles=handles, handler_map=handler, frameon=False, fontsize=8, ncol=len(handles), loc="center",
               bbox_to_anchor=(0.5, first.get_position().y1 + 0.42 / height))
    return fig
