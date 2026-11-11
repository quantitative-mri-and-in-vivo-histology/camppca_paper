"""Figure: the simulation on the subject phantom.

a   one acquisition at window blend 0.5 and maximum gain 2: image, residual and noise level
    for ground truth, undenoised, Standard MPPCA and Proposed.
b-g two sweeps, the gain field alone and the window alone, N_DRAW draws per cell, scored over
    the brain: the true noise level over the thermal level (b, c), sigma-hat over the true level
    (d, e), and the MSE over every volume split into variance and bias squared (f, g).

Each simulation is saved with its settings and rerun only when they change.

    python scripts/figures/main/simulation.py
"""


import camppca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from camppca_paper.paths import DATA_ROOT, WORK
from camppca_paper.simulation import cached, load_phantom, simulate_acquisition
from camppca_paper.style import CMP, COU, RCM
from camppca_paper.zoom import box, zoomed

PATCH = (5, 5, 5)
QUAL = (0.5, 2.0)                              # (f_c, f_d) of panel a
SHOW_B = 2000                                  # the shown volume's shell
ROI_CENTRE = (0.586, 0.774)    # zoom box centre, fractions of the panel, bottom-left origin
ROI_SIZE = 0.27                # zoom box side, fraction of the panel's shorter side
N_DRAW = 20
F_D = (1.0, 1.75, 2.5, 3.25, 4.0)              # gain sweep, window off
F_C = (0.2, 0.4, 0.6, 0.8)                     # window sweep, gain off

QUANT = WORK / "simulations_quant.npz"
QUAL_CACHE = WORK / "simulations_qual.npz"
OUT = DATA_ROOT / "results/main/simulation.pdf"

ARMS = (("standard", CMP, "Standard MPPCA"), ("proposed", COU, "Proposed"))
MIXES = (("level", "Spatially varying level only", "sd_d", r"Spread of the noise level (std($d$))"),
         ("corr", "Correlation only", "rho1", r"Nearest-neighbor correlation ($\rho(1)$)"))


# ---------------------------------------------------------------------------- simulations

def cells():
    """(mix, f_c, f_d) of every sweep cell. The white cell (first) also opens the window sweep."""
    return [("level", 0.0, v) for v in F_D] + [("corr", v, 1.0) for v in F_C]


def run_quant(signal, mask, bval):
    """Every cell, N_DRAW draws, both arms over the brain.

    Per cell: each arm's sigma-hat in the brain per draw, and its variance and bias squared over
    draws, pooled over brain voxels and volumes. Undenoised's too.
    """
    out = {}
    for i, (mix, f_c, f_d) in enumerate(cells()):
        s1 = {k: 0.0 for k in ("noisy", "standard", "proposed")}
        s2 = dict(s1)
        sig = {"standard": [], "proposed": []}
        y = None
        for r in range(N_DRAW):
            a = simulate_acquisition(signal, mask, bval, f_c, f_d, seed=r, y=y)
            y = a["y"]
            std = camppca.denoise_mppca(a["y_noisy"], mask=mask, patch=PATCH)
            prop = camppca.denoise_camppca(a["y_noisy"], bvals=bval, mask=mask, patch=PATCH, mask_only=True)
            for k, x in (("noisy", a["y_noisy"]), ("standard", std.denoised), ("proposed", prop.denoised)):
                x = x[mask].astype(float)
                s1[k] = s1[k] + x
                s2[k] = s2[k] + x ** 2
            sig["standard"].append(std.sigma[mask])
            sig["proposed"].append(prop.sigma[mask])
        truth = a["y"][mask].astype(float)
        for k in s1:
            mean = s1[k] / N_DRAW
            var = (s2[k] - N_DRAW * mean ** 2) / (N_DRAW - 1)
            out[f"{i}_{k}_var"] = float(var.mean())
            out[f"{i}_{k}_b2"] = float(((mean - truth) ** 2 - var / N_DRAW).mean())
        out |= {f"{i}_sigma_{k}": np.asarray(v, np.float32) for k, v in sig.items()}
        out |= {f"{i}_sigma": a["sigma"][mask], f"{i}_thermal": a["sigma_thermal"], f"{i}_rho1": a["rho1"],
                f"{i}_sd_d": a["sd_d"]}
        print(f"  {mix:5s} f_c {f_c:.2f} f_d {f_d:.2f}  {N_DRAW} draws", flush=True)
    return out


def run_qual(signal, mask, bval):
    """One acquisition at QUAL, both arms over the whole brain."""
    a = simulate_acquisition(signal, mask, bval, *QUAL)
    std = camppca.denoise_mppca(a["y_noisy"], mask=mask, patch=PATCH)
    prop = camppca.denoise_camppca(a["y_noisy"], bvals=bval, mask=mask, patch=PATCH)
    return dict(truth=a["y"], noisy=a["y_noisy"], standard=std.denoised,
                proposed=prop.denoised, sigma=(a["sigma"] * mask).astype(np.float32),
                sigma_standard=(std.sigma * mask).astype(np.float32),
                sigma_proposed=(prop.sigma * mask).astype(np.float32), mask=mask)


# ---------------------------------------------------------------------------- statistics

def quartiles(x):
    return np.percentile(x, (25, 50, 75))


def sweep_stats(q):
    """Per mix, per cell: x values, true level over thermal, sigma-hat over true, MSE split."""
    res = {"level": [], "corr": []}
    for i, (mix, _, _) in enumerate(cells()):
        sig = q[f"{i}_sigma"].astype(float)
        row = dict(sd_d=float(q[f"{i}_sd_d"]), rho1=float(q[f"{i}_rho1"]),
                   level=quartiles(sig / float(q[f"{i}_thermal"])))
        var_n, b2_n = float(q[f"{i}_noisy_var"]), float(q[f"{i}_noisy_b2"])
        for arm, _, _ in ARMS:
            var, b2 = float(q[f"{i}_{arm}_var"]), float(q[f"{i}_{arm}_b2"])
            row[arm] = dict(ratio=quartiles(q[f"{i}_sigma_{arm}"] / sig),
                            var=100 * var / (var_n + b2_n), mse=100 * (var + b2) / (var_n + b2_n))
        res[mix].append(row)
        if mix == "level" and i == 0:
            res["corr"].append(row)
    return res


# ---------------------------------------------------------------------------- panel a

COLS = ("Ground truth", "Undenoised", "Standard MPPCA", "Proposed")
ROWS = ("Image", "Residual", "Noise level")


def panel_a(s, bval):
    """Three rows (image, residual, noise level) by four columns (the arms)."""
    M4 = s["mask"]
    zc = M4.shape[2] // 2
    m2 = M4[:, :, zc]
    yy, xx = np.where(m2)
    sl2 = (slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))
    mc = m2[sl2]

    def cut(A2):
        return np.where(mc, A2[sl2], np.nan).T

    shown = np.where(np.abs(bval - SHOW_B) <= 100)[0]
    vol = int(shown[len(shown) // 2])
    r0, c0, RZ = box(cut(s["truth"][:, :, zc, vol]).shape, ROI_CENTRE, ROI_SIZE)

    ev = 2.0 * float((s["noisy"] - s["truth"])[M4].std())
    sv = np.concatenate([s[k][M4] for k in ("sigma", "sigma_standard", "sigma_proposed")])
    smin, smax = (float(v) for v in np.percentile(sv, (5, 95)))
    sticks = [t for t in MaxNLocator(nbins=3, steps=[1, 2, 5, 10]).tick_values(smin, smax)
              if smin <= t <= smax]

    ih, iw = mc.T.shape
    FIGW, L, RT, TOP, BOT = 10.4, 0.060, 0.996, 0.930, 0.010
    NC, NR, WS, HS = 4, 3, 0.030, 0.035
    u = (RT - L) * FIGW / (NC * (1 + WS * (NC - 1) / NC))
    figh = u * ih / iw * NR * (1 + HS * (NR - 1) / NR) / (TOP - BOT)
    fig = plt.figure(figsize=(FIGW, figh))
    gs = fig.add_gridspec(NR, NC, hspace=HS, wspace=WS, left=L, right=RT, top=TOP, bottom=BOT)

    lim = float(np.nanpercentile(s["truth"][:, :, zc, vol][m2], 99.5))
    gk = dict(vmin=0, vmax=lim, cmap="gray")
    rk = dict(vmin=-ev, vmax=ev, cmap=RCM)
    sk = dict(vmin=smin, vmax=smax, cmap="magma")
    bars = {(1, 0): (rk, "Residual vs\nground truth", [-ev, 0, ev], ["−2σ", "0", "+2σ"]),
            (2, 1): (sk, r"Noise level $\sigma(x)$", sticks, [f"{t:g}" for t in sticks])}
    arms = ("truth", "noisy", "standard", "proposed")
    grid = ([s[k][:, :, zc, vol] for k in arms],
            [None] + [(s[k] - s["truth"])[:, :, zc, vol] for k in arms[1:]],
            [s["sigma"][:, :, zc], None, s["sigma_standard"][:, :, zc],
             s["sigma_proposed"][:, :, zc]])
    rowax = []
    for i, cells_ in enumerate(grid):
        for k, A in enumerate(cells_):
            a = fig.add_subplot(gs[i, k])
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if k == 0:
                rowax.append(a)
            if i == 0:
                a.set_title(COLS[k], fontsize=11.5, fontweight="bold", pad=7)
            if A is None:
                kw, title, ticks, labels = bars[(i, k)]
                cax = a.inset_axes([0.08, 0.46, 0.84, 0.075])
                cax.set_title(title, fontsize=10, pad=2)
                cb = fig.colorbar(ScalarMappable(Normalize(kw["vmin"], kw["vmax"]), cmap=kw["cmap"]),
                                  cax=cax, orientation="horizontal")
                cb.set_ticks(ticks)
                cb.set_ticklabels(labels)
                cb.ax.tick_params(labelsize=9, length=2, pad=1.5)
                cb.outline.set_linewidth(0.6)
                continue
            zoomed(a, cut(A), (gk, rk, sk)[i], r0, c0, RZ)
    for a, name in zip(rowax, ROWS, strict=True):
        q = a.get_position()
        fig.text(L - 0.012, 0.5 * (q.y0 + q.y1), name, rotation=90, va="center", ha="center",
                 fontsize=11.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------- panels b-g

QUAL_SHARE = 0.5     # panel a's share of the width
SCALE = 0.68         # the whole figure is drawn, then shrunk; fonts keep their size
GAP = 1.35           # inches between panel a and b-g
LETTER_DROP = 0.006
LETTER_SHIFT = 0.008
WSPACE_A = 0.01      # gap between panel a's image columns, fraction of a column
TICK, LABEL, TITLE = 10, 11.5, 12.5


def tint(col, share):
    """`col` mixed with white, opaque."""
    return tuple(share * np.array(to_rgb(col)) + (1 - share))


class BarsKey(HandlerBase):
    """Legend key: one small bar per arm, light body under a dark cap."""

    def __init__(self, label, colours):
        super().__init__()
        self.label, self.colours = label, colours

    def get_label(self):
        return self.label

    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        out, w = [], 0.3 * width
        for k, col in enumerate(self.colours):
            x = x0 + 0.17 * width + k * (w + 0.06 * width)
            out.append(Rectangle((x, y0 - 0.25 * height), w, 1.05 * height, transform=trans,
                                 fc=col, alpha=0.3, ec=col, lw=0.6))
            out.append(Rectangle((x, y0 + 0.8 * height), w, 0.3 * height, transform=trans,
                                 fc=col, ec="k", lw=0.4))
        return out


def widen(fig):
    """Make panel a take QUAL_SHARE of a wider figure. Returns (a's width, total width, y of
    the column titles)."""
    w, _ = fig.get_size_inches()
    W = w / QUAL_SHARE
    fig.set_size_inches(W, fig.get_size_inches()[1])
    k = w / W
    gs = fig.axes[0].get_subplotspec().get_gridspec()
    gs.update(left=gs.left * k, right=gs.right * k)
    n = gs.ncols
    cell = (gs.right - gs.left) / (n + gs.wspace * (n - 1))
    right = gs.left + cell * (n + WSPACE_A * (n - 1))
    w -= (gs.right - right) * W
    gs.update(right=right, wspace=WSPACE_A)
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x * k, y))
    x_rows = min(t.get_position()[0] for t in fig.texts)
    y_titles = fig.axes[0].get_position().y1 + 0.012
    fig.text(x_rows, y_titles - LETTER_DROP, "a", fontsize=15, fontweight="bold",
             ha="center", va="bottom")
    return w, W, y_titles


def panels_bg(grid, res):
    line = dict(lw=1.6, ms=4, mec="k", mew=0.4)
    for j, (mix, title, xkey, xlabel) in enumerate(MIXES):
        rows = res[mix]
        x = [r[xkey] for r in rows]

        ax = grid[0, j]
        ax.axhline(1.0, color="0.6", lw=1.0, ls="--", zorder=0,
                   label=r"Thermal, $\sigma_{\mathrm{thermal}}$")
        q = np.array([r["level"] for r in rows])
        ax.fill_between(x, q[:, 0], q[:, 2], color="k", alpha=0.2, lw=0)
        ax.plot(x, q[:, 1], "-", color="k", lw=1.4, label=r"Image level, $\sigma$")
        ax.set_title(title, fontsize=TITLE, fontweight="bold")

        ax = grid[1, j]
        ax.axhline(1.0, color="0.4", lw=0.8, zorder=0)
        for arm, col, name in ARMS:
            q = np.array([r[arm]["ratio"] for r in rows])
            ax.fill_between(x, q[:, 0], q[:, 2], color=col, alpha=0.2, lw=0)
            ax.plot(x, q[:, 1], "-o", color=col, label=name, **line)

        ax = grid[2, j]
        for z, (arm, col, name) in enumerate(ARMS):
            var = np.array([r[arm]["var"] for r in rows])
            mse = np.array([r[arm]["mse"] for r in rows])
            ax.fill_between(x, 0, var, color=tint(col, 0.3), lw=0, zorder=1 + 3 * z)
            ax.fill_between(x, var, mse, color=tint(col, 0.75), lw=0, zorder=2 + 3 * z)
            ax.plot(x, mse, "-o", color=col, label=name, zorder=3 + 3 * z, **line)
        for art in (*ax.spines.values(), ax.xaxis, ax.yaxis):
            art.set_zorder(10)

        for ax in grid[:, j]:
            ax.tick_params(labelsize=TICK)
        grid[2, j].set_xlabel(xlabel, fontsize=LABEL)
        for ax in grid[:2, j]:
            ax.tick_params(labelbottom=False)
    grid[0, 0].set_ylabel(r"$\sigma(x)$ / $\sigma_{\mathrm{thermal}}$", fontsize=LABEL)
    grid[1, 0].set_ylabel(r"$\hat{\sigma}(x)$ / $\sigma(x)$", fontsize=LABEL)
    grid[2, 0].set_ylabel("MSE [% of Undenoised]", fontsize=LABEL)
    for ax in grid[1]:
        ax.legend(frameon=False, fontsize=TICK, loc="lower left")
    top = max(r[arm]["mse"] for m in res for r in res[m] for arm, _, _ in ARMS)
    grid[2, 0].set_ylim(0, 1.05 * top)
    key = BarsKey("Variance (bright)\nand bias$^2$ (dark)", [c for _, c, _ in ARMS])
    handles = grid[2, 0].get_legend_handles_labels()[0]
    for ax in grid[2]:
        ax.legend(handles=handles + [key], handler_map={key: key}, frameon=False,
                  fontsize=TICK, loc="upper left", handlelength=1.6)


def figure(qual, res, bval):
    fig = panel_a(qual, bval)
    h = fig.get_size_inches()[1]
    w, W, y_titles = widen(fig)
    right = fig.add_gridspec(3, 2, left=(w + GAP) / W, right=0.99, top=0.93, bottom=0.06,
                             wspace=0.05, hspace=0.12)
    grid = np.array([[fig.add_subplot(right[i, j]) for j in range(2)] for i in range(3)])
    for i in range(3):
        grid[i, 1].sharey(grid[i, 0])
        grid[i, 1].tick_params(labelleft=False)
    for j in range(2):
        for i in (0, 1):
            grid[i, j].sharex(grid[2, j])
    panels_bg(grid, res)
    for i in range(3):
        for j in range(2):
            x = ((w + GAP * 0.35) / W if j == 0 else grid[i, j].get_position().x0 - 0.012)
            y = (y_titles if i == 0 else grid[i, j].get_position().y1 + 0.012) - LETTER_DROP
            fig.text(x + LETTER_SHIFT, y, "bcdefg"[2 * i + j], fontsize=15, fontweight="bold",
                     ha="center", va="bottom")
    fig.set_size_inches(SCALE * W, SCALE * h)
    return fig


def main():
    signal, mask, bval = load_phantom()
    common = dict(patch=PATCH)
    qual = cached(QUAL_CACHE, dict(common, qual=QUAL),
                  lambda: run_qual(signal, mask, bval))
    quant = cached(QUANT, dict(common, scope="brain", n_draw=N_DRAW, f_d=F_D,
                               f_c=F_C), lambda: run_quant(signal, mask, bval))
    fig = figure(qual, sweep_stats(quant), bval)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
