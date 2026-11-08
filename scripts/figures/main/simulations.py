"""Figure: the simulation on the subject phantom.

a   one acquisition at window blend 0.5 and gain amplitude 1.0: image, residual and noise level
    for ground truth, noisy, Standard MPPCA and Proposed.
b-g two sweeps, the gain field alone and the window alone, N_DRAW draws per cell, scored at
    N_PATCHES patch centres: the true noise level over the thermal level (b, c), sigma-hat over
    the true level (d, e), and the MSE split into variance and bias squared (f, g).

Each simulation is saved with its settings and rerun only when they change.

    python scripts/figures/main/simulations.py
"""

import json

import camppca
import matplotlib.pyplot as plt
import numpy as np
from camppca.denoiser.camppca import denoise_whitened
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter
from scipy.spatial import ConvexHull

from camppca_paper.paths import DATA_ROOT, WORK
from camppca_paper.simulation import (
    PATCH,
    load_phantom,
    sample_centres,
    simulate_acquisition,
)
from camppca_paper.style import CMP, COU, RCM

SNR = 20
SEED = 5
N_PATCHES = 1500
N_DRAW = 10
F_D = (0.0, 0.75, 1.5, 2.25, 3.0)              # gain sweep, window off
F_CORR = (0.2, 0.4, 0.6, 0.8)                  # window sweep, gain off
QUAL = (0.5, 1.0)                              # (f, f_d) of panel a
SHOW_B = 2000                                  # panel a shows a volume of this shell

QUANT = WORK / "simulations_quant.npz"
QUAL_CACHE = WORK / "simulations_qual.npz"
OUT = DATA_ROOT / "results/main/simulations.pdf"

ARMS = (("standard", CMP, "Standard MPPCA"), ("proposed", COU, "Proposed"))
MIXES = (("level", "Spatially varying level only", "sd_d", r"Spread of the noise level (std($d$))"),
         ("corr", "Correlation only", "rho1", r"Nearest-neighbor correlation ($\rho(1)$)"))


# ---------------------------------------------------------------------------- simulations

def cached(path, settings, run):
    """`run()` saved to `path` with `settings`, or loaded from there if the settings match."""
    key = json.dumps(settings, sort_keys=True)
    if path.exists():
        with np.load(path) as z:
            if str(z["settings"]) == key:
                return {k: z[k] for k in z.files}
    out = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, settings=key, **out)
    return out


def cells():
    """(mix, f, f_d) of every sweep cell. The white cell (first) also opens the window sweep."""
    return [("level", 0.0, v) for v in F_D] + [("corr", f, 0.0) for f in F_CORR]


def denoise_camppca_at(data, mask, bval, where):
    """camppca with the noise model estimated on `mask`, denoised only at the voxels `where`.

    Returns `(denoised, sigma_hat)`, `sigma_hat` in data units.
    """
    model = camppca.estimate_noise_model(data, bvals=bval, mask=mask, patch=PATCH)
    d = model.effective_d()
    den, sig, _ = denoise_whitened(data, where, model.patch_correlation(PATCH), patch=PATCH,
                                   d_field=d)
    return den, sig * d


def run_quant(signal, mask, bval):
    """Every cell, N_DRAW draws, both arms at the patch centres."""
    where = sample_centres(mask, N_PATCHES)
    out = {}
    for i, (mix, f, f_d) in enumerate(cells()):
        rows = {k: [] for k in ("noisy", "standard", "proposed", "sigma_standard", "sigma_proposed")}
        for r in range(N_DRAW):
            a = simulate_acquisition(signal, mask, bval, SNR, f, f_d, SEED + 1000 * r)
            std = camppca.denoise_mppca(a["data"], mask=where, patch=PATCH)
            den_p, sig_p = denoise_camppca_at(a["data"], mask, bval, where)
            rows["noisy"].append(a["data"][where])
            rows["standard"].append(std.denoised[where])
            rows["proposed"].append(den_p[where])
            rows["sigma_standard"].append(std.sigma_hat[where])
            rows["sigma_proposed"].append(sig_p[where])
        out |= {f"{i}_{k}": np.asarray(v, np.float32) for k, v in rows.items()}
        out |= {f"{i}_truth": a["truth"][where], f"{i}_sigma": a["sigma"][where],
                f"{i}_thermal": a["thermal"], f"{i}_rho1": a["rho1"], f"{i}_sd_d": a["sd_d"]}
        print(f"  {mix:5s} f {f:.2f} f_d {f_d:.2f}  {N_DRAW} draws", flush=True)
    return out


def patch_mean(x, mask):
    """Each mask voxel's value averaged over the mask voxels of its patch."""
    num = uniform_filter(np.where(mask, x, 0).astype(np.float32), PATCH, mode="constant")
    den = uniform_filter(mask.astype(np.float32), PATCH, mode="constant")
    return np.where(mask, num / np.maximum(den, 1e-12), 0).astype(np.float32)


def run_qual(signal, mask, bval):
    """One acquisition at QUAL, both arms over the whole brain."""
    a = simulate_acquisition(signal, mask, bval, SNR, *QUAL, SEED)
    std = camppca.denoise_mppca(a["data"], mask=mask, patch=PATCH)
    prop = camppca.denoise_camppca(a["data"], bvals=bval, mask=mask, patch=PATCH)
    return dict(truth=a["truth"], noisy=a["data"], standard=std.denoised,
                proposed=prop.denoised, sigma=(a["sigma"] * mask).astype(np.float32),
                sigma_standard=patch_mean(std.sigma_hat, mask),
                sigma_proposed=patch_mean(prop.sigma_hat, mask), mask=mask)


# ---------------------------------------------------------------------------- statistics

def quartiles(x):
    return np.percentile(x, (25, 50, 75))


def bias_variance(X, truth):
    """Variance and bias squared over draws, pooled over centres and volumes."""
    X = X.astype(float)
    var = X.var(0, ddof=1)
    b2 = (X.mean(0) - truth) ** 2 - var / len(X)
    return float(var.mean()), float(b2.mean())


def sweep_stats(q):
    """Per mix, per cell: x values, true level over thermal, sigma-hat over true, MSE split."""
    res = {"level": [], "corr": []}
    for i, (mix, _, _) in enumerate(cells()):
        sig, truth = q[f"{i}_sigma"].astype(float), q[f"{i}_truth"].astype(float)
        row = dict(sd_d=float(q[f"{i}_sd_d"]), rho1=float(q[f"{i}_rho1"]),
                   level=quartiles(sig / float(q[f"{i}_thermal"])))
        var_n, b2_n = bias_variance(q[f"{i}_noisy"], truth)
        for arm, _, _ in ARMS:
            var, b2 = bias_variance(q[f"{i}_{arm}"], truth)
            row[arm] = dict(ratio=quartiles(q[f"{i}_sigma_{arm}"] / sig),
                            var=100 * var / (var_n + b2_n), mse=100 * (var + b2) / (var_n + b2_n))
        res[mix].append(row)
        if mix == "level" and i == 0:
            res["corr"].append(row)
    return res


# ---------------------------------------------------------------------------- panel a

ROI_CENTRE = (0.586, 0.774)    # zoom box centre, fractions of the panel, bottom-left origin
ROI_SIZE = 0.27                # zoom box side, fraction of the panel's shorter side
ZC = "#00c853"                 # zoom box colour
IZ = 0.552                     # magnifier size, fraction of the panel
INSET_DROP = 0.05              # how far the magnifier hangs below the panel
COLS = ("Ground truth", "Noisy", "Standard MPPCA", "Proposed")
ROWS = ("Image", "Residual", "Noise level")


def connect(a, box, ia, inset):
    """Dashed lines from the zoom box to the magnifier, along the outside of both squares."""
    x0, y0, side = box
    to_frac = a.transData + a.transAxes.inverted()
    unit = ((0, 0), (1, 0), (1, 1), (0, 1))
    small = to_frac.transform([(x0 + u * side, y0 + v * side) for u, v in unit])
    big = np.array([(inset[0] + u * inset[2], inset[1] + v * inset[2]) for u, v in unit])
    on = set(ConvexHull(np.vstack([small, big])).vertices)
    for k, (u, v) in enumerate(unit):
        if k in on and k + 4 in on:
            a.add_artist(ConnectionPatch(small[k], (u, v), coordsA=a.transAxes,
                                         coordsB=ia.transAxes, color=ZC, lw=1.2,
                                         ls=(0, (3, 2)), zorder=4))


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
    T2 = cut(s["truth"][:, :, zc, vol])
    RZ = max(10, int(ROI_SIZE * min(T2.shape)))
    h = RZ // 2 + 1
    cy = int(np.clip(round(ROI_CENTRE[0] * (T2.shape[0] - 1)), h, T2.shape[0] - h - 1))
    cx = int(np.clip(round(ROI_CENTRE[1] * (T2.shape[1] - 1)), h, T2.shape[1] - h - 1))
    r0, c0 = cy - RZ // 2, cx - RZ // 2
    roi = (slice(r0, r0 + RZ), slice(c0, c0 + RZ))
    ix, iy = 1 - IZ, -INSET_DROP

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
            B = cut(A)
            kw = (gk, rk, sk)[i]
            a.imshow(B, origin="lower", **kw)
            a.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), RZ, RZ, fill=False, ec=ZC, lw=1.9))
            ia = a.inset_axes([ix, iy, IZ, IZ])
            ia.imshow(B[roi], origin="lower", **kw)
            ia.set_xticks([])
            ia.set_yticks([])
            for sp in ia.spines.values():
                sp.set_color(ZC)
                sp.set_linewidth(1.9)
            connect(a, (c0 - 0.5, r0 - 0.5, RZ), ia, (ix, iy, IZ))
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
    grid[2, 0].set_ylabel("MSE [% of Noisy]", fontsize=LABEL)
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
    common = dict(snr=SNR, seed=SEED, patch=PATCH)
    qual = cached(QUAL_CACHE, dict(common, qual=QUAL),
                  lambda: run_qual(signal, mask, bval))
    quant = cached(QUANT, dict(common, n_patches=N_PATCHES, n_draw=N_DRAW, f_d=F_D,
                               f_corr=F_CORR), lambda: run_quant(signal, mask, bval))
    fig = figure(qual, sweep_stats(quant), bval)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
