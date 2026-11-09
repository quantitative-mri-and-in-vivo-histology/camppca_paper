"""Figure: the noise level correction within scanners against eddy's displacement
(data/results/on_harmony/within.json, `camppca_paper.registration.displacement`).

a     the correction, sigma_proposed / sigma_standard - 1, one point per session, a bar at each
      scanner's median
b-e   each displacement measure over the correction, both centered on the scanner's mean, one
      point per session: the spread across volumes of the total (b), the motion (c) and the
      eddy currents (d), and the mean susceptibility shift (e). Above each, the repeated-measures
      correlation r and its p, on N - k - 1 degrees of freedom (N sessions, k scanners)

    python scripts/figures/supp/invivo_repeat_harmony_correction.py
"""

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from camppca_paper.invivo import VENDOR, correction_sites, metrics
from camppca_paper.on_harmony import PARTICIPANTS, SITES, sessions
from camppca_paper.paths import RESULTS
from camppca_paper.registration import displacement

OUT = RESULTS / "supp/invivo_repeat_harmony_correction.pdf"
COLOURS = dict(zip(("OXF1PRI", "OXF2PRI", "OXF3TRI", "NOT1ACH", "NOT2ING", "OXF4GEP", "NOT4GEP"),
                   ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7", "#E41A1C", "#000000"), strict=True))
MARKERS = dict(zip(COLOURS, "osD^vPX", strict=True))
KINDS = (("total", "Total displacement", "Spread across volumes"), ("motion", "Motion", "Spread across volumes"),
         ("eddy", "Eddy currents", "Spread across volumes"), ("susc", "Susceptibility", "Mean shift"))


def measure(session):
    return session.name, displacement(session.native)


def rmcorr(x, y, groups):
    """Repeated-measures correlation (https://doi.org/10.3389/fpsyg.2017.00456): Pearson's r of x
    and y each centered on its group's mean, tested on N - k - 1 degrees of freedom.

    Returns:
        `(r, p, x centered, y centered)`.
    """
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    xc, yc = x.copy(), y.copy()
    for g in np.unique(groups):
        sel = groups == g
        xc[sel] -= x[sel].mean()
        yc[sel] -= y[sel].mean()
    df = len(x) - len(np.unique(groups)) - 1
    r = float(np.corrcoef(xc, yc)[0, 1])
    p = float(2 * student_t.sf(abs(r) * np.sqrt(df / (1 - r ** 2)), df))
    return r, p, xc, yc


def main():
    todo = [s for subject in PARTICIPANTS for s in sessions(subject, "within")]
    with Pool(16) as pool:
        moved = dict(pool.map(measure, todo))
    correction = {k: v for r in metrics("within").values() for k, v in r["correction"].items()}
    sites = correction_sites("within")
    names = [s.name for s in todo]
    groups = [s.site for s in todo]

    fig, grid = plt.subplots(2, 3, figsize=(9.0, 6.6), gridspec_kw=dict(wspace=0.32, hspace=0.62))
    axes = list(grid.ravel()[:5])
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, site in enumerate(sites):
        v = [correction[n] for n, g in zip(names, groups, strict=True) if g == site]
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, MARKERS[site], color=COLOURS[site], mec="k", mew=0.4,
                ms=4, zorder=3)
        ax.hlines(np.median(v), i - 0.2, i + 0.2, color=COLOURS[site], lw=2.2, zorder=1)
    ax.text(0.5, 1.13, "Noise level correction", transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
            fontweight="bold")
    ax.set_ylabel("$\\sigma_{proposed} / \\sigma_{standard} - 1$ [%]", fontsize=8)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels([SITES[s] for s in sites], fontsize=8, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(sites) - 0.4)
    ax.set_ylim(bottom=0)

    for ax, (key, title, what) in zip(axes[1:], KINDS, strict=True):
        r, p, xc, yc = rmcorr([correction[n] for n in names], [moved[n][key] for n in names], groups)
        print(f"  {key:6s} r {r:+.2f}  p {p:.2g}")
        for site in sites:
            sel = np.array([g == site for g in groups])
            ax.plot(xc[sel], yc[sel], MARKERS[site], color=COLOURS[site], mec="k", mew=0.4, ms=5)
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        ax.axvline(0, color="0.6", lw=0.8, zorder=0)
        xl, yl = 1.1 * np.abs(xc).max(), 1.1 * np.abs(yc).max()
        ax.set_xlim(-xl, xl)
        ax.set_ylim(-yl, yl)
        ax.set_title(f"r = {r:.2f}, p = {p:.2g}", fontsize=10)
        ax.text(0.5, 1.13, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{what} (centered) [mm]", fontsize=8)
        ax.set_xlabel("Correction (centered) [%]", fontsize=8)
    for ax, letter in zip(axes, "abcde", strict=True):
        ax.tick_params(labelsize=8)
        ax.text(-0.2, 1.12, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
    grid[1, 2].set_axis_off()
    grid[1, 2].legend(handles=[plt.Line2D([], [], ls="", marker=MARKERS[s], color=COLOURS[s], mec="k", mew=0.4, ms=6,
                                          label=f"{VENDOR[s]} {SITES[s]}") for s in sites],
                      frameon=False, fontsize=9, loc="center")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.08)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
