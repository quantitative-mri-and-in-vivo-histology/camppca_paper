"""Figure: the simulation's noise covariance across matrix aspect ratios.

The gain field alone (F_D) and the k-space window alone (F_C), as in the main figure, each on
N_DRAW draws, every draw denoised on volume subsets of each size in N_LIST that keep the shell
mix (`volume_subsets`). The patch stays at PATCH, so the aspect ratio changes through N alone.
Both arms are scored over the brain.

    a   sigma-hat over the true level, median over brain voxels and draws
    b   MSE against the truth over brain voxels and every volume, as % of Undenoised's

The scores are saved with their settings and rerun only when they change.

    python scripts/figures/supp/simulation_matrix_geometry.py
"""

import camppca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from camppca_paper.paths import RESULTS, WORK
from camppca_paper.simulation import cached, load_phantom, simulate_acquisition

PATCH = (5, 5, 5)
F_D = (1.0, 1.75, 2.5, 3.25, 4.0)                 # gain sweep, window off
F_C = (0.0, 0.2, 0.4, 0.6, 0.8)                   # window sweep, gain off
N_LIST = (30, 45, 60, 75, 90, 105)
N_DRAW = 20
CACHE = WORK / "simulation_matrix_geometry.npz"
OUT = RESULTS / "supp/simulation_matrix_geometry.pdf"
MIXES = (("level", "Spatially varying level only", "sd_d", r"Spread of the noise level (std($d$))"),
         ("corr", "Correlation only", "rho1", r"Nearest-neighbor correlation ($\rho(1)$)"))
ARMS = (("standard", "Standard MPPCA"), ("proposed", "Proposed"))
NORM = Normalize(0.65, 1.35)


def volume_subsets(bval, counts):
    """Volume subsets of several sizes, each holding the shell mix of the whole acquisition.

    Args:
        bval: (nv,) b-values.
        counts: Sizes to draw.

    Returns:
        `{n: index array}`, each sorted.
    """
    order = np.argsort(bval, kind="stable")
    return {int(n): np.sort(order[np.linspace(0, len(order) - 1, n).round().astype(int)]) for n in counts}


def cells():
    return [("level", 0.0, v) for v in F_D] + [("corr", v, 1.0) for v in F_C]


def run(signal, mask, bval):
    """Per cell and N: the median sigma-hat over sigma and the MSE as % of Undenoised's, per arm."""
    subsets = volume_subsets(bval, N_LIST)
    out = {}
    for i, (mix, f_c, f_d) in enumerate(cells()):
        y = None
        for n, vols in subsets.items():
            ratio = {arm: [] for arm, _ in ARMS}
            err = {arm: 0.0 for arm, _ in ARMS} | {"noisy": 0.0}
            for r in range(N_DRAW):
                a = simulate_acquisition(signal, mask, bval, f_c, f_d, seed=r, y=y)
                y = a["y"]
                data, truth = a["y_noisy"][..., vols], a["y"][..., vols]
                got = {"standard": camppca.denoise_mppca(data, mask=mask, patch=PATCH),
                       "proposed": camppca.denoise_camppca(data, bvals=bval[vols], mask=mask, patch=PATCH,
                                                           mask_only=True)}
                err["noisy"] += float(np.mean((data[mask] - truth[mask]) ** 2))
                for arm, res in got.items():
                    ratio[arm].append(res.sigma[mask] / a["sigma"][mask])
                    err[arm] += float(np.mean((res.denoised[mask] - truth[mask]) ** 2))
            for arm, _ in ARMS:
                out[f"{i}_{n}_{arm}_ratio"] = float(np.median(np.concatenate(ratio[arm])))
                out[f"{i}_{n}_{arm}_mse"] = 100 * err[arm] / err["noisy"]
        out[f"{i}_sd_d"], out[f"{i}_rho1"] = a["sd_d"], a["rho1"]
        print(f"  {mix:5s} f_c {f_c:.2f} f_d {f_d:.2f}", flush=True)
    return out


def heatmaps(fig, spec, res, key, title, cmap, norm, fmt):
    grid = spec.subgridspec(2, 2, wspace=0.05, hspace=0.08)
    M = int(np.prod(PATCH))
    for j, (mix, mix_title, xkey, xlabel) in enumerate(MIXES):
        idx = [i for i, c in enumerate(cells()) if c[0] == mix]
        for a, (arm, arm_title) in enumerate(ARMS):
            ax = fig.add_subplot(grid[a, j])
            v = np.array([[float(res[f"{i}_{n}_{arm}_{key}"]) for i in idx] for n in N_LIST])
            im = ax.imshow(v, origin="lower", cmap=cmap, norm=norm, aspect="auto")
            for (y, x), val in np.ndenumerate(v):
                shade = np.mean(im.cmap(im.norm(val))[:3])
                ax.text(x, y, fmt.format(val), ha="center", va="center", fontsize=6.5,
                        color="k" if shade > 0.5 else "w")
            ax.set_yticks(range(len(N_LIST)))
            if j == 0:
                ax.set_yticklabels([f"{n} ({n / M:.2f})" for n in N_LIST], fontsize=7.5)
                ax.set_ylabel("Volumes $N$  ($N/M$)", fontsize=8)
                ax.text(-0.42, 0.5, arm_title, transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=10, fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.set_xticks(range(len(idx)))
            if a == 1:
                ax.set_xticklabels([f"{float(res[f'{i}_{xkey}']):.2f}" for i in idx], fontsize=7.5)
                ax.set_xlabel(xlabel, fontsize=8)
            else:
                ax.set_xticklabels([])
                ax.set_title(mix_title, fontsize=10, fontweight="bold")
    cax = fig.add_axes([0.9, spec.get_position(fig).y0 + 0.01, 0.018, spec.get_position(fig).height - 0.02])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(title, fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)


def main():
    signal, mask, bval = load_phantom()
    res = cached(CACHE, dict(patch=PATCH, f_d=F_D, f_c=F_C, n_list=N_LIST,
                             scope="brain", n_draw=N_DRAW), lambda: run(signal, mask, bval))
    fig = plt.figure(figsize=(7.0, 10.4))
    outer = fig.add_gridspec(2, 1, left=0.17, right=0.87, top=0.94, bottom=0.05, hspace=0.2)
    heatmaps(fig, outer[0], res, "ratio", r"$\hat\sigma$ / $\sigma$", "BrBG", NORM, "{:.2f}")
    heatmaps(fig, outer[1], res, "mse", "MSE [% of Undenoised]", "viridis", Normalize(0, 100), "{:.0f}")
    for spec, letter, title in ((outer[0], "a", "Noise level estimation"), (outer[1], "b", "Mean squared error (MSE)")):
        y = spec.get_position(fig).y1 + 0.025
        fig.text(0.52, y, title, ha="center", fontsize=12, fontweight="bold")
        fig.text(0.02, y, letter, fontsize=14, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
