"""Figure: the simulation's noise covariance across matrix aspect ratios.

The gain field alone (F_D) and the k-space window alone (F_CORR), each on N_DRAW draws, every
draw denoised on volume subsets of each size in N_LIST that keep the shell mix
(`camppca_paper.simulation.volume_subsets`). The patch stays at PATCH, so the aspect ratio
changes through N alone. Both arms are scored at N_PATCHES patch centres.

    a   sigma-hat over the true level at the centres, median over centres and draws
    b   MSE against the truth at the centres, over every volume, as % of Noisy's

The scores are saved with their settings and rerun only when they change.

    python scripts/figures/supplementary/calibration_sim.py
"""

import camppca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from camppca_paper.paths import RESULTS, WORK
from camppca_paper.simulation import (
    PATCH,
    SEED,
    SNR,
    cached,
    load_phantom,
    sample_centres,
    simulate_acquisition,
    volume_subsets,
)

F_D = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)        # gain sweep, window off
F_CORR = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)           # window sweep, gain off
N_LIST = (30, 45, 60, 75, 90, 105)
N_PATCHES = 1500
N_DRAW = 3
CACHE = WORK / "simulations_calibration.npz"
OUT = RESULTS / "supplementary/calibration_sim.pdf"
MIXES = (("level", "Spatially varying level only", "sd_d", r"Spread of the noise level (std($d$))"),
         ("corr", "Correlation only", "rho1", r"Nearest-neighbor correlation ($\rho(1)$)"))
ARMS = (("standard", "Standard MPPCA"), ("proposed", "Proposed"))
NORM = Normalize(0.65, 1.35)


def cells():
    return [("level", 0.0, v) for v in F_D] + [("corr", f, 0.0) for f in F_CORR]


def run(signal, mask, bval):
    """Per cell and N: the median sigma-hat over sigma and the MSE as % of Noisy's, per arm."""
    where = sample_centres(mask, N_PATCHES)
    subsets = volume_subsets(bval, N_LIST)
    out = {}
    for i, (mix, f, f_d) in enumerate(cells()):
        for n, vols in subsets.items():
            ratio = {arm: [] for arm, _ in ARMS}
            err = {arm: 0.0 for arm, _ in ARMS} | {"noisy": 0.0}
            for r in range(N_DRAW):
                a = simulate_acquisition(signal, mask, bval, SNR, f, f_d, SEED + 1000 * r)
                data, truth = a["data"][..., vols], a["truth"][..., vols]
                got = {"standard": camppca.denoise_mppca(data, mask=where, patch=PATCH),
                       "proposed": camppca.denoise_camppca(data, bvals=bval[vols], mask=mask, patch=PATCH,
                                                           mask_only=True)}
                err["noisy"] += float(np.mean((data[where] - truth[where]) ** 2))
                for arm, res in got.items():
                    ratio[arm].append(res.sigma[where] / a["sigma"][where])
                    err[arm] += float(np.mean((res.denoised[where] - truth[where]) ** 2))
            for arm, _ in ARMS:
                out[f"{i}_{n}_{arm}_ratio"] = float(np.median(np.concatenate(ratio[arm])))
                out[f"{i}_{n}_{arm}_mse"] = 100 * err[arm] / err["noisy"]
        out[f"{i}_sd_d"], out[f"{i}_rho1"] = a["sd_d"], a["rho1"]
        print(f"  {mix:5s} f {f:.2f} f_d {f_d:.2f}", flush=True)
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
    res = cached(CACHE, dict(snr=SNR, seed=SEED, patch=PATCH, f_d=F_D, f_corr=F_CORR, n_list=N_LIST,
                             n_patches=N_PATCHES, n_draw=N_DRAW), lambda: run(signal, mask, bval))
    fig = plt.figure(figsize=(7.0, 10.4))
    outer = fig.add_gridspec(2, 1, left=0.17, right=0.87, top=0.94, bottom=0.05, hspace=0.2)
    heatmaps(fig, outer[0], res, "ratio", r"$\hat\sigma$ / $\sigma$", "BrBG", NORM, "{:.2f}")
    heatmaps(fig, outer[1], res, "mse", "MSE [% of Noisy]", "viridis", Normalize(0, 100), "{:.0f}")
    for spec, letter, title in ((outer[0], "a", "Noise level estimation"), (outer[1], "b", "Mean squared error (MSE)")):
        y = spec.get_position(fig).y1 + 0.025
        fig.text(0.52, y, title, ha="center", fontsize=12, fontweight="bold")
        fig.text(0.02, y, letter, fontsize=14, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
