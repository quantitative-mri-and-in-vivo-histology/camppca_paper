"""Figure: the simulation's qualitative setting over N_DRAW independent noise draws.

The ground truth on the left; Noisy, Standard MPPCA and Proposed in rows, one draw per column,
each on the middle slice at the middle b SHOW_B volume, with the simulation figure's zoom box
and magnifier. Draw r has seed r - 1, so draw 1 is the simulation figure's panel a.
The slices are saved with their settings and rerun only when they change.

    python scripts/figures/supp/simulation_noise_draws.py
"""

import camppca
import matplotlib.pyplot as plt
import numpy as np

from camppca_paper.paths import RESULTS, WORK
from camppca_paper.simulation import cached, load_phantom, simulate_acquisition
from camppca_paper.zoom import box, zoomed

PATCH = (5, 5, 5)
N_DRAW = 5
QUAL = (0.5, 2.0)  # (f_c, f_d) of the main figure's panel a
SHOW_B = 2000      # the shown volume's shell
ROI_CENTRE = (0.586, 0.774)    # zoom box centre, fractions of the panel, bottom-left origin
ROI_SIZE = 0.27                # zoom box side, fraction of the panel's shorter side
CACHE = WORK / "simulation_noise_draws.npz"
OUT = RESULTS / "supp/simulation_noise_draws.pdf"
ROWS = (("noisy", "Noisy"), ("standard", "Standard MPPCA"), ("proposed", "Proposed"))


def run(signal, mask, bval):
    """The shown slice of the truth and of every arm in every draw."""
    k = mask.shape[2] // 2
    shown = np.where(np.abs(bval - SHOW_B) <= 100)[0]
    v = int(shown[len(shown) // 2])
    out, y = {}, None
    for r in range(N_DRAW):
        a = simulate_acquisition(signal, mask, bval, *QUAL, seed=r, y=y)
        y = a["y"]
        std = camppca.denoise_mppca(a["y_noisy"], mask=mask, patch=PATCH)
        prop = camppca.denoise_camppca(a["y_noisy"], bvals=bval, mask=mask, patch=PATCH)
        for arm, x in (("noisy", a["y_noisy"]), ("standard", std.denoised), ("proposed", prop.denoised)):
            out[f"{r}_{arm}"] = np.asarray(x[:, :, k, v], np.float32)
        out["truth"] = np.asarray(a["y"][:, :, k, v], np.float32)
        print(f"  draw {r + 1} of {N_DRAW}", flush=True)
    return out | {"mask": mask[:, :, k]}


def main():
    signal, mask, bval = load_phantom()
    s = cached(CACHE, dict(patch=PATCH, qual=QUAL, n_draw=N_DRAW),
               lambda: run(signal, mask, bval))
    m2 = s["mask"].astype(bool)
    yy, xx = np.where(m2)
    sl = (slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))

    def cut(a):
        return np.where(m2[sl], a[sl], np.nan).T

    truth = cut(s["truth"])
    r0, c0, side = box(truth.shape, ROI_CENTRE, ROI_SIZE)
    kw = dict(vmin=0, vmax=float(np.nanpercentile(truth, 99.5)), cmap="gray")
    h, w = truth.shape
    widths = [1, 0.25] + [1] * N_DRAW
    fig_w = 10.4
    cell = fig_w * 0.92 / sum(widths)
    fig = plt.figure(figsize=(fig_w, 3 * cell * h / w * 1.08 + 0.35))
    gs = fig.add_gridspec(3, len(widths), width_ratios=widths, left=0.06, right=0.98, top=0.93, bottom=0.03,
                          wspace=0.04, hspace=0.06)
    panels = [(fig.add_subplot(gs[1, 0]), truth, "Ground truth", None)]
    for i, (arm, name) in enumerate(ROWS):
        for r in range(N_DRAW):
            panels.append((fig.add_subplot(gs[i, 2 + r]), cut(s[f"{r}_{arm}"]),
                           f"Draw {r + 1}" if i == 0 else None, name if r == 0 else None))
    for a, img, title, row in panels:
        zoomed(a, img, kw, r0, c0, side)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if title:
            a.set_title(title, fontsize=10, fontweight="bold")
        if row:
            a.set_ylabel(row, fontsize=10, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
