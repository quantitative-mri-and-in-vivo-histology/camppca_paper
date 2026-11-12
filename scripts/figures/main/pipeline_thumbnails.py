"""Thumbnails for the pipeline figure: EDDEN run-01 magnitude, one axial slice, cropped to the brain.

Undenoised and Proposed, one volume per shell, on one grey scale per shell. The initial noise
map d, as Proposed estimates it, and Proposed's final noise level, each on its own scale. d is
not stored by the analysis, so it is computed once with camppca's level pass and cached.

Writes PNGs without axes to data/results/main/pipeline/.

    python scripts/figures/main/pipeline_thumbnails.py
"""

import camppca
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from camppca.noise_model import _footprint_average, _level_pass
from camppca.patch import resolve_patch

from camppca_paper import edden
from camppca_paper.harmony_figure import SHELL, SLICE_AT, VOLUME
from camppca_paper.paths import RESULTS

PART, RUN = "magnitude", "run-01"
SHELLS = (0, 1000, 2000)
UP = 4                               # integer upscale of the voxel grid
NOISE_CMAP = "magma"
OUT = RESULTS / "main/pipeline"


def initial_d(native, data, mask):
    path = native / "d_initial.nii.gz"
    if not path.exists():
        patch = resolve_patch(None, data.shape)
        d = _footprint_average(*_level_pass(data, mask, patch, "cordero-grande", None), mask, patch)
        nib.save(nib.Nifti1Image(d, nib.load(native / "sigma_proposed.nii.gz").affine), path)
    return np.asarray(nib.load(path).dataobj, np.float32)


def shown_volumes(bval, bvec):
    """The first b0, and per shell the volume closest in direction to the main figure's."""
    target = bvec[:, np.where(np.abs(bval - SHELL) < 100)[0][VOLUME]]
    out = {0: int(np.where(bval < 100)[0][0])}
    for b in SHELLS[1:]:
        shell = np.where(np.abs(bval - b) < 100)[0]
        out[b] = int(shell[np.argmax(np.abs(target @ bvec[:, shell]))])
    return out


def save(img, name, lo, hi, cmap="gray"):
    rgba = plt.get_cmap(cmap)(plt.Normalize(lo, hi)(img))
    rgba[np.isnan(img)] = (0, 0, 0, 1)
    plt.imsave(OUT / f"{name}.png", np.repeat(np.repeat(rgba, UP, 0), UP, 1))


def main():
    native = edden.native(PART, RUN)
    mask = np.asarray(nib.load(native / "brain_mask_denoising.nii.gz").dataobj) > 0
    data = np.asarray(nib.load(edden.raw(PART, RUN)).dataobj, np.float32)
    d = initial_d(native, data, mask)
    zs = np.where(mask.any((0, 1)))[0]
    k = int(zs[0] + SLICE_AT * (zs[-1] - zs[0]))
    m2 = np.rot90(mask[:, :, k])
    rows, cols = np.where(m2)
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))

    def cut(a):
        return np.where(m2, np.rot90(np.asarray(a, np.float32)), np.nan)[box]

    OUT.mkdir(parents=True, exist_ok=True)
    proposed = nib.load(native / "proposed.nii.gz").dataobj
    vols = shown_volumes(np.loadtxt(edden.bval()).ravel(), np.loadtxt(edden.bvec()).reshape(3, -1))
    for b, v in vols.items():
        raw, den = cut(data[:, :, k, v]), cut(proposed[:, :, k, v])
        hi = np.nanpercentile(raw, 99)
        save(raw, f"undenoised_b{b}", 0, hi)
        save(den, f"denoised_b{b}", 0, hi)
    for name, a in (("d_initial", d), ("sigma_final", nib.load(native / "sigma_proposed.nii.gz").dataobj)):
        s = cut(np.asarray(a)[:, :, k])
        save(s, name, *np.nanpercentile(s, (1, 99)), cmap=NOISE_CMAP)
    print(f"  slice {k}, volumes {vols}, wrote {OUT}")


if __name__ == "__main__":
    main()
