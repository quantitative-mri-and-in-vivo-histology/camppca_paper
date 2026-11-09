"""Stage 1: the reference, the true noise level, and Standard MPPCA and Proposed.

    noisy       the two-average series
    reference   the twenty-average series over AVG_RATIO. The files hold the sums of 20 and 2
                averages, so the ratio is fixed by the acquisition
    sigma_true  per voxel, the SD over the noisy series' b0s, pooled by the root mean square
                over POOL^3 inside the mask
    arms        Standard MPPCA (camppca.denoise_mppca) and Proposed (camppca.denoise_camppca)
                on the noisy series inside the mask, with camppca's default patch

Out: data/analyses/exvivo_mouse/native/ (noisy, reference, sigma_true, standard, proposed,
sigma_standard, sigma_proposed, mask, all .nii.gz). Skipped when done.

    python scripts/analyses/exvivo_mouse/denoise.py
"""

import time

import camppca
import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter

from camppca_paper.exvivo_mouse import AVERAGE, BVAL, MASK, NATIVE, SERIES

AVG_RATIO = 10
POOL = 3
B0_MAX = 50
OUTPUTS = ("noisy", "reference", "sigma_true", "standard", "proposed", "sigma_standard",
           "sigma_proposed", "mask")


def pooled(sd, mask):
    """`sd` pooled by the root mean square over POOL^3, over mask voxels only."""
    num = uniform_filter(np.where(mask, sd ** 2, 0), POOL, mode="constant")
    den = uniform_filter(mask.astype(float), POOL, mode="constant")
    return np.where(mask, np.sqrt(num / np.maximum(den, 1e-12)), 0)


def save(name, data, affine):
    nib.save(nib.Nifti1Image(np.asarray(data, np.float32), affine), NATIVE / f"{name}.nii.gz")


if __name__ == "__main__":
    if not all((NATIVE / f"{n}.nii.gz").exists() for n in OUTPUTS):
        t0 = time.time()
        NATIVE.mkdir(parents=True, exist_ok=True)
        img = nib.load(SERIES)
        bval = np.loadtxt(BVAL).ravel()
        mask = np.asarray(nib.load(MASK).dataobj) > 0
        noisy = np.asarray(img.dataobj, np.float32)
        sd = np.zeros(mask.shape)
        sd[mask] = noisy[mask][:, bval <= B0_MAX].astype(np.float64).std(1, ddof=1)
        save("noisy", noisy, img.affine)
        save("reference", np.asarray(nib.load(AVERAGE).dataobj, np.float64) / AVG_RATIO, img.affine)
        save("sigma_true", pooled(sd, mask), img.affine)
        save("mask", mask, img.affine)
        for arm, fn in (("standard", lambda: camppca.denoise_mppca(noisy, mask=mask)),
                        ("proposed", lambda: camppca.denoise_camppca(noisy, bvals=bval, mask=mask))):
            got = fn()
            save(arm, got.denoised, img.affine)
            save(f"sigma_{arm}", got.sigma, img.affine)
            print(f"  {arm}: patch {got.provenance['patch']}  [{time.time() - t0:.0f}s]", flush=True)
