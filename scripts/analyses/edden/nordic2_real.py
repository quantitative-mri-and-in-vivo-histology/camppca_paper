"""NORDIC2's real part: its magnitude and phase output rotated by minus the smooth phase
NIFTI_NORDIC2 removed before denoising, which is rebuilt from the source data as NIFTI_NORDIC2
estimates it (its decorrelated kernel in plane, no mean phase).

Out: data/analyses/edden/nordic2/<run>_nordic2_real.nii.gz. Finished runs are skipped.

    python scripts/analyses/edden/nordic2_real.py
"""

import nibabel as nib
import numpy as np
from scipy import ndimage

from camppca_paper.edden import NORDIC2, RUNS, series

KERNEL = np.array([[0.03, 0.04, 0.05, 0.04, 0.03],
                   [0.04, 0.08, 0.01, 0.08, 0.04],
                   [0.05, 0.01, 0.00, 0.01, 0.05],
                   [0.04, 0.08, 0.01, 0.08, 0.04],
                   [0.03, 0.04, 0.05, 0.04, 0.03]])[:, :, None]
KERNEL = KERNEL / KERNEL.sum()


def load(path):
    return np.asarray(nib.load(path).dataobj, np.float32)


def real(r):
    out = NORDIC2 / f"{r}_nordic2_real.nii.gz"
    if out.exists():
        return
    phase = load(series("AP", r, "phase"))
    lo, hi = float(phase.min()), float(phase.max())

    def radians(x):
        """NIFTI_NORDIC2's mapping of the stored phase to radians, from the run's range."""
        return (x / (hi - lo) - (hi + lo) / (hi - lo) / 2) * 2 * np.pi

    magnitude = load(series("AP", r, "mag"))
    img = nib.load(NORDIC2 / f"{r}_nordic2_magn.nii.gz")
    den = np.asarray(img.dataobj, np.float32)
    den_phase = load(NORDIC2 / f"{r}_nordic2_phase.nii.gz")
    for n in range(den.shape[3]):
        z = magnitude[..., n] * np.exp(1j * radians(phase[..., n]))
        smooth = (ndimage.convolve(z.real, KERNEL, mode="constant")
                  + 1j * ndimage.convolve(z.imag, KERNEL, mode="constant"))
        den[..., n] *= np.cos(radians(den_phase[..., n]) - np.angle(smooth))
    nib.save(nib.Nifti1Image(den, img.affine, img.header), out)
    print(f"  {r}: NORDIC2 real part", flush=True)


if __name__ == "__main__":
    for r in RUNS:
        real(r)
