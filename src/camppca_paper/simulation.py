"""Simulated acquisitions on the subject phantom."""

import json

import camppca
import nibabel as nib
import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter

from .paths import BVAL, DWI

B0_MAX = 100
N_SLICES = 15             # the slab around the middle slice, which the figures show
FIELD_SEED = 11
FIELD_SMOOTH = 2.0


def load_phantom():
    """The phantom's `N_SLICES` central slices, on the full in-plane grid.

    Returns:
        signal: (ny, nx, nz, nv) float32, scaled so the median b0 in the brain is 1.
        mask: (ny, nx, nz) bool brain mask.
        bval: (nv,) b-values.
    """
    bval = np.loadtxt(BVAL).ravel()
    signal = np.asarray(nib.load(DWI).dataobj, np.float32)
    mask = camppca.brain_mask(signal[..., bval <= B0_MAX].mean(-1))
    z = mask.shape[2] // 2 - N_SLICES // 2
    signal, mask = signal[:, :, z:z + N_SLICES], mask[:, :, z:z + N_SLICES]
    signal /= np.median(signal[mask][:, bval <= B0_MAX])
    return signal, mask, bval


def create_hann_blend_window(ny, nx, f_c):
    """k-space window, a fraction of the way from flat to Hann, separable in-plane.

    Args:
        ny, nx: In-plane grid size.
        f_c: Blend fraction. 0 is flat, 1 is the full Hann window.

    Returns:
        (ny, nx) window with DC at the centre.
    """
    def w1(n):
        k = np.arange(n) - n // 2
        return (1 - f_c) + f_c * 0.5 * (1 + np.cos(np.pi * k / (n / 2)))
    return np.outer(w1(ny), w1(nx))


def apply_k_space_window(X, w):
    """Filter real data in-plane by a k-space window.

    Args:
        X: (ny, nx, nz, nv) real data.
        w: (ny, nx) window with DC at the centre.

    Returns:
        (ny, nx, nz, nv) float32 filtered data.
    """
    K = fft.fft2(np.asarray(X, np.float32), axes=(0, 1), workers=-1)
    K *= np.fft.ifftshift(w).astype(np.float32)[:, :, None, None]
    return fft.ifft2(K, axes=(0, 1), workers=-1, overwrite_x=True).real.astype(np.float32)


def create_gain_field(mask, f_d):
    """Smooth noise gain field from 1 to `f_d`.

    The pattern is fixed by `FIELD_SEED`, so `f_d` only scales it.

    Args:
        mask: (ny, nx, nz) bool brain mask. The pattern's 1st and 99th percentiles in it map
            to 1 and `f_d`.
        f_d: Maximum gain.

    Returns:
        (ny, nx, nz) float32 gain field.
    """
    z = gaussian_filter(np.random.default_rng(FIELD_SEED).standard_normal(mask.shape),
                        FIELD_SMOOTH)
    p1, p99 = np.percentile(z[mask], (1, 99))
    return (1 + (f_d - 1) * np.clip((z - p1) / (p99 - p1), 0, 1)).astype(np.float32)


def simulate_acquisition(s, mask, bval, f_c, f_d, snr=20, seed=0, y=None):
    """One noisy acquisition, y_noisy = y + sigma_thermal d F^-1{w F eps} with y = F^-1{w F s}.

    w is `create_hann_blend_window(f_c)` and d is `create_gain_field(f_d)`.

    Args:
        s: (ny, nx, nz, nv) noise-free phantom.
        mask: (ny, nx, nz) bool brain mask.
        bval: (nv,) b-values.
        f_c: Window blend fraction.
        f_d: Maximum gain of the gain field.
        snr: Median b0 of `s` in the brain over sigma_thermal.
        seed: Seed of eps.
        y: The noise-free signal of an earlier draw at the same `f_c`, to skip filtering `s` again.

    Returns:
        Dict with
            y: (ny, nx, nz, nv) noise-free signal, the phantom filtered by w.
            y_noisy: (ny, nx, nz, nv) y plus noise.
            sigma: (ny, nx, nz) true noise level per voxel, sigma_thermal d rms(w).
            sigma_thermal: Thermal noise level.
            rho1: Nearest-neighbour noise correlation.
            sd_d: Standard deviation of d in the brain.
    """
    ny, nx, nz, nv = s.shape
    sigma_thermal = float(np.median(s[mask][:, bval <= B0_MAX])) / snr
    w = create_hann_blend_window(ny, nx, f_c)
    d = create_gain_field(mask, f_d)
    eps = np.random.default_rng(seed).standard_normal((ny, nx, nz, nv), dtype=np.float32)
    if y is None:
        y = apply_k_space_window(s, w)
    noise = (sigma_thermal * d)[..., None] * apply_k_space_window(eps, w)
    acf = np.real(np.fft.ifft2(np.fft.ifftshift(w) ** 2))
    return dict(y=y, y_noisy=y + noise, sigma=sigma_thermal * float(np.sqrt(np.mean(w ** 2))) * d,
                sigma_thermal=sigma_thermal, rho1=float(acf[1, 0] / acf[0, 0]), sd_d=float(d[mask].std()))


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

