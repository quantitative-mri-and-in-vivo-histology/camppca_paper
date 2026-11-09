"""Simulated acquisitions on the subject phantom."""

import json

import camppca
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

from .paths import BVAL, DWI

PATCH = (5, 5, 5)
B0_MAX = 100
N_SLICES = 14
FIELD_SEED = 11
FIELD_SMOOTH = 2.0
SNR = 20
SEED = 5
QUAL = (0.5, 1.0)          # (f, f_d) of the qualitative examples
SHOW_B = 2000              # they show a volume of this shell


def load_phantom():
    """The phantom cut to `N_SLICES` central slices and the brain's bounding box.

    Returns:
        signal: (ny, nx, nz, nv) float32, scaled so the median b0 in the brain is 1.
        mask: (ny, nx, nz) bool brain mask.
        bval: (nv,) b-values.
    """
    bval = np.loadtxt(BVAL).ravel()
    signal = np.asarray(nib.load(DWI).dataobj, np.float32)
    mask = camppca.brain_mask(signal[..., bval <= B0_MAX].mean(-1))
    z = mask.shape[2] // 2
    sl = slice(z - N_SLICES // 2, z + N_SLICES // 2)
    signal, mask = signal[:, :, sl], mask[:, :, sl]
    ys, xs = np.where(mask.any(2))
    box = (slice(max(ys.min() - 3, 0), ys.max() + 4), slice(max(xs.min() - 3, 0), xs.max() + 4))
    signal, mask = signal[box], mask[box]
    signal /= np.median(signal[mask][:, bval <= B0_MAX])
    return signal, mask, bval


def create_hann_blend_window(ny, nx, f):
    """k-space window, a fraction of the way from flat to Hann, separable in-plane.

    Args:
        ny, nx: In-plane grid size.
        f: Blend fraction. 0 is flat, 1 is the full Hann window.

    Returns:
        (ny, nx) window with DC at the centre.
    """
    def w1(n):
        k = np.arange(n) - n // 2
        return (1 - f) + f * 0.5 * (1 + np.cos(np.pi * k / (n / 2)))
    return np.outer(w1(ny), w1(nx))


def apply_k_space_window(X, W):
    """Filter real data in-plane by a k-space window.

    Args:
        X: (ny, nx, nz, nv) real data.
        W: (ny, nx) window with DC at the centre.

    Returns:
        (ny, nx, nz, nv) float32 filtered data.
    """
    K = np.fft.fftshift(np.fft.fft2(X, axes=(0, 1)), axes=(0, 1))
    out = np.fft.ifft2(np.fft.ifftshift(K * W[:, :, None, None], axes=(0, 1)), axes=(0, 1))
    return np.real(out).astype(np.float32)


def create_gain_field(mask, f_d):
    """Smooth noise gain field from 1 to 1 + `f_d`.

    The pattern is fixed by `FIELD_SEED`, so `f_d` only scales it.

    Args:
        mask: (ny, nx, nz) bool brain mask. The pattern's 1st and 99th percentiles in it map
            to 1 and 1 + `f_d`.
        f_d: Amplitude.

    Returns:
        (ny, nx, nz) float32 gain field.
    """
    z = gaussian_filter(np.random.default_rng(FIELD_SEED).standard_normal(mask.shape),
                        FIELD_SMOOTH)
    p1, p99 = np.percentile(z[mask], (1, 99))
    return (1 + f_d * np.clip((z - p1) / (p99 - p1), 0, 1)).astype(np.float32)


def simulate_acquisition(signal, mask, bval, snr, f, f_d, seed):
    """One noisy acquisition.

    White noise at the thermal level is filtered by `create_hann_blend_window(f)`, then
    scaled by `create_gain_field(f_d)`. The window filters the signal too.

    Args:
        signal: (ny, nx, nz, nv) noiseless phantom.
        mask: (ny, nx, nz) bool brain mask.
        bval: (nv,) b-values.
        snr: Median b0 in the brain over the thermal noise level.
        f: Window blend fraction.
        f_d: Gain field amplitude.
        seed: Noise seed.

    Returns:
        Dict with
            truth: (ny, nx, nz, nv) filtered phantom.
            data: (ny, nx, nz, nv) truth plus noise.
            sigma: (ny, nx, nz) true noise level per voxel.
            thermal: Thermal noise level.
            rho1: Nearest-neighbour noise correlation.
            sd_d: Standard deviation of the gain field in the brain.
    """
    ny, nx, nz, nv = signal.shape
    thermal = float(np.median(signal[mask][:, bval <= B0_MAX])) / snr
    W = create_hann_blend_window(ny, nx, f)
    d = create_gain_field(mask, f_d)
    white = np.random.default_rng(seed).standard_normal((ny, nx, nz, nv)).astype(np.float32)
    truth = apply_k_space_window(signal, W)
    noise = (thermal * d)[..., None] * apply_k_space_window(white, W)
    acf = np.real(np.fft.ifft2(np.fft.ifftshift(W) ** 2))
    return dict(truth=truth, data=truth + noise,
                sigma=thermal * float(np.sqrt(np.mean(W ** 2))) * d, thermal=thermal,
                rho1=float(acf[1, 0] / acf[0, 0]), sd_d=float(d[mask].std()))


def sample_centres(mask, n, seed=0):
    """Random voxels whose whole patch lies in the mask.

    Args:
        mask: (ny, nx, nz) bool brain mask.
        n: Number of voxels.
        seed: Sampling seed.

    Returns:
        (ny, nx, nz) bool, true at the `n` chosen voxels.
    """
    inner = binary_erosion(mask, np.ones(PATCH), border_value=0)
    pick = np.random.default_rng(seed).choice(np.flatnonzero(inner), n, replace=False)
    out = np.zeros_like(mask)
    out.flat[pick] = True
    return out


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
