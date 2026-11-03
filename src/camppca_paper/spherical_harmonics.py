"""Spherical harmonics for diffusion signals."""

import numpy as np
import scipy.special as sp


def sh_basis(bvecs, order):
    """Real, antipodally symmetric spherical-harmonic basis of even degree up to `order`.

    Args:
        bvecs: (3, n) unit gradient directions.
        order: Maximum even degree.

    Returns:
        (n, k) design matrix.
    """
    x, y, z = bvecs
    polar, azimuth = np.arccos(np.clip(z, -1.0, 1.0)), np.arctan2(y, x)
    cols = []
    for degree in range(0, order + 1, 2):
        for m in range(-degree, degree + 1):
            harm = sp.sph_harm_y(degree, abs(m), polar, azimuth)
            cols.append(harm.real if m == 0 else
                        (np.sqrt(2) * harm.real if m > 0 else np.sqrt(2) * harm.imag))
    return np.stack(cols, 1)


def sh_fit(values, bvecs, order):
    """Least-squares coefficients of `values` over `bvecs`.

    Args:
        values: (..., n) signal, one column per direction.
        bvecs: (3, n) unit gradient directions.
        order: Maximum even degree.

    Returns:
        (..., k) coefficients.
    """
    basis = sh_basis(bvecs, order)
    flat = np.asarray(values, np.float64).reshape(-1, basis.shape[0])
    coef, *_ = np.linalg.lstsq(basis, flat.T, rcond=None)
    return coef.T.reshape(*np.shape(values)[:-1], basis.shape[1])


def sh_synthesise(coef, bvecs, order):
    """Evaluate coefficients at a set of directions.

    Args:
        coef: (..., k) coefficients from `sh_fit`.
        bvecs: (3, n) unit gradient directions.
        order: Maximum even degree, matching `coef`.

    Returns:
        (..., n) signal.
    """
    basis = sh_basis(bvecs, order)
    flat = np.asarray(coef, np.float64).reshape(-1, basis.shape[1])
    return (flat @ basis.T).reshape(*np.shape(coef)[:-1], basis.shape[0])


def sh_project(signal, mask, bval, bvec, order, b0_max=100.0):
    """Replace a 4-D signal by its spherical-harmonic fit, shell by shell.

    The b0 volumes are replaced by their mean.

    Args:
        signal: (ny, nx, nz, nv) data.
        mask: (ny, nx, nz) voxels to fit. Outside it the output is zero.
        bval: (nv,) b-values.
        bvec: (3, nv) gradient directions.
        order: Maximum even degree.
        b0_max: Volumes with b at or below this count as b0.

    Returns:
        `(projected, residuals)`. `projected` is (ny, nx, nz, nv) float32. `residuals` maps
        each shell's b-value to its relative fit residual, with the b0 volumes under 0.0.
    """
    out = np.zeros_like(signal, np.float32)
    flat = signal[mask].astype(np.float64)
    fit = np.empty_like(flat)
    residuals = {}

    is_b0 = bval <= b0_max
    fit[:, is_b0] = flat[:, is_b0].mean(1, keepdims=True)
    residuals[0.0] = float(np.linalg.norm(flat[:, is_b0] - fit[:, is_b0])
                           / np.linalg.norm(flat[:, is_b0]))

    for edge in np.unique(np.round(bval[~is_b0], -2)):
        col = ~is_b0 & (np.round(bval, -2) == edge)
        fit[:, col] = sh_synthesise(sh_fit(flat[:, col], bvec[:, col], order),
                                    bvec[:, col], order)
        residuals[float(edge)] = float(np.linalg.norm(flat[:, col] - fit[:, col])
                                       / np.linalg.norm(flat[:, col]))
    out[mask] = fit
    return out, residuals
