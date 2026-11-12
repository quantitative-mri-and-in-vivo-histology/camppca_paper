"""MPPCA's noise level on patches, read and predicted from a noise covariance model."""

import numpy as np
from camppca.denoiser.mp_threshold import mp_threshold
from camppca.noise_model.loop import _sample_centers
from camppca.patch import patch_offsets

ALGO = "cordero-grande"


def grid_patches(mask, patch):
    """Voxel indices of the patches camppca's loop fits on: a grid that tiles the mask, each
    patch wholly inside it.

    Returns:
        (n, P, 3) indices, rows in camppca's patch order.
    """
    (x, y, z), _ = _sample_centers(mask, np.asarray(patch))
    off = patch_offsets(patch) - np.asarray(patch) // 2
    return np.stack([x, y, z], -1)[:, None] + off[None]


def sigma(X):
    """MP-PCA's noise level of each (P, N) patch in a (B, P, N) stack, and its rank."""
    X = np.asarray(X, np.float64)
    P, N = X.shape[1:]
    G = X.transpose(0, 2, 1) @ X if P > N else X @ X.transpose(0, 2, 1)
    evals = np.maximum(np.linalg.eigvalsh(G)[:, ::-1], 0.0)
    rank, s = mp_threshold(evals, min(P, N), max(P, N), ALGO)
    return s, rank


def perp2(S):
    """Noise variance of a (P, P) covariance off the patch-mean mode, tr(Q S Q) / (P - 1)."""
    P = S.shape[0]
    return float((np.trace(S) - S.sum() / P) / (P - 1))


def shape(C, d):
    """D C D, with D the patch's level scaled to unit RMS."""
    d = np.maximum(np.asarray(d, np.float64), 1e-9)
    d = d / np.sqrt(np.mean(d ** 2))
    return d[:, None] * C * d[None, :]


def root(S, power=0.5):
    w, V = np.linalg.eigh(S)
    return (V * np.maximum(w, 1e-12) ** power) @ V.T


def fit(X, C, d):
    """Fit the model's covariance to a patch: whiten by D C D, split by MP, keep the signal.

    Args:
        X: (P, N) patch in data units.
        C: (P, P) patch correlation.
        d: (P,) level at the patch's voxels, any scale.

    Returns:
        `(Sigma, kept)`: the (P, P) fitted covariance v D C D, and the (P, N) kept signal.
    """
    X = np.asarray(X, np.float64)
    S = shape(C, d)
    _, s, Vt = np.linalg.svd(root(S, -0.5) @ X, full_matrices=False)
    rank, level = mp_threshold((s ** 2)[None], s.size, max(X.shape), ALGO)
    return max(float(level[0]) ** 2, 1e-24) * S, X @ Vt[:rank[0]].T @ Vt[:rank[0]]


def predicted(Sigma, kept, G):
    """MPPCA's noise level on a patch, predicted as its median over draws of the
    fitted noise plus the kept signal.

    Args:
        Sigma: (P, P) fitted covariance.
        kept: (P, N) kept signal.
        G: (n_draw, P, N) standard normal draws.
    """
    return float(np.median(sigma(np.matmul(root(Sigma), G) + kept[None])[0]))
