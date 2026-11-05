"""Standard MPPCA's noise level on patches, read and predicted from a noise covariance model."""

import numpy as np
from camppca.denoiser.mp_threshold import mp_threshold
from camppca.patch import patch_offsets

ALGO = "cordero-grande"


def sample_patches(mask, patch, n, rng):
    """Voxel indices of n patches drawn at random, each lying wholly inside the mask.

    Returns:
        (n, P, 3) indices, rows in camppca's patch order.
    """
    off = patch_offsets(patch) - np.asarray(patch) // 2
    shape = np.array(mask.shape)
    centres = np.argwhere(mask)
    out = []
    while len(out) < n:
        p = centres[rng.integers(len(centres))] + off
        if (p >= 0).all() and (p < shape).all() and mask[tuple(p.T)].all():
            out.append(p)
    return np.array(out)


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


def freeze(X, C, d):
    """Fit the model's covariance to each patch: whiten by D C D, split by MP, keep the signal.

    Args:
        X: (B, P, N) patches in data units.
        C: (P, P) patch correlation.
        d: (B, P) level at each patch's voxels, any scale.

    Returns:
        `(Sigma, kept)`: (B, P, P) fitted covariances v D C D, and (B, P, N) kept signal.
    """
    Sigma, kept = [], []
    for Xb, db in zip(np.asarray(X, np.float64), d, strict=True):
        S = shape(C, db)
        _, s, Vt = np.linalg.svd(root(S, -0.5) @ Xb, full_matrices=False)
        rank, level = mp_threshold((s ** 2)[None], s.size, max(Xb.shape), ALGO)
        Sigma.append(max(float(level[0]) ** 2, 1e-24) * S)
        kept.append(Xb @ Vt[:rank[0]].T @ Vt[:rank[0]])
    return np.array(Sigma), np.array(kept)


def predicted(Sigma, kept, G):
    """Standard MPPCA's noise level on a patch, predicted as its median over draws of the
    fitted noise plus the kept signal.

    Args:
        Sigma: (P, P) fitted covariance.
        kept: (P, N) kept signal.
        G: (n_draw, P, N) standard normal draws.
    """
    return float(np.median(sigma(np.matmul(root(Sigma), G) + kept[None])[0]))
