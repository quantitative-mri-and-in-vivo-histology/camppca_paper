"""DTI and DKI fits with dipy."""

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.reconst.dti import TensorModel

B0_MAX = 100
B_DTI = 1000
FLOOR = 1.0
BLOWUP = 3
MAPS = ("FA", "MD", "MK", "AK", "RK")


def _load(series, mask, bval, bvec):
    img = nib.load(series)
    mask = np.asarray(nib.load(mask).dataobj) > 0
    data = np.maximum(np.asarray(img.dataobj, np.float32)[mask], FLOOR).astype(np.float64)
    return img, mask, data, np.loadtxt(bval).ravel(), np.loadtxt(bvec)


def _volume(mask, values):
    vol = np.zeros(mask.shape, np.float32)
    vol[mask] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return vol


def dki_s0_fa(series, mask, bval, bvec, out):
    """DKI S0 and FA for building diffusion templates.

    dipy's DKI, weighted least squares, on every volume inside the mask, data clipped at
    FLOOR. The S0 fit diverges in a few voxels, so every S0 above BLOWUP times the brain's 99th
    percentile is replaced by the median of its 3x3x3 neighbourhood inside the mask.

    Args:
        series, mask, bval, bvec: The diffusion series, its brain mask, b-values and b-vectors.
        out: Folder for dki_S0.nii.gz and dki_FA.nii.gz.

    Returns:
        The number of S0 voxels replaced.
    """
    img, mask, data, b, g = _load(series, mask, bval, bvec)
    gtab = gradient_table(b, bvecs=g.T, b0_threshold=B0_MAX)
    fit = DiffusionKurtosisModel(gtab, fit_method="WLS", return_S0_hat=True).fit(data)
    s0, fa = _volume(mask, fit.S0_hat), _volume(mask, fit.fa)
    bad = mask & (s0 > BLOWUP * np.percentile(s0[mask], 99))
    good = mask & ~bad
    for i, j, k in np.argwhere(bad):
        box = tuple(slice(max(c - 1, 0), c + 2) for c in (i, j, k))
        s0[i, j, k] = np.median(s0[box][good[box]]) if good[box].any() else np.median(s0[good])
    nib.save(nib.Nifti1Image(s0, img.affine), out / "dki_S0.nii.gz")
    nib.save(nib.Nifti1Image(fa, img.affine), out / "dki_FA.nii.gz")
    return int(bad.sum())


def dti_dki_maps(series, mask, bval, bvec, prefix):
    """FA and MD from DTI, MK, AK and RK from DKI.

    DTI is log-linear least squares on the b0s and the b = B_DTI shell. DKI is weighted least
    squares on every volume, kurtosis bounded to 0..3. Data are clipped at FLOOR.

    Args:
        series, mask, bval, bvec: The diffusion series, its mask, b-values and b-vectors.
        prefix: Output path prefix. Writes <prefix>_{FA,MD,MK,AK,RK}.nii.gz.
    """
    img, mask, data, b, g = _load(series, mask, bval, bvec)
    idx = np.where((b < B0_MAX) | (np.abs(b - B_DTI) < 100))[0]
    t = TensorModel(gradient_table(b[idx], bvecs=g[:, idx].T, b0_threshold=B0_MAX),
                    fit_method="LS").fit(data[:, idx])
    k = DiffusionKurtosisModel(gradient_table(b, bvecs=g.T, b0_threshold=B0_MAX),
                               fit_method="WLS").fit(data)
    for name, v in (("FA", t.fa), ("MD", t.md), ("MK", k.mk(min_kurtosis=0, max_kurtosis=3)),
                    ("AK", k.ak(min_kurtosis=0, max_kurtosis=3)),
                    ("RK", k.rk(min_kurtosis=0, max_kurtosis=3))):
        nib.save(nib.Nifti1Image(_volume(mask, v), img.affine), f"{prefix}_{name}.nii.gz")
