"""Repeatability metrics for denoised series and diffusion maps in a common template."""

import nibabel as nib
import numpy as np

SHELLS = (0, 1000, 2000)
MAPS = ("FA", "MD", "MK", "AK", "RK")
MIN_MEAN = {"FA": 0.0, "MD": 0.0, "MK": 0.1, "AK": 0.1, "RK": 0.1}


def signal(frame, names, mask, arms, raw="raw"):
    """Signal error of each arm against the other sessions' undenoised mean, per shell.

    Each session is divided by its scale, its raw mean b0 in the mask over the sessions' mean.
    Per shell, over every volume and mask voxel:

        error     arm in session k minus the mean of the other sessions' raw data, squared, less
                  that reference's own noise s2 / (R - 1), s2 = R / (R - 1) times the raw
                  sessions' mean squared deviation from their mean
        variance  the arm's spread across sessions (ddof 1)
        bias2     error - variance

    Each also as % of the raw arm's error in that shell.

    Args:
        frame: Folder with <name>_<arm>.nii.gz and <name>.bval.
        names: Session names.
        mask: (X, Y, Z) bool.
        arms: Arm names, including `raw`.
        raw: The undenoised arm.
    """
    R = len(names)
    shell = np.round(np.loadtxt(frame / f"{names[0]}.bval").ravel(), -3)
    x = {(n, arm): np.asarray(nib.load(frame / f"{n}_{arm}.nii.gz").dataobj, np.float32)[mask]
         for n in names for arm in arms}
    level = np.array([x[n, raw][:, shell == 0].mean() for n in names])
    scale = level / level.mean()
    out = {"scale": scale.tolist()}
    for s in SHELLS:
        v = shell == s
        ref = np.stack([x[n, raw][:, v] / scale[i] for i, n in enumerate(names)]).astype(np.float64)
        s2 = R / (R - 1) * np.mean((ref - ref.mean(0)) ** 2)
        row = {}
        for arm in arms:
            a = np.stack([x[n, arm][:, v] / scale[i] for i, n in enumerate(names)]).astype(np.float64)
            err = np.mean([np.mean((a[k] - np.delete(ref, k, 0).mean(0)) ** 2)
                           for k in range(R)]) - s2 / (R - 1)
            var = np.mean(a.var(0, ddof=1))
            row[arm] = dict(mse=float(err), var=float(var), bias2=float(err - var))
        for arm in arms:
            row[arm] |= {f"{k}_pct": 100 * row[arm][k] / row[raw]["mse"]
                         for k in ("mse", "var", "bias2")}
        out[f"b{s}"] = row | {"n_volumes": int(v.sum())}
    return out


def cov(maps, names, wm, arms):
    """Coefficient of variation of each map across sessions, median over white matter.

    Per voxel, the SD across sessions over the arm's own mean. Only voxels where every arm's mean
    is above MIN_MEAN count, the same voxels for every arm.

    Args:
        maps: Folder with <name>_<arm>_<map>.nii.gz.
        names: Session names.
        wm: (X, Y, Z) bool white-matter mask.
        arms: Arm names.
    """
    out = {}
    for m in MAPS:
        v = {arm: np.stack([np.asarray(nib.load(maps / f"{n}_{arm}_{m}.nii.gz").dataobj,
                                       np.float64)[wm] for n in names]) for arm in arms}
        means = {arm: v[arm].mean(0) for arm in arms}
        ok = np.all([means[arm] > MIN_MEAN[m] for arm in arms], 0)
        out[m] = {arm: float(np.median(v[arm].std(0, ddof=1)[ok] / means[arm][ok])) for arm in arms}
    return out


def correction(native):
    """Proposed's noise level over Standard's, minus 1, in %, median over the denoising mask.

    Args:
        native: Folder with sigma_standard, sigma_proposed and brain_mask_denoising (.nii.gz).
    """
    mask = np.asarray(nib.load(native / "brain_mask_denoising.nii.gz").dataobj) > 0
    std, prop = (np.asarray(nib.load(native / f"sigma_{a}.nii.gz").dataobj)
                 for a in ("standard", "proposed"))
    ok = mask & (std > 0) & (prop > 0)
    return float(100 * (np.median(prop[ok] / std[ok]) - 1))
