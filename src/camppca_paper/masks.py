"""Brain masks of diffusion series."""

import tempfile

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

from .tools import run


def b0_brain_mask(b0, affine, f=0.3, fill_holes=False):
    """FSL bet with robust centre estimation (-R) on a mean b0.

    Args:
        b0: (X, Y, Z) mean b0.
        affine: Its affine.
        f: bet's fractional intensity threshold. Smaller gives a larger mask.
        fill_holes: Fill holes in 3D, then slice by slice along the third axis, so holes open
            at the edge of a slab are filled too.

    Returns:
        (X, Y, Z) bool brain mask.
    """
    with tempfile.TemporaryDirectory() as tmp:
        nib.save(nib.Nifti1Image(np.asarray(b0, np.float32), affine), f"{tmp}/b0.nii.gz")
        run("bet", f"{tmp}/b0.nii.gz", f"{tmp}/b0_brain", "-R", "-m", "-n", "-f", f)
        mask = np.asarray(nib.load(f"{tmp}/b0_brain_mask.nii.gz").dataobj) > 0
    if fill_holes:
        mask = binary_fill_holes(mask)
        for z in range(mask.shape[2]):
            mask[:, :, z] = binary_fill_holes(mask[:, :, z])
    return mask
