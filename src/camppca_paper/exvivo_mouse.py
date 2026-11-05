"""The ex vivo mouse brain: a two-average series and its twenty-average reference."""

import numpy as np
from scipy.ndimage import binary_erosion

from .paths import DATA_ROOT, EXVIVO

SERIES = EXVIVO / "MB_exp2.nii"
AVERAGE = EXVIVO / "MB_exp2_20averages.nii"
MASK = EXVIVO / "MB_exp2_mask.nii"
BVAL = EXVIVO / "MB.bval"
BVEC = EXVIVO / "MB.bvec"
WORK = DATA_ROOT / "analyses/exvivo_mouse"
NATIVE = WORK / "native"
MAPS = WORK / "maps"
ARMS = ("reference", "noisy", "standard", "proposed")
ERODE = 2


def scored(mask):
    """The tissue everything is scored on: the mask eroded in-plane by ERODE voxels. There are
    only nine slices, so there is no erosion through-slice."""
    return binary_erosion(mask, np.ones((3, 3, 1), bool), iterations=ERODE)
