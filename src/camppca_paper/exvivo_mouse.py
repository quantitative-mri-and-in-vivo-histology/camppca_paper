"""The ex vivo mouse brain: a two-average series and its twenty-average reference."""

import numpy as np

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


def shown(mask):
    """The middle slice and a function that crops a 2-D map of it to the brain, blanks the
    background and flips it along y."""
    k = mask.shape[2] // 2
    m = mask[:, :, k]
    xi, yi = np.where(m)
    sx, sy = slice(xi.min(), xi.max() + 1), slice(yi.min(), yi.max() + 1)

    def cut(a):
        return np.where(m[sx, sy], a[sx, sy], np.nan)[:, ::-1].T
    return k, (sy.stop - sy.start) / (sx.stop - sx.start), cut
