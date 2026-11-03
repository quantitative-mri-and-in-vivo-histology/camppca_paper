"""Build the simulation phantom from the downloaded complex average.

Each voxel is fitted with spherical harmonics up to order 4, one shell at a time. The b0
volumes are averaged into one. The fit makes each patch low rank.

The average still holds noise at a b0 SNR of about 105.

    python scripts/datasets/subject_phantom/prepare.py
"""

import shutil

import nibabel as nib
import numpy as np
from fetch import AVERAGE, BRAIN_MASK, BVAL_FILE, BVEC_FILE, ROOT

from camppca_paper.paths import BVAL, BVEC, DWI
from camppca_paper.spherical_harmonics import sh_project

SH_ORDER = 4
B0_MAX = 100.0


def main():
    bval_f, bvec_f = ROOT / BVAL_FILE, ROOT / BVEC_FILE
    DWI.parent.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(ROOT / AVERAGE))
    signal = np.asarray(img.dataobj, np.float32)
    bval = np.loadtxt(bval_f).ravel()
    bvec = np.loadtxt(bvec_f)
    # The brain mask that ships with the release.
    mask = np.asarray(nib.load(str(ROOT / BRAIN_MASK)).dataobj) > 0

    out, residuals = sh_project(signal, mask, bval, bvec, SH_ORDER, B0_MAX)
    for edge, r in sorted(residuals.items()):
        print(f"  b = {int(edge):5d}   relative fit residual {100 * r:.2f}%")

    nib.save(nib.Nifti1Image(out, img.affine, img.header), DWI)
    shutil.copy2(bval_f, BVAL)
    shutil.copy2(bvec_f, BVEC)
    print(f"  wrote {DWI}")


if __name__ == "__main__":
    main()
