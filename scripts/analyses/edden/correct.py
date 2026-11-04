"""Stage 2: topup and eddy on every run.

    topup  the first N_B0 b0s of the AP magnitude and of the PA magnitude of the same run
    mask   FSL bet -R -f 0.1 on topup's mean corrected b0, holes filled
           (`camppca_paper.masks.b0_brain_mask`), as eddy's mask
    eddy   eddy_cuda11.0 on the Standard MPPCA series (stage 1), so the alignment cannot favour
           another arm. Only eddy's estimates are used later, in stage 7

Out, per run in data/analyses/edden/native/<run>/: topup_*, b0_mean.nii.gz, mask_bet.nii.gz,
eddy.*. Finished steps are skipped.

    python scripts/analyses/edden/correct.py [--gpu-workers 2] [run ...]
"""

import argparse
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np

from camppca_paper.correct import B0_MAX, eddy, topup
from camppca_paper.edden import ACQP, RUNS, bval, bvec, native, series
from camppca_paper.masks import b0_brain_mask

N_B0 = 2


def correct(r):
    out = native(r)
    t0 = time.time()
    if not (out / "b0_mean.nii.gz").exists():
        b0s, rows, affine = [], [], None
        for direction in ("AP", "PA"):
            img = nib.load(series(direction, r))
            use = np.where(np.loadtxt(bval(direction)).ravel() < B0_MAX)[0][:N_B0]
            b0s += [np.asarray(img.dataobj[..., int(j)], np.float32) for j in use]
            rows += [ACQP[direction]] * len(use)
            affine = img.affine if affine is None else affine
        topup(b0s, rows, affine, out)
    if not (out / "mask_bet.nii.gz").exists():
        b0 = nib.load(out / "b0_mean.nii.gz")
        mask = b0_brain_mask(np.asarray(b0.dataobj, np.float32), b0.affine, f=0.1,
                             fill_holes=True)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), b0.affine), out / "mask_bet.nii.gz")
    if not (out / "eddy.eddy_parameters").exists():
        eddy(out / "standard.nii.gz", ACQP["AP"], bval(), bvec(), out)
    return f"{r}: topup, mask and eddy  [{time.time() - t0:.0f}s]"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-workers", type=int, default=2, help="runs corrected in parallel")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_intermixed_args()
    with Pool(a.gpu_workers) as pool:
        for line in pool.imap_unordered(correct, a.runs or RUNS):
            print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
