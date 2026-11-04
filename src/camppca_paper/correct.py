"""Susceptibility, eddy-current and motion correction with FSL's topup and eddy."""

import nibabel as nib
import numpy as np

from .tools import run

B0_MAX = 100
EDDY_FLAGS = ("--repol", "--data_is_shelled", "--dont_peas", "--flm=quadratic", "--slm=none",
              "--initrand=1")


def topup(b0s, rows, affine, out):
    """topup on b0 volumes of opposite phase encodes, and the mean corrected b0.

    Args:
        b0s: List of 3D b0 volumes.
        rows: Their acqp rows, `[x, y, z, readout]`.
        affine: The volumes' affine.
        out: Output folder. Writes topup_*, b0_corrected and b0_mean.nii.gz.
    """
    out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.stack(b0s, -1).astype(np.float32), affine), out / "b0_pairs.nii.gz")
    np.savetxt(out / "acqp_topup.txt", rows, fmt="%g")
    cnf = "b02b0.cnf" if all(n % 2 == 0 for n in b0s[0].shape) else "b02b0_1.cnf"
    run("topup", f"--imain={out / 'b0_pairs.nii.gz'}", f"--datain={out / 'acqp_topup.txt'}",
        f"--config={cnf}", f"--out={out / 'topup'}", f"--iout={out / 'b0_corrected'}",
        log=out / "topup.log")
    corrected = nib.load(out / "b0_corrected.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(corrected.dataobj, np.float32).mean(-1), corrected.affine),
             out / "b0_mean.nii.gz")


def bet_mask(out, f=0.2):
    """Plain bet on topup's mean corrected b0, written as mask_bet.nii.gz for eddy."""
    run("bet", out / "b0_mean.nii.gz", out / "b0_brain", "-m", "-f", f)
    (out / "b0_brain_mask.nii.gz").rename(out / "mask_bet.nii.gz")


def eddy(series, row, bval, bvec, out):
    """eddy_cuda11.0 on one series, with topup's field from the same folder.

    Args:
        series: The diffusion series.
        row: Its acqp row, `[x, y, z, readout]`.
        bval, bvec: Its b-values and b-vectors.
        out: The folder `topup` wrote to. Writes eddy.* there.
    """
    n = nib.load(series).shape[3]
    np.savetxt(out / "acqp.txt", [row], fmt="%g")
    np.savetxt(out / "index.txt", np.ones((1, n), int), fmt="%d")
    run("eddy_cuda11.0", f"--imain={series}", f"--mask={out / 'mask_bet.nii.gz'}",
        f"--acqp={out / 'acqp.txt'}", f"--index={out / 'index.txt'}", f"--bvecs={bvec}",
        f"--bvals={bval}", f"--topup={out / 'topup'}", f"--out={out / 'eddy'}", *EDDY_FLAGS,
        log=out / "eddy.log")
