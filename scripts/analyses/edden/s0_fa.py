"""Stage 3: DKI S0 and FA of every run in its own space, the inputs of the diffusion template.

`camppca_paper.fits.dki_s0_fa` on eddy's corrected Standard series (stage 2), with eddy's
rotated b-vectors, inside the bet mask.

Out, per run in data/analyses/edden/native/<run>/: dki_S0.nii.gz, dki_FA.nii.gz. Finished runs
are skipped.

    python scripts/analyses/edden/s0_fa.py
"""

from camppca_paper.edden import RUNS, bval, native
from camppca_paper.fits import dki_s0_fa

if __name__ == "__main__":
    for r in RUNS:
        out = native(r)
        if not (out / "dki_FA.nii.gz").exists():
            n = dki_s0_fa(out / "eddy.nii.gz", out / "mask_bet.nii.gz", bval(),
                          out / "eddy.eddy_rotated_bvecs", out)
            print(f"  {r}: S0 and FA, {n} S0 voxels repaired", flush=True)
