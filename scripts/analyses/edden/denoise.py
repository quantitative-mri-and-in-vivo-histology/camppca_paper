"""Stage 1: MPPCA and Proposed on every run's magnitude, and on its real part.

    real     complex only, `camppca phase` with the Siemens phase range PHASE_RANGE: magnitude
             and phase rotated onto the real axis
    mask     FSL bet -R -f 0.1 on the mean b0 of the input, holes filled
             (`camppca_paper.masks.b0_brain_mask`)
    arms     MPPCA (camppca.denoise_mppca) and Proposed (camppca.denoise_camppca), both
             on every voxel, with camppca's default patch. Proposed estimates its noise model
             inside the mask. NORDIC2 runs separately, see nordic2.m

Out, per run in data/analyses/edden/<part>/native/<run>/: real.nii.gz (complex), standard.nii.gz,
proposed.nii.gz, sigma_standard.nii.gz, sigma_proposed.nii.gz, brain_mask_denoising.nii.gz.
Finished runs are skipped.

    python scripts/analyses/edden/denoise.py [run ...]
"""

import argparse
import time

import camppca
import nibabel as nib
import numpy as np

from camppca_paper.edden import PARTS, RUNS, bval, native, raw, series
from camppca_paper.masks import b0_brain_mask
from camppca_paper.tools import run

B0_MAX = 100
PHASE_RANGE = (-4096, 4096)
OUTPUTS = ("standard", "proposed", "sigma_standard", "sigma_proposed", "brain_mask_denoising")


def denoise(part, r):
    out = native(part, r)
    if all((out / f"{name}.nii.gz").exists() for name in OUTPUTS):
        return
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    if part == "complex" and not (out / "real.nii.gz").exists():
        run("camppca", "phase", "--phase-range", *PHASE_RANGE, series("AP", r, "mag"), series("AP", r, "phase"),
            out / "real.nii.gz")
    img = nib.load(raw(part, r))
    data = np.asarray(img.dataobj, np.float32)
    b = np.loadtxt(bval()).ravel()
    mask = b0_brain_mask(data[..., b <= B0_MAX].mean(-1), img.affine, f=0.1, fill_holes=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine), out / "brain_mask_denoising.nii.gz")
    for arm, fn in (("standard", lambda: camppca.denoise_mppca(data)),
                    ("proposed", lambda: camppca.denoise_camppca(data, bvals=b, mask=mask))):
        got = fn()
        nib.save(nib.Nifti1Image(np.asarray(got.denoised, np.float32), img.affine, img.header),
                 out / f"{arm}.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(got.sigma, np.float32), img.affine),
                 out / f"sigma_{arm}.nii.gz")
        patch = got.provenance["patch"]
        del got
    print(f"  {part} {r}: {data.shape[3]} volumes, patch {patch}  [{time.time() - t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*")
    for part in PARTS:
        for r in ap.parse_args().runs or RUNS:
            denoise(part, r)


if __name__ == "__main__":
    main()
