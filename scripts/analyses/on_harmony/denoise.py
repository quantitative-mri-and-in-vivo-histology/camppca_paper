"""Stage 1: denoise every session in its own space, before any correction.

Standard MPPCA and Proposed on the AP series, both on every voxel. Proposed estimates its noise
model inside a brain mask from FSL bet -R -f 0.3 on the mean b0
(`camppca_paper.masks.b0_brain_mask`). The patch is camppca's default, dwidenoise's rule.

Out, per session in data/analyses/on_harmony/<subject>/native/<session>/: standard.nii.gz,
proposed.nii.gz, sigma_standard.nii.gz, sigma_proposed.nii.gz, brain_mask_denoising.nii.gz.
Sessions with all outputs are skipped.

    python scripts/analyses/on_harmony/denoise.py [subject ...]
"""

import argparse
import time

import camppca
import nibabel as nib
import numpy as np

from camppca_paper.masks import b0_brain_mask
from camppca_paper.on_harmony import PARTICIPANTS, SOURCE, all_sessions

B0_MAX = 100
OUTPUTS = ("standard", "proposed", "sigma_standard", "sigma_proposed", "brain_mask_denoising")


def denoise(session):
    out = session.native
    if all((out / f"{name}.nii.gz").exists() for name in OUTPUTS):
        return
    t0 = time.time()
    stem = SOURCE / session.dwi("AP")
    img = nib.load(f"{stem}.nii.gz")
    data = np.asarray(img.dataobj, np.float32)
    bval = np.loadtxt(f"{stem}.bval").ravel()
    mask = b0_brain_mask(data[..., bval <= B0_MAX].mean(-1), img.affine)

    std = camppca.denoise_mppca(data)
    prop = camppca.denoise_camppca(data, bvals=bval, mask=mask)

    out.mkdir(parents=True, exist_ok=True)
    for name, x in (("standard", std.denoised), ("proposed", prop.denoised)):
        nib.save(nib.Nifti1Image(np.asarray(x, np.float32), img.affine, img.header),
                 out / f"{name}.nii.gz")
    for name, x in (("sigma_standard", std.sigma_hat), ("sigma_proposed", prop.sigma_hat)):
        nib.save(nib.Nifti1Image(np.asarray(x, np.float32), img.affine), out / f"{name}.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine), out / "brain_mask_denoising.nii.gz")
    print(f"  {session.subject} {session.name}: {data.shape[3]} volumes, "
          f"patch {std.provenance['patch']}  [{time.time() - t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("subjects", nargs="*")
    for subject in ap.parse_args().subjects or list(PARTICIPANTS):
        for session in all_sessions(subject):
            denoise(session)


if __name__ == "__main__":
    main()
