"""Stage 1: a brain mask per scan, FSL bet -R -f 0.3 on the mean b0
(`camppca_paper.masks.b0_brain_mask`).

Out, per scan in data/analyses/panel/<config>/<subject>/: brain_mask_denoising.nii.gz,
b0_mean.nii.gz. Scans with both are skipped.

    python scripts/analyses/panel/masks.py [config ...]
"""

import argparse
from multiprocessing import Pool

import nibabel as nib
import numpy as np

from camppca_paper.masks import b0_brain_mask
from camppca_paper.panel import SCANS, load

B0_MAX = 100


def mask(scan):
    out = scan.work
    if (out / "brain_mask_denoising.nii.gz").exists() and (out / "b0_mean.nii.gz").exists():
        return f"  {scan.name}: done"
    data, bval, _, affine, _ = load(scan)
    b0 = data[..., bval <= B0_MAX].mean(-1)
    m = b0_brain_mask(b0, affine)
    out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(b0.astype(np.float32), affine), out / "b0_mean.nii.gz")
    nib.save(nib.Nifti1Image(m.astype(np.uint8), affine), out / "brain_mask_denoising.nii.gz")
    return f"  {scan.name}: {data.shape[3]} volumes, {int(m.sum())} mask voxels"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="*")
    configs = ap.parse_args().configs
    with Pool(8) as pool:
        for line in pool.imap(mask, [s for s in SCANS if not configs or s.config in configs]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
