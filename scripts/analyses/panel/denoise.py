"""Stage 2: denoise every scan, as ON-Harmony.

Standard MPPCA and Proposed on the series up to b SHELL_MAX (`camppca_paper.panel.load`), both
on every voxel. Proposed estimates its noise model inside brain_mask_denoising (stage 1). The
patch is camppca's default, dwidenoise's rule.

Out, per scan in data/analyses/panel/<config>/<subject>/: sigma_standard.nii.gz,
sigma_proposed.nii.gz, and noise_model.npz (Proposed's correlation table rho, level field d,
lag, patch). Scans with all outputs are skipped.

    python scripts/analyses/panel/denoise.py [config ...]
"""

import argparse
import time

import camppca
import nibabel as nib
import numpy as np

from camppca_paper.panel import SCANS, load

B0_MAX = 100
OUTPUTS = ("sigma_standard.nii.gz", "sigma_proposed.nii.gz", "noise_model.npz")


def denoise(scan):
    out = scan.work
    if all((out / name).exists() for name in OUTPUTS):
        return
    t0 = time.time()
    data, bval, _, affine, _ = load(scan)
    mask = np.asarray(nib.load(out / "brain_mask_denoising.nii.gz").dataobj) > 0

    std = camppca.denoise_mppca(data)
    prop = camppca.denoise_camppca(data, bvals=bval, mask=mask)

    for name, x in (("sigma_standard", std.sigma_hat), ("sigma_proposed", prop.sigma_hat)):
        nib.save(nib.Nifti1Image(np.asarray(x, np.float32), affine), out / f"{name}.nii.gz")
    m = prop.model
    np.savez(out / "noise_model.npz", rho=m.rho, d=m.d.astype(np.float32), lag=m.lag,
             patch=np.array(prop.provenance["patch"]))
    ok = mask & (std.sigma_hat > 0) & (prop.sigma_hat > 0)
    corr = 100 * (np.median(prop.sigma_hat[ok] / std.sigma_hat[ok]) - 1)
    print(f"  {scan.name}: {data.shape[3]} volumes, patch {prop.provenance['patch']}, "
          f"correction {corr:.1f} %  [{time.time() - t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="*")
    configs = ap.parse_args().configs
    for scan in SCANS:
        if not configs or scan.config in configs:
            denoise(scan)


if __name__ == "__main__":
    main()
