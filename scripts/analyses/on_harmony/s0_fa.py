"""Stage 3: DKI S0 and FA of every session in its own space, the inputs of the diffusion
templates.

dipy's DKI, weighted least squares, on eddy's corrected Standard series (stage 2) with eddy's
rotated b-vectors, inside the bet mask. Data are clipped at FLOOR. The S0 fit diverges in a few
voxels, so every S0 above BLOWUP times the brain's 99th percentile is replaced by the median of
its 3x3x3 neighbourhood inside the mask.

Out, per session in data/analyses/on_harmony/<subject>/native/<session>/: dki_S0.nii.gz,
dki_FA.nii.gz. Finished sessions are skipped.

    python scripts/analyses/on_harmony/s0_fa.py [--workers 8] [subject ...]
"""

import argparse
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.dki import DiffusionKurtosisModel

from camppca_paper.on_harmony import PARTICIPANTS, SOURCE, all_sessions

B0_MAX = 100
FLOOR = 1.0
BLOWUP = 3


def s0_fa(session):
    out = session.native
    img = nib.load(out / "eddy.nii.gz")
    mask = np.asarray(nib.load(out / "mask_bet.nii.gz").dataobj) > 0
    data = np.maximum(np.asarray(img.dataobj, np.float32)[mask], FLOOR)
    bval = np.loadtxt(f"{SOURCE / session.dwi('AP')}.bval").ravel()
    gtab = gradient_table(bval, bvecs=np.loadtxt(out / "eddy.eddy_rotated_bvecs").T,
                          b0_threshold=B0_MAX)
    fit = DiffusionKurtosisModel(gtab, fit_method="WLS", return_S0_hat=True).fit(data)
    maps = {}
    for name, v in (("S0", fit.S0_hat), ("FA", fit.fa)):
        maps[name] = np.zeros(mask.shape, np.float32)
        maps[name][mask] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    s0 = maps["S0"]
    bad = mask & (s0 > BLOWUP * np.percentile(s0[mask], 99))
    good = mask & ~bad
    for i, j, k in np.argwhere(bad):
        box = tuple(slice(max(c - 1, 0), c + 2) for c in (i, j, k))
        s0[i, j, k] = np.median(s0[box][good[box]]) if good[box].any() else np.median(s0[good])
    for name, x in maps.items():
        nib.save(nib.Nifti1Image(x, img.affine), out / f"dki_{name}.nii.gz")
    return f"{session.subject} {session.name}: S0 and FA, {int(bad.sum())} S0 voxels repaired"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("subjects", nargs="*")
    a = ap.parse_intermixed_args()
    todo = [s for subject in a.subjects or list(PARTICIPANTS) for s in all_sessions(subject)
            if not (s.native / "dki_FA.nii.gz").exists()]
    with Pool(a.workers) as pool:
        for line in pool.imap_unordered(s0_fa, todo):
            print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
