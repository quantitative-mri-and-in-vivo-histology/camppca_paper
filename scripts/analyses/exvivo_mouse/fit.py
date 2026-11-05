"""Stage 2: DTI and DKI of the reference, Noisy, Standard MPPCA and Proposed inside the mask
(`camppca_paper.fits.dti_dki_maps`).

Out: data/analyses/exvivo_mouse/maps/<arm>_{FA,MD,MK,AK,RK}.nii.gz. Finished arms are skipped.

    python scripts/analyses/exvivo_mouse/fit.py
"""

from multiprocessing import Pool

from camppca_paper.exvivo_mouse import ARMS, BVAL, BVEC, MAPS, NATIVE
from camppca_paper.fits import dti_dki_maps


def fit(arm):
    dti_dki_maps(NATIVE / f"{arm}.nii.gz", NATIVE / "mask.nii.gz", BVAL, BVEC, MAPS / arm)
    return arm


if __name__ == "__main__":
    MAPS.mkdir(parents=True, exist_ok=True)
    todo = [arm for arm in ARMS if not (MAPS / f"{arm}_RK.nii.gz").exists()]
    with Pool(len(ARMS)) as pool:
        for arm in pool.imap_unordered(fit, todo):
            print(f"  {arm} fitted", flush=True)
