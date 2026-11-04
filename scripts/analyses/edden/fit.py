"""Stage 8: DTI and DKI of every run and arm in the template (`camppca_paper.fits.dti_dki_maps`).

Out: data/analyses/edden/maps/<run>_<arm>_{FA,MD,MK,AK,RK}.nii.gz. Finished series are skipped.

    python scripts/analyses/edden/fit.py [--workers 2]
"""

import argparse
from multiprocessing import Pool

from camppca_paper.edden import RUNS, WORK
from camppca_paper.fits import dti_dki_maps

ARMS = ("raw", "standard", "proposed", "nordic2")


def fit(task):
    r, arm = task
    frame, maps = WORK / "frames", WORK / "maps"
    dti_dki_maps(frame / f"{r}_{arm}.nii.gz", frame / "mask.nii.gz", frame / f"{r}.bval",
                 frame / f"{r}.bvec", maps / f"{r}_{arm}")
    return f"{r} {arm}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    (WORK / "maps").mkdir(parents=True, exist_ok=True)
    tasks = [(r, arm) for r in RUNS for arm in ARMS
             if not (WORK / "maps" / f"{r}_{arm}_RK.nii.gz").exists()]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(fit, tasks):
            print(f"  {line} fitted", flush=True)
