"""Stage 8: DTI and DKI of every run and arm in the template (`camppca_paper.fits.dti_dki_maps`).

Out: data/analyses/edden/<part>/maps/<run>_<arm>_{FA,MD,MK,AK,RK}.nii.gz. Finished series are skipped.

    python scripts/analyses/edden/fit.py [--workers 2]
"""

import argparse
from multiprocessing import Pool

from camppca_paper.edden import ARMS, PARTS, RUNS, work
from camppca_paper.fits import dti_dki_maps


def fit(task):
    part, r, arm = task
    frame, maps = work(part) / "frames", work(part) / "maps"
    dti_dki_maps(frame / f"{r}_{arm}.nii.gz", frame / "mask.nii.gz", frame / f"{r}.bval",
                 frame / f"{r}.bvec", maps / f"{r}_{arm}")
    return f"{part} {r} {arm}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for part in PARTS:
        (work(part) / "maps").mkdir(parents=True, exist_ok=True)
    tasks = [(part, r, arm) for part in PARTS for r in RUNS for arm in ARMS[part]
             if not (work(part) / "maps" / f"{r}_{arm}_RK.nii.gz").exists()]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(fit, tasks):
            print(f"  {line} fitted", flush=True)
