"""Stage 8: DTI and DKI of every session and arm in its set's template (stage 7).

`camppca_paper.fits.dti_dki_maps`: FA and MD from DTI on the b0s and b1000, MK, AK and RK from
DKI on every volume, inside the frame's mask, with the b-vectors rotated into the template.

Out: data/analyses/on_harmony/<subject>/maps/<set>/<session>_<arm>_{FA,MD,MK,AK,RK}.nii.gz.
Finished series are skipped.

    python scripts/analyses/on_harmony/fit.py [--workers 16] [subject ...]
"""

import argparse
from multiprocessing import Pool

from camppca_paper.fits import dti_dki_maps
from camppca_paper.on_harmony import PARTICIPANTS, SETS, WORK, sessions

ARMS = ("raw", "standard", "proposed")


def fit(task):
    frame, maps, name, arm = task
    dti_dki_maps(frame / f"{name}_{arm}.nii.gz", frame / "mask.nii.gz", frame / f"{name}.bval",
                 frame / f"{name}.bvec", maps / f"{name}_{arm}")
    return f"{frame.parent.parent.name} {frame.name} {name} {arm}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("subjects", nargs="*")
    a = ap.parse_intermixed_args()
    tasks = []
    for subject in a.subjects or list(PARTICIPANTS):
        for which in SETS:
            frame = WORK / subject / "frames" / which
            maps = WORK / subject / "maps" / which
            maps.mkdir(parents=True, exist_ok=True)
            tasks += [(frame, maps, s.name, arm) for s in sessions(subject, which) for arm in ARMS
                      if not (maps / f"{s.name}_{arm}_RK.nii.gz").exists()]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(fit, tasks):
            print(f"  {line} fitted", flush=True)


if __name__ == "__main__":
    main()
