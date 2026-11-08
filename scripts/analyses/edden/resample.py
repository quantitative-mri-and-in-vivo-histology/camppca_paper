"""Stage 7: per input, every arm of every run into its diffusion template, in one resampling
(`camppca_paper.registration.resample`).

The arms are the input (raw), Standard and Proposed, and for the complex data NORDIC2's magnitude.

Out: data/analyses/edden/<part>/frames/ (<run>_<arm>.nii.gz, <run>.bval, <run>.bvec, mask.nii.gz,
wm_prob.nii.gz). Skipped when done.

    python scripts/analyses/edden/resample.py [--workers 3]
"""

import argparse

from camppca_paper.edden import ARMS, PARTS, RUNS, arm, bval, native, work
from camppca_paper.registration import resample

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3, help="series pulled in parallel")
    a = ap.parse_args()
    for part in PARTS:
        dest = work(part) / "frames"
        if all((dest / f"{r}_{x}.nii.gz").exists() for r in RUNS for x in ARMS[part]) \
                and (dest / "wm_prob.nii.gz").exists():
            continue
        todo = [(r, native(part, r), bval(), {x: arm(part, r, x) for x in ARMS[part]}) for r in RUNS]
        check = resample(work(part) / "dwi", dest, todo, a.workers)
        print(f"  {part} resampled, lowest FA check r {min(check.values()):.4f}", flush=True)
