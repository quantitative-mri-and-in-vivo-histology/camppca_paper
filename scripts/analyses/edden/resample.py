"""Stage 7: every arm of every run into the diffusion template, in one resampling
(`camppca_paper.registration.resample`).

The arms are the real part (raw), Standard, Proposed and NORDIC2. NORDIC2 is magnitude, written
in the input's units.

Out: data/analyses/edden/frames/ (<run>_<arm>.nii.gz, <run>.bval, <run>.bvec, mask.nii.gz,
wm_prob.nii.gz). Skipped when done.

    python scripts/analyses/edden/resample.py [--workers 3]
"""

import argparse

from camppca_paper.edden import NORDIC2, RUNS, WORK, bval, native
from camppca_paper.registration import resample

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3, help="series pulled in parallel")
    a = ap.parse_args()
    dest = WORK / "frames"
    if not (dest / "wm_prob.nii.gz").exists():
        todo = [(r, native(r), bval(), {"raw": native(r) / "real.nii.gz",
                                        "standard": native(r) / "standard.nii.gz",
                                        "proposed": native(r) / "proposed.nii.gz",
                                        "nordic2": NORDIC2 / f"{r}_nordic2_.nii.gz"})
                for r in RUNS]
        check = resample(WORK / "dwi", dest, todo, a.workers)
        print(f"  resampled, lowest FA check r {min(check.values()):.4f}", flush=True)
