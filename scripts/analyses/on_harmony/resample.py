"""Stage 7: every arm of every session into its set's diffusion template, in one resampling.

Each volume is pulled through eddy's estimated correction for it and the session's warp into
the template, with a cubic B-spline (`camppca_paper.registration`). The arms are the source
series (raw), Standard and Proposed. b-vectors are eddy's rotated ones, then rotated by the
session-to-template affine.

Out: data/analyses/on_harmony/<subject>/frames/<set>/ (<session>_<arm>.nii.gz, <session>.bval,
<session>.bvec, mask.nii.gz, wm_prob.nii.gz). Finished sets are skipped.

    python scripts/analyses/on_harmony/resample.py [--workers 8] [subject ...]
"""

import argparse

from camppca_paper.on_harmony import PARTICIPANTS, SETS, SOURCE, WORK, sessions
from camppca_paper.registration import resample

ARMS = ("raw", "standard", "proposed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8, help="series pulled in parallel")
    ap.add_argument("subjects", nargs="*")
    a = ap.parse_intermixed_args()
    for subject in a.subjects or list(PARTICIPANTS):
        for which in SETS:
            tpl = WORK / subject / "dwi" / which
            dest = WORK / subject / "frames" / which
            if (dest / "wm_prob.nii.gz").exists():
                continue
            todo = []
            for s in sessions(subject, which):
                stem = SOURCE / s.dwi("AP")
                arms = {"raw": f"{stem}.nii.gz", "standard": s.native / "standard.nii.gz",
                        "proposed": s.native / "proposed.nii.gz"}
                todo.append((s.name, s.native, f"{stem}.bval", arms))
            check = resample(tpl, dest, todo, a.workers)
            print(f"  {subject} {which}: resampled, lowest FA check r {min(check.values()):.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
