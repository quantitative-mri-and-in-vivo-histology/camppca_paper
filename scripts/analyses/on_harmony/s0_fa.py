"""Stage 3: DKI S0 and FA of every session in its own space, the inputs of the diffusion
templates.

`camppca_paper.fits.dki_s0_fa` on eddy's corrected Standard series (stage 2), with eddy's
rotated b-vectors, inside the bet mask.

Out, per session in data/analyses/on_harmony/<subject>/native/<session>/: dki_S0.nii.gz,
dki_FA.nii.gz. Finished sessions are skipped.

    python scripts/analyses/on_harmony/s0_fa.py [--workers 8] [subject ...]
"""

import argparse
from multiprocessing import Pool

from camppca_paper.fits import dki_s0_fa
from camppca_paper.on_harmony import PARTICIPANTS, SOURCE, all_sessions


def s0_fa(session):
    out = session.native
    n = dki_s0_fa(out / "eddy.nii.gz", out / "mask_bet.nii.gz",
                  f"{SOURCE / session.dwi('AP')}.bval", out / "eddy.eddy_rotated_bvecs", out)
    return f"{session.subject} {session.name}: S0 and FA, {n} S0 voxels repaired"


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
