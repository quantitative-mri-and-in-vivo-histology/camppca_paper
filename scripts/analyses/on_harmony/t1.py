"""Stage 4: brain mask and tissue segmentation of one T1w per participant
(`camppca_paper.t1.segment_t1`).

The T1w is the participant's first session on their own scanner, which is in both sets.

Out, per participant in data/analyses/on_harmony/<subject>/t1/: t1_n4.nii.gz, brain_mask.nii.gz,
t1_brain.nii.gz, seg.nii.gz, seg_post{1,2,3}.nii.gz. Finished participants are skipped.

    python scripts/analyses/on_harmony/t1.py [--workers 4] [--threads 8] [subject ...]
"""

import argparse
import time
from multiprocessing import Pool

from camppca_paper.on_harmony import PARTICIPANTS, SOURCE, WORK, sessions
from camppca_paper.t1 import segment_t1


def t1(task):
    subject, threads = task
    out = WORK / subject / "t1"
    if (out / "seg_post2.nii.gz").exists():
        return None
    t0 = time.time()
    segment_t1(SOURCE / sessions(subject, "within")[0].t1, out, threads)
    return f"{subject}: T1 segmentation  [{time.time() - t0:.0f}s]"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4, help="participants at the same time")
    ap.add_argument("--threads", type=int, default=8, help="ITK threads each")
    ap.add_argument("subjects", nargs="*")
    a = ap.parse_intermixed_args()
    with Pool(a.workers) as pool:
        for line in pool.imap_unordered(t1, [(s, a.threads) for s in a.subjects or list(PARTICIPANTS)]):
            if line:
                print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
