"""Stages 5 and 6: one diffusion template per participant and set, and the T1 registered to it.

Stage 5 builds a two-channel template from the sessions' DKI S0 and FA (stage 3) and registers
every session to it once more. Channel weights are S0 1 : FA 1 within a scanner, and
S0 0.33 : FA 0.67 across scanners, since S0 contrast differs between vendors.

Stage 6 registers the participant's T1 (stage 4) to each template and carries its brain mask
and white-matter posterior onto the template grid.

Out: data/analyses/on_harmony/<subject>/dwi/<set>/ (T_template{0,1}.nii.gz, to_template/, t1/).
Finished templates are skipped.

    python scripts/analyses/on_harmony/templates.py [--workers 4] [--threads 8] [subject ...]
"""

import argparse
import time
from multiprocessing import Pool

from camppca_paper.on_harmony import PARTICIPANTS, SETS, WORK, sessions
from camppca_paper.templates import dwi_template, t1_to_template

WEIGHTS = {"within": (1.0, 1.0), "cross": (0.33, 0.67)}


def one(task):
    subject, which, threads = task
    out = WORK / subject / "dwi" / which
    if (out / "t1" / "wm_prob.nii.gz").exists():
        return None
    t0 = time.time()
    inputs = {s.name: (s.native / "dki_S0.nii.gz", s.native / "dki_FA.nii.gz")
              for s in sessions(subject, which)}
    r = dwi_template(out, inputs, WEIGHTS[which], threads)
    t1 = WORK / subject / "t1"
    t1_to_template(out, t1 / "t1_brain.nii.gz", t1 / "brain_mask.nii.gz",
                   t1 / "seg_post2.nii.gz", threads)
    worst = min(min(v) for v in r.values())
    return (f"{subject} {which}: templates, lowest warped-vs-template r {worst:.3f}  "
            f"[{time.time() - t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4, help="templates built at the same time")
    ap.add_argument("--threads", type=int, default=8, help="ITK threads each")
    ap.add_argument("subjects", nargs="*")
    a = ap.parse_intermixed_args()
    tasks = [(s, w, a.threads) for s in a.subjects or list(PARTICIPANTS) for w in SETS]
    with Pool(a.workers) as pool:
        for line in pool.imap_unordered(one, tasks):
            if line:
                print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
