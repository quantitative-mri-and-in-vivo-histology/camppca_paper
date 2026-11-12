"""Stage 9: the within- and cross-scanner numbers (`camppca_paper.metrics`).

    signal      within only: each arm's error against the other sessions' undenoised mean, per
                shell, split into variance and bias squared
    CoV         per map, across the sessions (within) or scanners (cross), median over white
                matter: the T1's white-matter posterior above WM_PV, inside the frame's mask
    correction  per session, Proposed's noise level over MPPCA's, minus 1, in %
    noise       per session, each arm's noise level over the series' mean b0, in %

Out: data/results/on_harmony/{within,cross}.json, one entry per participant.

    python scripts/analyses/on_harmony/metrics.py [subject ...]
"""

import argparse
import json

import nibabel as nib
import numpy as np

from camppca_paper.metrics import MAPS, correction, cov, noise_level, signal
from camppca_paper.on_harmony import PARTICIPANTS, SETS, SOURCE, WORK, sessions
from camppca_paper.paths import RESULTS

ARMS = ("raw", "standard", "proposed")
WM_PV = 0.5
OUT = RESULTS / "on_harmony"


def noise(s):
    base = SOURCE / s.dwi("AP")
    return noise_level(s.native, f"{base}.nii.gz", np.loadtxt(f"{base}.bval").ravel())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("subjects", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for which in SETS:
        path = OUT / f"{which}.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        for subject in a.subjects or list(PARTICIPANTS):
            frame = WORK / subject / "frames" / which
            maps = WORK / subject / "maps" / which
            ses = sessions(subject, which)
            names = [s.name for s in ses]
            if not all((maps / f"{n}_{arm}_RK.nii.gz").exists() for n in names for arm in ARMS):
                print(f"  {subject} {which}: maps missing, skipped", flush=True)
                continue
            mask = np.asarray(nib.load(frame / "mask.nii.gz").dataobj) > 0
            wm = mask & (np.asarray(nib.load(frame / "wm_prob.nii.gz").dataobj) > WM_PV)
            entry = {"sessions": names, "cov": cov(maps, names, wm, ARMS),
                     "correction": {s.name: correction(s.native) for s in ses},
                     "noise": {s.name: noise(s) for s in ses}}
            if which == "within":
                entry["signal"] = signal(frame, names, mask, ARMS)
            res[subject] = entry
            c = entry["cov"]
            print(f"  {subject} {which}: CoV % " + "  ".join(
                f"{m} " + "/".join(f"{100 * c[m][arm]:.1f}" for arm in ARMS) for m in MAPS),
                flush=True)
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
