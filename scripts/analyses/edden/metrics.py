"""Stage 9: the repeatability numbers across the four runs (`camppca_paper.metrics`).

    signal      each arm's error against the other runs' undenoised mean (real part), per
                shell, split into variance and bias squared
    CoV         per map, across the runs, median over white matter: the T1's white-matter
                posterior above WM_PV, inside the frame's mask
    correction  per run, Proposed's noise level over Standard's, minus 1, in %
    noise       per run, each arm's noise level over the real part's mean b0, in %

Out: data/results/edden/within.json.

    python scripts/analyses/edden/metrics.py
"""

import json

import nibabel as nib
import numpy as np

from camppca_paper.edden import RUNS, WORK, bval, native
from camppca_paper.metrics import MAPS, correction, cov, noise_level, signal
from camppca_paper.paths import RESULTS

ARMS = ("raw", "standard", "proposed", "nordic2")
WM_PV = 0.5

if __name__ == "__main__":
    frame, maps = WORK / "frames", WORK / "maps"
    mask = np.asarray(nib.load(frame / "mask.nii.gz").dataobj) > 0
    wm = mask & (np.asarray(nib.load(frame / "wm_prob.nii.gz").dataobj) > WM_PV)
    res = {"runs": list(RUNS), "cov": cov(maps, list(RUNS), wm, ARMS),
           "correction": {r: correction(native(r)) for r in RUNS},
           "noise": {r: noise_level(native(r), native(r) / "real.nii.gz", np.loadtxt(bval()).ravel())
                     for r in RUNS},
           "signal": signal(frame, list(RUNS), mask, ARMS)}
    (RESULTS / "edden").mkdir(parents=True, exist_ok=True)
    (RESULTS / "edden" / "within.json").write_text(json.dumps(res, indent=1))
    c = res["cov"]
    print("  CoV % " + "  ".join(f"{m} " + "/".join(f"{100 * c[m][a]:.1f}" for a in ARMS)
                                 for m in MAPS), flush=True)
