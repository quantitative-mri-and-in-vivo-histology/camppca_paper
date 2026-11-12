"""Stage 9: the repeatability numbers across the four runs, per input (`camppca_paper.metrics`).

    signal      magnitude only, each arm's error against the other runs' undenoised mean, per
                shell, split into variance and bias squared
    CoV         per map, across the runs, median over white matter: the T1's white-matter
                posterior above WM_PV, inside the frame's mask
    correction  per run, Proposed's noise level over MPPCA's, minus 1, in %
    noise       per run, each arm's noise level over the input's mean b0, in %. NORDIC2's from
                its g-factor map (`camppca_paper.edden.nordic2_sigma`)

Out: data/results/edden/{magnitude,complex}.json.

    python scripts/analyses/edden/metrics.py
"""

import json

import nibabel as nib
import numpy as np

from camppca_paper.edden import ARMS, PARTS, RUNS, bval, native, nordic2_sigma, raw, work
from camppca_paper.metrics import MAPS, correction, cov, noise_level, signal
from camppca_paper.paths import RESULTS

WM_PV = 0.5

if __name__ == "__main__":
    (RESULTS / "edden").mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        frame, maps, arms = work(part) / "frames", work(part) / "maps", ARMS[part]
        mask = np.asarray(nib.load(frame / "mask.nii.gz").dataobj) > 0
        wm = mask & (np.asarray(nib.load(frame / "wm_prob.nii.gz").dataobj) > WM_PV)
        res = {"runs": list(RUNS), "cov": cov(maps, list(RUNS), wm, arms),
               "correction": {r: correction(native(part, r)) for r in RUNS},
               "noise": {r: noise_level(native(part, r), raw(part, r), np.loadtxt(bval()).ravel(),
                                        {"nordic2": nordic2_sigma(r)} if "nordic2" in arms else None)
                         for r in RUNS}}
        if part == "magnitude":
            res["signal"] = signal(frame, list(RUNS), mask, arms)
        (RESULTS / "edden" / f"{part}.json").write_text(json.dumps(res, indent=1))
        c = res["cov"]
        print(f"  {part} CoV % " + "  ".join(f"{m} " + "/".join(f"{100 * c[m][a]:.1f}" for a in arms)
                                            for m in MAPS), flush=True)
