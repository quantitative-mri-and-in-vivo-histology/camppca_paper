"""Stage 3: each arm against the reference, in the scored tissue (`camppca_paper.exvivo_mouse.scored`)
where both arms wrote a noise level.

    noise   sigma-hat over the true level, voxel by voxel: median, 10th and 90th percentile
    signal  per shell, the mean squared error against the reference as % of Noisy's. The
            reference's own noise, AVG_SHARE of the Noisy-minus-reference variance, is taken off
            both
    maps    per DTI and DKI map, the RMS difference from the reference's map in white matter
            (the reference's FA above WM_FA), as % of the reference map's mean

Out: data/results/exvivo_mouse/metrics.json.

    python scripts/analyses/exvivo_mouse/metrics.py
"""

import json

import nibabel as nib
import numpy as np

from camppca_paper.exvivo_mouse import BVAL, MAPS, NATIVE, scored
from camppca_paper.fits import MAPS as NAMES
from camppca_paper.paths import RESULTS

AVG_SHARE = 1 / 11
WM_FA = 0.25
OUT = RESULTS / "exvivo_mouse/metrics.json"


def load(name, where=None):
    data = np.asarray(nib.load(NATIVE / f"{name}.nii.gz").dataobj, np.float64)
    return data if where is None else data[where]


if __name__ == "__main__":
    mask = load("mask") > 0
    true = load("sigma_true")
    floor = 1e-3 * np.median(true[mask])
    valid = scored(mask) & (load("sigma_standard") > floor) & (load("sigma_proposed") > floor)
    res = {"n_voxels": int(valid.sum()), "noise": {}, "signal": {}, "maps": {}}
    for arm in ("standard", "proposed"):
        r = load(f"sigma_{arm}", valid) / true[valid]
        res["noise"][arm] = {"median": float(np.median(r)), "p10": float(np.percentile(r, 10)),
                             "p90": float(np.percentile(r, 90))}
    shell = np.round(np.loadtxt(BVAL).ravel(), -3)
    ref, noisy = load("reference", valid), load("noisy", valid)
    for b in np.unique(shell):
        v = shell == b
        n2 = np.mean((noisy[:, v] - ref[:, v]) ** 2)
        own = AVG_SHARE * n2
        res["signal"][f"b{int(b)}"] = {arm: float(100 * (np.mean((load(arm, valid)[:, v] - ref[:, v]) ** 2) - own)
                                                  / (n2 - own)) for arm in ("standard", "proposed")}

    def fitted(arm, m):
        return np.asarray(nib.load(MAPS / f"{arm}_{m}.nii.gz").dataobj, np.float64)

    wm = (fitted("reference", "FA") > WM_FA) & scored(mask)
    res["n_wm_voxels"] = int(wm.sum())
    for m in NAMES:
        ref_m = fitted("reference", m)[wm]
        res["maps"][m] = {arm: float(100 * np.sqrt(np.mean((fitted(arm, m)[wm] - ref_m) ** 2)) / ref_m.mean())
                          for arm in ("noisy", "standard", "proposed")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"  noise median {res['noise']['standard']['median']:.3f} / {res['noise']['proposed']['median']:.3f}",
          flush=True)
    for b, v in res["signal"].items():
        print(f"  signal {b}: {v['standard']:.1f} / {v['proposed']:.1f} %", flush=True)
    for m, v in res["maps"].items():
        print(f"  {m}: " + " / ".join(f"{x:.2f}" for x in v.values()), flush=True)
