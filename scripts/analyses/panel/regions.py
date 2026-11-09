"""Stage 4: does a correlation pattern transfer between brain regions of one scan?

    regions   brain_mask_denoising cut into thirds along the phase encode, the readout and the
              slice axis, at the tertiles of the mask voxels' positions on that axis
    patterns  camppca's noise model estimated on each region alone, at the scan's patch
    transfer  pure noise X = C_A^(1/2) G, N_DRAW draws at the scan's patch and volume count, with
              C_A a region's pattern. Standard's noise level of X whitened with each other region's
              pattern of the same cut, with the whole-brain pattern (noise_model.npz), and not at
              all, each over that of G

Out: regions.npz per scan in data/analyses/panel/<config>/<subject>/ (each region's rho, lag and
voxel count, and the volume count), skipped when present; data/results/panel/regions.json, one row per recipient region
and source.

    python scripts/analyses/panel/regions.py
"""

import json
import time
from multiprocessing import Pool

import camppca
import nibabel as nib
import numpy as np
from threadpoolctl import threadpool_limits

from camppca_paper.panel import SCANS, load
from camppca_paper.paths import RESULTS
from camppca_paper.prediction import root, sigma

AXES = ("phase_encode", "readout", "slice")
THIRDS = ("low", "middle", "high")
N_DRAW = 200
SEED = 0
WORKERS = 16
OUT = RESULTS / "panel/regions.json"


def estimate(scan):
    out = scan.work / "regions.npz"
    if out.exists():
        return
    t0 = time.time()
    data, bval, _, _, pe = load(scan)
    mask = np.asarray(nib.load(scan.work / "brain_mask_denoising.nii.gz").dataobj) > 0
    patch = tuple(int(p) for p in np.load(scan.work / "noise_model.npz")["patch"])
    keep = {}
    for axis, ax in zip(AXES, (pe, 1 - pe, 2), strict=True):
        cuts = np.percentile(np.nonzero(mask)[ax], (100 / 3, 200 / 3))
        coord = np.indices(mask.shape)[ax]
        for third, (lo, hi) in zip(THIRDS, ((-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], mask.shape[ax])),
                                   strict=True):
            region = mask & (coord > lo) & (coord <= hi)
            m = camppca.estimate_noise_model(data, bvals=bval, mask=region, patch=patch)
            keep |= {f"rho_{axis}_{third}": m.rho, f"n_{axis}_{third}": int(region.sum()),
                     f"fallback_{axis}_{third}": bool(m.provenance.get("fallback", False))}
    np.savez(out, lag=m.lag, patch=np.array(patch), n_volumes=data.shape[3], **keep)
    print(f"  {scan.name}: patch {patch}  [{time.time() - t0:.0f}s]", flush=True)


def transfer(scan):
    z = np.load(scan.work / "regions.npz")
    lag, patch = int(z["lag"]), tuple(int(p) for p in z["patch"])
    whole = np.load(scan.work / "noise_model.npz")
    N = int(z["n_volumes"])

    def C(rho, lag):
        return camppca.NoiseModel(rho=np.asarray(rho), d=np.ones(1), mask=np.ones(1, bool),
                                  lag=lag).patch_correlation(patch)

    G = np.random.default_rng(SEED).standard_normal((N_DRAW, int(np.prod(patch)), N))
    ref = np.median(sigma(G)[0])
    rows = []
    for axis in AXES:
        for third in THIRDS:
            X = np.matmul(root(C(z[f"rho_{axis}_{third}"], lag)), G)
            sources = [("region", other, C(z[f"rho_{axis}_{other}"], lag)) for other in THIRDS if other != third]
            sources += [("whole_brain", None, C(whole["rho"], int(whole["lag"]))), ("none", None, None)]
            for source, donor, Cb in sources:
                Y = X if Cb is None else np.matmul(root(Cb, -0.5), X)
                rows.append(dict(scan=scan.name, config=scan.config, axis=axis, region=third, source=source,
                                 donor=donor, value=float(np.median(sigma(Y)[0]) / ref),
                                 fallback=bool(z[f"fallback_{axis}_{third}"])))
    print(f"  {scan.name}: region " + " ".join(f"{np.median([r['value'] for r in rows if r['source'] == s]):.3f}"
                                              for s in ("region", "whole_brain", "none")), flush=True)
    return rows


def main():
    for scan in SCANS:
        estimate(scan)
    with Pool(WORKERS, initializer=threadpool_limits, initargs=(1,)) as pool:
        rows = [r for out in pool.imap(transfer, SCANS) for r in out]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    for source in ("region", "whole_brain", "none"):
        v = np.array([r["value"] for r in rows if r["source"] == source])
        print(f"  {source:12s} n {len(v):4d}  median {np.median(v):.3f}  range {v.min():.3f}-{v.max():.3f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
