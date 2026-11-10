"""Stage 5: how many volumes and how many patches a correlation pattern needs.

    volumes   camppca's noise model estimated on N_VOLUMES volumes, N_SUBSET["volumes"] random
              subsets each, every shell keeping its share, at the scan's patch
    patches   the loop alone (`camppca.noise_model.loop.fit_rho`, from white noise) on
              N_PATCHES patches, N_SUBSET["patches"] random subsets each of the tiling the loop
              fits on, with the full model's level map. A fit costs about as much as the full
              loop, hence one subset
    price     pure noise X = C^(1/2) G, N_DRAW draws at the scan's patch and volume count, with
              C the whole series' pattern (noise_model.npz). Standard's noise level of X
              whitened with each subset's pattern, and not at all, over that of G

Out: convergence.npz per scan in data/analyses/panel/<config>/<subject>/ (each subset's rho and
fallback, the volume and tile count), skipped when present; data/results/panel/convergence.json,
one row per scan, kind, size and subset.

    python scripts/analyses/panel/convergence.py
"""

import json
import time
import warnings
from multiprocessing import Pool

import camppca
import nibabel as nib
import numpy as np
from camppca.noise_model import _good_level, _white_rho
from camppca.noise_model.loop import _sample_centers, fit_rho
from threadpoolctl import threadpool_limits

from camppca_paper.panel import SCANS, load
from camppca_paper.paths import RESULTS
from camppca_paper.prediction import root, sigma

N_VOLUMES = (5, 10, 20, 40, 60)
N_PATCHES = (5, 10, 25, 50, 100)
N_SUBSET = {"volumes": 3, "patches": 1}
N_DRAW = 200
SEED = 0
WORKERS = 16
OUT = RESULTS / "panel/convergence.json"


def volume_subset(bval, n, rng):
    """n volumes drawn at random, every shell (b rounded to 100) keeping its share."""
    shell = np.round(bval, -2)
    values, counts = np.unique(shell, return_counts=True)
    want = n * counts / counts.sum()
    take = np.floor(want).astype(int)
    take[np.argsort(take - want)[: n - take.sum()]] += 1
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(shell == v), k, replace=False)
                                   for v, k in zip(values, take, strict=True)]))


def estimate(scan):
    out = scan.work / "convergence.npz"
    if out.exists():
        return
    t0 = time.time()
    data, bval, *_ = load(scan)
    data = np.ascontiguousarray(data, np.float32)
    mask = np.asarray(nib.load(scan.work / "brain_mask_denoising.nii.gz").dataobj) > 0
    full = np.load(scan.work / "noise_model.npz")
    patch, lag, d = tuple(int(p) for p in full["patch"]), int(full["lag"]), full["d"]
    rng = np.random.default_rng(SEED)
    keep = {}
    for n in N_VOLUMES:
        for r in range(N_SUBSET["volumes"]):
            idx = volume_subset(bval, n, rng)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = camppca.estimate_noise_model(data[..., idx], bvals=bval[idx], mask=mask, patch=patch)
            keep[f"rho_volumes_{n}_{r}"] = m.rho
            keep[f"fallback_volumes_{n}_{r}"] = bool(m.provenance.get("fallback"))
    tiles, _ = _sample_centers(_good_level(d, mask), np.asarray(patch), min_centers=0)
    for n in N_PATCHES:
        for r in range(N_SUBSET["patches"]):
            pick = np.sort(rng.choice(tiles[0].size, min(n, tiles[0].size), replace=False))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, info = fit_rho(data, mask, d, _white_rho(full["rho"].shape), np.asarray(patch), lag,
                                    centers=tuple(c[pick] for c in tiles))
            keep[f"rho_patches_{n}_{r}"] = rho
            keep[f"fallback_patches_{n}_{r}"] = bool(info["fallback"])
    np.savez(out, n_volumes=data.shape[3], n_tiles=tiles[0].size, **keep)
    print(f"  {scan.name}: patch {patch}, {tiles[0].size} tiles  [{time.time() - t0:.0f}s]", flush=True)


def price(scan):
    z = np.load(scan.work / "convergence.npz")
    full = np.load(scan.work / "noise_model.npz")
    patch, lag, N = tuple(int(p) for p in full["patch"]), int(full["lag"]), int(z["n_volumes"])

    def C(rho):
        return camppca.NoiseModel(rho=np.asarray(rho), d=np.ones(1), mask=np.ones(1, bool),
                                  lag=lag).patch_correlation(patch)

    G = np.random.default_rng(SEED).standard_normal((N_DRAW, int(np.prod(patch)), N))
    ref = np.median(sigma(G)[0])
    X = np.matmul(root(C(full["rho"])), G)
    rows = [dict(scan=scan.name, config=scan.config, kind="none", size=None, subset=None,
                 value=float(np.median(sigma(X)[0]) / ref), fallback=False)]
    for kind, sizes in (("volumes", N_VOLUMES), ("patches", N_PATCHES)):
        for n in sizes:
            for r in range(N_SUBSET[kind]):
                try:
                    Y = np.matmul(root(C(z[f"rho_{kind}_{n}_{r}"]), -0.5), X)
                    value = float(np.median(sigma(Y)[0]) / ref)
                except ValueError:                 # a table camppca refuses to whiten with
                    value = float("nan")
                rows.append(dict(scan=scan.name, config=scan.config, kind=kind, size=n, subset=r, value=value,
                                 fallback=bool(z[f"fallback_{kind}_{n}_{r}"]),
                                 n_tiles=int(z["n_tiles"]), n_volumes=N))
    print(f"  {scan.name}: " + "  ".join(
        f"{k} " + " ".join(f"{np.nanmedian([r['value'] for r in rows if r['kind'] == k and r['size'] == n]):.3f}"
                           for n in s) for k, s in (("volumes", N_VOLUMES), ("patches", N_PATCHES))), flush=True)
    return rows


def main():
    for scan in SCANS:
        estimate(scan)
    with Pool(WORKERS, initializer=threadpool_limits, initargs=(1,)) as pool:
        rows = [r for out in pool.imap(price, SCANS) for r in out]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    for kind, sizes in (("volumes", N_VOLUMES), ("patches", N_PATCHES)):
        for n in sizes:
            v = np.array([r["value"] for r in rows if r["kind"] == kind and r["size"] == n])
            f = sum(r["fallback"] for r in rows if r["kind"] == kind and r["size"] == n)
            print(f"  {kind:8s} {n:4d}  median {np.nanmedian(v):.3f}  range {np.nanmin(v):.3f}-{np.nanmax(v):.3f}"
                  f"  fallbacks {f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
