"""Stage 3: what each scan's noise covariance model says (`camppca_paper.prediction`).

    correction  median over the mask of sigma_proposed / sigma_standard - 1, in %
    prediction  on the patches camppca's loop fits on, the model fitted to each (`fit`)
                predicts Standard's noise level from N_DRAW draws. Error: RMS over patches of
                the predicted over the read level, minus 1, in %
    components  the fitted level off the patch-mean mode over Standard's reading, RMS over the
                patches, minus 1, in %: the full model, the level only (C = I) and the
                correlation only (D flat)
    transfer    pure noise X = C^(1/2) G, N_DRAW draws at the scan's patch and volume count.
                Standard's noise level of X whitened with the other subject's C, with the mean
                table of the other configurations, and not at all, each over that of G. The
                mean is over every other configuration whose table reaches the scan's lag,
                cropped to it, with the phase encode on the first axis
    rho         the correlation table and the phase-encode axis, for the profiles

Out: data/results/panel/noise.json, one entry per scan.

    python scripts/analyses/panel/noise.py
"""

import json
from multiprocessing import Pool

import camppca
import nibabel as nib
import numpy as np
from threadpoolctl import threadpool_limits

from camppca_paper.panel import SCANS, load, pe_axis
from camppca_paper.paths import RESULTS
from camppca_paper.prediction import fit, grid_patches, perp2, predicted, root, sigma

N_DRAW = 200
SEED = 0
WORKERS = 16
OUT = RESULTS / "panel/noise.json"


def model(scan):
    z = np.load(scan.work / "noise_model.npz")
    mask = np.asarray(nib.load(scan.work / "brain_mask_denoising.nii.gz").dataobj) > 0
    return camppca.NoiseModel(rho=z["rho"], d=z["d"], mask=mask, lag=int(z["lag"])), tuple(z["patch"])


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def per_scan(scan):
    m, patch = model(scan)
    data, *_ = load(scan)
    idx = grid_patches(m.mask, patch)
    X = data[idx[..., 0], idx[..., 1], idx[..., 2]]
    d = m.d[idx[..., 0], idx[..., 1], idx[..., 2]]
    del data
    C = m.patch_correlation(patch)
    eye = np.eye(len(C))
    rng = np.random.default_rng(SEED)
    read, pred, comp = [], [], {"full": [], "level": [], "correlation": []}
    for Xb, db in zip(X, d, strict=True):
        read.append(sigma(Xb[None])[0][0])
        S, kept = fit(Xb, C, db)
        pred.append(predicted(S, kept, rng.standard_normal((N_DRAW, *Xb.shape))))
        comp["full"].append(perp2(S))
        comp["level"].append(perp2(fit(Xb, eye, db)[0]))
        comp["correlation"].append(perp2(fit(Xb, C, np.ones_like(db))[0]))
    read = rms(read)
    comp = {arm: 100 * (rms(np.sqrt(v)) / read - 1) for arm, v in comp.items()}
    sig = {a: np.asarray(nib.load(scan.work / f"sigma_{a}.nii.gz").dataobj) for a in ("standard", "proposed")}
    ok = m.mask & (sig["standard"] > 0) & (sig["proposed"] > 0)
    out = dict(config=scan.config, sub=scan.sub, n_volumes=int(X.shape[2]), patch=list(map(int, patch)),
               n_patches=len(X), pe_axis=pe_axis(scan), rho=m.rho.tolist(),
               correction=float(100 * (np.median(sig["proposed"][ok] / sig["standard"][ok]) - 1)),
               prediction=100 * (rms(pred) / read - 1), components=comp)
    print(f"  {scan.name}: {len(X)} patches, correction {out['correction']:.1f} %, prediction "
          f"{out['prediction']:.2f} %, level {comp['level']:.1f}, correlation {comp['correlation']:.1f}, "
          f"full {comp['full']:.1f}", flush=True)
    return scan.name, out


def in_pe_frame(rho, pe):
    return np.swapaxes(rho, 0, 1) if pe == 1 else rho


def crop(rho, lag):
    L = rho.shape[0] // 2
    return rho[L - lag:L + lag + 1, L - lag:L + lag + 1, :lag + 1]


def transfer(task):
    name, res = task
    r = res[name]
    lag, patch, N = r["patch"][0] - 1, r["patch"], r["n_volumes"]

    def C(rho):
        return camppca.NoiseModel(rho=np.asarray(rho), d=np.ones(1), mask=np.ones(1, bool),
                                  lag=lag).patch_correlation(patch)

    other = [x for k, x in res.items() if x["config"] == r["config"] and k != name][0]
    tables = [crop(in_pe_frame(np.asarray(x["rho"]), x["pe_axis"]), lag) for x in res.values()
              if x["config"] != r["config"] and np.asarray(x["rho"]).shape[0] // 2 >= lag]
    mean = in_pe_frame(np.mean(tables, axis=0), r["pe_axis"])
    G = np.random.default_rng(SEED).standard_normal((N_DRAW, int(np.prod(patch)), N))
    ref = np.median(sigma(G)[0])
    X = np.matmul(root(C(r["rho"])), G)
    out = {"none": float(np.median(sigma(X)[0]) / ref)}
    for arm, rho in (("other_subject", other["rho"]), ("mean", mean)):
        out[arm] = float(np.median(sigma(np.matmul(root(C(rho), -0.5), X))[0]) / ref)
    print(f"  {name}: transfer other subject {out['other_subject']:.3f}, mean {out['mean']:.3f} "
          f"({len(tables)} tables), none {out['none']:.3f}", flush=True)
    return name, out | {"n_mean": len(tables)}


def main():
    with Pool(WORKERS, initializer=threadpool_limits, initargs=(1,)) as pool:
        res = dict(pool.imap_unordered(per_scan, SCANS))
        res = {s.name: res[s.name] for s in SCANS}
        for name, t in pool.imap(transfer, [(s.name, res) for s in SCANS]):
            res[name]["transfer"] = t
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
