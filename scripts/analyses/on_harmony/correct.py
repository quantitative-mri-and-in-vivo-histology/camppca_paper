"""Stage 2: topup and eddy on every session.

    topup  the first N_B0 b0s of the AP and of the PA series. acqp rows from each series'
           PhaseEncodingDirection, readout time READOUT_S. bet -f 0.2 on the mean corrected b0
    eddy   eddy_cuda11.0 on the Standard MPPCA series (stage 1), so the alignment cannot favour
           Proposed. Only eddy's estimates are used later, in stage 7

Topups run on the CPU, many sessions in parallel. Each session's eddy starts on the GPU as soon
as its topup is done, a few sessions at a time.

Out, per session in data/analyses/on_harmony/<subject>/native/<session>/: topup_*, b0_corrected,
b0_mean, mask_bet.nii.gz, eddy.*. Finished steps are skipped.

    python scripts/analyses/on_harmony/correct.py [--workers 16] [--gpu-workers 3] [subject ...]
"""

import argparse
import json
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np

from camppca_paper.on_harmony import PARTICIPANTS, SOURCE, all_sessions
from camppca_paper.tools import run

B0_MAX = 100
N_B0 = 2
READOUT_S = 0.05
EDDY_FLAGS = ("--repol", "--data_is_shelled", "--dont_peas", "--flm=quadratic", "--slm=none",
              "--initrand=1")


def acqp_row(stem):
    """The acqp row of a series, from the PhaseEncodingDirection in its JSON sidecar."""
    ped = json.loads(stem.with_name(stem.name + ".json").read_text())["PhaseEncodingDirection"]
    v = [0, 0, 0]
    v["ijk".index(ped[0])] = -1 if ped.endswith("-") else 1
    return [*v, READOUT_S]


def topup(session):
    out = session.native
    if (out / "mask_bet.nii.gz").exists():
        return session, None
    t0 = time.time()
    b0s, rows, affine = [], [], None
    for direction in ("AP", "PA"):
        stem = SOURCE / session.dwi(direction)
        img = nib.load(f"{stem}.nii.gz")
        bval = np.loadtxt(f"{stem}.bval").ravel()
        use = np.where(bval < B0_MAX)[0][:N_B0]
        b0s += [np.asarray(img.dataobj[..., int(j)], np.float32) for j in use]
        rows += [acqp_row(stem)] * len(use)
        affine = img.affine if affine is None else affine
    out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.stack(b0s, -1), affine), out / "b0_pairs.nii.gz")
    np.savetxt(out / "acqp_topup.txt", rows, fmt="%g")
    cnf = "b02b0.cnf" if all(n % 2 == 0 for n in b0s[0].shape) else "b02b0_1.cnf"
    run("topup", f"--imain={out / 'b0_pairs.nii.gz'}", f"--datain={out / 'acqp_topup.txt'}",
        f"--config={cnf}", f"--out={out / 'topup'}", f"--iout={out / 'b0_corrected'}",
        log=out / "topup.log")
    corrected = nib.load(out / "b0_corrected.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(corrected.dataobj, np.float32).mean(-1), corrected.affine),
             out / "b0_mean.nii.gz")
    run("bet", out / "b0_mean.nii.gz", out / "b0_brain", "-m", "-f", 0.2)
    (out / "b0_brain_mask.nii.gz").rename(out / "mask_bet.nii.gz")
    return session, f"{session.subject} {session.name}: topup  [{time.time() - t0:.0f}s]"


def eddy(session):
    out = session.native
    if (out / "eddy.eddy_parameters").exists():
        return
    t0 = time.time()
    stem = SOURCE / session.dwi("AP")
    n = nib.load(out / "standard.nii.gz").shape[3]
    np.savetxt(out / "acqp.txt", [acqp_row(stem)], fmt="%g")
    np.savetxt(out / "index.txt", np.ones((1, n), int), fmt="%d")
    run("eddy_cuda11.0", f"--imain={out / 'standard.nii.gz'}", f"--mask={out / 'mask_bet.nii.gz'}",
        f"--acqp={out / 'acqp.txt'}", f"--index={out / 'index.txt'}", f"--bvecs={stem}.bvec",
        f"--bvals={stem}.bval", f"--topup={out / 'topup'}", f"--out={out / 'eddy'}",
        *EDDY_FLAGS, log=out / "eddy.log")
    print(f"  {session.subject} {session.name}: eddy  [{time.time() - t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16, help="topups in parallel")
    ap.add_argument("--gpu-workers", type=int, default=3, help="eddys in parallel")
    ap.add_argument("subjects", nargs="*")
    a = ap.parse_intermixed_args()
    todo = [s for subject in a.subjects or list(PARTICIPANTS) for s in all_sessions(subject)]
    with Pool(a.workers) as cpu, Pool(a.gpu_workers) as gpu:
        jobs = []
        for session, line in cpu.imap_unordered(topup, todo):
            if line:
                print(f"  {line}", flush=True)
            jobs.append(gpu.apply_async(eddy, (session,)))
        for job in jobs:
            job.get()


if __name__ == "__main__":
    main()
