"""T1w brain mask and tissue segmentation with eTPM priors (ANTs, FSL).

    n4        N4BiasFieldCorrection, shrink factor 4
    brain     antsBrainExtraction.sh with FSL's MNI152_T1_1mm head, brain mask and dilated mask,
              started from a rigid and affine MI registration of a rough bet brain (-f BET_F) to
              MNI152's brain
    priors    the eTPM tissue maps (GM, WM, CSF) through the same MNI registration, divided by
              their sum inside the brain mask
    segment   Atropos with those priors (weight PRIOR_WEIGHT), MRF [0.1,1x1x1], 5 iterations,
              inside the brain mask. Classes 1 GM, 2 WM, 3 CSF
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np

from .paths import ETPM
from .tools import run

FSL_STANDARD = Path(os.environ.get("FSLDIR", "")) / "data/standard"
MNI_HEAD = FSL_STANDARD / "MNI152_T1_1mm.nii.gz"
MNI_BRAIN = FSL_STANDARD / "MNI152_T1_1mm_brain.nii.gz"
MNI_MASK = FSL_STANDARD / "MNI152_T1_1mm_brain_mask.nii.gz"
MNI_MASK_DIL = FSL_STANDARD / "MNI152_T1_1mm_brain_mask_dil.nii.gz"
BET_F = 0.5
PRIOR_WEIGHT = 0.25
CLASSES = ("GM", "WM", "CSF")


def segment_t1(t1, out, threads=8):
    """N4, brain extraction and eTPM-prior segmentation of one T1w.

    Args:
        t1: The T1w image.
        out: Output folder.
        threads: ITK threads.

    Writes t1_n4.nii.gz, brain_mask.nii.gz, t1_brain.nii.gz, seg.nii.gz and
    seg_post{1,2,3}.nii.gz (GM, WM, CSF), with the intermediate steps in brain/ and priors/.
    """
    env = {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads)}
    abe, pri = out / "brain", out / "priors"
    abe.mkdir(parents=True, exist_ok=True)
    pri.mkdir(exist_ok=True)

    img = out / "t1_n4.nii.gz"
    run("N4BiasFieldCorrection", "-d", 3, "-i", t1, "-s", 4, "-o", img, env=env)

    run("bet", img, abe / "start_bet", "-f", BET_F)
    moving = abe / "start_bet.nii.gz"
    metric = f"MI[{MNI_BRAIN},{moving},1,32,Regular,0.25]"
    stages = ("-c", "[1000x500x250x0,1e-6,10]", "-f", "6x4x2x1", "-s", "4x2x1x0")
    run("antsRegistration", "-d", 3, "--float", 1, "-u", 0, "-w", "[0.005,0.995]", "-z", 1,
        "-r", f"[{MNI_BRAIN},{moving},1]", "-t", "Rigid[0.1]", "-m", metric, *stages,
        "-t", "Affine[0.1]", "-m", metric, *stages, "-o", abe / "start_", env=env)
    run("antsBrainExtraction.sh", "-d", 3, "-a", img, "-e", MNI_HEAD, "-m", MNI_MASK,
        "-f", MNI_MASK_DIL, "-r", abe / "start_0GenericAffine.mat", "-o", f"{abe}/", "-k", 1,
        "-q", 1, "-u", 0, "-s", "nii.gz", env=env, log=abe / "abe.log")
    if not (abe / "BrainExtractionMask.nii.gz").exists():   # the script exits 0 on some failures
        raise RuntimeError(f"antsBrainExtraction.sh wrote no mask, see {abe / 'abe.log'}")

    t1_img = nib.load(img)
    brain = np.asarray(nib.load(abe / "BrainExtractionMask.nii.gz").dataobj) > 0
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), t1_img.affine), out / "brain_mask.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(t1_img.dataobj, np.float32) * brain, t1_img.affine),
             out / "t1_brain.nii.gz")

    etpm = nib.load(ETPM)
    priors = []
    for i, name in enumerate(CLASSES):
        src, dst = pri / f"etpm_{name}.nii.gz", pri / f"prior{i + 1}.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(etpm.dataobj[..., i], np.float32), etpm.affine), src)
        # MNI is the fixed image of the brain extraction's registration, so use its inverse
        run("antsApplyTransforms", "-d", 3, "-i", src, "-r", img, "-n", "Linear", "-o", dst,
            "-t", f"[{abe}/BrainExtractionPrior0GenericAffine.mat,1]",
            "-t", abe / "BrainExtractionPrior1InverseWarp.nii.gz", env=env)
        priors.append(np.asarray(nib.load(dst).dataobj, np.float32))
    p = np.stack(priors)
    total = p.sum(0)
    p = np.where(brain & (total > 0), p / np.maximum(total, 1e-6), 0)
    for i in range(len(CLASSES)):
        nib.save(nib.Nifti1Image(p[i].astype(np.float32), t1_img.affine), pri / f"prior{i + 1}.nii.gz")
    run("Atropos", "-d", 3, "-a", img, "-x", out / "brain_mask.nii.gz",
        "-i", f"PriorProbabilityImages[3,{pri}/prior%d.nii.gz,{PRIOR_WEIGHT}]", "-c", "[5,0]",
        "-m", "[0.1,1x1x1]", "-o", f"[{out}/seg.nii.gz,{out}/seg_post%d.nii.gz]", env=env)
