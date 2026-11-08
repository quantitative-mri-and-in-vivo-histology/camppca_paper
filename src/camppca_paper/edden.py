"""The EDDEN 0.9 mm session: four repeat runs of one subject on one scanner, one T1w."""

import nibabel as nib
import numpy as np

from .paths import DATA_ROOT, EDDEN

RUNS = ("run-01", "run-02", "run-03", "run-04")
DWI = EDDEN / "sub-01/ses-0p9mm/dwi"
T1 = EDDEN / "sub-01/ses-0p9mm/anat/sub-01_ses-0p9mm_T1w.nii.gz"
WORK = DATA_ROOT / "analyses/edden"
NORDIC2 = WORK / "nordic2"
# The denoisers' input: the source magnitude, or the real part of the complex data.
PARTS = ("magnitude", "complex")
# NORDIC2 takes the complex data, so it is an arm of the complex set only.
ARMS = {"magnitude": ("raw", "standard", "proposed"), "complex": ("raw", "standard", "proposed", "nordic2")}

# Phase encode along +j for AP, -j for PA. Readout time from the echo spacing and GRAPPA factor.
READOUT_S = 0.07945
ACQP = {"AP": [0, 1, 0, READOUT_S], "PA": [0, -1, 0, READOUT_S]}


def series(direction, run, part="mag"):
    """A source series, e.g. the AP magnitude of run-01."""
    return DWI / f"sub-01_ses-0p9mm_dir-{direction}_{run}_part-{part}_dwi.nii.gz"


def bval(direction="AP"):
    return DWI / f"sub-01_ses-0p9mm_dir-{direction}_dwi.bval"


def bvec(direction="AP"):
    return DWI / f"sub-01_ses-0p9mm_dir-{direction}_dwi.bvec"


def work(part):
    """The pipeline's folder for one input."""
    return WORK / part


def native(part, run):
    """Where the pipeline writes a run's outputs in its own space."""
    return work(part) / "native" / run


def raw(part, run):
    """A run's undenoised input: the AP magnitude, or its real part."""
    return series("AP", run) if part == "magnitude" else native(part, run) / "real.nii.gz"


def nordic2_sigma(run):
    """NORDIC2's noise level of a run, per channel: its g-factor map times its scale, the smallest
    nonzero magnitude of the first volume, over sqrt(2)."""
    first = np.asarray(nib.load(series("AP", run)).dataobj[..., 0], np.float64)
    g = np.asarray(nib.load(NORDIC2 / f"gfactor_{run}_nordic2_.nii.gz").dataobj, np.float64)
    return g * first[first != 0].min() / np.sqrt(2)


def arm(part, run, name):
    """One arm's series of a run in its own space. NORDIC2's is its magnitude output."""
    if name == "raw":
        return raw(part, run)
    if name == "nordic2":
        return NORDIC2 / f"{run}_nordic2_magn.nii.gz"
    return native(part, run) / f"{name}.nii.gz"
