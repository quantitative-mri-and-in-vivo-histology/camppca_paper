"""The EDDEN 0.9 mm session: four repeat runs of one subject on one scanner, one T1w."""

from .paths import DATA_ROOT, EDDEN

RUNS = ("run-01", "run-02", "run-03", "run-04")
DWI = EDDEN / "sub-01/ses-0p9mm/dwi"
T1 = EDDEN / "sub-01/ses-0p9mm/anat/sub-01_ses-0p9mm_T1w.nii.gz"
WORK = DATA_ROOT / "analyses/edden"
NORDIC2 = WORK / "nordic2"

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


def native(run):
    """Where the pipeline writes a run's outputs in its own space."""
    return WORK / "native" / run
