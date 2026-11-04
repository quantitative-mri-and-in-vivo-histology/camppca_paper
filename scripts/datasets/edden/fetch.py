"""Download the 0.9 mm session of EDDEN from OpenNeuro ds004666 1.0.8.

Gets the four AP runs, magnitude and phase, the four PA runs, magnitude only, their bval and
bvec files, and the T1w. Each file is checked against the SHA256 the release records.

Dataset: https://doi.org/10.18112/openneuro.ds004666.v1.0.8
Paper:   Manzano-Patron et al., Imaging Neuroscience, 2024. https://doi.org/10.1162/imag_a_00060

    python scripts/datasets/edden/fetch.py
"""

from camppca_paper.openneuro import fetch
from camppca_paper.paths import EDDEN

ACCESSION = "ds004666"
VERSION = "1.0.8"
SES = "sub-01/ses-0p9mm"
RUNS = ("01", "02", "03", "04")


def files():
    dwi = f"{SES}/dwi/sub-01_ses-0p9mm"
    out = [f"{dwi}_dir-{d}_dwi.{ext}" for d in ("AP", "PA") for ext in ("bval", "bvec")]
    out += [f"{dwi}_dir-AP_run-{r}_part-{p}_dwi.nii.gz" for r in RUNS for p in ("mag", "phase")]
    out += [f"{dwi}_dir-PA_run-{r}_part-mag_dwi.nii.gz" for r in RUNS]
    return out + [f"{SES}/anat/sub-01_ses-0p9mm_T1w.nii.gz"]


if __name__ == "__main__":
    fetch(ACCESSION, VERSION, EDDEN, files())
