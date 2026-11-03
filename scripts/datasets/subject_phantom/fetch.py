"""Download the phantom source from OpenNeuro ds004666 1.0.8.

Gets the complex-averaged DWI of the 2 mm session, its brain mask, and the bval and bvec files.
Each file is checked against the SHA256 the release records.

Dataset: https://doi.org/10.18112/openneuro.ds004666.v1.0.8
Paper:   Manzano-Patron et al., Imaging Neuroscience, 2024. https://doi.org/10.1162/imag_a_00060

    python scripts/datasets/subject_phantom/fetch.py
"""

from camppca_paper.openneuro import fetch
from camppca_paper.paths import DATA_ROOT

ACCESSION = "ds004666"
VERSION = "1.0.8"
ROOT = DATA_ROOT / "source/subject_phantom" / ACCESSION

AVERAGE = "derivatives/ses-2mm/dwi/AVG_complex/sub-01_ses-2mm_dir-AP_AVG_complex_dwi_processed.nii.gz"
BRAIN_MASK = "derivatives/ses-2mm/masks/nodif_brain_mask.nii.gz"
BVAL_FILE = "sub-01/ses-2mm/dwi/sub-01_ses-2mm_dir-AP_dwi.bval"
BVEC_FILE = "sub-01/ses-2mm/dwi/sub-01_ses-2mm_dir-AP_dwi.bvec"


if __name__ == "__main__":
    fetch(ACCESSION, VERSION, ROOT, [AVERAGE, BRAIN_MASK, BVAL_FILE, BVEC_FILE])
