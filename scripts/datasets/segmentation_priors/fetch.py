"""Download the eTPM tissue probability maps from the hMRI toolbox, release v1.0.0.

The file is checked against its SHA256.

Toolbox: https://github.com/hMRI-group/hMRI-toolbox
Paper:   Lorio et al., Human Brain Mapping, 2016. https://doi.org/10.1002/hbm.23137

    python scripts/datasets/segmentation_priors/fetch.py
"""

import subprocess

from camppca_paper.openneuro import sha256
from camppca_paper.paths import ETPM

URL = "https://raw.githubusercontent.com/hMRI-group/hMRI-toolbox/v1.0.0/etpm/eTPM.nii"
SHA256 = "7c618c9612002861ab31d8d196d1c4e99e0fb49d67db47490712439c1d48fef2"


if __name__ == "__main__":
    if not ETPM.exists():
        subprocess.run(["curl", "-fsSL", "--retry", "3", "--create-dirs", "-o", str(ETPM), URL],
                       check=True)
    if sha256(ETPM) != SHA256:
        raise SystemExit(f"  sha256 mismatch: {ETPM}")
    print(f"  ok  {ETPM}")
