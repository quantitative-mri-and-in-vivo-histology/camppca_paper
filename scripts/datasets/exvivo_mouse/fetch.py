"""Download the ex vivo mouse brain from Zenodo record 10200469.

Gets the two-average series, the twenty-average series, the mask, and the bval and bvec files.
Each file is checked against the MD5 the record lists.

Dataset: https://doi.org/10.5281/zenodo.10200469
Paper:   Henriques et al., Imaging Neuroscience, 2023. https://doi.org/10.1162/imag_a_00049

    python scripts/datasets/exvivo_mouse/fetch.py
"""

import subprocess

from camppca_paper.openneuro import checksum
from camppca_paper.paths import EXVIVO

URL = "https://zenodo.org/records/10200469/files/{}?download=1"
MD5 = {"MB_exp2.nii": "643ea21b8acf715047785d19422ecbf1",
       "MB_exp2_20averages.nii": "48219aa96390effa6bb00f2bff6131cc",
       "MB_exp2_mask.nii": "011dd2e8d3582d98d264df02b5ca59bf",
       "MB.bval": "098038aa4c05b51bf9e84cd9c05be42f",
       "MB.bvec": "866cdb66cff4e69a645de497e06dcfb7"}


if __name__ == "__main__":
    for name, md5 in MD5.items():
        path = EXVIVO / name
        if not path.exists():
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--create-dirs", "-o", str(path),
                            URL.format(name)], check=True)
        if checksum(path, "md5") != md5:
            raise SystemExit(f"  md5 mismatch: {path}")
        print(f"  ok  {path}")
