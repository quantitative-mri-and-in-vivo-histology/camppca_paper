"""Download the phantom source from OpenNeuro ds004666 1.0.8.

Gets the complex-averaged DWI of the 2 mm session, its brain mask, and the bval and bvec files.
The release is cloned shallow for its file list. Each file is then downloaded and checked
against the SHA256 the release records.

Dataset: https://doi.org/10.18112/openneuro.ds004666.v1.0.8
Paper:   Manzano-Patron et al., Imaging Neuroscience, 2024. https://doi.org/10.1162/imag_a_00060

    python scripts/datasets/subject_phantom/fetch.py
"""

import hashlib
import re
import subprocess

from camppca_paper.paths import DATA_ROOT

ACCESSION = "ds004666"
VERSION = "1.0.8"
ROOT = DATA_ROOT / "source/subject_phantom" / ACCESSION

AVERAGE = "derivatives/ses-2mm/dwi/AVG_complex/sub-01_ses-2mm_dir-AP_AVG_complex_dwi_processed.nii.gz"
BRAIN_MASK = "derivatives/ses-2mm/masks/nodif_brain_mask.nii.gz"
BVAL_FILE = "sub-01/ses-2mm/dwi/sub-01_ses-2mm_dir-AP_dwi.bval"
BVEC_FILE = "sub-01/ses-2mm/dwi/sub-01_ses-2mm_dir-AP_dwi.bvec"

# git-annex key in a pointer's symlink target: size and SHA256 of the content.
KEY = re.compile(r"SHA256E-s(\d+)--([0-9a-f]{64})")


def annexed(rel):
    """`(path, size, sha256)` of an annexed file, or None if git stores it directly."""
    link = ROOT / rel
    if not link.is_symlink():
        return None
    size, sha = KEY.search(str(link.readlink())).groups()
    return (link.parent / link.readlink()).resolve(), int(size), sha


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if not (ROOT / ".git").exists():
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet",
                        "--depth", "1", "--branch", VERSION,
                        f"https://github.com/OpenNeuroDatasets/{ACCESSION}.git", str(ROOT)],
                       check=True)

    for rel in (AVERAGE, BRAIN_MASK, BVAL_FILE, BVEC_FILE):
        entry = annexed(rel)
        if entry is None:
            print(f"  ok  {rel}")
            continue
        dest, size, sha = entry
        if not (dest.exists() and dest.stat().st_size == size):
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--create-dirs", "-o", str(dest),
                            f"https://s3.amazonaws.com/openneuro.org/{ACCESSION}/{rel}"],
                           check=True)
        if sha256(dest) != sha:
            raise SystemExit(f"  sha256 mismatch: {rel}")
        print(f"  ok  {rel}")


if __name__ == "__main__":
    main()
