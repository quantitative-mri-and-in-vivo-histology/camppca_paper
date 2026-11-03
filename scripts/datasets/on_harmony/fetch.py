"""Download the ON-Harmony sessions the in vivo analyses use, from OpenNeuro ds004712 2.0.1.

Per session: the AP and PA DWI series with their bval, bvec and JSON sidecar, and the T1w.
Each file is checked against the SHA256 the release records.

Dataset: https://doi.org/10.18112/openneuro.ds004712.v2.0.1

    python scripts/datasets/on_harmony/fetch.py
"""

from camppca_paper.on_harmony import ACCESSION, PARTICIPANTS, SOURCE, VERSION, all_sessions
from camppca_paper.openneuro import fetch


def files():
    out = []
    for subject in PARTICIPANTS:
        for s in all_sessions(subject):
            for direction in ("AP", "PA"):
                stem = s.dwi(direction)
                out += [stem.with_name(stem.name + ext)
                        for ext in (".nii.gz", ".bval", ".bvec", ".json")]
            out.append(s.t1)
    return [str(f) for f in out]


if __name__ == "__main__":
    fetch(ACCESSION, VERSION, SOURCE, files())
