"""Where the repo and the data live. `DATA_PATH` sets the data root and defaults to the repo's
`data/`.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_PATH") or ROOT / "data")

# OpenNeuro releases, one folder per accession.
OPENNEURO = DATA_ROOT / "source/openneuro"
# EDDEN, ds004666. The phantom and the 0.9 mm runs come from it.
EDDEN = OPENNEURO / "ds004666"

# Subject phantom, built by scripts/datasets/subject_phantom/.
DATA = DATA_ROOT / "raw/subject_phantom"
DWI = DATA / "phantom_sh4.nii.gz"
BVAL = DATA / "phantom_sh4.bval"
BVEC = DATA / "phantom_sh4.bvec"
WORK = DATA_ROOT / "analyses/subject_phantom"

# eTPM tissue priors, from scripts/datasets/segmentation_priors/.
ETPM = DATA_ROOT / "source/segmentation_priors/eTPM.nii"
