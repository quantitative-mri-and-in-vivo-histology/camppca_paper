"""Where the repo and the data live. `DATA_PATH` sets the data root and defaults to the repo's
`data/`.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_PATH") or ROOT / "data")

# Subject phantom, built by scripts/datasets/subject_phantom/.
DATA = DATA_ROOT / "raw/subject_phantom"
DWI = DATA / "phantom_sh4.nii.gz"
BVAL = DATA / "phantom_sh4.bval"
BVEC = DATA / "phantom_sh4.bvec"
WORK = DATA_ROOT / "analyses/subject_phantom"
