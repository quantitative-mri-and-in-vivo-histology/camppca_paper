"""Stage 4: brain mask and tissue segmentation of the session's T1w
(`camppca_paper.t1.segment_t1`).

Out: data/analyses/edden/t1/ (t1_n4.nii.gz, brain_mask.nii.gz, t1_brain.nii.gz, seg.nii.gz,
seg_post{1,2,3}.nii.gz). Skipped when done.

    python scripts/analyses/edden/t1.py
"""

from camppca_paper.edden import T1, WORK
from camppca_paper.t1 import segment_t1

if __name__ == "__main__":
    out = WORK / "t1"
    if not (out / "seg_post2.nii.gz").exists():
        segment_t1(T1, out, threads=16)
        print("  T1 segmentation", flush=True)
