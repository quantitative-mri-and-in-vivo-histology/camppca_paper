"""Stages 5 and 6: the diffusion template of the four runs, and the T1 registered to it.

Stage 5 builds a two-channel template from the runs' DKI S0 and FA (stage 3), weights 1 : 1,
and registers every run to it once more. Stage 6 registers the T1 (stage 4) to the template and
carries its brain mask and white-matter posterior onto the template grid.

Out: data/analyses/edden/dwi/ (T_template{0,1}.nii.gz, to_template/, t1/). Skipped when done.

    python scripts/analyses/edden/templates.py
"""

from camppca_paper.edden import RUNS, WORK, native
from camppca_paper.templates import dwi_template, t1_to_template

if __name__ == "__main__":
    out = WORK / "dwi"
    if not (out / "t1" / "wm_prob.nii.gz").exists():
        inputs = {r: (native(r) / "dki_S0.nii.gz", native(r) / "dki_FA.nii.gz") for r in RUNS}
        r = dwi_template(out, inputs, (1.0, 1.0), threads=16)
        t1 = WORK / "t1"
        t1_to_template(out, t1 / "t1_brain.nii.gz", t1 / "brain_mask.nii.gz",
                       t1 / "seg_post2.nii.gz", threads=16)
        print(f"  templates, lowest warped-vs-template r {min(min(v) for v in r.values()):.3f}",
              flush=True)
