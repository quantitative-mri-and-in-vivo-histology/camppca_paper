"""Diffusion templates from per-session S0 and FA, and a T1 registered to them (ANTs)."""

import nibabel as nib
import numpy as np

from .tools import run

ITERATIONS = 4
PAD = 8
CHANNELS = ("S0", "FA")
LINEAR = ("-c", "[1000x500x250x0,1e-6,10]", "-f", "6x4x2x1", "-s", "4x2x1x0")


def dwi_template(out, inputs, weights, threads=8):
    """Build a two-channel template and register every session to it once more.

    Args:
        out: Output folder.
        inputs: `{name: (S0 path, FA path)}`, one entry per session.
        weights: Weight of the S0 and the FA channel.
        threads: ITK threads.

    Returns:
        `{name: (r_S0, r_FA)}`, each warped image's correlation with the template inside its brain.

    Writes T_template0 (S0) and T_template1 (FA), and per session to_template/<name>_
    {0GenericAffine.mat, 1Warp.nii.gz, S0_Warped.nii.gz, FA_Warped.nii.gz}.
    """
    env = {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads)}
    out.mkdir(parents=True, exist_ok=True)
    tpl = [out / f"T_template{i}.nii.gz" for i in range(len(CHANNELS))]
    # The template script rewrites T_template* after every iteration, so a marker says it finished.
    if not (out / "template_done").exists():
        (out / "inputs.csv").write_text("".join(",".join(map(str, p)) + "\n" for p in inputs.values()))
        for i, c in enumerate(CHANNELS):
            start = out / f"start_{c}.nii.gz"
            run("AverageImages", 3, start, 1, *[p[i] for p in inputs.values()], env=env)
            run("ImageMath", 3, start, "PadImage", start, PAD, env=env)
        run("antsMultivariateTemplateConstruction2.sh", "-d", 3, "-o", f"{out}/T_",
            "-k", len(CHANNELS), "-w", "x".join(f"{w:g}" for w in weights), "-i", ITERATIONS,
            "-t", "SyN", "-m", "CC", "-m", "CC", "-r", 1, "-n", 0, "-c", 0,
            *[a for c in CHANNELS for a in ("-z", out / f"start_{c}.nii.gz")], out / "inputs.csv",
            env=env, log=out / "template.log", cwd=out)
        (out / "template_done").touch()

    dest = out / "to_template"
    dest.mkdir(exist_ok=True)
    for name, imgs in inputs.items():
        prefix = dest / f"{name}_"
        if (dest / f"{name}_FA_Warped.nii.gz").exists():
            continue
        mi = [a for t, m, w in zip(tpl, imgs, weights, strict=True)
              for a in ("-m", f"MI[{t},{m},{w},32,Regular,0.25]")]
        cc = [a for t, m, w in zip(tpl, imgs, weights, strict=True) for a in ("-m", f"CC[{t},{m},{w},4]")]
        run("antsRegistration", "-d", 3, "--float", 1, "-u", 0, "-w", "[0.01,0.99]", "-z", 1,
            "-r", f"[{tpl[0]},{imgs[0]},1]", "-t", "Rigid[0.1]", *mi, *LINEAR,
            "-t", "Affine[0.1]", *mi, *LINEAR,
            "-t", "SyN[0.2,3,0]", *cc, "-c", "[100x100x70x20,1e-9,10]", "-f", "6x4x2x1",
            "-s", "3x2x1x0", "-o", prefix, env=env)
        for c, img in zip(CHANNELS, imgs, strict=True):
            run("antsApplyTransforms", "-d", 3, "-i", img, "-r", tpl[0],
                "-t", f"{prefix}1Warp.nii.gz", "-t", f"{prefix}0GenericAffine.mat", "-n", "Linear",
                "-o", f"{prefix}{c}_Warped.nii.gz", env=env)

    t = [np.asarray(nib.load(p).dataobj, np.float64) for p in tpl]
    brain = t[0] > 0.1 * np.percentile(t[0][t[0] > 0], 99)
    return {name: tuple(float(np.corrcoef(np.asarray(nib.load(dest / f"{name}_{c}_Warped.nii.gz")
                                                     .dataobj, np.float64)[brain], t[i][brain])[0, 1])
                        for i, c in enumerate(CHANNELS))
            for name in inputs}


BRAIN_FRAC = 0.1


def t1_to_template(dwi, t1_brain, t1_mask, wm_post, threads=8):
    """Register a skull-stripped T1 to a diffusion template and carry its masks over.

    Args:
        dwi: Folder of the diffusion template (`dwi_template`'s `out`).
        t1_brain: Skull-stripped T1.
        t1_mask: Its brain mask.
        wm_post: Its white-matter posterior.
        threads: ITK threads.

    Writes into dwi/t1/: T1.nii.gz, brain_prob.nii.gz and wm_prob.nii.gz on the S0 template's
    grid, and the registration t1_to_s0_*. Rigid and affine with MI, then SyN with MI, each image
    inside its brain.
    """
    env = {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads)}
    dest = dwi / "t1"
    dest.mkdir(exist_ok=True)
    s0 = dwi / "T_template0.nii.gz"
    img = nib.load(s0)
    x = np.asarray(img.dataobj, np.float32)
    nib.save(nib.Nifti1Image((x > BRAIN_FRAC * np.percentile(x[x > 0], 99)).astype(np.uint8),
                             img.affine), dest / "s0_brain.nii.gz")
    prefix = dest / "t1_to_s0_"
    mi = f"MI[{s0},{t1_brain},1,32,Regular,0.25]"
    stages = ("-c", "[1000x500x250,1e-6,10]", "-f", "4x2x1", "-s", "2x1x0")
    run("antsRegistration", "-d", 3, "--float", 1, "-u", 0, "-w", "[0.005,0.995]", "-z", 1,
        "-x", f"[{dest / 's0_brain.nii.gz'},{t1_mask}]", "-r", f"[{s0},{t1_brain},1]",
        "-t", "Rigid[0.1]", "-m", mi, *stages, "-t", "Affine[0.1]", "-m", mi, *stages,
        "-t", "SyN[0.1,3,0]", "-m", f"MI[{s0},{t1_brain},1,32]", "-c", "[100x70x20,1e-9,10]",
        "-f", "4x2x1", "-s", "2x1x0", "-o", prefix, env=env)
    for src, name in ((t1_brain, "T1"), (t1_mask, "brain_prob"), (wm_post, "wm_prob")):
        run("antsApplyTransforms", "-d", 3, "-i", src, "-r", s0, "-t", f"{prefix}1Warp.nii.gz",
            "-t", f"{prefix}0GenericAffine.mat", "-n", "Linear", "-o", dest / f"{name}.nii.gz",
            env=env)
