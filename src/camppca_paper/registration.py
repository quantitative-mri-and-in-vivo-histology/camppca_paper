"""Moving a diffusion series into a template in one resampling.

Each volume goes through two maps: eddy's estimated correction for that volume (rigid movement,
eddy-current field and topup's susceptibility field), and the session's affine and SyN warp into
the template. The maps are composed on points and the data are interpolated once, by a cubic
B-spline. The ANTs warp enters as the composite displacement field antsApplyTransforms writes on
the template grid (LPS mm). eddy's correction is built from its parameters and topup's spline
coefficients at the points, with intensity scaled by its Jacobian determinant, as eddy does. The
template warp brings no Jacobian, because the signal is not a density.

FSL's convention for b-vectors and matrices is "scaled voxels", with x flipped when the affine has
a positive determinant. `fsl_scale` is that map.
"""

import shutil
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates, spline_filter

from .tools import run

LPS = np.diag([-1.0, -1.0, 1.0])


def fsl_scale(img: nib.Nifti1Image) -> np.ndarray:
    """FSL's voxel-to-scaled-millimetre matrix: voxel size, x flipped for a positive determinant.

    Args:
        img: Any NIfTI image.

    Returns:
        (4, 4) matrix taking voxel indices to FSL's scaled millimetres.
    """
    z = np.array(img.header.get_zooms()[:3], float)
    S = np.diag([*z, 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        S[0, 0], S[0, 3] = -z[0], (img.shape[0] - 1) * z[0]
    return S


def itk_linear(path: Path) -> np.ndarray:
    """The 3 x 3 linear part of an ANTs/ITK affine (.mat), which maps fixed points to moving ones (LPS).

    Args:
        path: The 0GenericAffine.mat written by antsRegistration.

    Returns:
        (3, 3) matrix.
    """
    itk = loadmat(path)
    p = next(v for key, v in itk.items() if key.startswith("AffineTransform")).ravel()
    return p[:9].reshape(3, 3)


def points_from_field(field: nib.Nifti1Image, native: np.ndarray) -> np.ndarray:
    """Every voxel of a template grid as a voxel position in a session's native (eddy) grid.

    Args:
        field: The composite displacement field antsApplyTransforms writes with `-o [field,1]` on
            the template grid: shape (*grid, 1, 3), LPS millimetres, template point plus field =
            the session's point.
        native: (4, 4) the native grid's affine.

    Returns:
        (3, P) voxel coordinates in the native grid, P the template's voxels in C order.
    """
    shape = field.shape[:3]
    A = field.affine
    disp = np.asarray(field.dataobj, np.float64).reshape(-1, 3).T
    vox = np.indices(shape, dtype=np.float64).reshape(3, -1)
    ras = LPS @ (LPS @ (A[:3, :3] @ vox + A[:3, 3:4]) + disp)
    M = np.linalg.inv(native)
    return M[:3, :3] @ ras + M[:3, 3:4]


def rotate_bvecs(rot: np.ndarray, native: nib.Nifti1Image, linear: np.ndarray, A: np.ndarray) -> np.ndarray:
    """eddy's rotated b-vectors into a template grid's FSL frame, through the affine's rotation.

    Args:
        rot: (3, V) eddy's rotated b-vectors, FSL frame of the native grid.
        native: The native (eddy) grid's image.
        linear: (3, 3) the template affine's linear part (`itk_linear`), fixed (template) to
            moving (session), LPS.
        A: (4, 4) the template grid's affine.

    Returns:
        (3, V) unit b-vectors in FSL's convention for the template grid; zero where `rot` is zero.
    """
    U, _, Vt = np.linalg.svd(np.linalg.inv(linear))
    v = native.affine[:3, :3] @ np.linalg.inv(fsl_scale(native)[:3, :3]) @ rot
    v = LPS @ (U @ Vt) @ LPS @ v
    v = fsl_scale(nib.Nifti1Image(np.zeros((1, 1, 1)), A))[:3, :3] @ np.linalg.inv(A[:3, :3]) @ v
    v = v / np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-9)
    v[:, np.linalg.norm(rot, axis=0) < 1e-6] = 0
    return v


def _fsl_rotation(angle: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """FSL's `make_rot` (miscmaths.cc): rotation about `angle`'s axis by its norm, about `centre` (mm)."""
    rot = np.eye(4)
    theta = float(np.linalg.norm(angle))
    if theta < 1e-8:
        return rot
    x1 = angle / theta
    x2 = np.array([-x1[1], x1[0], 0.0])
    if np.linalg.norm(x2) <= 0.0:
        x2 = np.array([1.0, 0.0, 0.0])
    x2 = x2 / np.linalg.norm(x2)
    x3 = np.cross(x1, x2)
    x3 = x3 / np.linalg.norm(x3)
    basis = np.stack([x2, x3, x1], 1)
    core = np.eye(3)
    core[0, 0] = core[1, 1] = np.cos(theta)
    core[0, 1], core[1, 0] = np.sin(theta), -np.sin(theta)
    rot[:3, :3] = basis @ core @ basis.T
    rot[:3, 3] = (np.eye(3) - rot[:3, :3]) @ centre
    return rot


def movement_matrix(mp: np.ndarray, shape: tuple, vox: np.ndarray) -> np.ndarray:
    """eddy's forward movement matrix R for one volume, on FSL's sampling grid (mm = voxel x size).

    `TOPUP::MovePar2Matrix` (topup_matrices.cpp) with `MISCMATHS::construct_rotmat_euler`
    (miscmaths.cc): R = Rx . Ry . Rz about the volume centre ((n - 1) x size / 2), then the
    translations added.

    Args:
        mp: (6,) eddy's movement parameters: x, y, z translation (mm), x, y, z rotation (rad).
        shape: The eddy grid's shape.
        vox: (3,) its voxel size.

    Returns:
        (4, 4) matrix acting on FSL sampling coordinates.
    """
    centre = (np.asarray(shape[:3], float) - 1) * vox / 2
    R = np.eye(4)
    for a in range(3):
        angle = np.zeros(3)
        angle[a] = mp[3 + a]
        R = R @ _fsl_rotation(angle, centre)
    R[:3, 3] += mp[:3]
    return R


def _bspline3(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FSL's cubic B-spline kernel and its derivative (`Spline1D::kernel_value`, fsl_splines.h)."""
    ax = np.abs(x)
    val = np.where(ax <= 1, 2 / 3 + ax * ax * (ax / 2 - 1), np.where(ax < 2, (2 - ax) ** 3 / 6, 0.0))
    der = np.where(ax <= 1, 1.5 * ax * ax - 2 * ax, np.where(ax < 2, -0.5 * (2 - ax) ** 2, 0.0)) * np.sign(x)
    return val, der


def topup_at(coef: nib.Nifti1Image, p: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """topup's off-resonance field (Hz) at voxel positions, and its derivative along one voxel axis.

    The field is the cubic B-spline topup writes as coefficients (`topup_fieldcoef`): coefficient
    c sits at voxel (c - 1) x knot spacing (`Spline1D::OffsetIntoField`, fsl_splines.h, order 3),
    the knot spacing in voxels is the coefficient file's pixdim. Evaluated at the points
    themselves, as eddy does on its own grid.

    Args:
        coef: The topup_fieldcoef image.
        p: (3, P) voxel positions on the grid topup ran on.
        axis: The voxel axis to differentiate along (the phase encode).

    Returns:
        ((P,) field in Hz, (P,) its derivative in Hz per voxel along `axis`).
    """
    c = np.asarray(coef.dataobj, np.float64)
    ksp = np.array(coef.header.get_zooms()[:3], float)
    u = p / ksp[:, None] + 1.0
    first = np.floor(u).astype(int) - 1
    weights, derivs, index = [], [], []
    for a in range(3):
        idx = first[a][None, :] + np.arange(4)[:, None]
        w, d = _bspline3(u[a][None, :] - idx)
        inside = (idx >= 0) & (idx < c.shape[a])
        weights.append(np.where(inside, w, 0.0))
        derivs.append(np.where(inside, d / ksp[a], 0.0))
        index.append(np.clip(idx, 0, c.shape[a] - 1))
    field = np.zeros(p.shape[1])
    grad = np.zeros(p.shape[1])
    for i in range(4):
        for j in range(4):
            for k in range(4):
                v = c[index[0][i], index[1][j], index[2][k]]
                w = [weights[0][i], weights[1][j], weights[2][k]]
                field += v * w[0] * w[1] * w[2]
                w[axis] = derivs[axis][(i, j, k)[axis]]
                grad += v * w[0] * w[1] * w[2]
    return field, grad


def _ec_field(ep: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """eddy's quadratic EC field (Hz) at centred mm positions, and its gradient (Hz/mm).

    `QuadraticScanECModel::ECField` (ECModels.cpp): ep[0..2] x, y, z; ep[3..5] x2, y2, z2;
    ep[6..8] xy, xz, yz; ep[9] the offset.
    """
    x, y, z = q
    f = (ep[0] * x + ep[1] * y + ep[2] * z + ep[3] * x * x + ep[4] * y * y + ep[5] * z * z + ep[6] * x * y
         + ep[7] * x * z + ep[8] * y * z + ep[9])
    g = np.stack([ep[0] + 2 * ep[3] * x + ep[6] * y + ep[7] * z,
                  ep[1] + 2 * ep[4] * y + ep[6] * x + ep[8] * z,
                  ep[2] + 2 * ep[5] * z + ep[7] * x + ep[8] * y])
    return f, g


def eddy_coords_from_params(base: Path, j: int, p: np.ndarray,
                            susc: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points through eddy's correction for volume j, assembled from its parameters, and its Jacobian.

    The same map eddy writes with `--dfields`, built from `eddy.eddy_parameters` (6 movement and
    10 quadratic EC parameters, `--flm=quadratic --slm=none`), the acqp row and topup's spline
    coefficients, and evaluated exactly at the points, so no stored field is interpolated. Per
    `ECScan::field_for_scan_to_model_transform` and `NEWIMAGE::general_transform`
    (ECScanClasses.cpp, warpfns.h), for a model-space point x (FSL sampling mm, s = voxel size):

        Hz(x)   = EC(R x - centre) + susceptibility(x)    the EC field sits in the scanner frame,
                                                           so it is evaluated where the head moved
                                                           x to; the susceptibility moves with the
                                                           head (model space)
        d(x)    = pe x readout x Hz(x)                      voxels, along the phase encode
                                                           (`FieldUtils::Hz2VoxelDisplacements`)
        source  = s^-1 (R x) + d(x)                        voxel position in the acquired volume
        J(x)    = 1 + d d_pe / d x_pe                      `FieldUtils::GetJacobian` (eddy takes it
                                                           from a spline of d sampled on the grid;
                                                           here it is the analytic derivative)

    with R from `movement_matrix`. Outlier slices eddy replaced (`--repol`) are not replaced.

    Args:
        base: eddy's folder: eddy.eddy_parameters, acqp.txt, index.txt, topup_fieldcoef.nii.gz and
            the bet mask `mask_bet.nii.gz` (for the grid).
        j: Volume index, from 0.
        p: (3, P) voxel positions in the eddy grid (model space).
        susc: (field, derivative along the phase encode) at p from `topup_at`, if already known;
            it is the same for every volume, so a caller looping over volumes passes it once.

    Returns:
        ((3, P) positions in the acquired volume's voxel grid, (P,) Jacobian determinant).
    """
    grid = nib.load(base / "mask_bet.nii.gz")
    vox = np.array(grid.header.get_zooms()[:3], float)
    shape = grid.shape[:3]
    params = np.atleast_2d(np.loadtxt(base / "eddy.eddy_parameters"))[j]
    row = int(np.atleast_1d(np.loadtxt(base / "index.txt"))[j]) - 1
    acq = np.atleast_2d(np.loadtxt(base / "acqp.txt"))[row]
    pe, readout = acq[:3], acq[3]
    axis = int(np.flatnonzero(pe)[0])
    if susc is None:
        susc = topup_at(nib.load(base / "topup_fieldcoef.nii.gz"), p, axis)
    R = movement_matrix(params[:6], shape, vox)
    moved = R[:3, :3] @ (p * vox[:, None]) + R[:3, 3:4]
    centre = (np.asarray(shape, float) - 1) * vox / 2
    ec, ec_grad = _ec_field(params[6:16], moved - centre[:, None])
    hz = ec + susc[0]
    dhz = ec_grad.T @ (R[:3, axis] * vox[axis]) + susc[1]
    coords = moved / vox[:, None]
    coords[axis] += pe[axis] * readout * hz
    return coords, 1.0 + pe[axis] * readout * dhz


def pe_axis(base: Path) -> int:
    """The voxel axis of the phase encode, from eddy's acqp (every row of a session shares it)."""
    acq = np.atleast_2d(np.loadtxt(base / "acqp.txt"))[0]
    return int(np.argmax(np.abs(acq[:3])))


def pull_series(task: tuple) -> str:
    """One native series into a template in one pull: every volume through eddy's estimated correction.

    The correction is assembled from eddy's parameters and topup's coefficients at the points
    themselves (`eddy_coords_from_params`); the susceptibility part is the same for every volume,
    so it is evaluated once.

    Args:
        task: (src, base, points, dst, shape, affine): the native series (any NIfTI), eddy's folder,
            an .npy of (3, P) native voxel positions of the template's voxels (`points_from_field`),
            the destination (.nii.gz), and the template grid's shape and affine.

    Returns:
        The destination's file name.
    """
    src, base, points, dst, shape, affine = task
    x = np.asarray(nib.load(src).dataobj, np.float32)
    p = np.load(points)
    susc = topup_at(nib.load(base / "topup_fieldcoef.nii.gz"), p, pe_axis(base))
    n = x.shape[3]
    out = np.empty(tuple(shape) + (n,), np.float32)
    for j in range(n):
        coords, J = eddy_coords_from_params(base, j, p, susc)
        y = map_coordinates(spline_filter(x[..., j].astype(np.float64), order=3, mode="grid-wrap"), coords,
                            order=3, mode="grid-wrap", prefilter=False) * J
        out[..., j] = y.reshape(shape)
    nib.save(nib.Nifti1Image(out, affine), str(dst))
    return Path(dst).name


def resample(tpl, dest, sessions, workers=8):
    """Pull every arm of every session of one set into its diffusion template.

    Args:
        tpl: The template folder (`templates.dwi_template`'s `out`, with t1/ from
            `templates.t1_to_template`).
        dest: Output folder.
        sessions: List of `(name, base, bval, arms)`: the session's name in `tpl/to_template`,
            its eddy folder, its bval file, and `{arm: series path}` in the eddy grid.
        workers: Series pulled in parallel.

    Returns:
        `{name: r}`, the correlation inside the mask between the session's DKI FA pulled through
        these points and antsApplyTransforms' own pull, a check of the transform handling.

    Writes <name>_<arm>.nii.gz, <name>.bval, <name>.bvec, mask.nii.gz (the T1 brain above 0.5
    inside every session's field of view) and wm_prob.nii.gz.
    """
    dest.mkdir(parents=True, exist_ok=True)
    ref = nib.load(tpl / "T_template0.nii.gz")
    shape, A = ref.shape, ref.affine
    fov = np.ones(int(np.prod(shape)), bool)
    tasks = []
    for name, base, bval, arms in sessions:
        warp = tpl / "to_template" / name
        field = dest / f"field_{name}.nii.gz"
        run("antsApplyTransforms", "-d", 3, "-r", tpl / "T_template0.nii.gz",
            "-t", f"{warp}_1Warp.nii.gz", "-t", f"{warp}_0GenericAffine.mat", "-o", f"[{field},1]")
        native = nib.load(base / "mask_bet.nii.gz")
        p = points_from_field(nib.load(field), native.affine)
        field.unlink()
        np.save(dest / f"points_{name}.npy", p)
        fov &= np.all((p >= 0) & (p <= np.array(native.shape)[:, None] - 1), 0)
        rot = np.loadtxt(base / "eddy.eddy_rotated_bvecs")
        np.savetxt(dest / f"{name}.bvec",
                   rotate_bvecs(rot, native, itk_linear(f"{warp}_0GenericAffine.mat"), A), fmt="%.6f")
        shutil.copy(bval, dest / f"{name}.bval")
        tasks += [(src, base, dest / f"points_{name}.npy", dest / f"{name}_{arm}.nii.gz", shape, A)
                  for arm, src in arms.items() if not (dest / f"{name}_{arm}.nii.gz").exists()]
    brain = np.asarray(nib.load(tpl / "t1" / "brain_prob.nii.gz").dataobj) > 0.5
    mask = brain & fov.reshape(shape)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), A), dest / "mask.nii.gz")

    with Pool(workers) as pool:
        list(pool.imap_unordered(pull_series, tasks))

    check = {}
    for name, base, _, _ in sessions:
        warp = tpl / "to_template" / name
        tmp = dest / f"check_{name}.nii.gz"
        run("antsApplyTransforms", "-d", 3, "-i", base / "dki_FA.nii.gz",
            "-r", tpl / "T_template0.nii.gz", "-t", f"{warp}_1Warp.nii.gz",
            "-t", f"{warp}_0GenericAffine.mat", "-n", "Linear", "-o", tmp)
        theirs = np.asarray(nib.load(tmp).dataobj, np.float64)
        tmp.unlink()
        fa = np.asarray(nib.load(base / "dki_FA.nii.gz").dataobj, np.float64)
        p = dest / f"points_{name}.npy"
        ours = map_coordinates(fa, np.load(p), order=1).reshape(shape)
        p.unlink()
        check[name] = float(np.corrcoef(ours[mask], theirs[mask])[0, 1])
    # Written last, so its presence means the set is complete.
    shutil.copy(tpl / "t1" / "wm_prob.nii.gz", dest / "wm_prob.nii.gz")
    return check
