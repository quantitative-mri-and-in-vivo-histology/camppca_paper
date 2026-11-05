"""The in vivo panel: 28 scanner configurations without interpolation, two subjects each."""

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .paths import DATA_ROOT, OPENNEURO

HCP = DATA_ROOT / "source/hcp"
WORK = DATA_ROOT / "analyses/panel"
SHELL_MAX = 2500
PHILIPS = ("onh04_achieva_sw56", "onh05_achieva_sw53", "onh06_ingenia_r15", "onh07_ingenia_r2")
GE = ("ideas01_mr750_qs", "onh08_mr750", "onh09_premier48", "onh10_premier21", "sdsu01_mr750_8ch",
      "unam02_mr750_hb")
# Phase-encode array axis where the release records none.
PE_AXIS = {"hcp01_skyra_sense1": 0, "hcp02_7t_grappa3": 1, "ideas01_mr750_qs": 1}


@dataclass(frozen=True)
class Scan:
    """One subject of one configuration.

    For OpenNeuro, `series` are the BIDS entities between the session and `_dwi`, one per
    series. An empty string is a series with no further entities. For HCP, they are the
    series names in the HCP file names.
    """

    config: str
    source: str     # OpenNeuro accession, or "HCP_1200"
    version: str
    sub: str
    ses: str
    series: tuple

    def stems(self):
        """Each series' path relative to its release root, without extension."""
        if self.source == "HCP_1200":
            return [f"{self.sub}/unprocessed/{self.version}/Diffusion/"
                    f"{self.sub}_{self.version}_DWI_{s}" for s in self.series]
        folder = "/".join(p for p in (self.sub, self.ses, "dwi") if p)
        return [f"{folder}/" + "_".join(p for p in (self.sub, self.ses, s) if p) + "_dwi"
                for s in self.series]

    @property
    def root(self):
        return HCP if self.source == "HCP_1200" else OPENNEURO / self.source

    @property
    def name(self):
        return f"{self.config}_{self.sub}"

    @property
    def work(self):
        """Where the analyses write this scan's outputs."""
        return WORK / self.config / self.sub


# config, source, version, session, series, subjects
_CONFIGS = [
    ("acc01_mb1_r1", "ds006676", "1.0.0", "ses-001", ("acq-S1P1_dir-AP",), ("sub-001", "sub-002")),
    ("acc02_mb3_r1", "ds006676", "1.0.0", "ses-001", ("acq-S3P1_dir-AP",), ("sub-003", "sub-004")),
    ("acc03_mb3_r2", "ds006676", "1.0.0", "ses-001", ("acq-S3P2_dir-AP",), ("sub-005", "sub-006")),
    ("acc04_mb6_r1", "ds006676", "1.0.0", "ses-001", ("acq-S6P1_dir-AP",), ("sub-801", "sub-803")),
    ("acc05_mb6_r2", "ds006676", "1.0.0", "ses-001", ("acq-S6P2_dir-AP",), ("sub-806", "sub-808")),
    ("alem01_skyra_p2", "ds005364", "1.0.0", "", ("",), ("sub-001", "sub-003")),
    ("c2_connectom72", "ds007739", "1.0.0", "", ("run-05",), ("sub-02", "sub-03")),
    ("ccbbi01_mb2_b2000", "ds005901", "1.0.0", "", ("",), ("sub-s001", "sub-s002")),
    ("hcp01_skyra_sense1", "HCP_1200", "3T", "", ("dir95_LR", "dir96_LR", "dir97_LR"),
     ("100206", "103818")),
    ("hcp02_7t_grappa3", "HCP_1200", "7T", "", ("dir71_AP", "dir72_AP"), ("104416", "114823")),
    ("ideas01_mr750_qs", "ds007401", "1.0.0", "ses-1", ("",), ("sub-1", "sub-103")),
    ("leip01_verio_pf875", "ds000221", "1.0.0", "ses-01", ("",), ("sub-010002", "sub-010003")),
    ("nyu01_prisma64_mb3", "ds007283", "1.1.0", "ses-nyu3t02", ("acq-138vols",),
     ("sub-wlsubj016", "sub-wlsubj119")),
    ("onh01_prisma32", "ds004712", "2.0.1", "ses-OXF1PRI001", ("dir-AP",), ("sub-16842", "sub-16974")),
    ("onh02_prisma64", "ds004712", "2.0.1", "ses-OXF2PRI001", ("dir-AP",), ("sub-16975", "sub-16981")),
    ("onh03_trio", "ds004712", "2.0.1", "ses-OXF3TRI001", ("dir-AP",), ("sub-10975", "sub-12813")),
    ("onh04_achieva_sw56", "ds004712", "2.0.1", "ses-NOT1ACH001", ("dir-AP",), ("sub-14482", "sub-16841")),
    ("onh05_achieva_sw53", "ds004712", "2.0.1", "ses-NOT1ACH001", ("dir-AP",), ("sub-13192", "sub-13305")),
    ("onh06_ingenia_r15", "ds004712", "2.0.1", "ses-NOT2ING001", ("dir-AP",), ("sub-14221", "sub-16794")),
    ("onh07_ingenia_r2", "ds004712", "2.0.1", "ses-NOT2ING001", ("dir-AP",), ("sub-14229", "sub-14230")),
    ("onh08_mr750", "ds004712", "2.0.1", "ses-NOT3GEM001", ("dir-AP",), ("sub-03286", "sub-03997")),
    ("onh09_premier48", "ds004712", "2.0.1", "ses-NOT4GEP001", ("dir-AP",), ("sub-16745", "sub-16793")),
    ("onh10_premier21", "ds004712", "2.0.1", "ses-OXF4GEP001", ("dir-AP",), ("sub-15320", "sub-16766")),
    ("sdsu01_mr750_8ch", "ds005664", "1.1.2", "ses-cfmri1", ("acq-1shell61dir_dir-PA",),
     ("sub-ts003", "sub-ts005")),
    ("sdsu02_prisma_hcpls", "ds005664", "1.1.2", "ses-sdsu1",
     ("acq-hcplifespan98dir_dir-AP", "acq-hcplifespan99dir_dir-AP"), ("sub-ts007", "sub-ts008")),
    ("sral01_prisma_freedir", "ds004024", "1.0.1", "ses-mri", ("",), ("sub-CON015", "sub-CON025")),
    ("unam02_mr750_hb", "ds006837", "1.0.1", "", ("acq-hb",), ("sub-15464", "sub-700")),
    ("uq01_7t_grappa3", "ds007036", "1.1.3", "ses-01",
     ("acq-b2500_dir-AP_run-1", "acq-b1000_dir-AP_run-1"), ("sub-H0101", "sub-H0102")),
]

SCANS = [Scan(c, src, v, sub, ses, series)
         for c, src, v, ses, series, subs in _CONFIGS for sub in subs]
CONFIGS = [c[0] for c in _CONFIGS]


def vendor(config):
    return "Philips" if config in PHILIPS else "GE" if config in GE else "Siemens"


def load(scan):
    """One scan's series, its parts joined, up to b SHELL_MAX.

    A shell counts by its b-value rounded to 100, so b 2510 is b 2500.

    Returns:
        `(data (X, Y, Z, N) float32, bval (N,), bvec (3, N), affine, phase-encode axis)`.
    """
    parts, bval, bvec, affine = [], [], [], None
    for stem in scan.stems():
        p = scan.root / stem
        img = nib.load(f"{p}.nii.gz")
        if affine is not None and not np.allclose(img.affine, affine, atol=1e-3):
            raise ValueError(f"{scan.name}: parts on different grids")
        affine = img.affine
        parts.append(np.asarray(img.dataobj, np.float32))
        bval.append(np.loadtxt(f"{p}.bval").ravel())
        v = np.atleast_2d(np.loadtxt(f"{p}.bvec"))
        bvec.append(v if v.shape[0] == 3 else v.T)
    bval, bvec = np.concatenate(bval), np.concatenate(bvec, axis=1)
    keep = np.round(bval, -2) <= SHELL_MAX
    data = np.concatenate(parts, axis=3)[..., keep]
    return data, bval[keep], bvec[:, keep], affine, pe_axis(scan)


def pe_axis(scan):
    """The phase-encode array axis, from the sidecars or PE_AXIS."""
    axes = set()
    for stem in scan.stems():
        side = Path(f"{scan.root / stem}.json")
        meta = json.loads(side.read_text()) if side.exists() else {}
        ped = meta.get("PhaseEncodingDirection") or meta.get("PhaseEncodingAxis")
        axes.add("ijk".index(ped[0]) if ped else PE_AXIS[scan.config])
    if len(axes) != 1:
        raise ValueError(f"{scan.name}: parts differ in phase-encode axis")
    return axes.pop()
