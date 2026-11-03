"""The ON-Harmony participants and sessions the in vivo analyses use.

Seven participants were scanned six times on one scanner, their repeat scanner, and once on
every other scanner of their phase. The within-scanner set is the six repeats. The
cross-scanner set is one session per scanner, the repeat scanner's first.
"""

from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_ROOT

ACCESSION = "ds004712"
VERSION = "2.0.1"
SOURCE = DATA_ROOT / "source/on_harmony" / ACCESSION
WORK = DATA_ROOT / "analyses/on_harmony"

SITES = {"OXF1PRI": "Prisma 32ch", "OXF2PRI": "Prisma 64ch", "OXF3TRI": "Trio",
         "NOT1ACH": "Achieva", "NOT2ING": "Ingenia", "NOT4GEP": "Premier 48ch",
         "OXF4GEP": "Premier 21ch"}
PHASES = {"A": ("OXF1PRI", "OXF2PRI", "OXF3TRI", "NOT1ACH", "NOT2ING"),
          "B": ("OXF1PRI", "OXF2PRI", "NOT1ACH", "NOT2ING", "NOT4GEP", "OXF4GEP")}
# subject: (repeat scanner, phase)
PARTICIPANTS = {"sub-13192": ("OXF1PRI", "A"), "sub-14229": ("OXF2PRI", "A"),
                "sub-14230": ("OXF3TRI", "A"), "sub-14482": ("NOT1ACH", "A"),
                "sub-15320": ("NOT2ING", "B"), "sub-16745": ("NOT4GEP", "B"),
                "sub-16975": ("OXF4GEP", "B")}
REPEATS = (1, 2, 3, 4, 5, 6)
SETS = ("within", "cross")


@dataclass(frozen=True)
class Session:
    """One ON-Harmony session of one participant."""

    subject: str
    site: str
    run: int

    @property
    def name(self) -> str:
        """The BIDS session label without its prefix, e.g. NOT2ING001."""
        return f"{self.site}{self.run:03d}"

    def dwi(self, direction="AP") -> Path:
        """The DWI series of one phase-encoding direction, relative to the release root,
        without its extension."""
        ses = f"ses-{self.name}"
        return Path(self.subject, ses, "dwi", f"{self.subject}_{ses}_dir-{direction}_dwi")

    @property
    def native(self) -> Path:
        """Where the pipeline writes this session's outputs in its own space."""
        return WORK / self.subject / "native" / self.name

    @property
    def t1(self) -> Path:
        """The session's T1w, relative to the release root."""
        ses = f"ses-{self.name}"
        return Path(self.subject, ses, "anat", f"{self.subject}_{ses}_T1w.nii.gz")


def sessions(subject, which):
    """The sessions of one set.

    Args:
        subject: A key of `PARTICIPANTS`.
        which: "within" or "cross".

    Returns:
        List of `Session`, the repeat scanner's first.
    """
    own, phase = PARTICIPANTS[subject]
    if which == "within":
        return [Session(subject, own, k) for k in REPEATS]
    if which == "cross":
        return [Session(subject, own, 1)] + [Session(subject, s, 1) for s in PHASES[phase]
                                             if s != own]
    raise ValueError(f"unknown set {which!r}, expected one of {SETS}")


def all_sessions(subject):
    """Every session either set needs, each once."""
    out = sessions(subject, "within")
    return out + [s for s in sessions(subject, "cross") if s not in out]
