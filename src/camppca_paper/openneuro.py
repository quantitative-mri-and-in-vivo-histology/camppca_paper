"""Download files of an OpenNeuro release, checked against the SHA256 the release records.

The release is cloned shallow for its file tree, which holds a git-annex pointer per file. Each
file is then downloaded from OpenNeuro's S3 bucket by its path, and checked against the size
and SHA256 in its pointer.
"""

import hashlib
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

# git-annex key in a pointer's symlink target: size and SHA256 of the content.
KEY = re.compile(r"SHA256E-s(\d+)--([0-9a-f]{64})")


def clone(accession, version, root):
    """Shallow clone of the release's file tree into `root`, without file contents.

    Args:
        accession: OpenNeuro accession, e.g. "ds004666".
        version: Release tag.
        root: Where the clone goes. Nothing happens if it exists.
    """
    if not (root / ".git").exists():
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet",
                        "--depth", "1", "--branch", version,
                        f"https://github.com/OpenNeuroDatasets/{accession}.git", str(root)],
                       check=True)


def annexed(root, rel):
    """Where an annexed file's content goes, and its size and SHA256.

    Args:
        root: The clone.
        rel: Path relative to the release root.

    Returns:
        `(path, size, sha256)`, or None if git stores the file directly.
    """
    link = root / rel
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


def fetch_file(accession, root, rel):
    """Download one annexed file unless it is there already, then check it.

    Raises:
        RuntimeError: If the content does not match the release's SHA256.
    """
    entry = annexed(root, rel)
    if entry is None:
        return
    dest, size, sha = entry
    if not (dest.exists() and dest.stat().st_size == size):
        subprocess.run(["curl", "-fsSL", "--retry", "3", "--create-dirs", "-o", str(dest),
                        f"https://s3.amazonaws.com/openneuro.org/{accession}/{rel}"], check=True)
    if sha256(dest) != sha:
        raise RuntimeError(f"sha256 mismatch: {rel}")


def fetch(accession, version, root, files, workers=8):
    """Clone a release and download the given files.

    Args:
        accession: OpenNeuro accession.
        version: Release tag.
        root: Where the clone goes.
        files: Paths relative to the release root.
        workers: Parallel downloads.

    Raises:
        SystemExit: If a file is not in the release.
    """
    clone(accession, version, root)
    missing = [f for f in files if not ((root / f).is_symlink() or (root / f).exists())]
    if missing:
        raise SystemExit("  not in the release:\n" + "".join(f"    {f}\n" for f in missing))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda rel: fetch_file(accession, root, rel), files))
    print(f"  ok  {len(files)} files of {accession} {version} in {root}")
