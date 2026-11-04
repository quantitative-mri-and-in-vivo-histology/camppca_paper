"""Download files of an OpenNeuro release, checked against the hash the release records.

The release is cloned shallow for its file tree, which holds a git-annex pointer per file. Each
file is then downloaded from OpenNeuro's S3 bucket by its path, and checked against the size
and hash in its pointer. Most releases record SHA256, some older ones MD5.
"""

import hashlib
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

# git-annex key in a pointer's symlink target: hash type, size and hash of the content.
KEY = re.compile(r"(SHA256|MD5)E-s(\d+)--([0-9a-f]{64}|[0-9a-f]{32})")


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
    """Where an annexed file's content goes, its size, and its hash.

    Args:
        root: The clone.
        rel: Path relative to the release root.

    Returns:
        `(path, size, algorithm, hash)`, or None if git stores the file directly.
        `algorithm` is "sha256" or "md5".
    """
    link = root / rel
    if not link.is_symlink():
        return None
    algorithm, size, digest = KEY.search(str(link.readlink())).groups()
    return (link.parent / link.readlink()).resolve(), int(size), algorithm.lower(), digest


def checksum(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch_file(accession, root, rel):
    """Download one annexed file unless it is there already, then check it.

    Raises:
        RuntimeError: If the content does not match the release's hash.
    """
    entry = annexed(root, rel)
    if entry is None:
        return
    dest, size, algorithm, digest = entry
    if not (dest.exists() and dest.stat().st_size == size):
        subprocess.run(["curl", "-fsSL", "--retry", "3", "--create-dirs", "-o", str(dest),
                        f"https://s3.amazonaws.com/openneuro.org/{accession}/{rel}"], check=True)
    if checksum(dest, algorithm) != digest:
        raise RuntimeError(f"{algorithm} mismatch: {rel}")


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
