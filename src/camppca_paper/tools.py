"""Running external tools such as FSL and ANTs."""

import os
import subprocess


def run(*args, log=None, env=None, cwd=None):
    """Run a command, its output written to `log` or discarded.

    Args:
        *args: The program, then its arguments.
        log: File for the command line and its output.
        env: Extra environment variables.
        cwd: Working directory.

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    cmd = [str(a) for a in args]
    with open(log or os.devnull, "w") as f:
        if log:
            f.write(" ".join(cmd) + "\n")
            f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, cwd=cwd,
                       env=dict(os.environ, **(env or {})))
