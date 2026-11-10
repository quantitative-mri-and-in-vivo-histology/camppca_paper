"""Runs the panel stages in order. Every stage but noise skips finished work.

    python scripts/analyses/panel/run.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("masks", "denoise", "noise", "regions", "convergence")

if __name__ == "__main__":
    for stage in STAGES:
        print(f"== {stage}", flush=True)
        subprocess.run([sys.executable, HERE / f"{stage}.py"], check=True)
