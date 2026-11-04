"""Runs the ON-Harmony stages in order. Every stage skips finished work.

    python scripts/analyses/on_harmony/run.py [subject ...]
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("denoise", "correct", "s0_fa", "t1", "templates", "resample", "fit", "metrics")

if __name__ == "__main__":
    for stage in STAGES:
        print(f"== {stage}", flush=True)
        subprocess.run([sys.executable, HERE / f"{stage}.py", *sys.argv[1:]], check=True)
