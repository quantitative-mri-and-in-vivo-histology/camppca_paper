"""Runs the EDDEN stages in order. Every stage skips finished work.

NORDIC2 (nordic2.m) runs separately, in MATLAB on the host, before resample.

    python scripts/analyses/edden/run.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("denoise", "correct", "s0_fa", "t1", "templates", "resample", "fit", "metrics")

if __name__ == "__main__":
    for stage in STAGES:
        print(f"== {stage}", flush=True)
        subprocess.run([sys.executable, HERE / f"{stage}.py"], check=True)
