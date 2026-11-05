"""Runs the ex vivo stages in order. Every stage skips finished work.

    python scripts/analyses/exvivo_mouse/run.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("denoise", "fit", "metrics")

if __name__ == "__main__":
    for stage in STAGES:
        print(f"== {stage}", flush=True)
        subprocess.run([sys.executable, HERE / f"{stage}.py"], check=True)
