"""Figure: in vivo repeatability within scanners (ON-Harmony, six repeats on each participant's
own scanner, seven participants). Drawing in `camppca_paper.invivo`.

ARROWS mark texture Standard MPPCA keeps and Proposed removes, per session as (row, column,
tail row offset, tail column offset) in the region. They are placed by eye.

    python scripts/figures/main/invivo_within_scanner_on_harmony.py
"""

from camppca_paper.invivo import figure
from camppca_paper.paths import RESULTS

OUT = RESULTS / "main/invivo_within_scanner_on_harmony.pdf"
ARROWS = {1: [(2, 4, 4.5, 4.5)], 2: [(23, 14, -4.5, 4.5)], 3: [(5, 6, 4.5, 4.5)],
          4: [(4, 3, 4.5, 4.5)], 5: [(9, 3, -4.5, -4.5)], 6: [(3, 7, 0, 5.5)]}

if __name__ == "__main__":
    fig = figure("within", correction=False, fig_w=0.9 * 8.64, top_h=0.3, row_h=1.15 * 1.62,
                 bars_key_inside=True, left=0.06, arrows=ARROWS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")
