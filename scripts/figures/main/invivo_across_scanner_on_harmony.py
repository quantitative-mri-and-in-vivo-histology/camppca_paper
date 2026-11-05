"""Figure: in vivo repeatability across scanners (ON-Harmony, one session per scanner of each
participant's phase, seven participants). Drawing in `camppca_paper.invivo`.

ARROWS mark texture Standard MPPCA keeps and Proposed removes, per scanner as (row, column,
tail row offset, tail column offset) in the region. They are placed by eye.

    python scripts/figures/main/invivo_across_scanner.py
"""

from camppca_paper.invivo import figure
from camppca_paper.paths import RESULTS

OUT = RESULTS / "main/invivo_across_scanner.pdf"
ARROWS = {"OXF1PRI": [(23, 9, 4.5, 4.5)], "OXF2PRI": [(21, 11, 4.5, 4.5)],
          "NOT1ACH": [(24, 11, 4.5, 4.5)], "NOT2ING": [(10, 5, 4.5, 4.5)],
          "OXF4GEP": [(8, 14, 4.5, 4.5)], "NOT4GEP": [(24, 18, 4.5, -4.5)]}

if __name__ == "__main__":
    fig = figure("cross", fig_w=0.9 * 8.64, top_h=0.3, row_h=1.15 * 1.62, left=0.075,
                 arrows=ARROWS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"  wrote {OUT}")
