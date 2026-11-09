"""Figure: in vivo repeatability at 0.9 mm (EDDEN), the denoisers on the magnitude. Drawing in
`camppca_paper.edden_figure`.

    python scripts/figures/main/invivo_repeat_edden.py
"""

import matplotlib.pyplot as plt

from camppca_paper.edden_figure import figure
from camppca_paper.paths import RESULTS

OUT = RESULTS / "main/invivo_repeat_edden.pdf"

if __name__ == "__main__":
    fig = figure("magnitude")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        fig.savefig(OUT)
    print(f"  wrote {OUT}")
