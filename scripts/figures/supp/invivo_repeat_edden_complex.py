"""Figure: in vivo repeatability at 0.9 mm (EDDEN), the denoisers on the real part of the complex
data, with NORDIC on the complex data, without the signal error. Drawing in
`camppca_paper.edden_figure`.

    python scripts/figures/supp/invivo_repeat_edden_complex.py
"""

import matplotlib.pyplot as plt

from camppca_paper.edden_figure import figure
from camppca_paper.paths import RESULTS

OUT = RESULTS / "supp/invivo_repeat_edden_complex.pdf"

if __name__ == "__main__":
    fig = figure("complex")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        fig.savefig(OUT)
    print(f"  wrote {OUT}")
