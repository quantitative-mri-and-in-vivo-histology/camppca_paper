"""A slice with a boxed region and a magnifier of it, as the simulation figures draw them."""

import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.spatial import ConvexHull

ROI_CENTRE = (0.586, 0.774)    # zoom box centre, fractions of the panel, bottom-left origin
ROI_SIZE = 0.27                # zoom box side, fraction of the panel's shorter side
ZC = "#00c853"                 # zoom box colour
IZ = 0.552                     # magnifier size, fraction of the panel
INSET_DROP = 0.05              # how far the magnifier hangs below the panel


def box(shape):
    """The zoom box on an image of `shape`: `(row, column, side)` of its lower-left corner."""
    side = max(10, int(ROI_SIZE * min(shape)))
    h = side // 2 + 1
    cy = int(np.clip(round(ROI_CENTRE[0] * (shape[0] - 1)), h, shape[0] - h - 1))
    cx = int(np.clip(round(ROI_CENTRE[1] * (shape[1] - 1)), h, shape[1] - h - 1))
    return cy - side // 2, cx - side // 2, side


def connect(a, square, ia, inset):
    """Dashed lines from the zoom box to the magnifier, along the outside of both squares."""
    x0, y0, side = square
    to_frac = a.transData + a.transAxes.inverted()
    unit = ((0, 0), (1, 0), (1, 1), (0, 1))
    small = to_frac.transform([(x0 + u * side, y0 + v * side) for u, v in unit])
    big = np.array([(inset[0] + u * inset[2], inset[1] + v * inset[2]) for u, v in unit])
    on = set(ConvexHull(np.vstack([small, big])).vertices)
    for k, (u, v) in enumerate(unit):
        if k in on and k + 4 in on:
            a.add_artist(ConnectionPatch(small[k], (u, v), coordsA=a.transAxes,
                                         coordsB=ia.transAxes, color=ZC, lw=1.2,
                                         ls=(0, (3, 2)), zorder=4))


def zoomed(a, B, kw, r0, c0, side):
    """`B` on `a` with the zoom box at `(r0, c0, side)` and its magnifier."""
    a.imshow(B, origin="lower", **kw)
    a.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), side, side, fill=False, ec=ZC, lw=1.9))
    ix, iy = 1 - IZ, -INSET_DROP
    ia = a.inset_axes([ix, iy, IZ, IZ])
    ia.imshow(B[r0:r0 + side, c0:c0 + side], origin="lower", **kw)
    ia.set_xticks([])
    ia.set_yticks([])
    for sp in ia.spines.values():
        sp.set_color(ZC)
        sp.set_linewidth(1.9)
    connect(a, (c0 - 0.5, r0 - 0.5, side), ia, (ix, iy, IZ))
