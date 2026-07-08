"""Visualize ibm01 initial placement: hard macros colored by size, soft macros
faint, and overlapping hard-macro pairs outlined in red."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch

from macro_place.loader import load_benchmark_from_dir

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
out = sys.argv[2] if len(sys.argv) > 2 else f"/home/laz/partcl/my-macro-placer/notes/{name}_initial.png"

bench, _ = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
pos = bench.macro_positions
sz = bench.macro_sizes
nh = bench.num_hard_macros

def corners(i):
    cx, cy = pos[i, 0].item(), pos[i, 1].item()
    w, h = sz[i, 0].item(), sz[i, 1].item()
    return cx - w / 2, cy - h / 2, w, h

# Find overlapping hard-macro pairs (axis-aligned rectangle overlap, tiny tolerance)
tol = 1e-4
overlapping = set()
boxes = [corners(i) for i in range(nh)]
for i in range(nh):
    xi, yi, wi, hi = boxes[i]
    for j in range(i + 1, nh):
        xj, yj, wj, hj = boxes[j]
        ox = min(xi + wi, xj + wj) - max(xi, xj)
        oy = min(yi + hi, yj + hj) - max(yi, yj)
        if ox > tol and oy > tol:
            overlapping.add(i); overlapping.add(j)

fig, ax = plt.subplots(figsize=(9, 9))
ax.add_patch(Rectangle((0, 0), bench.canvas_width, bench.canvas_height,
                       fill=False, ec="black", lw=1.5))

# Soft macros: faint gray dots (they may overlap; just show where cell-mass sits)
ax.scatter(pos[nh:, 0], pos[nh:, 1], s=3, c="0.75", alpha=0.4, label="soft (894)")

# Hard macros: color by area (log scale to show 33x spread)
areas = (sz[:nh, 0] * sz[:nh, 1]).numpy()
norm = mcolors.LogNorm(vmin=areas.min(), vmax=areas.max())
cmap = cm.viridis
for i in range(nh):
    x, y, w, h = boxes[i]
    ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(norm(areas[i])),
                           alpha=0.85, ec="none"))
for i in overlapping:  # outline overlappers on top
    x, y, w, h = boxes[i]
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="red", lw=1.2))

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="hard macro area (um^2, log)")
ax.set_xlim(-1, bench.canvas_width + 1); ax.set_ylim(-1, bench.canvas_height + 1)
ax.set_aspect("equal")
ax.set_title(f"{name} initial placement — {nh} hard macros, "
             f"{len(overlapping)} in overlapping pairs (red)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"saved {out}")
print(f"hard macros in an overlap: {len(overlapping)} / {nh}")
