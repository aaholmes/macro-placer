"""Static placement-layout comparison: legalized reference vs our personal best,
for a few benchmarks. Hard macros drawn as area-colored rectangles, soft macros
as light dots. Personal-best placement is read from notes/polish (fallback
notes/validation). Saves notes/fig_layout_<nm>.png (one per benchmark).

Usage: python plot_layouts.py [ibm01 ibm09 ibm18]
"""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm, matplotlib.colors as mcolors
import numpy as np, torch
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.legalizer import legalize

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
NOTES = "/home/laz/partcl/my-macro-placer/notes"
benches = sys.argv[1:] or ["ibm01", "ibm09", "ibm18"]


def draw(ax, bench, pos, sz, nh, title):
    ax.add_patch(Rectangle((0, 0), bench.canvas_width, bench.canvas_height,
                           fill=False, ec="black", lw=1.2))
    ax.scatter(pos[nh:, 0], pos[nh:, 1], s=2, c="0.7", alpha=0.35)
    areas = sz[:nh, 0] * sz[:nh, 1]
    norm = mcolors.LogNorm(vmin=areas.min(), vmax=areas.max())
    for i in range(nh):
        w, h = sz[i]; x, y = pos[i]
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                               facecolor=cm.viridis(norm(areas[i])), alpha=0.85, ec="none"))
    ax.set_xlim(-1, bench.canvas_width + 1); ax.set_ylim(-1, bench.canvas_height + 1)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


for nm in benches:
    bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    sz = bench.macro_sizes.numpy(); nh = bench.num_hard_macros
    ref, _, _ = legalize(bench.macro_positions.numpy().astype(np.float64), sz, nh,
                         bench.canvas_width, bench.canvas_height, gap=0.01)
    src = f"{NOTES}/polish/{nm}.npz" if os.path.exists(f"{NOTES}/polish/{nm}.npz") \
        else f"{NOTES}/validation/{nm}.npz"
    best = np.load(src)["placement"].astype(np.float64)
    pr = float(compute_proxy_cost(torch.tensor(ref, dtype=torch.float32), bench, plc)["proxy_cost"])
    pb = float(compute_proxy_cost(torch.tensor(best, dtype=torch.float32), bench, plc)["proxy_cost"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
    draw(ax[0], bench, ref, sz, nh, f"{nm} — reference (proxy {pr:.3f})")
    draw(ax[1], bench, best, sz, nh, f"{nm} — ours (proxy {pb:.3f}, {pb/pr:.2f}× ref)")
    fig.tight_layout(); fig.savefig(f"{NOTES}/fig_layout_{nm}.png", dpi=115)
    print(f"saved fig_layout_{nm}.png  ref={pr:.3f} ours={pb:.3f} ratio={pb/pr:.3f}")
