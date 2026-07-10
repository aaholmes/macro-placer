"""Before/after placement plots for selected benchmarks: legalized reference vs
optimized checkpoint. Hard macros colored by area, soft macros as gray dots."""
import sys, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm, matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from placers.legalizer import legalize

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
NOTES = "/home/laz/partcl/my-macro-placer/notes"
CKPT = f"{NOTES}/checkpoints"
benches = sys.argv[1:] or ["ibm01", "ibm09", "ibm18"]

def draw(ax, bench, pos, sz, nh, title):
    ax.add_patch(Rectangle((0, 0), bench.canvas_width, bench.canvas_height,
                           fill=False, ec="black", lw=1.2))
    ax.scatter(pos[nh:, 0], pos[nh:, 1], s=2, c="0.7", alpha=0.35)
    areas = (sz[:nh, 0] * sz[:nh, 1])
    norm = mcolors.LogNorm(vmin=areas.min(), vmax=areas.max())
    for i in range(nh):
        w, h = sz[i]; x, y = pos[i]
        ax.add_patch(Rectangle((x - w/2, y - h/2), w, h,
                               facecolor=cm.viridis(norm(areas[i])), alpha=0.85, ec="none"))
    ax.set_xlim(-1, bench.canvas_width + 1); ax.set_ylim(-1, bench.canvas_height + 1)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11)

n = len(benches)
fig, axes = plt.subplots(n, 2, figsize=(12, 6 * n))
if n == 1:
    axes = axes[None, :]
for r, name in enumerate(benches):
    bench, _ = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
    sz = bench.macro_sizes.numpy(); nh = bench.num_hard_macros
    legal, _, _ = legalize(bench.macro_positions.numpy().astype(np.float64), sz, nh,
                           bench.canvas_width, bench.canvas_height, gap=0.01)
    # copy checkpoint to avoid racing the live writer
    src = f"{CKPT}/{name}.npz"
    d = None
    if os.path.exists(src):
        tmp = f"{NOTES}/_tmp_{name}.npz"
        try:
            shutil.copy(src, tmp); d = np.load(tmp); os.remove(tmp)
        except Exception as e:
            print(f"{name}: checkpoint read failed ({e})")
    opt = d["best_pos"].astype(np.float64) if d is not None else legal
    it = int(d["iters_done"]) if d is not None else 0
    bc = float(d["best_cost"]) if d is not None else 0.0
    draw(axes[r, 0], bench, legal, sz, nh, f"{name} initial (legalized)")
    draw(axes[r, 1], bench, opt, sz, nh, f"{name} optimized ({it//1000}k iters, proxy={bc:.3f})")
    print(f"{name}: {it} iters, proxy {bc:.4f}")

fig.suptitle("Before / after: hard macros (area-colored) + soft macros (gray). Soft co-optimization.",
             fontsize=13, y=1.0)
fig.tight_layout()
OUT = f"{NOTES}/before_after_" + "_".join(benches) + ".png"
fig.savefig(OUT, dpi=90)
print(f"saved {OUT}")
