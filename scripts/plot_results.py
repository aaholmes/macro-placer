"""Reproduce the greedy ibm01 run, capturing the full convergence trajectory,
then render (1) before/after placement and (2) score-per-eval with reference
lines."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import torch

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
NOTES = "/home/laz/partcl/my-macro-placer/notes"
name = "ibm01"
bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
fe = FastEval(bench, plc)
nh = bench.num_hard_macros
sz = bench.macro_sizes.numpy()

base = bench.macro_positions.numpy().astype(np.float64)
legal, _, _ = legalize(base, sz, nh, bench.canvas_width, bench.canvas_height, gap=0.01)

# capture full trajectory (log_every=1, silent)
best, hist = optimize(fe, legal, iters=2500, seed=1, T0=0.0,
                      jump_frac=0.3, jitter_sigma=1.0, log_every=1, logf=lambda *a: None)
np.save(f"{NOTES}/{name}_opt.npy", best)
evals = np.array([h[0] for h in hist]); scores = np.array([h[1] for h in hist])
print(f"final best fast_eval = {scores[-1]:.4f}  ({len(evals)} points)")

# ---------- Figure 1: before / after placement ----------
def draw(ax, pos, title):
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
    ax.set_aspect("equal"); ax.set_title(title)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7))
draw(a1, legal, f"{name} initial (legalized)  proxy=1.0387")
draw(a2, best, f"{name} optimized  proxy={scores[-1]:.4f}")
fig.suptitle("Hard macros (colored by area) + soft macros (gray). Greedy + soft co-optimization.",
             fontsize=11)
fig.tight_layout(); fig.savefig(f"{NOTES}/{name}_before_after.png", dpi=100)
print(f"saved {name}_before_after.png")

# ---------- Figure 2: convergence ----------
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(evals, scores, color="#2563eb", lw=2, label="our greedy run (best proxy)")
refs = [
    (1.0387, "#d97706", "initial / legalized reference (1.0387)"),
    (0.9976, "#16a34a", "RePlAce baseline, ibm01 (0.9976)"),
    (1.3166, "#999999", "SA baseline, ibm01 (1.3166)"),
    (0.7253, "#dc2626", "best team single-benchmark (~0.725, bench not disclosed)"),
]
for val, col, lab in refs:
    ax.axhline(val, ls=":", lw=1.5, color=col, label=lab)
ax.set_xlabel("cost evaluations (accepted + rejected moves)")
ax.set_ylabel("proxy cost (lower is better)")
ax.set_title(f"{name}: local-search convergence vs reference lines")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25)
fig2.tight_layout(); fig2.savefig(f"{NOTES}/{name}_convergence.png", dpi=100)
print(f"saved {name}_convergence.png")
