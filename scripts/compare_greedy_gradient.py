"""Compare soft-macro optimization approaches on ibm01 from the same legalized
start, plotting proxy vs wall-clock time:
  - greedy  : single-move local search (optimize_fast, soft-only)
  - gradient: continuous density force field (optimize_gradient)
  - hybrid  : short gradient spread, then greedy
Wall-clock is the fair axis (one greedy step moves 1 macro; one gradient step
moves ~all soft macros)."""
import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.force import optimize_gradient

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0

bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
fe = FastEval(bench, plc); sz = bench.macro_sizes.numpy()
legal, _, _ = legalize(bench.macro_positions.numpy().astype(np.float64), sz,
                       bench.num_hard_macros, bench.canvas_width, bench.canvas_height, gap=0.01)

def greedy_curve(start, budget, t_offset=0.0):
    pos = start.copy(); curve = []; t0 = time.time()
    while time.time() - t0 < budget:
        best, _ = optimize_fast(fe, pos, iters=5000, seed=1, T0=0.0,
                                move_hard=False, refresh=5000, log_every=10**9,
                                logf=lambda *a: None)
        pos = best
        p = fe.wirelength_cost(pos) + 0.5*fe.density_cost(pos) + 0.5*fe.congestion_cost(pos)
        curve.append((t_offset + time.time() - t0, p))
    return pos, curve

print("greedy..."); _, g = greedy_curve(legal, BUDGET)
print("gradient..."); _, ghist = optimize_gradient(fe, legal, iters=400, step0=0.5,
        w_wl=0.0, w_dens=1.0, w_cong=0.0, log_every=5, logf=lambda *a: None)
grad = [(t, p) for t, p in ghist if t <= BUDGET]
print("hybrid..."); gp, ghist2 = optimize_gradient(fe, legal, iters=40, step0=0.5,
        w_wl=0.0, w_dens=1.0, w_cong=0.0, log_every=2, logf=lambda *a: None)
t_grad = ghist2[-1][0]
_, hg = greedy_curve(gp, BUDGET - t_grad, t_offset=t_grad)
hybrid = [(t, p) for t, p in ghist2] + hg

fig, ax = plt.subplots(figsize=(10, 6))
for label, curve, col in [("greedy (single-move)", g, "#2563eb"),
                          ("gradient (density field)", grad, "#d97706"),
                          ("hybrid (gradient->greedy)", hybrid, "#16a34a")]:
    xs = [c[0] for c in curve]; ys = [c[1] for c in curve]
    ax.plot(xs, ys, "-o", ms=3, color=col, label=f"{label}  (final {ys[-1]:.4f})")
ax.set_xlabel("wall-clock time (s)"); ax.set_ylabel("proxy cost")
ax.set_title(f"{name}: soft-macro optimization — greedy vs gradient vs hybrid")
ax.grid(alpha=0.25); ax.legend()
OUT = f"/home/laz/partcl/my-macro-placer/notes/{name}_greedy_vs_gradient.png"
fig.tight_layout(); fig.savefig(OUT, dpi=100)
print("saved", OUT)
for label, curve in [("greedy", g), ("gradient", grad), ("hybrid", hybrid)]:
    print(f"{label:>9}: final {curve[-1][1]:.4f} at {curve[-1][0]:.0f}s")
