"""Validate incremental single-move eval against full fast_eval recompute, and
time it. Runs a sequence of random moves (soft + hard), applying each to the
incremental state and comparing cost_current() to a fresh full recompute. Also
checks undo restores state exactly."""
import sys, time
import numpy as np

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
fe = FastEval(bench, plc)
nh = bench.num_hard_macros; nm = bench.num_macros
sz = bench.macro_sizes.numpy()
W, H = bench.canvas_width, bench.canvas_height
rng = np.random.default_rng(0)

pos = bench.macro_positions.numpy().astype(np.float64)
inc = fe.init_incremental(pos)

def full(p):
    return fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p)

f0 = full(fe.ipos)
print(f"init: incremental={inc:.8f}  full={f0:.8f}  rel={abs(inc-f0)/f0:.2e}")

# --- correctness over a sequence of accepted moves ---
worst = 0.0
for step in range(60):
    i = int(rng.integers(nm))
    x = float(np.clip(pos[i, 0] + rng.normal(0, 2.0), sz[i, 0] / 2, W - sz[i, 0] / 2))
    y = float(np.clip(pos[i, 1] + rng.normal(0, 2.0), sz[i, 1] / 2, H - sz[i, 1] / 2))
    fe.apply_move(i, x, y)                      # commit into incremental state
    ic = fe.cost_current()
    fc = full(fe.ipos)                          # independent full recompute
    worst = max(worst, abs(ic - fc) / fc)
print(f"after 60 committed moves: worst rel err (incremental vs full) = {worst:.2e}")

# --- undo correctness ---
snap = fe.ipos.copy(); c_before = fe.cost_current()
u = fe.apply_move(5, 3.0, 3.0)
fe.undo_move(u)
c_after = fe.cost_current()
print(f"undo: pos restored={np.allclose(snap, fe.ipos)}  "
      f"cost restored rel={abs(c_after-c_before)/c_before:.2e}")

# --- timing: incremental try+undo vs full recompute ---
fe.init_incremental(pos)
N = 2000
t = time.time()
for _ in range(N):
    i = int(rng.integers(nm))
    x = float(np.clip(pos[i, 0] + rng.normal(0, 2.0), sz[i, 0] / 2, W - sz[i, 0] / 2))
    y = float(np.clip(pos[i, 1] + rng.normal(0, 2.0), sz[i, 1] / 2, H - sz[i, 1] / 2))
    u = fe.apply_move(i, x, y)
    _ = fe.cost_current()
    fe.undo_move(u)                              # reject (typical case)
inc_ms = (time.time() - t) / N * 1000
t = time.time()
for _ in range(20):
    full(pos)
full_ms = (time.time() - t) / 20 * 1000
print(f"incremental try+cost+undo: {inc_ms:.3f} ms/move")
print(f"full recompute:            {full_ms:.1f} ms/move   -> speedup {full_ms/inc_ms:.0f}x")
