"""Incremental-eval local search: legalize -> optimize_fast -> certify (TILOS)."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 40000

bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
fe = FastEval(bench, plc)
sz = bench.macro_sizes.numpy()
base = bench.macro_positions.numpy().astype(np.float64)
legal, _, _ = legalize(base, sz, bench.num_hard_macros,
                       bench.canvas_width, bench.canvas_height, gap=0.01)

def tilos(p):
    c = compute_proxy_cost(torch.tensor(p, dtype=torch.float32), bench, plc)
    return c["proxy_cost"], c["overlap_count"]

t0 = time.time()
best, hist = optimize_fast(fe, legal, iters=iters, seed=1, T0=0.0,
                           move_hard=False, move_soft=True)
dt = time.time() - t0
tb, ov = tilos(best)
print(f"--- {name}: {iters} iters in {dt:.0f}s ({1000*dt/iters:.2f} ms/iter) ---")
print(f"fast_eval best = {hist[-1][1]:.4f}   TILOS = {tb:.4f}  overlaps={ov}")
np.save(f"/home/laz/partcl/my-macro-placer/notes/{name}_fast.npy", best)
