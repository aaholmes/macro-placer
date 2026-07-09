"""Task 2 demo: legalize ibm01, run local search on the true proxy, then certify
the result with the ground-truth TILOS evaluator."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
fe = FastEval(bench, plc)

def tilos(pos):
    c = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), bench, plc)
    return c["proxy_cost"], c["overlap_count"]

# start = legalized reference
base = bench.macro_positions.numpy().astype(np.float64)
legal, _, _ = legalize(base, bench.macro_sizes.numpy(), bench.num_hard_macros,
                       bench.canvas_width, bench.canvas_height, gap=0.01)
t_ref, ov_ref = tilos(base)
t_leg, ov_leg = tilos(legal)
print(f"{name}: TILOS reference   proxy={t_ref:.4f} overlaps={ov_ref}")
print(f"{name}: TILOS legalized   proxy={t_leg:.4f} overlaps={ov_leg}")
print(f"start fast_eval proxy = {fe.wirelength_cost(legal)+0.5*fe.density_cost(legal)+0.5*fe.congestion_cost(legal):.4f}")
print(f"--- local search ({iters} iters) ---")

t0 = time.time()
best, hist = optimize(fe, legal, iters=iters, seed=1, T0=0.004)
dt = time.time() - t0

t_opt, ov_opt = tilos(best)
print(f"--- results ({dt:.0f}s) ---")
print(f"fast_eval best proxy = {hist[-1][1]:.4f}")
print(f"TILOS  optimized proxy = {t_opt:.4f}  overlaps={ov_opt}")
print(f"improvement vs legalized reference: {t_leg:.4f} -> {t_opt:.4f} "
      f"({100*(t_leg-t_opt)/t_leg:+.1f}%)")
np.save(f"/home/laz/partcl/my-macro-placer/notes/{name}_opt.npy", best)
