"""Legalize the initial reference placement and report proxy cost before/after,
across all 17 IBM benchmarks. This turns the illegal-but-cheap reference into a
legal, scoreable baseline."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from placers.legalizer import legalize
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
BENCHES = [f"ibm{n:02d}" for n in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
only = sys.argv[1:] or BENCHES

print(f"{'bench':>7} | {'init_cost':>9} {'init_ov':>7} | {'legal_cost':>10} {'legal_ov':>8} "
      f"{'d_cost':>7} {'iters':>5} {'sec':>5}")
print("-" * 78)
tot_init = tot_legal = 0.0
for name in only:
    bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
    init = bench.macro_positions.clone()
    ci = compute_proxy_cost(init, bench, plc)

    t0 = time.time()
    legal_np, iters, resid = legalize(
        init.numpy(), bench.macro_sizes.numpy(), bench.num_hard_macros,
        bench.canvas_width, bench.canvas_height, gap=0.01)
    dt = time.time() - t0
    legal = torch.tensor(legal_np, dtype=torch.float32)
    cl = compute_proxy_cost(legal, bench, plc)

    tot_init += ci['proxy_cost']; tot_legal += cl['proxy_cost']
    print(f"{name:>7} | {ci['proxy_cost']:9.4f} {ci['overlap_count']:7d} | "
          f"{cl['proxy_cost']:10.4f} {cl['overlap_count']:8d} "
          f"{cl['proxy_cost']-ci['proxy_cost']:+7.4f} {iters:5d} {dt:5.1f}")
print("-" * 78)
n = len(only)
print(f"{'AVG':>7} | {tot_init/n:9.4f} {'':7} | {tot_legal/n:10.4f}")
