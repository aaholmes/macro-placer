"""Task 1: verify fast_eval matches TILOS on PERTURBED placements, not just the
initial config. This is the failure mode that matters — bugs that only appear
once macros move would make the optimizer chase phantom gains.

For ibm01 we test several scenarios and report max relative error per term:
  - initial placement
  - small random jitter of hard macros (in-bounds)
  - large random relocation of hard macros (in-bounds, likely overlapping)
  - macros pushed hard against canvas edges
Each scenario is checked against the TILOS ground truth for WL / density /
congestion / combined proxy.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
fe = FastEval(bench, plc)
nh = bench.num_hard_macros
sz = bench.macro_sizes.numpy()
W, H = bench.canvas_width, bench.canvas_height
rng = np.random.default_rng(0)


def clamp(pos):
    p = pos.copy()
    p[:nh, 0] = np.clip(p[:nh, 0], sz[:nh, 0] / 2, W - sz[:nh, 0] / 2)
    p[:nh, 1] = np.clip(p[:nh, 1], sz[:nh, 1] / 2, H - sz[:nh, 1] / 2)
    return p


def gt(pos):
    c = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), bench, plc)
    return c["wirelength_cost"], c["density_cost"], c["congestion_cost"], c["proxy_cost"]


def mine(pos):
    wl = fe.wirelength_cost(pos); d = fe.density_cost(pos); c = fe.congestion_cost(pos)
    return wl, d, c, wl + 0.5 * d + 0.5 * c


base = bench.macro_positions.numpy().astype(np.float64)
scenarios = {"initial": base.copy()}
p = base.copy(); p[:nh] += rng.normal(0, 0.5, (nh, 2)); scenarios["jitter_small"] = clamp(p)
p = base.copy(); p[:nh] += rng.normal(0, 3.0, (nh, 2)); scenarios["jitter_large"] = clamp(p)
p = base.copy()
p[:nh, 0] = rng.uniform(sz[:nh, 0] / 2, W - sz[:nh, 0] / 2)
p[:nh, 1] = rng.uniform(sz[:nh, 1] / 2, H - sz[:nh, 1] / 2)
scenarios["random"] = clamp(p)
p = base.copy(); p[:nh, 0] = sz[:nh, 0] / 2; p[:nh, 1] = sz[:nh, 1] / 2  # all to bottom-left corner
scenarios["edge"] = clamp(p)

print(f"=== {name}: fast_eval vs TILOS (relative error) ===")
print(f"{'scenario':>13} | {'WL':>10} {'density':>10} {'congest':>10} {'proxy':>10}")
print("-" * 62)
worst = 0.0
for sname, pos in scenarios.items():
    g = gt(pos); m = mine(pos)
    rel = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(g, m)]
    worst = max(worst, rel[0], rel[1], max(rel[3], 0))  # WL/density/proxy (skip cong bias)
    print(f"{sname:>13} | {rel[0]:10.2e} {rel[1]:10.2e} {rel[2]:10.2e} {rel[3]:10.2e}")
print("-" * 62)
print(f"worst WL/density/proxy rel err (excl. known congestion bias): {worst:.2e}")
