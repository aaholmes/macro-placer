"""Explore a single benchmark: macros, sizes, nets, initial placement, cost."""
import sys
import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"

bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")

print(f"=== {bench.name} ===")
print(f"canvas         : {bench.canvas_width:.2f} x {bench.canvas_height:.2f} um")
print(f"grid           : {bench.grid_rows} rows x {bench.grid_cols} cols")
print(f"macros total   : {bench.num_macros}")
print(f"  hard macros  : {bench.num_hard_macros}  (indices [0, {bench.num_hard_macros}))")
print(f"  soft macros  : {bench.num_soft_macros}  (indices [{bench.num_hard_macros}, {bench.num_macros}))")
print(f"fixed macros   : {int(bench.macro_fixed.sum())}")
print(f"nets           : {bench.num_nets}")

sz = bench.macro_sizes
hard = slice(0, bench.num_hard_macros)
soft = slice(bench.num_hard_macros, bench.num_macros)
harea = (sz[hard, 0] * sz[hard, 1])
sarea = (sz[soft, 0] * sz[soft, 1])
canvas_area = bench.canvas_width * bench.canvas_height
print(f"\nhard macro area: min={harea.min():.3f}  max={harea.max():.3f}  "
      f"ratio={harea.max()/harea.min():.1f}x  sum={harea.sum():.1f} um^2")
print(f"hard util      : {100*harea.sum()/canvas_area:.1f}% of canvas")
print(f"soft macro area: sum={sarea.sum():.1f} um^2  ({100*sarea.sum()/canvas_area:.1f}% of canvas)")
print(f"total util     : {100*(harea.sum()+sarea.sum())/canvas_area:.1f}%")

# Cost of the initial (hand-crafted reference) placement
costs = compute_proxy_cost(bench.macro_positions.clone(), bench, plc)
print("\n=== proxy cost of INITIAL placement ===")
for k in ("proxy_cost", "wirelength_cost", "density_cost", "congestion_cost", "overlap_count"):
    print(f"  {k:16s}: {costs[k]}")
