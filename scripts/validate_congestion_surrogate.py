"""Validate the L-route congestion surrogate: across many perturbed placements,
does it CORRELATE with the true TILOS congestion metric -- and does it correlate
BETTER than gameable RUDY? Strong positive correlation (and beating RUDY) is the
prerequisite for using it as a loss term.

Usage: python validate_congestion_surrogate.py [ibm01 ibm17 ...]
"""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.analytical import DifferentiablePlacer

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
benches = sys.argv[1:] or ["ibm01", "ibm17"]


def pearson(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1])


for nm in benches:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    src = f"{ROOT}/notes/polish/{nm}.npz"
    base = (np.load(src)["placement"] if os.path.exists(src)
            else legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                          b.canvas_width, b.canvas_height, gap=0.01)[0]).astype(np.float64)
    nh = b.num_hard_macros
    lr_vals, rudy_vals, true_vals = [], [], []
    rng = np.random.default_rng(0)
    for i in range(30):
        pos = base.copy()
        # perturb soft macros (they dominate) by a spread of magnitudes -> varied congestion
        mag = (i / 29.0) * 0.25 * min(b.canvas_width, b.canvas_height)
        pos[nh:, :2] += rng.normal(0, mag, size=pos[nh:, :2].shape)
        pos[nh:, 0] = np.clip(pos[nh:, 0], 0, b.canvas_width)
        pos[nh:, 1] = np.clip(pos[nh:, 1], 0, b.canvas_height)
        coord = torch.tensor(pos[:, :2], dtype=torch.float32, device=P.dev)
        with torch.no_grad():
            lr_vals.append(float(P.congestion_lroute(coord, tau=0.02, blockage_w=1.0)))
            rudy_vals.append(float(P.congestion(coord, tau=0.02, topk="lapsum")))
        true_vals.append(float(compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)["congestion_cost"]))
    c_lr = pearson(lr_vals, true_vals); c_rudy = pearson(rudy_vals, true_vals)
    print(f"{nm}: L-route corr={c_lr:+.3f}  RUDY corr={c_rudy:+.3f}  "
          f"(true cong range {min(true_vals):.3f}..{max(true_vals):.3f})", flush=True)
    json.dump({"benchmark": nm, "lroute_corr": c_lr, "rudy_corr": c_rudy},
              open(f"{ROOT}/notes/cong_validate_{nm}.json", "w"))
