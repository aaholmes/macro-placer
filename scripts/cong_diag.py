"""Diagnose the congestion loss: for one benchmark, sweep tiny lam_c, run ONE
diff placement each, legalize, and report the TRUE proxy breakdown (WL / density
/ congestion) AND the surrogate value at the result. Distinguishes:
  - surrogate DOWN + true-cong UP  => gaming (optimizer exploits the relaxation)
  - both UP / broken               => miscalibrated weight (WL/density wrecked)
  - surrogate DOWN + true-cong DOWN => it works (find the useful weight)
No greedy -- isolates the diff stage (if diff can't lower congestion, greedy won't).
"""
import os, sys, json, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="ibm06")
ap.add_argument("--lam-cs", default="0,0.001,0.003,0.01,0.03,0.1")
ap.add_argument("--iters", type=int, default=15000)
a = ap.parse_args()
nm = a.bench
b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
p = json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"]
lde = p.get("lam_d_end_o")
base = proxy_loss(gamma_hi_mult=p["gamma"], lam_d0=lde * p["lam_d_ratio"], lam_d1=lde, lam_c0=0.0,
                 lam_c1=0.0, tau_d=p["tau_d"], tau_c=p["tau_c"], target=p["target"], topk="lapsum",
                 dmode="overflow", lam_wl0=p["lam_wl0"], ramp_p=p["ramp_p"])
ref = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
               b.canvas_width, b.canvas_height, gap=0.01)[0]
with torch.no_grad():
    norm = max(1e-6, float(P.congestion_lroute(torch.tensor(ref[:, :2], dtype=torch.float32, device=P.dev))))
print(f"{nm}: norm(cong_lroute at ref)={norm:.1f}")
print(f"{'lam_c':>7} {'proxy':>7} {'WL':>6} {'dens':>6} {'cong':>6} {'surrogate':>10}")
for lc in [float(x) for x in a.lam_cs.split(",")]:
    def loss(Pl, coord, t):
        L = base(Pl, coord, t)
        if lc > 0:
            L = L + lc * t * Pl.congestion_lroute(coord) / norm
        return L
    raw = P.place(loss, pos0=None, iters=a.iters, lr=p["lr"], seed=0, logf=None)
    lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    c = compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)
    with torch.no_grad():
        surr = float(P.congestion_lroute(torch.tensor(lg[:, :2], dtype=torch.float32, device=P.dev)))
    print(f"{lc:>7} {float(c['proxy_cost']):>7.3f} {float(c['wirelength_cost']):>6.3f} "
          f"{0.5*float(c['density_cost']):>6.3f} {0.5*float(c['congestion_cost']):>6.3f} {surr:>10.1f}",
          flush=True)
