"""Low-congestion-init test (the viable form of the two-phase idea).

Phase 1: density(spread) + congestion, NO wirelength, hard-overlap unpenalized ->
a spread, low-congestion arrangement (density prevents the congestion-only collapse
to a point). Phase 2: our usual tuned schedule (WL + density + overlap), warm-started
from the phase-1 arrangement. Both arms then legalize + greedy with the SAME seed.

Paired question: does starting phase 2 from a low-congestion init give a lower FINAL
proxy than the standard init? Motivated by multi-start showing init matters ~6%, so a
systematically better init could bias that spread favourably. Negative dproxy = win.

Usage: python low_cong_init.py [ibm17 ibm08 ...] [--n 4] [--p1-iters 4000] [--lam-c1 1.0]
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
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
ap = argparse.ArgumentParser()
ap.add_argument("benches", nargs="*", default=["ibm17"])
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--p1-iters", type=int, default=4000)
ap.add_argument("--lam-c1", type=float, default=1.0)   # phase-1 congestion weight (cg/norm ~ O(1))
ap.add_argument("--iters", type=int, default=15000)    # phase-2 iters
a = ap.parse_args()
P0 = json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"]
lde = P0.get("lam_d_end_o")


def greedy(fe, pos, seed):
    return optimize_fast(fe, pos, iters=100000, seed=seed, T0=0.0, move_hard=False,
                         move_soft=True, refresh=20000, log_every=10**9, logf=lambda *x: None)[0]


for nm in (a.benches or ["ibm17"]):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); Pl = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    # phase 2 = current tuned pipeline (WL from t=0, density ramp, no congestion term)
    base = proxy_loss(gamma_hi_mult=P0["gamma"], lam_d0=lde * P0["lam_d_ratio"], lam_d1=lde,
                      lam_c0=0.0, lam_c1=0.0, tau_d=P0["tau_d"], tau_c=P0["tau_c"], target=P0["target"],
                      topk="lapsum", dmode="overflow", lam_wl0=P0["lam_wl0"], ramp_p=P0["ramp_p"])
    ref = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                   b.canvas_width, b.canvas_height, gap=0.01)[0]
    with torch.no_grad():
        norm = max(1e-6, float(Pl.congestion_lroute(torch.tensor(ref[:, :2], dtype=torch.float32, device=Pl.dev))))

    def phase1_loss(P, coord, t):
        # spread + congestion, no WL, no overlap -> low-congestion spread (no collapse)
        L = lde * P.density(coord, tau=P0["tau_d"], target=P0["target"], topk="lapsum", mode="overflow")
        L = L + a.lam_c1 * (P.congestion_lroute(coord) / norm)
        return L

    def final_from(pos0, seed):
        raw = Pl.place(base, pos0=pos0, iters=a.iters, lr=P0["lr"], seed=seed, logf=lambda *x: None)
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        pol = greedy(fe, lg, seed)
        c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
        return float(c["proxy_cost"]), 0.5 * float(c["congestion_cost"])

    dpx, dcg = [], []
    for s in range(a.n):
        sp, sc = final_from(None, s)                                   # standard init
        p1 = Pl.place(phase1_loss, pos0=None, iters=a.p1_iters, lr=P0["lr"], seed=s, logf=lambda *x: None)
        lp, lc = final_from(p1, s)                                     # low-congestion init
        dpx.append(lp - sp); dcg.append(lc - sc)
        print(f"  {nm} seed {s}: std proxy={sp:.4f} cong={sc:.4f} | low-init proxy={lp:.4f} cong={lc:.4f} "
              f"| dproxy={lp-sp:+.4f} dcong={lc-sc:+.4f}", flush=True)
    dpx, dcg = np.array(dpx), np.array(dcg)
    print(f"{nm}: PAIRED mean dproxy={dpx.mean():+.4f}+-{dpx.std():.4f} "
          f"(low-init wins {int((dpx < 0).sum())}/{a.n}) | mean dcong={dcg.mean():+.4f} "
          f"(wins {int((dcg < 0).sum())}/{a.n})", flush=True)
    json.dump({"benchmark": nm, "p1_iters": a.p1_iters, "lam_c1": a.lam_c1,
               "dproxy": dpx.tolist(), "dcong": dcg.tolist(),
               "mean_dproxy": float(dpx.mean()), "mean_dcong": float(dcg.mean())},
              open(f"{ROOT}/notes/low_cong_init_{nm}.json", "w"))
