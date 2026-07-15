"""Isolation test (the gate): does a congestion-aware diff give greedy a LOWER
FINAL congestion/proxy? PAIRED design -- for each seed, run plain-diff and
congestion-diff from the SAME init, then greedy from each with the SAME greedy
seed. Only difference is the congestion term, so seed noise cancels and a small
real effect is detectable.

Usage: python isolation_test.py [ibm17 ibm08 ...] [--n 8] [--lam-c 0.003] [--log]
"""
import os, sys, json, time, argparse
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
ap.add_argument("benches", nargs="*", default=["ibm17", "ibm08"])
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--lam-c", type=float, default=0.003)
ap.add_argument("--log", action="store_true")
ap.add_argument("--iters", type=int, default=15000)
a = ap.parse_args()
P0 = json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"]
lde = P0.get("lam_d_end_o")


def greedy(fe, pos, seed):
    return optimize_fast(fe, pos, iters=100000, seed=seed, T0=0.0, move_hard=False,
                         move_soft=True, refresh=20000, log_every=200000, logf=lambda *x: None)[0]


for nm in (a.benches or ["ibm17", "ibm08"]):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); Pl = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    base = proxy_loss(gamma_hi_mult=P0["gamma"], lam_d0=lde * P0["lam_d_ratio"], lam_d1=lde,
                      lam_c0=0.0, lam_c1=0.0, tau_d=P0["tau_d"], tau_c=P0["tau_c"], target=P0["target"],
                      topk="lapsum", dmode="overflow", lam_wl0=P0["lam_wl0"], ramp_p=P0["ramp_p"])
    ref = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                   b.canvas_width, b.canvas_height, gap=0.01)[0]
    with torch.no_grad():
        norm = max(1e-6, float(Pl.congestion_lroute(torch.tensor(ref[:, :2], dtype=torch.float32, device=Pl.dev))))

    def cong_loss(P, coord, t):
        L = base(P, coord, t)
        cg = P.congestion_lroute(coord)
        term = torch.log(cg + 1.0) if a.log else cg / norm
        return L + a.lam_c * (t ** P0["ramp_p"]) * term

    def final(loss_fn, seed):
        raw = Pl.place(loss_fn, pos0=None, iters=a.iters, lr=P0["lr"], seed=seed, logf=None)
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        pol = greedy(fe, lg, seed)
        c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
        return float(c["proxy_cost"]), 0.5 * float(c["congestion_cost"])

    dpx, dcg = [], []   # paired diffs (cong - plain)
    for s in range(a.n):
        pp, pc = final(base, s)
        cp, cc = final(cong_loss, s)
        dpx.append(cp - pp); dcg.append(cc - pc)
        print(f"  {nm} seed {s}: plain proxy={pp:.4f} cong={pc:.4f} | cong-diff proxy={cp:.4f} cong={cc:.4f} "
              f"| dproxy={cp-pp:+.4f} dcong={cc-pc:+.4f}", flush=True)
    dpx, dcg = np.array(dpx), np.array(dcg)
    print(f"{nm}: PAIRED mean dproxy={dpx.mean():+.4f}+-{dpx.std():.4f} (cong-diff wins {int((dpx<0).sum())}/{a.n}) "
          f"| mean dcong={dcg.mean():+.4f} (wins {int((dcg<0).sum())}/{a.n})", flush=True)
    json.dump({"benchmark": nm, "dproxy": dpx.tolist(), "dcong": dcg.tolist(),
               "mean_dproxy": float(dpx.mean()), "mean_dcong": float(dcg.mean())},
              open(f"{ROOT}/notes/isolation_{nm}.json", "w"))
