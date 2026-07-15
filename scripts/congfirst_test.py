"""Congestion-FIRST paired test (the real version of the hypothesis).

Plain arm  = current pipeline: WL from t=0, density ramps, no congestion term.
Cong-first arm = a DIFFERENT curriculum: congestion carries t=0 at strong weight
to set the global topology, then WL and density ramp in *late*. A weak density
floor from the start prevents collapse/overlap blow-up.

Paired: for each seed, both arms share the same diff init seed and the same greedy
seed, so noise cancels and a small real effect is visible. If the cong-first arm
wins the FINAL (post-greedy) proxy, it reached a topology greedy can't get to from
the plain-diff basin -- the one way global congestion could actually pay off.

Usage: python congfirst_test.py [ibm17 ibm08 ...] [--n 8] [--lam-c 0.1]
       [--wl-ton 0.4] [--d-ton 0.15] [--iters 15000]
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
from placers.analytical import DifferentiablePlacer, proxy_loss, anneal

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
ap = argparse.ArgumentParser()
ap.add_argument("benches", nargs="*", default=["ibm17", "ibm08"])
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--lam-c", type=float, default=0.1)      # congestion weight, strong, from t=0
ap.add_argument("--wl-ton", type=float, default=0.4)     # WL entry time (late)
ap.add_argument("--wl-p", type=float, default=2.0)
ap.add_argument("--d-ton", type=float, default=0.15)     # density entry time (after congestion)
ap.add_argument("--d-p", type=float, default=2.0)
ap.add_argument("--d-floor", type=float, default=0.1)    # density floor from t=0 (x lam_d_end), anti-collapse
ap.add_argument("--iters", type=int, default=15000)
a = ap.parse_args()
P0 = json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"]
lde = P0.get("lam_d_end_o")


def ramp(t, ton, p):
    s = max(0.0, (t - ton) / (1.0 - ton + 1e-6))
    return s ** p


def greedy(fe, pos, seed):
    return optimize_fast(fe, pos, iters=100000, seed=seed, T0=0.0, move_hard=False,
                         move_soft=True, refresh=20000, log_every=200000, logf=lambda *x: None)[0]


for nm in (a.benches or ["ibm17", "ibm08"]):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); Pl = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    # plain arm = current pipeline (WL from t=0, no congestion term)
    base = proxy_loss(gamma_hi_mult=P0["gamma"], lam_d0=lde * P0["lam_d_ratio"], lam_d1=lde,
                      lam_c0=0.0, lam_c1=0.0, tau_d=P0["tau_d"], tau_c=P0["tau_c"], target=P0["target"],
                      topk="lapsum", dmode="overflow", lam_wl0=P0["lam_wl0"], ramp_p=P0["ramp_p"])
    ref = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                   b.canvas_width, b.canvas_height, gap=0.01)[0]
    with torch.no_grad():
        norm = max(1e-6, float(Pl.congestion_lroute(torch.tensor(ref[:, :2], dtype=torch.float32, device=Pl.dev))))
    ld_end = lde
    ld_floor = a.d_floor * lde

    def congfirst_loss(P, coord, t):
        te = t ** P0["ramp_p"]
        # congestion: strong, from t=0 -- sets the global topology first
        cg = P.congestion_lroute(coord)
        L = a.lam_c * (cg / norm)
        # density: weak floor from t=0 (anti-collapse), ramps to full late
        d_s = ramp(t, a.d_ton, a.d_p)
        lam_d = ld_floor + (ld_end - ld_floor) * d_s
        L = L + lam_d * P.density(coord, tau=P0["tau_d"], target=P0["target"], topk="lapsum", mode="overflow")
        # wirelength: ramps in from ~0, late; gamma sharpens on te
        wl_s = ramp(t, a.wl_ton, a.wl_p)
        if wl_s > 0:
            gamma = anneal(te, P.gw * P0["gamma"], P.gw, geometric=True)
            L = L + wl_s * P.wirelength(coord, gamma)
        return L

    def final(loss_fn, seed):
        raw = Pl.place(loss_fn, pos0=None, iters=a.iters, lr=P0["lr"], seed=seed, logf=None)
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        pol = greedy(fe, lg, seed)
        c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
        return float(c["proxy_cost"]), 0.5 * float(c["congestion_cost"]), int(c["overlap_count"])

    dpx, dcg = [], []
    for s in range(a.n):
        pp, pc, po = final(base, s)
        cp, cc, co = final(congfirst_loss, s)
        dpx.append(cp - pp); dcg.append(cc - pc)
        print(f"  {nm} seed {s}: plain proxy={pp:.4f} cong={pc:.4f} ov={po} | "
              f"cong-first proxy={cp:.4f} cong={cc:.4f} ov={co} | "
              f"dproxy={cp-pp:+.4f} dcong={cc-pc:+.4f}", flush=True)
    dpx, dcg = np.array(dpx), np.array(dcg)
    print(f"{nm}: PAIRED mean dproxy={dpx.mean():+.4f}+-{dpx.std():.4f} "
          f"(cong-first wins {int((dpx < 0).sum())}/{a.n}) | mean dcong={dcg.mean():+.4f} "
          f"(wins {int((dcg < 0).sum())}/{a.n})", flush=True)
    json.dump({"benchmark": nm, "lam_c": a.lam_c, "wl_ton": a.wl_ton, "d_ton": a.d_ton,
               "dproxy": dpx.tolist(), "dcong": dcg.tolist(),
               "mean_dproxy": float(dpx.mean()), "mean_dcong": float(dcg.mean())},
              open(f"{ROOT}/notes/congfirst_{nm}.json", "w"))
