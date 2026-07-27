"""Corrected operator A/B: FIXED sigma everywhere (round 1 showed adaptive sigma
hurts and confounded the operators). Matched wall-time, paired on shared candidates.
  A  baseline           : jitter only (records wall time T)
  D2 optmove            : 30% closed-form optimal-location proposals
  E  optflat            : 30% samples inside the WL-flat span (free congestion search)
  F  routeshift         : 20% surgical route-shift off hot rows/columns
  G  optflat+routeshift : both (25% / 20%)
Usage: python ab_greedy_ops2.py [ibm01 ibm17] [--k 6]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json, time, argparse
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import numpy as np
import bayes_swap as bs
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
ap = argparse.ArgumentParser()
ap.add_argument("benches", nargs="*", default=["ibm01", "ibm17"])
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--moves", type=int, default=100000)
a = ap.parse_args()

pool = json.load(open(f"{ROOT}/notes/best100_pool.json"))
cfg = next(p for p in pool if p["_label"] == "t94_win+swap")
dflt = dict(ov_hold=0.0, ov_ramp=0.0, swap_T0=0.0, swap_frac=0.3, gdiff_D0=0.0, gdiff_frac=0.3)
hp = {**bs.hp_from({**dflt, **cfg}), "iters": 15000}

ARMS = [("D2_optmove", dict(opt_frac=0.3)),
        ("E_optflat", dict(opt_frac=0.3, opt_flat=True)),
        ("F_routeshift", dict(route_frac=0.2)),
        ("G_flat+route", dict(opt_frac=0.25, opt_flat=True, route_frac=0.2))]


def proxy(fe, p):
    return float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))


summary = {}
for nm in a.benches:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    res = {"A": []}; res.update({lbl: [] for lbl, _ in ARMS})
    for s in range(a.k):
        raw = bs.diff_place(P, s, hp)
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        t0 = time.perf_counter()
        pa, _ = optimize_fast(fe, lg, iters=a.moves, seed=s, T0=0.0, move_hard=False, move_soft=True,
                              refresh=20000, log_every=10**9, logf=lambda *x: None)
        T = time.perf_counter() - t0
        va = proxy(fe, pa); res["A"].append(va)
        line = f"{nm} cand{s}: A={va:.4f} "
        for lbl, kw in ARMS:
            p_, _ = optimize_fast(fe, lg, iters=10**9, seed=s, T0=0.0, move_hard=False, move_soft=True,
                                  refresh=20000, log_every=10**9, logf=lambda *x: None,
                                  time_budget_s=T, **kw)
            v = proxy(fe, p_); res[lbl].append(v)
            line += f" {lbl}={v - va:+.4f}"
        print(line + f"  [T={T:.0f}s]", flush=True)
    A = np.array(res["A"]); summary[nm] = res
    parts = [f"{lbl}: {np.mean(np.array(res[lbl]) - A):+.4f} ({(np.array(res[lbl]) < A).sum()}/{a.k})"
             for lbl, _ in ARMS]
    print(f"== {nm} (mean A={A.mean():.4f}):  " + "  ".join(parts) + "\n", flush=True)

print("=== overall paired deltas vs baseline (negative = better) ===", flush=True)
for lbl, _ in ARMS:
    d = np.concatenate([np.array(summary[nm][lbl]) - np.array(summary[nm]["A"]) for nm in a.benches])
    print(f"  {lbl:14s}: mean {d.mean():+.4f}  win {(d < 0).sum()}/{len(d)}", flush=True)
