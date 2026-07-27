"""A/B/C the new greedy operators at MATCHED WALL-TIME, paired on shared candidates.

For each benchmark: generate K diff placements with the deployed winner config
(window+swap), legalize once, then polish each with
  A baseline   : fixed jitter sigma=1.0, single moves, 100k moves (records its wall time T)
  B adapt      : adaptive canvas-scaled sigma, time budget T
  C adapt+group: adaptive sigma + 15% coordinated k-NN group moves, time budget T
All arms use the (new, exactness-verified) fast evaluator, so the only variables are
the proposal operators. Reports per-candidate paired deltas vs A. -> stdout
Usage: python ab_greedy_ops.py [ibm01 ibm17] [--k 6]
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

def proxy(fe, p):
    return float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))

summary = {}
for nm in a.benches:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    res = {"A": [], "B": [], "C": [], "D": []}
    for s in range(a.k):
        raw = bs.diff_place(P, s, hp)
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        t0 = time.perf_counter()
        pa, _ = optimize_fast(fe, lg, iters=a.moves, seed=s, T0=0.0, move_hard=False, move_soft=True,
                              refresh=20000, log_every=10**9, logf=lambda *x: None)
        T = time.perf_counter() - t0
        va = proxy(fe, pa)
        pb, _ = optimize_fast(fe, lg, iters=10**9, seed=s, T0=0.0, move_hard=False, move_soft=True,
                              refresh=20000, log_every=10**9, logf=lambda *x: None,
                              adapt_sigma=True, time_budget_s=T)
        vb = proxy(fe, pb)
        pc, _ = optimize_fast(fe, lg, iters=10**9, seed=s, T0=0.0, move_hard=False, move_soft=True,
                              refresh=20000, log_every=10**9, logf=lambda *x: None,
                              adapt_sigma=True, group_frac=0.15, group_k=8, time_budget_s=T)
        vc = proxy(fe, pc)
        pd, _ = optimize_fast(fe, lg, iters=10**9, seed=s, T0=0.0, move_hard=False, move_soft=True,
                              refresh=20000, log_every=10**9, logf=lambda *x: None,
                              adapt_sigma=True, opt_frac=0.3, time_budget_s=T)
        vd = proxy(fe, pd)
        res["A"].append(va); res["B"].append(vb); res["C"].append(vc); res["D"].append(vd)
        print(f"{nm} cand{s}: A={va:.4f}  B(adapt)={vb-va:+.4f}  C(+group)={vc-va:+.4f}  "
              f"D(+optmove)={vd-va:+.4f}   [T={T:.0f}s]", flush=True)
    A = np.array(res["A"]); B = np.array(res["B"]); C = np.array(res["C"]); D = np.array(res["D"])
    summary[nm] = (A, B, C, D)
    print(f"== {nm}: mean A={A.mean():.4f}  B-A={np.mean(B-A):+.4f} ({(B<A).sum()}/{a.k})  "
          f"C-A={np.mean(C-A):+.4f} ({(C<A).sum()}/{a.k})  D-A={np.mean(D-A):+.4f} ({(D<A).sum()}/{a.k})\n", flush=True)

print("=== overall paired deltas (negative = new operator better) ===", flush=True)
for arm, lbl in [("B", "adaptive sigma"), ("C", "adapt+group"), ("D", "adapt+optmove")]:
    d = np.concatenate([summary[nm][{"B": 1, "C": 2, "D": 3}[arm]] - summary[nm][0] for nm in a.benches])
    print(f"  {lbl:15s}: mean {d.mean():+.4f}  win {(d<0).sum()}/{len(d)}", flush=True)
