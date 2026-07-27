"""Top-up test on SAVED best candidates, all 17 benchmarks (CPU-only, parallel).

For each benchmark, load the best already-polished placement from the gate-run
artifact dirs and add 100k more greedy moves under three arms:
  a_more   : baseline jitter          (was the original 100k converged?)
  b_ops    : route-shift + flat-span  (do the new operators beat the plateau?)
  c_ops_sa : ops + small SA temperature (does smart-proposal SA escape it?)
Reports per-bench start -> per-arm final. -> notes/repolish.json + stdout
Usage: python repolish_best.py [--moves 100000]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, glob, argparse
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np
import multiprocessing as mp

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
DIRS = ["fullrun_pool5swap", "fullrun_gate114", "gate_window", "gate_221"]
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
ARMS = [("a_more", dict()),
        ("b_ops", dict(opt_frac=0.25, opt_flat=True, route_frac=0.2)),
        ("c_ops_sa", dict(opt_frac=0.25, opt_flat=True, route_frac=0.2, T0=3e-4, Tend=1e-6))]


def best_saved(nm):
    best = (1e9, None)
    for d in DIRS:
        for f in glob.glob(f"{ROOT}/notes/{d}/{nm}_p*_final.json"):
            r = json.load(open(f))
            npz = f.replace("_final.json", "_polished.npz")
            if int(r.get("overlaps", 1)) == 0 and r["final"] < best[0] and os.path.exists(npz):
                best = (r["final"], npz)
    return best


def run_bench(args):
    nm, moves = args
    import torch as T
    T.set_num_threads(1)
    from macro_place.loader import load_benchmark_from_dir
    from placers.fast_eval import FastEval
    from placers.local_search import optimize_fast
    score0, npz = best_saved(nm)
    if npz is None:
        return nm, None
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc)
    start = np.load(npz)["placement"].astype(np.float64)
    v0 = float(fe.wirelength_cost(start) + 0.5 * fe.density_cost(start) + 0.5 * fe.congestion_cost(start))
    out = {"start": v0, "saved_final": score0}
    for lbl, kw in ARMS:
        p, _ = optimize_fast(fe, start, iters=moves, seed=7, move_hard=False, move_soft=True,
                             refresh=20000, log_every=10**9, logf=lambda *x: None, **({"T0": 0.0} | kw))
        out[lbl] = float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))
    return nm, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--moves", type=int, default=100000)
    a = ap.parse_args()
    pool = mp.get_context("spawn").Pool(max(2, min(14, (os.cpu_count() or 4) - 2)))
    res = dict(r for r in pool.map(run_bench, [(nm, a.moves) for nm in ALL]) if r[1] is not None)
    pool.close()
    print(f"{'bench':7s} {'start':>8s} {'a_more':>8s} {'b_ops':>8s} {'c_ops_sa':>9s}")
    for nm in ALL:
        if nm not in res: continue
        r = res[nm]
        print(f"{nm:7s} {r['start']:8.4f} {r['a_more']:8.4f} {r['b_ops']:8.4f} {r['c_ops_sa']:9.4f}")
    agg = {k: float(np.mean([res[nm][k] for nm in res])) for k in ["start", "a_more", "b_ops", "c_ops_sa"]}
    print(f"\nAVG    start={agg['start']:.4f}  a_more={agg['a_more']:.4f} ({agg['a_more']-agg['start']:+.4f})  "
          f"b_ops={agg['b_ops']:.4f} ({agg['b_ops']-agg['start']:+.4f})  "
          f"c_ops_sa={agg['c_ops_sa']:.4f} ({agg['c_ops_sa']-agg['start']:+.4f})")
    json.dump({"per_bench": res, "avg": agg}, open(f"{ROOT}/notes/repolish.json", "w"), indent=1)
    print("saved notes/repolish.json")
