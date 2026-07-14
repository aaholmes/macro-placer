"""Decompose the randomness: how much does the FINAL proxy vary across greedy
seeds, holding the diff placement FIXED? Polish one good diff placement N times
with independent greedy seeds; report the spread and the best-of-N gain. This
decides whether sweeping greedy seeds (separately from diff seeds) is worth it.

Usage: python greedy_variance.py [ibm06 ibm17 ...] [--n 8]
"""
import os, sys, glob, json, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
ap = argparse.ArgumentParser()
ap.add_argument("benches", nargs="*", default=["ibm06", "ibm17"])
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()


def proxy(pos, b, plc):
    return float(compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)["proxy_cost"])


for nm in (a.benches or ["ibm06", "ibm17"]):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); sz = b.macro_sizes.numpy()
    # a fixed diff placement (legalized), from the cash-in candidates or validation
    cand = sorted(glob.glob(f"{ROOT}/notes/fullrun/{nm}_p*.npz"))
    cand = [c for c in cand if "_polished" not in c and "_final" not in c]
    src = cand[0] if cand else f"{ROOT}/notes/validation/{nm}.npz"
    start = np.load(src)["placement"].astype(np.float64)
    finals = []
    for s in range(a.n):
        pol, _ = optimize_fast(fe, start, iters=100000, seed=1 + s * 137, T0=0.0, move_hard=False,
                               move_soft=True, refresh=20000, log_every=200000, logf=lambda *x: None)
        finals.append(proxy(pol, b, plc))
    fa = np.array(finals)
    print(f"{nm}: greedy-seed finals (same diff): mean={fa.mean():.4f} std={fa.std():.4f} "
          f"({100*fa.std()/fa.mean():.2f}%) | best={fa.min():.4f} worst={fa.max():.4f} "
          f"| best-of-{a.n} gain vs single={100*(fa[0]-fa.min())/fa[0]:+.2f}%", flush=True)
    json.dump({"benchmark": nm, "finals": finals, "std_pct": float(100 * fa.std() / fa.mean())},
              open(f"{ROOT}/notes/greedy_var_{nm}.json", "w"))
