"""Paired A/B for the congestion-directed detailed placer.

Isolates the DETAILED stage: both arms start from the SAME placement (legalized
reference) and run optimize_fast with the SAME greedy seed -- only cong_directed
differs. So any paired difference is the operator's, with seed noise cancelled.
Baseline = cong_directed=0.0 (current pipeline). Reports paired mean dproxy/dcong
per directed fraction; negative = directed wins.

Usage: python ab_cong_directed.py [ibm17 ibm08 ...] [--n 8] [--fracs 0.3,0.6]
       [--iters 100000]
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

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
ap = argparse.ArgumentParser()
ap.add_argument("benches", nargs="*", default=["ibm17", "ibm08"])
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--fracs", default="0.3,0.6")
ap.add_argument("--iters", type=int, default=100000)
a = ap.parse_args()
fracs = [float(x) for x in a.fracs.split(",")]


def run(fe, start, sz, b, plc, seed, cd):
    pos, _ = optimize_fast(fe, start, iters=a.iters, seed=seed, T0=0.0, move_hard=False,
                           move_soft=True, refresh=20000, log_every=10**9,
                           logf=lambda *x: None, cong_directed=cd)
    c = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)
    return float(c["proxy_cost"]), 0.5 * float(c["congestion_cost"]), int(c["overlap_count"])


for nm in (a.benches or ["ibm17", "ibm08"]):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); sz = b.macro_sizes.numpy()
    start = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                     b.canvas_width, b.canvas_height, gap=0.01)[0]
    res = {f: {"dpx": [], "dcg": []} for f in fracs}
    for s in range(a.n):
        bp, bc, bo = run(fe, start, sz, b, plc, s, 0.0)          # baseline arm
        line = f"  {nm} seed {s}: base proxy={bp:.4f} cong={bc:.4f} ov={bo}"
        for f in fracs:
            dp, dc, do = run(fe, start, sz, b, plc, s, f)
            res[f]["dpx"].append(dp - bp); res[f]["dcg"].append(dc - bc)
            line += f" | cd{f}: proxy={dp:.4f}({dp-bp:+.4f}) cong={dc:.4f}({dc-bc:+.4f}) ov={do}"
        print(line, flush=True)
    print(f"{nm}: baseline vs directed (paired, n={a.n}):", flush=True)
    out = {"benchmark": nm, "n": a.n, "fracs": {}}
    for f in fracs:
        dpx = np.array(res[f]["dpx"]); dcg = np.array(res[f]["dcg"])
        print(f"    cd={f}: mean dproxy={dpx.mean():+.4f}+-{dpx.std():.4f} "
              f"(wins {int((dpx < 0).sum())}/{a.n}) | mean dcong={dcg.mean():+.4f} "
              f"(wins {int((dcg < 0).sum())}/{a.n})", flush=True)
        out["fracs"][str(f)] = {"mean_dproxy": float(dpx.mean()), "mean_dcong": float(dcg.mean()),
                                "dproxy": dpx.tolist(), "dcong": dcg.tolist()}
    json.dump(out, open(f"{ROOT}/notes/ab_cong_directed_{nm}.json", "w"), indent=2)
