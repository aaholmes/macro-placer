"""Parallel checkpointed sweep: all 17 IBM benchmarks as independent greedy
chains across a worker pool. ~N-x throughput (N = #benchmarks running at once).

- single-threaded numpy per worker (no core oversubscription)
- per-benchmark checkpoint (resume) + per-benchmark result JSON (no shared-file
  races)
- lightweight plateau stop: if a chunk improves < eps for 2 chunks in a row,
  the benchmark is considered converged and frees its worker.

Usage: python parallel_run.py [--iter-cap N] [--time-cap-min M] [--workers W]
Aggregate anytime with: python show_results.py
"""
import os
# must set BEFORE numpy import so each worker uses one core
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import sys, time, json, argparse, multiprocessing as mp
import numpy as np
import torch
torch.set_num_threads(1)

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
CKPT = f"{ROOT}/notes/checkpoints"
RES = f"{ROOT}/notes/results"
ALL = [f"ibm{n:02d}" for n in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]


def run_one(args):
    name, iter_cap, time_cap_s, chunk, plateau_eps = args
    os.makedirs(CKPT, exist_ok=True); os.makedirs(RES, exist_ok=True)
    ck = f"{CKPT}/{name}.npz"; rj = f"{RES}/{name}.json"
    try:
        bench, plc = load_benchmark_from_dir(
            f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
        fe = FastEval(bench, plc); sz = bench.macro_sizes.numpy()
        if os.path.exists(ck):
            d = np.load(ck)
            pos = d["best_pos"].astype(np.float64)
            iters_done = int(d["iters_done"]); elapsed = float(d["elapsed"])
        else:
            pos, _, _ = legalize(bench.macro_positions.numpy().astype(np.float64), sz,
                                 bench.num_hard_macros, bench.canvas_width,
                                 bench.canvas_height, gap=0.01)
            iters_done = 0; elapsed = 0.0

        prev_best = None; flat = 0; t_start = time.time()
        while iters_done < iter_cap and elapsed < time_cap_s:
            best, hist = optimize_fast(fe, pos, iters=chunk, seed=1000 + iters_done,
                                       T0=0.0, move_hard=False, move_soft=True,
                                       refresh=chunk, log_every=chunk + 1,
                                       logf=lambda *a: None)
            pos = best; iters_done += chunk
            elapsed += time.time() - t_start; t_start = time.time()
            bc = hist[-1][1]
            np.savez(ck, best_pos=pos, iters_done=iters_done, elapsed=elapsed, best_cost=bc)
            c = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), bench, plc)
            json.dump({"benchmark": name, "iters": iters_done, "fast": bc,
                       "tilos": float(c["proxy_cost"]), "overlaps": int(c["overlap_count"]),
                       "seconds": elapsed}, open(rj, "w"))
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {iters_done} iters "
                  f"tilos={c['proxy_cost']:.4f} ov={c['overlap_count']} ({elapsed:.0f}s)", flush=True)
            # plateau detection
            if prev_best is not None and (prev_best - bc) < plateau_eps:
                flat += 1
                if flat >= 2:
                    print(f"[{time.strftime('%H:%M:%S')}] {name}: plateaued, stopping", flush=True)
                    break
            else:
                flat = 0
            prev_best = bc
        return name, iters_done, elapsed
    except Exception as e:
        print(f"{name}: ERROR {type(e).__name__}: {e}", flush=True)
        return name, -1, 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-cap", type=int, default=5_000_000)
    ap.add_argument("--time-cap-min", type=float, default=180.0)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--plateau-eps", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=min(len(ALL), max(1, mp.cpu_count() - 2)))
    ap.add_argument("--benches", type=str, default=",".join(ALL))
    a = ap.parse_args()
    benches = a.benches.split(",")
    jobs = [(n, a.iter_cap, a.time_cap_min * 60, a.chunk, a.plateau_eps) for n in benches]
    print(f"=== parallel sweep: {len(benches)} benches, {a.workers} workers, "
          f"iter_cap={a.iter_cap}, time_cap={a.time_cap_min}min ===", flush=True)
    with mp.Pool(a.workers) as pool:
        for name, it, el in pool.imap_unordered(run_one, jobs):
            print(f"[done] {name}: {it} iters, {el:.0f}s", flush=True)
    print("=== parallel sweep finished ===", flush=True)
