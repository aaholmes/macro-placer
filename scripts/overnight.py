"""Checkpointed overnight sweep of all 17 IBM benchmarks (soft-only greedy).

Resumable: each benchmark's checkpoint stores its best positions + iters_done +
elapsed. On restart we reload the positions and CONTINUE (init_incremental
rebuilds all live state from positions). Raising --iter-cap in a later block
simply extends benchmarks that hit the previous cap.

Usage:
    python overnight.py [--iter-cap N] [--time-cap-min M] [--deadline-hours H]
    python overnight.py --benches ibm01,ibm05     # subset
Progress is written continuously to notes/overnight_results.csv and
notes/overnight.log; checkpoints to notes/checkpoints/<name>.npz.
"""
import sys, os, time, argparse
import numpy as np
import torch

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
CKPT = f"{ROOT}/notes/checkpoints"
RESULTS = f"{ROOT}/notes/overnight_results.csv"
LOG = f"{ROOT}/notes/overnight.log"
ALL = [f"ibm{n:02d}" for n in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]

ap = argparse.ArgumentParser()
ap.add_argument("--iter-cap", type=int, default=400000)
ap.add_argument("--time-cap-min", type=float, default=30.0)
ap.add_argument("--deadline-hours", type=float, default=11.5)
ap.add_argument("--chunk", type=int, default=20000)
ap.add_argument("--benches", type=str, default=",".join(ALL))
args = ap.parse_args()
BENCHES = args.benches.split(",")
os.makedirs(CKPT, exist_ok=True)
DEADLINE = time.time() + args.deadline_hours * 3600


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def write_result(name, iters, fast_best, tilos_cost, overlaps, elapsed):
    header = not os.path.exists(RESULTS)
    # rewrite the row for this benchmark (keep latest)
    rows = {}
    if os.path.exists(RESULTS):
        for ln in open(RESULTS).read().splitlines()[1:]:
            if ln.strip():
                rows[ln.split(",")[0]] = ln
    rows[name] = f"{name},{iters},{fast_best:.4f},{tilos_cost:.4f},{overlaps},{elapsed:.0f}"
    with open(RESULTS, "w") as f:
        f.write("benchmark,iters,fast_best,tilos,overlaps,seconds\n")
        for n in ALL:
            if n in rows:
                f.write(rows[n] + "\n")


def run_benchmark(name):
    path = f"{CKPT}/{name}.npz"
    bench, plc = load_benchmark_from_dir(
        f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
    fe = FastEval(bench, plc)
    sz = bench.macro_sizes.numpy()

    if os.path.exists(path):
        d = np.load(path)
        pos = d["best_pos"].astype(np.float64)
        iters_done = int(d["iters_done"]); elapsed = float(d["elapsed"])
        log(f"{name}: resume from {iters_done} iters, best={float(d['best_cost']):.4f}")
    else:
        pos, _, _ = legalize(bench.macro_positions.numpy().astype(np.float64), sz,
                             bench.num_hard_macros, bench.canvas_width,
                             bench.canvas_height, gap=0.01)
        iters_done = 0; elapsed = 0.0
        log(f"{name}: fresh start from legalized reference")

    bench_start = time.time()
    while iters_done < args.iter_cap and elapsed < args.time_cap_min * 60:
        if time.time() > DEADLINE:
            log("global deadline reached — stopping"); return False
        chunk = min(args.chunk, args.iter_cap - iters_done)
        best, hist = optimize_fast(fe, pos, iters=chunk, seed=1000 + iters_done,
                                   T0=0.0, move_hard=False, move_soft=True,
                                   refresh=chunk, log_every=chunk + 1,
                                   logf=lambda *a: None)
        pos = best
        iters_done += chunk
        elapsed += time.time() - bench_start; bench_start = time.time()
        best_cost = hist[-1][1]
        np.savez(path, best_pos=pos, iters_done=iters_done, elapsed=elapsed,
                 best_cost=best_cost)
        c = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), bench, plc)
        write_result(name, iters_done, best_cost, c["proxy_cost"], c["overlap_count"], elapsed)
        log(f"{name}: {iters_done} iters, fast={best_cost:.4f} "
            f"tilos={c['proxy_cost']:.4f} ov={c['overlap_count']} ({elapsed:.0f}s)")
    log(f"{name}: DONE ({iters_done} iters, {elapsed:.0f}s)")
    return True


log(f"=== overnight sweep start: {len(BENCHES)} benches, "
    f"iter_cap={args.iter_cap}, time_cap={args.time_cap_min}min, "
    f"deadline={args.deadline_hours}h ===")
for name in BENCHES:
    if time.time() > DEADLINE:
        log("deadline reached before " + name); break
    try:
        if not run_benchmark(name):
            break
    except Exception as e:
        log(f"{name}: ERROR {type(e).__name__}: {e}")
log("=== sweep finished ===")
