"""Stage 3: greedy detailed-placement polish of the differentiable-placer output.

Warm-starts soft-only greedy local search (move_hard=False, so the diff-placer's
zero-overlap hard placement is preserved) from each benchmark's validation
placement (notes/validation/{nm}.npz), runs to plateau, and records diff-alone
vs diff+greedy proxy. Parallel across benchmarks (CPU, one core each).

Guard: if the polish ever raises hard-overlap above 0, the diff placement is
kept instead -- this stage can only improve or tie the scored proxy.

Usage: python greedy_polish.py [--chunk N] [--time-cap-min M]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import sys, time, json, argparse, multiprocessing as mp
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.local_search import optimize_fast

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
VAL = f"{ROOT}/notes/validation"
OUT = f"{ROOT}/notes/polish"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]


def proxy(pos, b, plc):
    c = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)
    return float(c["proxy_cost"]), int(c["overlap_count"])


def run_one(args):
    nm, chunk, time_cap_s = args
    os.makedirs(OUT, exist_ok=True)
    npz = f"{VAL}/{nm}.npz"
    if not os.path.exists(npz):
        return nm, "no-input"
    try:
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        fe = FastEval(b, plc)
        pos0 = np.load(npz)["placement"].astype(np.float64)
        p_diff, ov0 = proxy(pos0, b, plc)

        best = pos0.copy(); best_proxy = p_diff
        prev = None; flat = 0; done = 0; t0 = time.time()
        while time.time() - t0 < time_cap_s:
            cand, hist = optimize_fast(fe, best, iters=chunk, seed=1000 + done, T0=0.0,
                                       move_hard=False, move_soft=True, refresh=chunk,
                                       log_every=chunk + 1, logf=lambda *a: None)
            done += chunk
            pc, ov = proxy(cand, b, plc)
            if ov == 0 and pc < best_proxy:      # only accept legal improvements
                best, best_proxy = cand, pc
            bc = hist[-1][1]
            if prev is not None and (prev - bc) < 5e-4:
                flat += 1
                if flat >= 2:
                    break
            else:
                flat = 0
            prev = bc

        np.savez(f"{OUT}/{nm}.npz", placement=best)
        p_pol, ovf = proxy(best, b, plc)
        json.dump({"benchmark": nm, "diff_proxy": p_diff, "polished_proxy": p_pol,
                   "gain": p_diff - p_pol, "overlaps": ovf, "iters": done,
                   "seconds": time.time() - t0}, open(f"{OUT}/{nm}.json", "w"))
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: diff={p_diff:.4f} -> polished={p_pol:.4f} "
              f"(gain {p_diff-p_pol:+.4f}) ov={ovf} ({done} iters)", flush=True)
        return nm, "ok"
    except Exception as e:
        print(f"{nm}: ERROR {type(e).__name__}: {e}", flush=True)
        return nm, "error"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--time-cap-min", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=min(len(ALL), max(1, mp.cpu_count() - 2)))
    a = ap.parse_args()
    jobs = [(nm, a.chunk, a.time_cap_min * 60) for nm in ALL]
    print(f"=== greedy polish: {len(jobs)} benches, {a.workers} workers, "
          f"time-cap {a.time_cap_min}min/bench ===", flush=True)
    with mp.Pool(a.workers) as pool:
        for nm, st in pool.imap_unordered(run_one, jobs):
            pass
    print("=== greedy polish finished ===", flush=True)
