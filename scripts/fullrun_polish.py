"""Full-run stage B (CPU, parallel): greedy-polish EVERY candidate placement
(soft-only, preserves hard legality). Selecting the winner on the post-greedy
proxy is the point -- the best diff candidate is not always the best diff+greedy
candidate. Parallel across all (benchmark, candidate) pairs.

Saves notes/fullrun/{nm}_p{idx}_final.json. Checkpointed/resumable.

Usage: python fullrun_polish.py [--time-cap-min 15]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import sys, json, time, glob, re, argparse, multiprocessing as mp
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
torch.set_num_threads(1)
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.local_search import optimize_fast

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
OUT = os.environ.get("FULLRUN_DIR", f"{ROOT}/notes/fullrun")


def run_one(args):
    nm, idx, time_cap_s = args
    fin = f"{OUT}/{nm}_p{idx}_final.json"
    if os.path.exists(fin):
        return
    try:
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        fe = FastEval(b, plc)
        best = np.load(f"{OUT}/{nm}_p{idx}.npz")["placement"].astype(np.float64)
        best_p = float(compute_proxy_cost(torch.tensor(best, dtype=torch.float32), b, plc)["proxy_cost"])
        prev = None; flat = 0; done = 0; t0 = time.time()
        while time.time() - t0 < time_cap_s:
            cand, hist = optimize_fast(fe, best, iters=20000, seed=1000 + done, T0=0.0,
                                       move_hard=False, move_soft=True, refresh=20000,
                                       log_every=20001, logf=lambda *a: None)
            done += 20000
            c = compute_proxy_cost(torch.tensor(cand, dtype=torch.float32), b, plc)
            if int(c["overlap_count"]) == 0 and float(c["proxy_cost"]) < best_p:
                best, best_p = cand, float(c["proxy_cost"])
            bc = hist[-1][1]
            if prev is not None and (prev - bc) < 5e-4:
                flat += 1
                if flat >= 2:
                    break
            else:
                flat = 0
            prev = bc
        np.savez(f"{OUT}/{nm}_p{idx}_polished.npz", placement=best)
        ovf = int(compute_proxy_cost(torch.tensor(best, dtype=torch.float32), b, plc)["overlap_count"])
        json.dump({"benchmark": nm, "idx": idx, "final": best_p, "overlaps": ovf}, open(fin, "w"))
    except Exception as e:
        print(f"{nm} p{idx}: ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-cap-min", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    a = ap.parse_args()
    tasks = []
    for npz in sorted(glob.glob(f"{OUT}/*_p*.npz")):
        base = os.path.basename(npz)
        if "_polished" in base:
            continue
        m = re.match(r"(ibm\d+)_p(\d+)\.npz", base)
        if m:
            tasks.append((m.group(1), int(m.group(2)), a.time_cap_min * 60))
    print(f"=== stage B (polish): {len(tasks)} candidates, {a.workers} workers ===", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for _ in pool.imap_unordered(run_one, tasks):
            pass
    print("=== stage B done ===", flush=True)
