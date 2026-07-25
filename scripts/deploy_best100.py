"""Best-of-N deployment run over an arbitrary config pool, FAITHFUL to window/swap/gdiff
(uses bayes_swap.diff_place, so swap configs deploy WITH their swap; window-off configs run
with all mechanisms off). Per-benchmark wall budget = DEADLINE_MIN; within it, cycle
configs x seeds, dispatching each candidate's FULL greedy to a spawn pool asynchronously so
greedy overlaps the GPU diffs across all 17 benchmarks. Keeps the per-benchmark best.

Pool = notes/best100_pool.json : a list of {"_label": str, ...full param dict...}.
-> notes/best100_report.json (+ stdout). Compare after_avg to 1.0155.
Usage: DEADLINE_MIN=60 python deploy_best100.py
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json, time
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import numpy as np, torch, multiprocessing as mp
import bayes_swap as bs
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.analytical import DifferentiablePlacer

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
if os.environ.get("BENCHES"): ALL = os.environ["BENCHES"].split(",")
DEADLINE_MIN = float(os.environ.get("DEADLINE_MIN", "60"))   # GPU-diff budget per benchmark
ITERS = int(os.environ.get("ITERS", "15000"))                # match the 1.0155 deployment
MOVES = int(os.environ.get("MOVES", "100000"))               # full greedy


def main():
    pool = json.load(open(f"{ROOT}/notes/best100_pool.json"))
    dflt = dict(ov_hold=0.0, ov_ramp=0.0, swap_T0=0.0, swap_frac=0.3, gdiff_D0=0.0, gdiff_frac=0.3)
    hps = [{**bs.hp_from({**dflt, **p}), "iters": ITERS} for p in pool]   # window-off configs lack window/mech keys
    labels = [p.get("_label", str(i)) for i, p in enumerate(pool)]
    print(f"pool ({len(hps)} configs): {labels}\nDEADLINE_MIN={DEADLINE_MIN} ITERS={ITERS} MOVES={MOVES}\n", flush=True)
    workers = max(2, min(30, (os.cpu_count() or 4) - 2))
    gp = mp.get_context("spawn").Pool(workers)
    results = {}; wins = {l: 0 for l in labels}
    t0 = time.time()
    for nm in ALL:
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        P = DifferentiablePlacer(FastEval(b, plc))
        deadline = time.time() + DEADLINE_MIN * 60
        futs = []; cnt = 0
        while time.time() < deadline:
            ci = cnt % len(hps); seed = cnt // len(hps)
            try:
                raw = bs.diff_place(P, seed, hps[ci])
            except Exception as e:
                print(f"  {nm} cand{cnt} ({labels[ci]}) diff FAILED: {type(e).__name__}", flush=True); cnt += 1; continue
            futs.append((cnt, ci, gp.apply_async(bs.greedy_worker, ((nm, raw, cnt, MOVES),))))
            cnt += 1
        best = 1e9; bl = None
        for c, ci, f in futs:                        # wait for the pipelined greedy to drain
            try:
                pr = f.get()[2]
            except Exception:
                continue
            if pr < best:
                best, bl = pr, labels[ci]
        results[nm] = best; wins[bl] = wins.get(bl, 0) + 1
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: {cnt} cands, best={best:.4f} (won by {bl}) "
              f"| running avg={np.mean(list(results.values())):.4f}", flush=True)
    avg = float(np.mean(list(results.values())))
    json.dump({"per_bench": results, "after_avg": avg, "pool": labels, "wins": wins,
               "deadline_min": DEADLINE_MIN, "iters": ITERS, "moves": MOVES},
              open(f"{ROOT}/notes/best100_report.json", "w"), indent=2)
    print(f"\nAFTER_AVG = {avg:.4f}   (baseline 1.0155, delta {avg-1.0155:+.4f})", flush=True)
    print(f"per-config wins: {wins}", flush=True)
    print(f"total wall: {(time.time()-t0)/3600:.1f}h", flush=True)
    gp.close()


if __name__ == "__main__":
    main()
