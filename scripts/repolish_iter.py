"""Iterated top-up bank: keep the best-known placement per benchmark in
notes/bank/{nm}.npz and repeatedly top each up with smart-proposal SA greedy,
updating the bank when improved. Stops when a round's average gain < 5e-4 (or
--max-rounds). Bank initialized from deploy_v2 winners + the best saved gate-run
candidates. CPU-only, parallel across benchmarks. -> notes/bank_log.json
Usage: python repolish_iter.py [--moves 200000] [--max-rounds 6]
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
BANK = f"{ROOT}/notes/bank"
DIRS = ["fullrun_pool5swap", "fullrun_gate114", "gate_window", "gate_221"]
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
KW = dict(opt_frac=0.25, opt_flat=True, route_frac=0.2, T0=3e-4, Tend=1e-6)


def _load(nm):
    from macro_place.loader import load_benchmark_from_dir
    from placers.fast_eval import FastEval
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    return b, plc, FastEval(b, plc)


def _score(fe, p):
    return float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))


def init_bank(nm):
    """Best of deploy_v2 winner + saved gate candidates -> bank."""
    import torch as T
    T.set_num_threads(1)
    b, plc, fe = _load(nm)
    cands = []
    v2 = f"{ROOT}/notes/deploy_v2/{nm}_best.npz"
    if os.path.exists(v2):
        cands.append(np.load(v2)["placement"].astype(np.float64))
    best = (1e9, None)
    for d in DIRS:
        for f in glob.glob(f"{ROOT}/notes/{d}/{nm}_p*_final.json"):
            r = json.load(open(f)); npz = f.replace("_final.json", "_polished.npz")
            if int(r.get("overlaps", 1)) == 0 and r["final"] < best[0] and os.path.exists(npz):
                best = (r["final"], npz)
    if best[1]:
        cands.append(np.load(best[1])["placement"].astype(np.float64))
    scored = sorted((( _score(fe, p), p) for p in cands), key=lambda t: t[0])
    v, p = scored[0]
    np.savez(f"{BANK}/{nm}.npz", placement=p)
    return nm, v


def round_bench(args):
    nm, moves, seed = args
    import torch as T
    T.set_num_threads(1)
    from placers.local_search import optimize_fast
    b, plc, fe = _load(nm)
    start = np.load(f"{BANK}/{nm}.npz")["placement"].astype(np.float64)
    v0 = _score(fe, start)
    p, _ = optimize_fast(fe, start, iters=moves, seed=seed, move_hard=False, move_soft=True,
                         refresh=20000, log_every=10**9, logf=lambda *x: None, **KW)
    v1 = _score(fe, p)
    if v1 < v0 - 1e-9:
        np.savez(f"{BANK}/{nm}.npz", placement=p)
        return nm, v0, v1
    return nm, v0, v0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=int, default=200000)
    ap.add_argument("--max-rounds", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(BANK, exist_ok=True)
    pool = mp.get_context("spawn").Pool(max(2, min(14, (os.cpu_count() or 4) - 2)))
    log = {"rounds": []}
    init = dict((nm, v) for nm, v in pool.map(init_bank, ALL))
    avg = float(np.mean(list(init.values())))
    print(f"bank initialized: avg={avg:.4f}", flush=True)
    log["init_avg"] = avg
    for rnd in range(a.max_rounds):
        res = pool.map(round_bench, [(nm, a.moves, 100 + rnd) for nm in ALL])
        new = {nm: v1 for nm, v0, v1 in res}
        navg = float(np.mean(list(new.values())))
        gain = avg - navg
        print(f"round {rnd}: avg {avg:.4f} -> {navg:.4f} (gain {gain:+.4f})", flush=True)
        log["rounds"].append({"avg": navg, "gain": gain})
        avg = navg
        if gain < 5e-4:
            print("converged; stopping", flush=True)
            break
    json.dump(log, open(f"{ROOT}/notes/bank_log.json", "w"), indent=1)
    print(f"FINAL bank avg = {avg:.4f}  (combined-before 1.0037, deploy_v2 1.0123)", flush=True)
    pool.close()
