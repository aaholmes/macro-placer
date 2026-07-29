"""FINAL-HOUR protocol: one benchmark, ONE process, HARD 60-min wall-clock
end-to-end (bench loading included) on strictly-slower-than-judge hardware, so
the result is a conservative lower bound for the contest budget.

Allocation (all validated levers):
  A GENERATE  (~15 min GPU): candidates from the winning config family
              (window+swap, high-lr, wo87, wo95) + diff-from-reference, each
              dispatched to a CPU screen (60k moves, tuned SA knobs) as it lands.
  B SELECT    : deep-polish (250k) the two best screens in parallel.
  C COMPOUND  (rest of the hour): parallel top-up rounds -- 12 seeded 200k-move
              SA replicas of the current best across the cores, keep the min,
              repeat until the wall (with a safety margin from the measured
              round time). GPU keeps streaming screened candidates between
              rounds; any that beats the current best becomes the new base.
-> notes/final_hour/{nm}_best.npz + {nm}.json
Usage: BUDGET_S=3600 python final_hour.py ibm01
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json, time
T_START = time.time()                                  # wall starts BEFORE any loading
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import numpy as np, torch, multiprocessing as mp
import bayes_swap as bs

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
NM = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
BUDGET = float(os.environ.get("BUDGET_S", "3600"))
KW = json.load(open(f"{ROOT}/notes/greedy_knobs_best.json"))["kw"]
GEN_S = min(900.0, 0.25 * BUDGET)                      # phase-A GPU window (scales for smoke runs)
SCREEN, DEEP, ROUND = 60000, 250000, 200000
REPL = 12

_CACHE = {}


def _bench(nm):
    if nm not in _CACHE:
        from macro_place.loader import load_benchmark_from_dir
        from placers.fast_eval import FastEval
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        _CACHE[nm] = (b, plc, FastEval(b, plc), b.macro_sizes.numpy())
    return _CACHE[nm]


def polish_worker(args):
    nm, raw, seed, moves, kw, legal = args
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch as T
    T.set_num_threads(1)
    from placers.legalizer import legalize
    from placers.local_search import optimize_fast
    b, plc, fe, sz = _bench(nm)
    lg = raw if legal else legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)[0]
    p, _ = optimize_fast(fe, lg, iters=moves, seed=seed, move_hard=False, move_soft=True,
                         refresh=20000, log_every=10**9, logf=lambda *x: None, **({"T0": 0.0} | kw))
    v = float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))
    return v, p


def left():
    return BUDGET - (time.time() - T_START)


def main():
    from macro_place.loader import load_benchmark_from_dir
    from placers.fast_eval import FastEval
    from placers.legalizer import legalize
    from placers.analytical import DifferentiablePlacer
    from deploy_v2 import diff_from
    os.makedirs(f"{ROOT}/notes/final_hour", exist_ok=True)
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{NM}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    pool_cfg = json.load(open(f"{ROOT}/notes/best100_pool.json"))
    dflt = dict(ov_hold=0.0, ov_ramp=0.0, swap_T0=0.0, swap_frac=0.3, gdiff_D0=0.0, gdiff_frac=0.3)
    by = {p["_label"]: {**bs.hp_from({**dflt, **p}), "iters": 15000} for p in pool_cfg}
    GEN = [("t94_win+swap", 0), ("ref", 0), ("wo221_hi", 0), ("wo87", 0),
           ("t94_win+swap", 1), ("wo95", 0), ("t94_win+swap", 2), ("wo221_hi", 1)]
    gp = mp.get_context("spawn").Pool(max(2, min(14, (os.cpu_count() or 4) - 2)))
    print(f"{NM}: wall starts, budget {BUDGET:.0f}s (load took {time.time()-T_START:.0f}s)", flush=True)

    # ---- A: generate + screen ----
    ref, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                         b.canvas_width, b.canvas_height, gap=0.01)
    screens = []
    for lbl, seed in GEN:
        if time.time() - T_START > GEN_S or left() < 0.25 * BUDGET:
            break
        raw = diff_from(P, seed, by["t94_win+swap"] if lbl == "ref" else by[lbl],
                        x0=ref if lbl == "ref" else None)
        screens.append((lbl, gp.apply_async(polish_worker, ((NM, raw, seed, SCREEN, KW, False),))))
    scr = []
    for lbl, f in screens:
        v, p = f.get()
        scr.append((v, lbl, p))
        print(f"  screen {lbl}: {v:.4f}  [t={time.time()-T_START:.0f}s]", flush=True)
    scr.sort(key=lambda t: t[0])

    # ---- B: deep-polish top-2 ----
    deep = [gp.apply_async(polish_worker, ((NM, p, 5 + i, DEEP, KW, True),)) for i, (_, _, p) in enumerate(scr[:2])]
    best_v, best_p = 1e9, None
    for f in deep:
        v, p = f.get()
        if v < best_v:
            best_v, best_p = v, p
    print(f"  deep best: {best_v:.4f}  [t={time.time()-T_START:.0f}s]", flush=True)

    # ---- C: parallel top-up rounds until the wall ----
    rnd = 0; round_est = 400.0
    while left() > 1.2 * round_est + 60:
        t0 = time.time()
        futs = [gp.apply_async(polish_worker, ((NM, best_p, 1000 + rnd * REPL + r, ROUND, KW, True),))
                for r in range(REPL)]
        # keep the GPU busy: one fresh screened candidate per round
        lbl, seed = GEN[rnd % len(GEN)][0], 10 + rnd
        raw = diff_from(P, seed, by[lbl] if lbl != "ref" else by["t94_win+swap"], x0=ref if lbl == "ref" else None)
        futs.append(gp.apply_async(polish_worker, ((NM, raw, seed, SCREEN, KW, False),)))
        for f in futs:
            v, p = f.get()
            if v < best_v - 1e-9:
                best_v, best_p = v, p
        round_est = time.time() - t0
        rnd += 1
        print(f"  round {rnd}: best={best_v:.4f}  ({round_est:.0f}s/round, {left():.0f}s left)", flush=True)

    elapsed = time.time() - T_START
    np.savez(f"{ROOT}/notes/final_hour/{NM}_best.npz", placement=best_p)
    json.dump({"bench": NM, "best": best_v, "elapsed_s": elapsed, "rounds": rnd,
               "budget_s": BUDGET}, open(f"{ROOT}/notes/final_hour/{NM}.json", "w"))
    print(f"{NM} FINAL: {best_v:.4f} in {elapsed:.0f}s ({rnd} rounds)", flush=True)
    gp.close()


if __name__ == "__main__":
    main()
