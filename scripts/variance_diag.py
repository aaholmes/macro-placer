"""Variance diagnostic: does the overlap-window REDUCE seed-to-seed variance of the
final (post-greedy) proxy while leaving best-of-N (the min) unchanged? That signature
means the window is redundant with multi-start+best-of-N (it fixes the mean, not the
tail). If instead the whole distribution barely moves, ordering is just flat.

Baseline (221, window off) vs Window (same 221 cluster + hold=0.24/ramp=0.52) -- identical
except the window -- N seeds each, full pipeline (diff@5000 -> full greedy), on ibm01 & ibm17.
Diffs on GPU; greedy in a spawn pool. Usage: python variance_diag.py
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import numpy as np, torch, multiprocessing as mp
import bayes_window_greedy as wg           # reuse diff_place (windowed) + greedy_worker + hp_from
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.analytical import DifferentiablePlacer

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
BENCHES = ["ibm01", "ibm17"]
N = 16
MOVES = 100000                             # full greedy = deployment fidelity
P0 = json.load(open("/home/laz/partcl/my-macro-placer/notes/bayes_final.json"))["params"]


def hp(window):
    p = dict(ov_hold=0.24 if window else 0.0, ov_ramp=0.52 if window else 0.0, diff_D0=0.0,
             lam_d_end_o=P0["lam_d_end_o"], lam_d_ratio=P0["lam_d_ratio"], ramp_p=P0["ramp_p"],
             target=P0["target"], gamma=P0["gamma"], lam_wl0=P0["lam_wl0"], lr=P0["lr"], tau_d=P0["tau_d"])
    return wg.hp_from(p)


def main():
    env = {}; ref = {}
    for nm in BENCHES:
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        P = DifferentiablePlacer(FastEval(b, plc)); sz = b.macro_sizes.numpy()
        env[nm] = (b, plc, P, sz)
        lg, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                            b.canvas_width, b.canvas_height, gap=0.01)
        ref[nm] = float(compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)["proxy_cost"])
    pool = mp.get_context("spawn").Pool(max(2, min(14, (os.cpu_count() or 4) - 2)))
    print(f"variance diagnostic: N={N} seeds, full greedy ({MOVES}), diff@5000\n", flush=True)
    for nm in BENCHES:
        P = env[nm][2]
        res = {}
        for label, window in [("baseline(win-off)", False), ("window(.24/.52)", True)]:
            h = hp(window)
            jobs = [(nm, wg.diff_place(P, s, h), s, MOVES) for s in range(N)]
            vals = np.array([r[2] for r in pool.map(wg.greedy_worker, jobs)]) / ref[nm]  # ratio vs ref
            res[label] = np.sort(vals)
            print(f"{nm} {label:18s}: mean={vals.mean():.4f} std={vals.std():.4f} "
                  f"min={vals.min():.4f} max={vals.max():.4f}", flush=True)
        a, w = res["baseline(win-off)"], res["window(.24/.52)"]
        print(f"  -> window vs baseline:  d_mean={w.mean()-a.mean():+.4f}  "
              f"d_std={w.std()-a.std():+.4f}  d_min(best-of-N)={w.min()-a.min():+.4f}", flush=True)
        print(f"  baseline sorted: {[round(x,3) for x in a]}", flush=True)
        print(f"  window   sorted: {[round(x,3) for x in w]}\n", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
