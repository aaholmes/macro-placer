"""Attribution: isolate the SA-swap's contribution to config #94. Same config, two arms
-- swap ON (its tuned swap_T0) vs swap OFF (swap_T0=0) -- N seeds each, best-of-16 +
distribution (mean/std/min) per benchmark. If swap-off is clearly worse, the swap is doing
real work; if ~equal, #94's edge is lr/window (which overlaps the washed #114). -> stdout."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import numpy as np, torch, optuna, multiprocessing as mp
import bayes_swap as bs
from optuna.trial import TrialState
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.analytical import DifferentiablePlacer

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
BENCHES = ["ibm01", "ibm13", "ibm17"]
N = 16
MOVES = 12000                                 # match the finalize's greedy length


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.load_study(study_name="diffplace_swap",
                           storage="sqlite:////home/laz/partcl/my-macro-placer/notes/bayes_swap.db")
    t94 = min([t for t in st.trials if t.number == 94], key=lambda t: t.number)
    base = dict(t94.params)
    print(f"#94 params: swap_T0={base['swap_T0']:.4f} lr={base['lr']:.3f} gamma={base['gamma']:.2f} "
          f"ov_ramp={base['ov_ramp']:.2f}\n", flush=True)
    env = {}; ref = {}
    for nm in BENCHES:
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        P = DifferentiablePlacer(FastEval(b, plc)); sz = b.macro_sizes.numpy()
        env[nm] = P
        lg, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                            b.canvas_width, b.canvas_height, gap=0.01)
        ref[nm] = float(compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)["proxy_cost"])
    pool = mp.get_context("spawn").Pool(max(2, min(14, (os.cpu_count() or 4) - 2)))
    arms = {"swap-ON": bs.hp_from(base), "swap-OFF": bs.hp_from({**base, "swap_T0": 0.0})}
    bestof = {}
    for nm in BENCHES:
        P = env[nm]; print(f"--- {nm} ---", flush=True)
        for label, hp in arms.items():
            jobs = [(nm, bs.diff_place(P, s, hp), s, MOVES) for s in range(N)]
            vals = np.sort(np.array([r[2] for r in pool.map(bs.greedy_worker, jobs)]) / ref[nm])
            bestof.setdefault(label, []).append(vals.min())
            print(f"  {label:9s}: mean={vals.mean():.4f} std={vals.std():.4f} min(best-of-{N})={vals.min():.4f}", flush=True)
    print(f"\nbest-of-{N} ratio-mean over {len(BENCHES)} benches:", flush=True)
    for label in arms:
        print(f"  {label:9s}: {np.mean(bestof[label]):.4f}", flush=True)
    d = np.mean(bestof["swap-ON"]) - np.mean(bestof["swap-OFF"])
    print(f"\nSWAP effect (ON - OFF) = {d:+.4f}  -> "
          + ("swap HELPS (negative)" if d < -0.002 else "swap HURTS" if d > 0.002 else "swap ~INERT (within noise)"),
          flush=True)
    pool.close()


if __name__ == "__main__":
    main()
