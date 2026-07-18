"""Measure the per-stage wall-clock split (GPU diff vs CPU greedy polish) for one
candidate, on this machine, so we can scale to the challenge hardware. Uses the
winning config (bayes_final = trial 221). One small and one large benchmark.
"""
import os, sys, json, time
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
P0 = json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"]
lde = P0["lam_d_end_o"]
print(f"torch cuda: {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

for nm in ["ibm01", "ibm17"]:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    loss = proxy_loss(gamma_hi_mult=P0["gamma"], lam_d0=lde * P0["lam_d_ratio"], lam_d1=lde,
                      lam_c0=0.0, lam_c1=0.0, tau_d=P0["tau_d"], tau_c=P0["tau_c"], target=P0["target"],
                      topk="lapsum", dmode="overflow", lam_wl0=P0["lam_wl0"], ramp_p=P0["ramp_p"])
    # GPU diff stage (15k iters, as in the full run)
    t0 = time.time()
    raw = P.place(loss, pos0=None, iters=15000, lr=P0["lr"], seed=0, logf=lambda *x: None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_diff = time.time() - t0
    lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    # CPU greedy polish (100k iters single pass)
    t0 = time.time()
    optimize_fast(fe, lg, iters=100000, seed=0, T0=0.0, move_hard=False, move_soft=True,
                  refresh=20000, log_every=10**9, logf=lambda *x: None)
    t_polish = time.time() - t0
    print(f"{nm}: n_macros={b.num_macros} n_hard={b.num_hard_macros} | "
          f"diff(15k, GPU)={t_diff:.0f}s  polish(100k, CPU)={t_polish:.0f}s  "
          f"total={t_diff + t_polish:.0f}s", flush=True)
print("DONE", flush=True)
