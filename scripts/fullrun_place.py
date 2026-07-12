"""Full-run stage A (GPU): for each benchmark, run N differentiable seeds with
the tuned config, legalize each, and save every seed's legalized placement. No
selection here -- selection happens AFTER greedy (stage B/C), so the winner is
chosen on the true post-greedy proxy, not the pre-greedy surrogate.

Saves notes/fullrun/{nm}_s{k}.npz (placement) + notes/fullrun/{nm}_diff.json
(per-seed diff-alone proxy). Checkpointed/resumable.

Usage: python fullrun_place.py --iters 15000 --seeds 16 [--benches ...]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json, time, argparse
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
OUT = f"{ROOT}/notes/fullrun"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]


def hp_from_params(p):
    dmode = p["dmode"]
    lam_d_end = p.get("lam_d_end_e") if dmode == "electrostatic" else p.get("lam_d_end_o")
    lam_c0 = lam_c1 = 0.0
    if p.get("use_cong"):
        lam_c1 = p["lam_c1"]; lam_c0 = lam_c1 * p["lam_c_startfrac"]
    return dict(dmode=dmode, lam_d_end=lam_d_end, lam_c0=lam_c0, lam_c1=lam_c1,
                lam_d_ratio=p["lam_d_ratio"], gamma=p["gamma"], tau_d=p["tau_d"],
                tau_c=p["tau_c"], lam_wl0=p["lam_wl0"], target=p["target"], lr=p["lr"],
                iters=p["iters"], ramp_p=p["ramp_p"], lsteps=p["lsteps_n"])


ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, required=True)
ap.add_argument("--seeds", type=int, default=16)
ap.add_argument("--benches", default=",".join(ALL))
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
print(f"=== stage A (place): iters={a.iters} seeds={a.seeds} ===", flush=True)

for nm in a.benches.split(","):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    ref_lg, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                            b.canvas_width, b.canvas_height, gap=0.01)
    ref = float(compute_proxy_cost(torch.tensor(ref_lg, dtype=torch.float32), b, plc)["proxy_cost"])
    loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"],
                      tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"], topk="lapsum",
                      dmode=hp["dmode"], lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
                      legalize_steps=hp["lsteps"])
    dj = f"{OUT}/{nm}_diff.json"
    diffs = json.load(open(dj)) if os.path.exists(dj) else {"benchmark": nm, "reference": ref, "seeds": {}}
    for k in range(a.seeds):
        ck = f"{OUT}/{nm}_s{k}.npz"
        if os.path.exists(ck) and str(k) in diffs["seeds"]:
            continue
        t0 = time.time()
        raw = P.place(loss, pos0=None, iters=a.iters, lr=hp["lr"], seed=k, logf=None)
        if hp["lsteps"]:
            raw[:, :2] = P.legalize_soft(torch.tensor(raw[:, :2], dtype=torch.float32, device=P.dev),
                                         steps=hp["lsteps"]).detach().cpu().numpy()
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        np.savez(ck, placement=lg)
        pd = float(compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)["proxy_cost"])
        diffs["seeds"][str(k)] = pd
        json.dump(diffs, open(dj, "w"))
        print(f"[{time.strftime('%H:%M:%S')}] {nm} seed {k}: diff={pd:.4f} ({time.time()-t0:.0f}s)", flush=True)
print("=== stage A done ===", flush=True)
