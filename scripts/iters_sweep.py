"""Step-1 probe: does running the differentiable stage longer than 5000 iters
improve the FINAL (diff+greedy) proxy? The tuned config pinned iters=5000, the
top of the search range, so there may be unrealized headroom.

For each (benchmark, iters), run diff best-of-N -> legalize -> greedy polish and
record diff-alone and diff+greedy proxy. Cheap subset first; informs the full-17
re-run. Results -> notes/iters_sweep.json.

Usage: python iters_sweep.py [--benches ibm01,ibm09,ibm17] [--iters 5000,15000,30000] [--seeds 8]
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
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"


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


def proxy(pos, b, plc):
    c = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)
    return float(c["proxy_cost"]), int(c["overlap_count"])


ap = argparse.ArgumentParser()
ap.add_argument("--benches", default="ibm01,ibm09,ibm17")
ap.add_argument("--iters", default="5000,15000,30000")
ap.add_argument("--seeds", type=int, default=8)
a = ap.parse_args()
hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
iters_list = [int(x) for x in a.iters.split(",")]

rows = []
for nm in a.benches.split(","):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"],
                      tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"], topk="lapsum",
                      dmode=hp["dmode"], lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
                      legalize_steps=hp["lsteps"])

    def score(p):
        lg, _, _ = legalize(p, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        c = compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)
        return float(c["proxy_cost"]) + (1.0 if c["overlap_count"] else 0.0)

    for it in iters_list:
        t0 = time.time()
        best_pos, _ = P.place_multistart(loss, score, n_starts=a.seeds, pos0=None,
                                         iters=it, lr=hp["lr"], logf=None)
        lg, _, _ = legalize(best_pos, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        p_diff, _ = proxy(lg, b, plc)
        pol, _ = optimize_fast(fe, lg, iters=200000, seed=0, T0=0.0, move_hard=False,
                               move_soft=True, refresh=200000, log_every=200001, logf=lambda *a: None)
        p_pol, ovf = proxy(pol, b, plc)
        r = {"benchmark": nm, "iters": it, "diff": p_diff, "diff_greedy": p_pol,
             "overlaps": ovf, "seconds": time.time() - t0}
        rows.append(r)
        json.dump(rows, open(f"{ROOT}/notes/iters_sweep.json", "w"), indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] {nm} iters={it:5d}: diff={p_diff:.4f} "
              f"diff+greedy={p_pol:.4f} ov={ovf} ({time.time()-t0:.0f}s)", flush=True)

print("\n=== summary (diff+greedy by iters) ===", flush=True)
for nm in a.benches.split(","):
    line = f"{nm}: " + "  ".join(f"{it}={next(r['diff_greedy'] for r in rows if r['benchmark']==nm and r['iters']==it):.4f}"
                                 for it in iters_list)
    print(line, flush=True)
