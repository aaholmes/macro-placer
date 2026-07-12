"""Held-out validation: take the finalized differentiable-placer config
(notes/bayes_final.json) and evaluate it, unchanged, on all 17 IBM benchmarks
using best-of-N multi-start. Reports the ABSOLUTE average proxy over 17 (the
quantity the leaderboard ranks by) and the per-benchmark ratio vs the reference.

The 3 benchmarks used for tuning (ibm01/13/17) are marked so in-sample vs
held-out performance can be read off directly.

Usage: python validate_all.py [--seeds N] [--config path.json]
Results are checkpointed per-benchmark to notes/validation/ so progress is
visible and the run is resumable.
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
OUT = f"{ROOT}/notes/validation"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
TUNED = {"ibm01", "ibm13", "ibm17"}


def hp_from_params(p):
    """Reconstruct the loss hyperparameters from a stored trial's params."""
    dmode = p["dmode"]
    lam_d_end = p.get("lam_d_end_e") if dmode == "electrostatic" else p.get("lam_d_end_o")
    if p.get("use_cong"):
        lam_c1 = p["lam_c1"]; lam_c0 = lam_c1 * p["lam_c_startfrac"]
    else:
        lam_c0 = lam_c1 = 0.0
    return dict(dmode=dmode, lam_d_end=lam_d_end, lam_c0=lam_c0, lam_c1=lam_c1,
                lam_d_ratio=p["lam_d_ratio"], gamma=p["gamma"], tau_d=p["tau_d"],
                tau_c=p["tau_c"], lam_wl0=p["lam_wl0"], target=p["target"], lr=p["lr"],
                iters=p["iters"], ramp_p=p["ramp_p"], lsteps=p["lsteps_n"],
                lam_disp=p.get("lam_disp", 0.0))


def run_one(nm, hp, n_seeds):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    ref_lg, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                            b.canvas_width, b.canvas_height, gap=0.01)
    ref = float(compute_proxy_cost(torch.tensor(ref_lg, dtype=torch.float32), b, plc)["proxy_cost"])

    loss = proxy_loss(
        gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
        lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"],
        tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"], topk="lapsum",
        dmode=hp["dmode"], lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
        legalize_steps=hp["lsteps"], lam_disp=hp["lam_disp"])

    def score(p):
        pp = p.copy()
        if hp["lsteps"]:
            pp[:, :2] = P.legalize_soft(torch.tensor(p[:, :2], dtype=torch.float32, device=P.dev),
                                        steps=hp["lsteps"]).detach().cpu().numpy()
        lg, _, _ = legalize(pp, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        c = compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)
        return float(c["proxy_cost"]) + (1.0 if c["overlap_count"] else 0.0)

    best_pos, best = P.place_multistart(loss, score, n_starts=n_seeds, pos0=None,
                                        iters=hp["iters"], lr=hp["lr"], logf=None)
    # reconstruct the zero-overlap legalized placement that achieved `best`, so
    # the greedy stage can warm-start from it.
    pp = best_pos.copy()
    if hp["lsteps"]:
        pp[:, :2] = P.legalize_soft(torch.tensor(best_pos[:, :2], dtype=torch.float32, device=P.dev),
                                    steps=hp["lsteps"]).detach().cpu().numpy()
    lg, _, _ = legalize(pp, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    return {"benchmark": nm, "proxy": best, "reference": ref, "ratio": best / ref,
            "tuned": nm in TUNED, "seeds": n_seeds}, lg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--config", type=str, default=f"{ROOT}/notes/bayes_final.json")
    ap.add_argument("--benches", type=str, default=",".join(ALL))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(a.config))
    hp = hp_from_params(cfg["params"])
    print(f"=== held-out validation: best-of-{a.seeds} on {a.benches.count(',')+1} benches ===", flush=True)
    print(f"config: trial {cfg.get('trial')}  params={cfg['params']}", flush=True)

    rows = []
    for nm in a.benches.split(","):
        ck = f"{OUT}/{nm}.json"
        if os.path.exists(ck):
            r = json.load(open(ck)); rows.append(r)
            print(f"[cached] {nm}: proxy={r['proxy']:.4f} ratio={r['ratio']:.3f}", flush=True)
            continue
        t0 = time.time()
        r, placement = run_one(nm, hp, a.seeds)
        np.savez(f"{OUT}/{nm}.npz", placement=placement)   # for the greedy stage
        json.dump(r, open(ck, "w"))
        rows.append(r)
        tag = "TUNED" if r["tuned"] else "heldout"
        print(f"[{time.strftime('%H:%M:%S')}] {nm} ({tag}): proxy={r['proxy']:.4f} "
              f"ref={r['reference']:.4f} ratio={r['ratio']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    held = [r for r in rows if not r["tuned"]]
    abs_all = float(np.mean([r["proxy"] for r in rows]))
    abs_ref = float(np.mean([r["reference"] for r in rows]))
    print("\n=== SUMMARY ===", flush=True)
    print(f"absolute avg proxy over {len(rows)} benches: ours={abs_all:.4f}  reference={abs_ref:.4f}", flush=True)
    print(f"  (leaderboard #1 'Archgen' = 0.9507, #2 = 0.9522, ranked by this absolute avg)", flush=True)
    print(f"mean ratio (ours/ref)  all={np.mean([r['ratio'] for r in rows]):.4f}  "
          f"held-out-only={np.mean([r['ratio'] for r in held]):.4f}", flush=True)
    json.dump({"rows": rows, "abs_avg": abs_all, "abs_ref": abs_ref,
               "mean_ratio_all": float(np.mean([r["ratio"] for r in rows])),
               "mean_ratio_heldout": float(np.mean([r["ratio"] for r in held])),
               "config": cfg["params"]}, open(f"{ROOT}/notes/validate_all.json", "w"), indent=2)
    print("saved notes/validate_all.json", flush=True)
