"""Full-run stage A (GPU): place many candidates per benchmark, to be selected
AFTER greedy (stage B/C). Candidates come from a pool of the top-K Bayes configs
x many random seeds. Each benchmark gets an equal wall-time slice, so it produces
as many candidates as its size allows within the budget.

No selection here. Saves notes/fullrun/{nm}_p{idx}.npz (legalized placement) and
notes/fullrun/{nm}_meta.json ([{idx, trial, config_i, seed, diff}]). Checkpointed
and resumable (re-run continues adding candidates within the remaining slice).

Usage: python fullrun_place.py --iters 15000 --topk 5 --deadline-min 300
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json, time, argparse
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
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
                ramp_p=p["ramp_p"], lsteps=p["lsteps_n"], iters=p["iters"])


def build_loss(hp):
    return proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"],
                      tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"], topk="lapsum",
                      dmode=hp["dmode"], lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
                      legalize_steps=hp["lsteps"])


ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=0, help="0 = use each config's own tuned iters")
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--configs", default="",
                help="explicit comma-sep trial numbers (overrides --topk); default uses "
                     "the finalize-verified set, NOT top-k by search value (which excludes "
                     "the multistart winner trial 221 -> the 1.0455 regression)")
ap.add_argument("--deadline-min", type=float, required=True)
ap.add_argument("--benches", default=",".join(ALL))
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

study = optuna.load_study(study_name="diffplace",
                          storage="sqlite:////home/laz/partcl/my-macro-placer/notes/bayes.db")
# Config selection: finalize-verified configs by DEFAULT (best-of-16 multistart ranking),
# not top-k by raw search value. Search-value top-k excludes trial 221/222 (the actual
# multistart winners) and caused the earlier 1.0455 vs 1.0376 regression.
FINALIZE = [221, 97, 95, 222, 87]
byn = {t.number: t for t in study.trials}
if a.configs:
    ids = [int(x) for x in a.configs.split(",")]
elif FINALIZE and all(n in byn for n in FINALIZE):
    ids = FINALIZE[:a.topk]
else:
    ids = [t.number for t in sorted([t for t in study.trials if t.value is not None],
                                    key=lambda t: t.value)[:a.topk]]
top = [byn[n] for n in ids]
configs = [(t.number, hp_from_params(t.params), build_loss(hp_from_params(t.params))) for t in top]
benches = a.benches.split(",")
per_bench = a.deadline_min * 60 / len(benches)
print(f"=== stage A: iters={a.iters or 'per-config'} topk={a.topk} "
      f"(trials {[t.number for t in top]}, their iters {[c[1]['iters'] for c in configs]}) "
      f"per-bench {per_bench/60:.0f}min ===", flush=True)

for nm in benches:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    mj = f"{OUT}/{nm}_meta.json"
    meta = json.load(open(mj)) if os.path.exists(mj) else []
    cnt = len(meta)
    bdl = time.time() + per_bench
    while time.time() < bdl:
        ci = cnt % len(configs); seed = cnt // len(configs)
        trial, hp, loss = configs[ci]
        it = a.iters if a.iters else hp["iters"]
        raw = P.place(loss, pos0=None, iters=it, lr=hp["lr"], seed=seed * 101 + ci, logf=None)
        if hp["lsteps"]:
            raw[:, :2] = P.legalize_soft(torch.tensor(raw[:, :2], dtype=torch.float32, device=P.dev),
                                         steps=hp["lsteps"]).detach().cpu().numpy()
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        np.savez(f"{OUT}/{nm}_p{cnt}.npz", placement=lg)
        pd = float(compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)["proxy_cost"])
        meta.append({"idx": cnt, "trial": trial, "config_i": ci, "seed": seed, "diff": pd})
        json.dump(meta, open(mj, "w"))
        cnt += 1
    print(f"[{time.strftime('%H:%M:%S')}] {nm}: {cnt} candidates placed "
          f"(best diff {min(m['diff'] for m in meta):.4f})", flush=True)
print("=== stage A done ===", flush=True)
