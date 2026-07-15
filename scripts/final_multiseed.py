"""Final multi-seed best-of-N-AFTER-greedy run.

The one un-exhausted lever: greedy has a ~6% per-seed spread, and best-of-N after
greedy takes the min over that spread, so more seeds systematically lowers the
average. We run the top finalize-verified configs (trials 221/97/95/222/87), each
with N seeds, varying BOTH the diff seed and the greedy seed (the existing run_cell
fixed greedy seed=0 -- that harvested only diff noise, not the noisy stage). Per
benchmark we take the min over all (config, seed), then element-wise min with the
greedy-from-reference floor. Checkpointed per (bench, config, seed) so it resumes.

Usage: python final_multiseed.py [--seeds 12] [--configs 221,97,95,222,87]
       [--iters 15000] [--benches ibm01,...]
"""
import os, sys, json, glob, time, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import numpy as np, torch, optuna
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss
from congestion_experiment import hp_from_params

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
OUT = f"{ROOT}/notes/final_multiseed"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
PRIOR = 1.0376

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--configs", default="221,97,95")
ap.add_argument("--iters", type=int, default=0, help="override diff iters; 0 = use each config's tuned iters")
ap.add_argument("--benches", default=",".join(ALL))
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
benches = a.benches.split(",")
cfg_ids = [int(x) for x in a.configs.split(",")]

# pull the top configs' hyperparameters from the tuning study
study = optuna.load_study(study_name="diffplace", storage=f"sqlite:///{ROOT}/notes/bayes.db")
byn = {t.number: t for t in study.trials}
CONFIGS = {c: hp_from_params(byn[c].params) for c in cfg_ids}

# greedy-from-reference floor (element-wise min guard: never worse than reference-polish)
gref = {}
for f in glob.glob(f"{ROOT}/notes/results/*.json"):
    d = json.load(open(f))
    if d.get("method", "greedy") == "greedy" and d.get("overlaps", 1) == 0:
        gref[d["benchmark"]] = d["tilos"]


_ENV = {}
def env(nm):
    """Load benchmark + build placer IR ONCE per benchmark, reuse across all
    (config, seed) cells -- avoids rebuilding the expensive wirelength/congestion
    IR on every cell."""
    if nm not in _ENV:
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
        _ENV[nm] = (b, plc, fe, P, sz)
    return _ENV[nm]


def one(nm, hp, seed, iters):
    """One diff(seed)->legalize->greedy(seed) chain; both stages use the same seed
    so best-of-N varies diff AND greedy noise together."""
    b, plc, fe, P, sz = env(nm)
    loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=0.0, lam_c1=0.0, tau_d=hp["tau_d"],
                      tau_c=hp["tau_c"], target=hp["target"], topk="lapsum", dmode=hp["dmode"],
                      lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
                      legalize_steps=hp["lsteps"])
    raw = P.place(loss, pos0=None, iters=iters, lr=hp["lr"], seed=seed, logf=lambda *x: None)
    lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    pol, _ = optimize_fast(fe, lg, iters=100000, seed=seed, T0=0.0, move_hard=False,
                           move_soft=True, refresh=20000, log_every=10**9, logf=lambda *x: None)
    c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
    return dict(proxy=float(c["proxy_cost"]), cong=0.5 * float(c["congestion_cost"]),
                overlaps=int(c["overlap_count"]))


def report(tag):
    """Best-of-whatever-is-on-disk so far: per-bench min over all cached cells,
    element-wise min with the greedy-ref floor. Callable after every seed sweep so
    partial runs still yield a full 17-benchmark number."""
    rows = []
    for nm in benches:
        cells = [json.load(open(f)) for f in glob.glob(f"{OUT}/{nm}_c*_s*.json")]
        if not cells:
            continue
        legal = [r for r in cells if r["overlaps"] == 0] or cells
        bc = min(legal, key=lambda r: r["proxy"])
        floor = gref.get(nm)
        final = min([x for x in [bc["proxy"], floor] if x is not None])
        rows.append(dict(benchmark=nm, diff_best=bc["proxy"], best_config=bc["config"],
                         best_seed=bc["seed"], greedy_ref=floor, final=final, n_cells=len(cells)))
    if len(rows) < len(benches):
        print(f"  [{tag}] only {len(rows)}/{len(benches)} benchmarks have cells yet", flush=True)
        return
    avg = float(np.mean([r["final"] for r in rows]))
    avg_do = float(np.mean([r["diff_best"] for r in rows]))
    print(f"\n=== [{tag}] MULTI-SEED best-so-far (configs={cfg_ids}) ===")
    print(f"  diff-pipeline avg:      {avg_do:.4f}")
    print(f"  + greedy-ref floor min: {avg:.4f}")
    print(f"  prior personal best {PRIOR:.4f}  ({100*(PRIOR-avg)/PRIOR:+.1f}% vs prior)\n", flush=True)
    json.dump(dict(configs=cfg_ids, seeds=a.seeds, avg=avg, avg_diffonly=avg_do, prior=PRIOR,
                   rows=rows), open(f"{ROOT}/notes/final_multiseed_report.json", "w"), indent=2)


# seed-OUTER: complete seed 0 across all 17 benchmarks first, then seed 1, ...
# so any interruption still yields a full-coverage (lower-N) best-of number.
for s in range(a.seeds):
    for nm in benches:
        for c in cfg_ids:
            ck = f"{OUT}/{nm}_c{c}_s{s}.json"
            if os.path.exists(ck):
                continue
            t0 = time.time(); r = one(nm, CONFIGS[c], s, a.iters or CONFIGS[c]["iters"])
            r["seconds"] = time.time() - t0; r["config"] = c; r["seed"] = s
            json.dump(r, open(ck, "w"))
            print(f"[{time.strftime('%H:%M:%S')}] {nm} cfg{c} seed{s}: proxy={r['proxy']:.4f} "
                  f"cong={r['cong']:.4f} ov={r['overlaps']} ({r['seconds']:.0f}s)", flush=True)
    report(f"after seed sweep {s} ({(s+1)*len(cfg_ids)} draws/bench)")

print("DONE", flush=True)
