"""Tune the greedy-stage knobs (SA temperature/cooling, operator fractions, jitter)
on a PAIRED objective: the same fixed legalized-diff starting placements every
trial, polished at deployment depth (200k moves), scored exactly. No GPU, no seed
noise across trials. Starts span THREE diff-config families (window+swap / plain
window-off / high-lr) so per-start results probe diff-greedy correlation: uniform
winners => the stages factor; family-dependent winners => joint tuning is needed.

CPU-parallel (9 jobs/trial). -> notes/greedy_knobs.db, notes/greedy_knobs_best.json
Usage: HOURS=10 python bayes_greedy_knobs.py
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, glob, time
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np
import multiprocessing as mp

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
BENCHES = ["ibm01", "ibm08", "ibm17"]
MOVES = int(os.environ.get("MOVES", "200000"))
SRC = f"{ROOT}/notes/fullrun_pool5swap"          # has {nm}_meta.json with per-candidate config


def pick_starts():
    """Per bench: one legalized diff candidate from each of 3 config families."""
    starts = []
    for nm in BENCHES:
        meta = {e["idx"]: e for e in json.load(open(f"{SRC}/{nm}_meta.json"))}
        fams = {}
        for idx, e in sorted(meta.items()):
            fam = ("winswap" if e["trial"] == -1 else "base")
            fams.setdefault(fam, []).append(idx)
        chosen = []
        if fams.get("winswap"):
            chosen.append(fams["winswap"][0])
        chosen += fams.get("base", [])[:3 - len(chosen)]
        for idx in chosen[:3]:
            f = f"{SRC}/{nm}_p{idx}.npz"
            if os.path.exists(f):
                starts.append((nm, f, "winswap" if meta[idx]["trial"] == -1 else f"cfg{meta[idx]['trial']}"))
    return starts


def polish_job(args):
    nm, npz, kw, moves = args
    import torch as T
    T.set_num_threads(1)
    from macro_place.loader import load_benchmark_from_dir
    from placers.fast_eval import FastEval
    from placers.local_search import optimize_fast
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc)
    start = np.load(npz)["placement"].astype(np.float64)
    p, _ = optimize_fast(fe, start, iters=moves, seed=11, move_hard=False, move_soft=True,
                         refresh=20000, log_every=10**9, logf=lambda *x: None, **kw)
    return float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    starts = pick_starts()
    print(f"fixed starts ({len(starts)}): {[(nm, fam) for nm, _, fam in starts]}", flush=True)
    pool = mp.get_context("spawn").Pool(min(len(starts), max(2, (os.cpu_count() or 4) - 2)))

    def objective(trial):
        T0 = trial.suggest_float("T0", 1e-5, 3e-3, log=True)
        kw = dict(T0=T0,
                  Tend=T0 * trial.suggest_float("tend_ratio", 1e-4, 0.3, log=True),
                  opt_frac=trial.suggest_float("opt_frac", 0.0, 0.5),
                  opt_flat=True,
                  route_frac=trial.suggest_float("route_frac", 0.0, 0.4),
                  jitter_sigma=trial.suggest_float("sigma", 0.4, 2.5, log=True))
        vals = pool.map(polish_job, [(nm, f, kw, MOVES) for nm, f, _ in starts])
        trial.set_user_attr("per_start", {f"{nm}:{fam}": round(v, 5)
                                          for (nm, _, fam), v in zip(starts, vals)})
        return float(np.mean(vals))

    study = optuna.create_study(study_name="greedy_knobs", direction="minimize",
                                storage=f"sqlite:///{ROOT}/notes/greedy_knobs.db",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=16, seed=0,
                                                                    multivariate=True))
    if len(study.trials) == 0:
        study.enqueue_trial(dict(T0=3e-4, tend_ratio=3.3e-3, opt_frac=0.25, route_frac=0.2, sigma=1.0))  # hand config
        study.enqueue_trial(dict(T0=1e-5, tend_ratio=1e-4, opt_frac=0.0, route_frac=0.0, sigma=1.0))     # ~baseline greedy
        study.enqueue_trial(dict(T0=1e-5, tend_ratio=1e-4, opt_frac=0.3, route_frac=0.2, sigma=1.0))     # ops, ~no SA
        study.enqueue_trial(dict(T0=1e-3, tend_ratio=1e-3, opt_frac=0.25, route_frac=0.2, sigma=1.0))    # hotter SA
    deadline = time.time() + float(os.environ.get("HOURS", "10")) * 3600

    def cb(study, trial):
        if trial.number % 5 == 0:
            b = study.best_trial
            print(f"[{time.strftime('%H:%M:%S')}] trial {trial.number}: best={b.value:.5f} "
                  f"T0={b.params['T0']:.1e} opt={b.params['opt_frac']:.2f} route={b.params['route_frac']:.2f} "
                  f"sigma={b.params['sigma']:.2f}", flush=True)
        if time.time() > deadline:
            study.stop()

    study.optimize(objective, callbacks=[cb], n_jobs=1)
    b = study.best_trial
    kw = dict(T0=b.params["T0"], Tend=b.params["T0"] * b.params["tend_ratio"],
              opt_frac=b.params["opt_frac"], opt_flat=True,
              route_frac=b.params["route_frac"], jitter_sigma=b.params["sigma"])
    json.dump({"value": b.value, "trial": b.number, "kw": kw, "per_start": b.user_attrs.get("per_start")},
              open(f"{ROOT}/notes/greedy_knobs_best.json", "w"), indent=1)
    print(f"DONE best={b.value:.5f} kw={kw}", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
