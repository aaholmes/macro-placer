"""Bayesian (TPE) hyperparameter search for the differentiable placer.

Goal: find a from-scratch differentiable-placement config (no greedy polish)
whose legalized proxy beats the reference the challenge provides. Searches the
whole space -- density mode, soft-top-k operator, per-term weights and annealing
schedules (so loss-homotopy schedules are explored), gamma/tau, WL schedule
(cluster-first vs spread-first), lr, and the differentiable legalizer.

Objective per trial = mean legalized proxy over a small benchmark set (robust,
avoids overfitting one benchmark). Resumable (SQLite). Runs unattended.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, time, json
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
import optuna
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
BENCHES = ["ibm01", "ibm09", "ibm13"]          # easy/low/mid mix
ENV = {}
REF = {}
for nm in BENCHES:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe)
    sz = b.macro_sizes.numpy()
    ENV[nm] = (b, plc, fe, P, sz)
    lg, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                        b.canvas_width, b.canvas_height, gap=0.01)
    REF[nm] = float(compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)["proxy_cost"])
REF_MEAN = float(np.mean(list(REF.values())))
print(f"reference legalized: {REF}  mean={REF_MEAN:.4f}", flush=True)


def evaluate(nm, hp):
    b, plc, fe, P, sz = ENV[nm]
    loss = proxy_loss(
        gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
        lam_d1=hp["lam_d_end"], lam_c=hp["lam_c"], tau_d=hp["tau_d"], tau_c=hp["tau_c"],
        target=hp["target"], topk=hp["topk"], dmode=hp["dmode"],
        lam_wl0=hp["lam_wl0"], lam_wl1=1.0,
        legalize_steps=hp["lsteps"], lam_disp=hp["lam_disp"])
    out = P.place(loss, pos0=None, iters=hp["iters"], lr=hp["lr"], logf=None)
    if hp["lsteps"]:
        outL = P.legalize_soft(torch.tensor(out[:, :2], dtype=torch.float32, device=P.dev),
                               steps=hp["lsteps"]).detach().cpu().numpy()
        out[:, :2] = outL
    lg, _, resid = legalize(out, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    c = compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)
    return float(c["proxy_cost"]) + (1.0 if c["overlap_count"] else 0.0)


def objective(trial):
    dmode = trial.suggest_categorical("dmode", ["overflow", "electrostatic"])
    # density weight scale differs by ~100x between modes; give each its own range
    if dmode == "electrostatic":
        lam_d_end = trial.suggest_float("lam_d_end_e", 5e-5, 2e-2, log=True)
    else:
        lam_d_end = trial.suggest_float("lam_d_end_o", 5e-3, 0.3, log=True)
    hp = dict(
        dmode=dmode, lam_d_end=lam_d_end,
        topk=trial.suggest_categorical("topk", ["softmax", "lapsum"]),
        lam_d_ratio=trial.suggest_float("lam_d_ratio", 0.02, 1.0, log=True),
        lam_c=(trial.suggest_float("lam_c", 1e-5, 5e-3, log=True)
               if trial.suggest_categorical("use_cong", [0, 1]) else 0.0),
        gamma=trial.suggest_float("gamma", 1.5, 30.0, log=True),
        tau_d=trial.suggest_float("tau_d", 0.01, 0.5, log=True),
        tau_c=trial.suggest_float("tau_c", 0.005, 0.1, log=True),
        lam_wl0=trial.suggest_float("lam_wl0", 0.02, 1.0, log=True),   # <1 = spread-first
        target=trial.suggest_float("target", 0.6, 1.0),
        lr=trial.suggest_float("lr", 0.02, 0.3, log=True),
        iters=trial.suggest_int("iters", 1000, 2000, step=250),
        lsteps=trial.suggest_categorical("lsteps", [0, 8, 15]),
    )
    hp["lam_disp"] = trial.suggest_float("lam_disp", 1e-3, 1.0, log=True) if hp["lsteps"] else 0.0
    try:
        vals = [evaluate(nm, hp) for nm in BENCHES]
    except Exception as e:
        return 10.0
    m = float(np.mean(vals))
    trial.set_user_attr("per_bench", vals)
    return m


if __name__ == "__main__":
    storage = "sqlite:////home/laz/partcl/my-macro-placer/notes/bayes.db"
    study = optuna.create_study(study_name="diffplace", storage=storage,
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=40, seed=0))
    if len(study.trials) == 0:   # anchor with best-known configs (~1.13 region)
        study.enqueue_trial({"dmode": "overflow", "lam_d_end_o": 0.05, "topk": "lapsum",
            "lam_d_ratio": 0.05, "use_cong": 0, "gamma": 10.0, "tau_d": 0.05, "tau_c": 0.02,
            "lam_wl0": 1.0, "target": 0.8, "lr": 0.1, "iters": 1500, "lsteps": 0})
        study.enqueue_trial({"dmode": "electrostatic", "lam_d_end_e": 0.002, "topk": "lapsum",
            "lam_d_ratio": 0.1, "use_cong": 0, "gamma": 10.0, "tau_d": 0.05, "tau_c": 0.02,
            "lam_wl0": 1.0, "target": 0.8, "lr": 0.1, "iters": 1500, "lsteps": 0})
    t0 = time.time(); deadline = t0 + float(os.environ.get("HOURS", "8")) * 3600

    def cb(study, trial):
        if trial.number % 20 == 0:
            b = study.best_trial
            print(f"[{time.strftime('%H:%M:%S')}] trial {trial.number}: "
                  f"best={b.value:.4f} (ref {REF_MEAN:.4f}) params={b.params}", flush=True)
            json.dump({"best": b.value, "ref": REF_MEAN, "params": b.params,
                       "per_bench": b.user_attrs.get("per_bench"), "n": trial.number},
                      open("/home/laz/partcl/my-macro-placer/notes/bayes_best.json", "w"), indent=2)
        if time.time() > deadline:
            study.stop()

    study.optimize(objective, callbacks=[cb], n_jobs=1)
    b = study.best_trial
    print(f"DONE best={b.value:.4f} vs ref {REF_MEAN:.4f}  params={b.params}", flush=True)
