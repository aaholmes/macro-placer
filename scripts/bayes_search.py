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
BENCHES = ["ibm01", "ibm13", "ibm17"]          # easy / mid / hard (congestion-heavy)
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
        lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"],
        tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"], topk="lapsum",
        dmode=hp["dmode"], lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
        legalize_steps=hp["lsteps"], lam_disp=hp["lam_disp"])
    out = P.place(loss, pos0=None, iters=hp["iters"], lr=hp["lr"], logf=None)
    if hp["lsteps"]:
        outL = P.legalize_soft(torch.tensor(out[:, :2], dtype=torch.float32, device=P.dev),
                               steps=hp["lsteps"]).detach().cpu().numpy()
        out[:, :2] = outL
    lg, _, resid = legalize(out, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    c = compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)
    return float(c["proxy_cost"]) + (1.0 if c["overlap_count"] else 0.0)


def hp_from_params(p):
    """Reconstruct the evaluate() hp dict from a stored trial's params."""
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


def evaluate_ms(nm, hp, n_seeds):
    """Best-of-n_seeds multi-start legalized proxy for one benchmark."""
    b, plc, fe, P, sz = ENV[nm]
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

    _, best = P.place_multistart(loss, score, n_starts=n_seeds, pos0=None,
                                 iters=hp["iters"], lr=hp["lr"], logf=None)
    return best


def finalize(study, top_k=5, n_seeds=16):
    """Take the top_k configs, multi-start each (best-of-n_seeds) per benchmark,
    and report the best -- this is where the 'best of random seeds' pays off."""
    comp = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:top_k]
    print(f"\n=== FINALIZE: multi-start (best-of-{n_seeds}) on top {len(comp)} configs ===", flush=True)
    best = None
    for t in comp:
        hp = hp_from_params(t.params)
        vals = [evaluate_ms(nm, hp, n_seeds) for nm in BENCHES]
        ratios = [v / REF[nm] for v, nm in zip(vals, BENCHES)]
        rm = float(np.mean(ratios))
        print(f"  trial {t.number}: search={t.value:.4f} -> multistart ratio-mean={rm:.4f} "
              f"per-bench-ratio={[round(x,3) for x in ratios]}", flush=True)
        if best is None or rm < best["ratio_mean"]:
            best = {"ratio_mean": rm, "per_bench_proxy": vals, "per_bench_ratio": ratios,
                    "params": t.params, "trial": t.number, "n_seeds": n_seeds}
    json.dump({"benches": BENCHES, "reference": REF, **best},
              open("/home/laz/partcl/my-macro-placer/notes/bayes_final.json", "w"), indent=2)
    print(f"FINAL BEST ratio-mean={best['ratio_mean']:.4f} (<1 beats ref) "
          f"per-bench {[round(x,3) for x in best['per_bench_ratio']]}", flush=True)
    return best


def objective(trial):
    dmode = trial.suggest_categorical("dmode", ["overflow", "electrostatic"])
    # density weight scale differs by ~100x between modes; give each its own range
    if dmode == "electrostatic":
        lam_d_end = trial.suggest_float("lam_d_end_e", 5e-5, 2e-2, log=True)
    else:
        lam_d_end = trial.suggest_float("lam_d_end_o", 5e-3, 0.3, log=True)
    # congestion: annealed lam_c0 -> lam_c1 (start can be 0, ramp in late via ramp_p)
    if trial.suggest_categorical("use_cong", [0, 1]):
        lam_c1 = trial.suggest_float("lam_c1", 1e-5, 5e-3, log=True)
        lam_c0 = lam_c1 * trial.suggest_float("lam_c_startfrac", 0.0, 1.0)  # 0 => ramp from 0
    else:
        lam_c0 = lam_c1 = 0.0
    hp = dict(
        dmode=dmode, lam_d_end=lam_d_end, lam_c0=lam_c0, lam_c1=lam_c1,
        lam_d_ratio=trial.suggest_float("lam_d_ratio", 0.02, 1.0, log=True),
        gamma=trial.suggest_float("gamma", 1.5, 30.0, log=True),
        tau_d=trial.suggest_float("tau_d", 0.01, 0.5, log=True),
        tau_c=trial.suggest_float("tau_c", 0.005, 0.1, log=True),
        lam_wl0=trial.suggest_float("lam_wl0", 0.02, 1.0, log=True),   # <1 = spread-first
        target=trial.suggest_float("target", 0.6, 1.0),
        lr=trial.suggest_float("lr", 0.02, 0.3, log=True),
        iters=trial.suggest_int("iters", 1000, 30000, step=1000),
        ramp_p=trial.suggest_float("ramp_p", 0.5, 4.0, log=True),      # >1 = delay ramps
        lsteps=trial.suggest_int("lsteps_n", 0, 16),                   # ordered; 0 = off
    )
    hp["lam_disp"] = trial.suggest_float("lam_disp", 1e-3, 1.0, log=True) if hp["lsteps"] else 0.0
    try:
        vals = [evaluate(nm, hp) for nm in BENCHES]
    except Exception as e:
        return 10.0
    # DIFFICULTY-NORMALIZED objective: mean of ratios ours/reference, so no single
    # (hard, high-magnitude) benchmark dominates. <1.0 => beats the reference.
    ratios = [v / REF[nm] for v, nm in zip(vals, BENCHES)]
    trial.set_user_attr("per_bench", vals)
    trial.set_user_attr("ratios", ratios)
    return float(np.mean(ratios))


if __name__ == "__main__":
    storage = "sqlite:////home/laz/partcl/my-macro-placer/notes/bayes.db"
    study = optuna.create_study(study_name="diffplace", storage=storage,
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=18, seed=0))
    if len(study.trials) == 0:   # anchor with best-known configs (~1.13 region)
        study.enqueue_trial({"dmode": "overflow", "lam_d_end_o": 0.05, "lam_d_ratio": 0.05,
            "use_cong": 0, "gamma": 10.0, "tau_d": 0.05, "tau_c": 0.02, "lam_wl0": 1.0,
            "target": 0.8, "lr": 0.1, "iters": 1500, "ramp_p": 1.0, "lsteps_n": 0})
        study.enqueue_trial({"dmode": "electrostatic", "lam_d_end_e": 0.002, "lam_d_ratio": 0.1,
            "use_cong": 0, "gamma": 10.0, "tau_d": 0.05, "tau_c": 0.02, "lam_wl0": 1.0,
            "target": 0.8, "lr": 0.1, "iters": 1500, "ramp_p": 1.0, "lsteps_n": 0})
    t0 = time.time(); deadline = t0 + float(os.environ.get("HOURS", "8")) * 3600

    def cb(study, trial):
        if trial.number % 10 == 0:
            b = study.best_trial
            print(f"[{time.strftime('%H:%M:%S')}] trial {trial.number}: best ratio-mean="
                  f"{b.value:.4f} (<1 beats ref) per-bench-ratio="
                  f"{[round(x,3) for x in b.user_attrs.get('ratios') or []]}", flush=True)
            json.dump({"best_ratio_mean": b.value, "benches": BENCHES,
                       "per_bench_proxy": b.user_attrs.get("per_bench"),
                       "per_bench_ratio": b.user_attrs.get("ratios"),
                       "reference": REF, "params": b.params, "n": trial.number},
                      open("/home/laz/partcl/my-macro-placer/notes/bayes_best.json", "w"), indent=2)
        if time.time() > deadline:
            study.stop()

    if "--finalize" in sys.argv:          # run multi-start on the existing study only
        finalize(study, top_k=5, n_seeds=16)
    else:
        study.optimize(objective, callbacks=[cb], n_jobs=1)
        b = study.best_trial
        print(f"DONE search best ratio-mean={b.value:.4f}  per-bench {b.user_attrs.get('ratios')}", flush=True)
        if "--finalize-after" in sys.argv:     # opt-in; the full-run does selection now
            finalize(study, top_k=5, n_seeds=16)
