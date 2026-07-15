"""Schedule Bayes (DRAFT -- fire only if the isolation test shows congestion-aware
diff lowers the FINAL proxy). Tunes the *schedule*: for each term (WL, density,
congestion) an entry time, ramp shape, and weights, plus the congestion
formulation. Objective = FINAL (diff+greedy) proxy, mean ratio-to-reference over
the hard benchmarks -- NOT diff-alone, so we don't optimize gains greedy erases.

Congestion entry time (cong_ton) is the key knob: cong_ton~0 = congestion-aware
from the start (shape the global arrangement); cong_ton~0.8 = late nudge.

Usage: HOURS=8 python schedule_bayes.py
"""
import os, sys, json, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss, anneal

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
BENCHES = ["ibm06", "ibm08", "ibm17"]           # congestion-bound
ENV, REF = {}, {}
for nm in BENCHES:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    ref = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                   b.canvas_width, b.canvas_height, gap=0.01)[0]
    with torch.no_grad():
        norm = max(1e-6, float(P.congestion_lroute(torch.tensor(ref[:, :2], dtype=torch.float32, device=P.dev))))
    REF[nm] = float(compute_proxy_cost(torch.tensor(ref, dtype=torch.float32), b, plc)["proxy_cost"])
    ENV[nm] = (b, plc, fe, P, sz, norm)
print(f"reference proxies: {REF}", flush=True)


def ramp(t, ton, p):
    s = max(0.0, (t - ton) / (1.0 - ton + 1e-6))
    return s ** p


def make_loss(hp, norm):
    def loss(P, coord, t):
        te = t ** hp["ramp_p"]
        gamma = anneal(te, P.gw * hp["gamma"], P.gw, geometric=True)
        lam_wl = anneal(te, max(hp["lam_wl0"], 1e-9), 1.0, geometric=True)
        L = lam_wl * P.wirelength(coord, gamma)
        d_s = ramp(t, hp["d_ton"], hp["d_p"])
        lam_d = hp["lam_d0"] + (hp["lam_d_end"] - hp["lam_d0"]) * d_s
        L = L + lam_d * P.density(coord, tau=hp["tau_d"], target=hp["target"], topk="lapsum", mode="overflow")
        if hp["lam_c"] > 0:
            cg = P.congestion_lroute(coord)
            term = torch.log(cg + 1.0) if hp["cong_log"] else cg / norm
            L = L + hp["lam_c"] * ramp(t, hp["cong_ton"], hp["cong_p"]) * term
        return L
    return loss


def evaluate(nm, hp):
    b, plc, fe, P, sz, norm = ENV[nm]
    loss = make_loss(hp, norm)
    raw = P.place(loss, pos0=None, iters=hp["iters"], lr=hp["lr"], seed=0, logf=None)
    lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    pol, _ = optimize_fast(fe, lg, iters=80000, seed=0, T0=0.0, move_hard=False, move_soft=True,
                           refresh=20000, log_every=200000, logf=lambda *x: None)
    c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
    return float(c["proxy_cost"]) + (1.0 if int(c["overlap_count"]) else 0.0)


def objective(trial):
    lam_d_end = trial.suggest_float("lam_d_end", 0.03, 0.15)
    hp = dict(
        ramp_p=trial.suggest_float("ramp_p", 1.5, 4.5),
        gamma=trial.suggest_float("gamma", 2.0, 12.0),
        lam_wl0=trial.suggest_float("lam_wl0", 0.2, 1.0),
        lam_d_end=lam_d_end,
        lam_d0=lam_d_end * trial.suggest_float("lam_d_ratio", 0.02, 0.5),
        d_ton=trial.suggest_float("d_ton", 0.0, 0.5),
        d_p=trial.suggest_float("d_p", 1.0, 4.0),
        tau_d=trial.suggest_float("tau_d", 0.03, 0.2),
        target=trial.suggest_float("target", 0.6, 0.9),
        lam_c=trial.suggest_float("lam_c", 0.0, 0.01),
        cong_ton=trial.suggest_float("cong_ton", 0.0, 0.9),      # KEY: congestion entry time
        cong_p=trial.suggest_float("cong_p", 1.0, 4.0),
        cong_log=trial.suggest_categorical("cong_log", [False, True]),
        lr=trial.suggest_float("lr", 0.05, 0.3),
        iters=trial.suggest_int("iters", 10000, 20000, step=1000),
    )
    try:
        vals = [evaluate(nm, hp) for nm in BENCHES]
    except Exception:
        return 10.0
    ratios = [v / REF[nm] for v, nm in zip(vals, BENCHES)]
    trial.set_user_attr("ratios", ratios)
    return float(np.mean(ratios))


if __name__ == "__main__":
    storage = "sqlite:////home/laz/partcl/my-macro-placer/notes/schedule_bayes.db"
    study = optuna.create_study(study_name="schedule", storage=storage, direction="minimize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(n_startup_trials=15, seed=0))
    if len(study.trials) == 0:
        # anchor 1: current best (te-ramp congestion late, off by default)
        study.enqueue_trial(dict(ramp_p=3.8, gamma=9.7, lam_wl0=0.58, lam_d_end=0.078, lam_d_ratio=0.04,
                                 d_ton=0.0, d_p=3.8, tau_d=0.11, target=0.71, lam_c=0.003, cong_ton=0.5,
                                 cong_p=3.8, cong_log=False, lr=0.27, iters=15000))
        # anchor 2: congestion + WL EARLY, density late (the hypothesis)
        study.enqueue_trial(dict(ramp_p=3.8, gamma=9.7, lam_wl0=0.9, lam_d_end=0.078, lam_d_ratio=0.04,
                                 d_ton=0.3, d_p=3.0, tau_d=0.11, target=0.71, lam_c=0.003, cong_ton=0.0,
                                 cong_p=2.0, cong_log=False, lr=0.27, iters=15000))
    t0 = time.time(); deadline = t0 + float(os.environ.get("HOURS", "8")) * 3600

    def cb(study, trial):
        if trial.number % 5 == 0 and study.best_trial.value is not None:
            b = study.best_trial
            print(f"[{time.strftime('%H:%M:%S')}] trial {trial.number}: best ratio-mean={b.value:.4f} "
                  f"cong_ton={b.params.get('cong_ton'):.2f} lam_c={b.params.get('lam_c'):.4f} "
                  f"per-bench={[round(x,3) for x in b.user_attrs.get('ratios') or []]}", flush=True)
            json.dump({"best": b.value, "params": b.params, "ratios": b.user_attrs.get("ratios")},
                      open(f"{ROOT}/notes/schedule_bayes_best.json", "w"), indent=2)
        if time.time() > deadline:
            study.stop()

    study.optimize(objective, callbacks=[cb], n_jobs=1)
    b = study.best_trial
    print(f"DONE best ratio-mean={b.value:.4f} params={b.params}", flush=True)
