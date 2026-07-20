"""Warm-started TPE search over the overlap-ramp WINDOW + the density/schedule
cluster it interacts with, on top of config 221.

Config 221's differentiable loss is WL + overflow-density only (use_cong=0). We add
the overlap window from split_experiment: the deep-overlap penalty is held at 0
until ov_tstart, then ramps to full by ov_tfull (macros may stack -> pass through
each other -> resort ordering early, spreading force preserved throughout). We
tune {ov_tstart, ov_tfull} jointly with {lam_d_ratio, ramp_p, target, gamma,
lam_d_end}; everything else is pinned to 221 (dmode, lam_wl0, lr, tau_d, tau_c,
iters=5000, lsteps=0). Trial 0 is enqueued as 221 + (ov_tstart=0, ov_tfull=0.6) --
the confirmed split_.6 -- so the study is anchored on the known-good point and can
only improve on it.

Separate study (diffplace_window) + DB -- does NOT touch the deployed diffplace
study or bayes_final.json. Search objective is diff+legalize (no greedy), matching
how 221 was tuned; --finalize does best-of-n_seeds on the top configs.
Usage: HOURS=12 python bayes_window.py   |   python bayes_window.py --finalize
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, time, json
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.analytical import DifferentiablePlacer, anneal, lapsum_topk_mean

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
BENCHES = ["ibm01", "ibm13", "ibm17"]                 # same set as the deployed study
DB = "sqlite:////home/laz/partcl/my-macro-placer/notes/bayes_window.db"
P0 = json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"]   # config 221
# Structural pins (NOT tuned): dmode=overflow (the window loss is defined for overflow
# density), use_cong=0 and lsteps=0 (both decisively settled off by the 252-trial search).
# iters fixed at 5000 -- 221's Bayes chose 5000 from [1000,30000] (not a fidelity floor, the
# tuned optimum); fixing it keeps every trial same-cost for a clean ranking. The final full-run
# re-confirms the winner at higher iters. Everything else (window + cluster + lam_wl0/lr/tau_d).
ITERS = 5000
CL = ["lam_d_end_o", "lam_d_ratio", "ramp_p", "target", "gamma", "lam_wl0", "lr", "tau_d"]

ENV = {}; REF = {}
for nm in BENCHES:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    ENV[nm] = (b, plc, fe, P, sz)
    lg, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                        b.canvas_width, b.canvas_height, gap=0.01)
    REF[nm] = float(compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)["proxy_cost"])
print(f"reference legalized: { {k: round(v,4) for k,v in REF.items()} }", flush=True)


def wdensity(P, coord, t, hp):
    """Overflow density with the deep-overlap penalty windowed. w = 0 during the
    flat hold [0, ov_hold], then ramps to 1 over ov_ramp. (ov_hold=0, ov_ramp=0)
    => w=1 for all t>0 => exactly the 221 baseline (density fully on)."""
    cap = 1.0                                          # cap>=target; spread<=cap always penalized
    h, r = hp["ov_hold"], hp["ov_ramp"]
    rho = P.density_field(coord)
    spread = torch.clamp(torch.clamp(rho, max=cap) - hp["target"], min=0.0)
    overlap = torch.clamp(rho - cap, min=0.0)
    if t <= h:
        w = 0.0
    elif r <= 1e-9:
        w = 1.0
    else:
        w = min(1.0, (t - h) / r)
    return lapsum_topk_mean(spread + w * overlap, 0.10, hp["tau_d"])


def place(P, seed, hp):
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(P.b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    coord = c0.to(P.dev).requires_grad_(True)
    opt = torch.optim.Adam([coord], lr=hp["lr"])
    LD0 = hp["lam_d_end"] * hp["lam_d_ratio"]; LD1 = hp["lam_d_end"]
    for it in range(hp["iters"]):
        t = it / hp["iters"]; te = t ** hp["ramp_p"]
        gamma = anneal(te, P.gw * hp["gamma"], P.gw, geometric=True)
        lam_d = anneal(te, LD0, LD1); lam_wl = anneal(te, max(hp["lam_wl0"], 1e-9), 1.0, geometric=True)
        opt.zero_grad()
        (lam_wl * P.wirelength(coord, gamma) + lam_d * wdensity(P, coord, t, hp)).backward()
        opt.step()
        with torch.no_grad():
            coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
    return coord.detach().cpu().numpy()


def score_one(nm, hp, seed):
    b, plc, fe, P, sz = ENV[nm]
    raw = place(P, seed, hp)
    lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    c = compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)
    return float(c["proxy_cost"]) + (1.0 if c["overlap_count"] else 0.0)


def hp_from(p):
    return dict(lam_wl0=p["lam_wl0"], lr=p["lr"], tau_d=p["tau_d"], iters=ITERS,
                lam_d_end=p["lam_d_end_o"], lam_d_ratio=p["lam_d_ratio"], ramp_p=p["ramp_p"],
                target=p["target"], gamma=p["gamma"], ov_hold=p["ov_hold"], ov_ramp=p["ov_ramp"])


def objective(trial):
    p = dict(ov_hold=trial.suggest_float("ov_hold", 0.0, 0.5),
             ov_ramp=trial.suggest_float("ov_ramp", 0.0, 0.6),
             lam_d_end_o=trial.suggest_float("lam_d_end_o", 5e-3, 0.3, log=True),
             lam_d_ratio=trial.suggest_float("lam_d_ratio", 0.02, 1.0, log=True),
             ramp_p=trial.suggest_float("ramp_p", 0.5, 5.0, log=True),
             target=trial.suggest_float("target", 0.6, 1.0),
             gamma=trial.suggest_float("gamma", 1.5, 30.0, log=True),
             lam_wl0=trial.suggest_float("lam_wl0", 0.02, 1.0, log=True),   # early WL vs density balance
             lr=trial.suggest_float("lr", 0.02, 0.3, log=True),
             tau_d=trial.suggest_float("tau_d", 0.01, 0.5, log=True))
    hp = hp_from(p)
    try:
        vals = [score_one(nm, hp, 0) for nm in BENCHES]
    except Exception:
        return 10.0
    ratios = [v / REF[nm] for v, nm in zip(vals, BENCHES)]
    trial.set_user_attr("per_bench", vals); trial.set_user_attr("ratios", ratios)
    return float(np.mean(ratios))


def finalize(study, top_k=6, n_seeds=16):
    comp = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:top_k]
    print(f"\n=== FINALIZE window study: best-of-{n_seeds} on top {len(comp)} configs ===", flush=True)
    best = None
    for t in comp:
        hp = hp_from(t.params)
        vals = [min(score_one(nm, hp, s) for s in range(n_seeds)) for nm in BENCHES]
        ratios = [v / REF[nm] for v, nm in zip(vals, BENCHES)]
        rm = float(np.mean(ratios))
        print(f"  trial {t.number}: search={t.value:.4f} -> best-of-{n_seeds} ratio-mean={rm:.4f} "
              f"win=(hold={t.params['ov_hold']:.2f},ramp={t.params['ov_ramp']:.2f}) "
              f"per-bench-ratio={[round(x,3) for x in ratios]}", flush=True)
        if best is None or rm < best["ratio_mean"]:
            best = {"ratio_mean": rm, "per_bench_proxy": vals, "per_bench_ratio": ratios,
                    "params": t.params, "trial": t.number, "n_seeds": n_seeds}
    json.dump({"benches": BENCHES, "reference": REF, **best},
              open(f"{ROOT}/notes/bayes_window_final.json", "w"), indent=2)
    print(f"FINAL window best ratio-mean={best['ratio_mean']:.4f} "
          f"per-bench {[round(x,3) for x in best['per_bench_ratio']]}", flush=True)
    return best


if __name__ == "__main__":
    study = optuna.create_study(study_name="diffplace_window", storage=DB, direction="minimize",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=45, seed=0))
    if len(study.trials) == 0:      # structured warm-start (see notes below)
        def cfg(src):               # tuned-param subset (iters is fixed, not a seed param)
            return {k: src[k] for k in CL}
        prior = optuna.load_study(study_name="diffplace",
                                  storage="sqlite:////home/laz/partcl/my-macro-placer/notes/bayes.db")
        ok = [t for t in prior.trials if t.value is not None
              and t.params.get("dmode") == "overflow" and not t.params.get("use_cong")]
        # unique full-configs, best-first, with 221 (P0) guaranteed first
        clusters = [cfg(P0)]; seen = {tuple(round(P0[k], 6) for k in CL)}
        for t in sorted(ok, key=lambda t: t.value):
            key = tuple(round(t.params[k], 6) for k in CL)
            if key not in seen:
                seen.add(key); clusters.append(cfg(t.params))
        # 1. baseline anchors: top-6 clusters at (hold=0, ramp=0) -> transfers prior landscape
        #    (kept small so mediocre window-off points don't skew TPE's good/bad split; the
        #    n_startup_trials random phase after these seeds provides full-box coverage)
        for c in clusters[:6]:
            study.enqueue_trial({"ov_hold": 0.0, "ov_ramp": 0.0, **c})
        # 2. window sweep on the best cluster (221) -> clean 1-D read of the window effect
        for h, r in [(0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.2, 0.4), (0.3, 0.5)]:
            study.enqueue_trial({"ov_hold": h, "ov_ramp": r, **clusters[0]})
        # 3. window generalization: a few other good clusters at the confirmed split_.6
        for c in clusters[1:4]:
            study.enqueue_trial({"ov_hold": 0.0, "ov_ramp": 0.6, **c})
    if "--finalize" in sys.argv:
        finalize(study); sys.exit(0)
    deadline = time.time() + float(os.environ.get("HOURS", "12")) * 3600

    def cb(study, trial):
        if trial.number % 10 == 0:
            b = study.best_trial
            print(f"[{time.strftime('%H:%M:%S')}] trial {trial.number}: best ratio-mean={b.value:.4f} "
                  f"(hold={b.params.get('ov_hold'):.2f},ramp={b.params.get('ov_ramp'):.2f}) "
                  f"per-bench={[round(x,3) for x in b.user_attrs.get('ratios') or []]}", flush=True)
        if time.time() > deadline:
            study.stop()

    study.optimize(objective, callbacks=[cb], n_jobs=1)
    b = study.best_trial
    print(f"DONE search best ratio-mean={b.value:.4f} (hold={b.params['ov_hold']:.2f},"
          f"ramp={b.params['ov_ramp']:.2f}) per-bench {b.user_attrs.get('ratios')}", flush=True)
    finalize(study)
