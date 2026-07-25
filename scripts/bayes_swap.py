"""Warm-started TPE search whose objective is the DEPLOYMENT metric: best-of-N
seeds AFTER a (short) greedy polish -- not single-seed pre-greedy. This fixes the
mean-vs-tail mismatch: single-seed-no-greedy rewards a good typical placement,
but we deploy best-of-N-after-greedy, which rewards a good left tail (and thus
diversity). Greedy is short (GREEDY_MOVES) and run in PARALLEL across CPU cores
(spawn pool) while the GPU does the diffs, so greedy is ~free relative to the
3x-seed diff cost.

Objective per trial = mean over benches of (best-of-N_SEEDS post-greedy proxy) /
reference. Tunes the (hold,ramp) window + density/schedule cluster + lam_wl0/lr/
tau_d (iters fixed; overflow/no-cong/no-softlegalize pinned), warm-started from
221 + the window study's seeds. Separate study (diffplace_wg) + DB.
Usage: HOURS=12 python bayes_window_greedy.py | --finalize | --smoke
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, time, json
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
import multiprocessing as mp

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
BENCHES = ["ibm01", "ibm13", "ibm17"]
DB = "sqlite:////home/laz/partcl/my-macro-placer/notes/bayes_swap.db"
ITERS = 5000
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
GREEDY_MOVES = int(os.environ.get("GREEDY_MOVES", "12000"))
CL = ["lam_d_end_o", "lam_d_ratio", "ramp_p", "target", "gamma", "lam_wl0", "lr", "tau_d"]

# ---- CPU greedy worker (spawn; no CUDA in workers). Per-process benchmark cache. ----
_CACHE = {}


def _bench(nm):
    if nm not in _CACHE:
        from macro_place.loader import load_benchmark_from_dir
        from placers.fast_eval import FastEval
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        _CACHE[nm] = (b, plc, FastEval(b, plc), b.macro_sizes.numpy())
    return _CACHE[nm]


def greedy_worker(args):
    """(nm, raw_placement, seed, moves) -> (nm, seed, post-greedy proxy)."""
    nm, raw, seed, moves = args
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch as T
    T.set_num_threads(1)
    from macro_place.objective import compute_proxy_cost
    from placers.legalizer import legalize
    from placers.local_search import optimize_fast
    b, plc, fe, sz = _bench(nm)
    lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    pol, _ = optimize_fast(fe, lg, iters=moves, seed=seed, T0=0.0, move_hard=False,
                           move_soft=True, refresh=20000, log_every=10**9, logf=lambda *x: None)
    c = compute_proxy_cost(T.tensor(pol, dtype=T.float32), b, plc)
    return (nm, seed, float(c["proxy_cost"]) + (1.0 if c["overlap_count"] else 0.0))


# ---- main-process (GPU) globals, set in main() ----
ENV = {}; REF = {}; P0 = None; POOL = None


SWAP_EVERY = 100
SWAP_ATTEMPTS = 40
SWAP_T0_MAX = 0.004         # calibrated: adjacent same-size swap |dWL| ~ 2e-4..3e-3, so T0 up to
                            # ~max|dWL| spans off -> aggressive (higher would be near-random reshuffle)


def _swap_groups(P):
    """same-size macro groups (>=2 members); cached on P. Same-size swaps leave density
    invariant, so the SA Metropolis needs only wirelength."""
    if not hasattr(P, "_sgroups"):
        from collections import defaultdict
        hw = P.hw.detach().cpu().numpy(); hh = P.hh.detach().cpu().numpy()
        d = defaultdict(list)
        for m in range(P.b.num_macros):
            d[(round(float(hw[m]), 3), round(float(hh[m]), 3))].append(m)
        P._sgroups = [torch.tensor(v, device=P.dev, dtype=torch.long) for v in d.values() if len(v) >= 2]
    return P._sgroups


def hp_from(p):
    return dict(lam_wl0=p["lam_wl0"], lr=p["lr"], tau_d=p["tau_d"], iters=ITERS,
                lam_d_end=p["lam_d_end_o"], lam_d_ratio=p["lam_d_ratio"], ramp_p=p["ramp_p"],
                target=p["target"], gamma=p["gamma"], ov_hold=p["ov_hold"], ov_ramp=p["ov_ramp"],
                gdiff_D0=p.get("gdiff_D0", 0.0), gdiff_frac=p.get("gdiff_frac", 0.3),
                swap_T0=p.get("swap_T0", 0.0), swap_frac=p.get("swap_frac", 0.3))


def diff_place(P, seed, hp):
    """Diff stage: overlap window + (opt) independent global Langevin diffusion (gdiff_D0
    high -> 0 by gdiff_frac) + (opt) SA swaps of adjacent same-size macros (swap_T0 -> 0 by
    swap_frac). Density is invariant under same-size swaps, so swap Metropolis uses WL only."""
    from placers.analytical import anneal, lapsum_topk_mean
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(P.b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    coord = c0.to(P.dev).requires_grad_(True)
    opt = torch.optim.Adam([coord], lr=hp["lr"])
    LD0 = hp["lam_d_end"] * hp["lam_d_ratio"]; LD1 = hp["lam_d_end"]
    cap = 1.0
    gD0 = hp.get("gdiff_D0", 0.0); gfrac = max(1e-6, hp.get("gdiff_frac", 0.3))
    sT0 = hp.get("swap_T0", 0.0); sfrac = max(1e-6, hp.get("swap_frac", 0.3))
    csz = float(min(P.W, P.H))
    ar = getattr(P, "area", None)
    mob = ((ar.median() / ar).sqrt().clamp(max=5.0).unsqueeze(1) if ar is not None
           else torch.ones(P.b.num_macros, 1, device=P.dev))
    ngen = torch.Generator(device=P.dev).manual_seed(seed * 1000 + 7)
    rs = np.random.RandomState(seed * 7 + 3)
    groups = _swap_groups(P) if sT0 > 0 else []
    for it in range(hp["iters"]):
        t = it / hp["iters"]; te = t ** hp["ramp_p"]
        gamma = anneal(te, P.gw * hp["gamma"], P.gw, geometric=True)
        lam_d = anneal(te, LD0, LD1); lam_wl = anneal(te, max(hp["lam_wl0"], 1e-9), 1.0, geometric=True)
        rho = P.density_field(coord)
        spread = torch.clamp(torch.clamp(rho, max=cap) - hp["target"], min=0.0)
        overlap = torch.clamp(rho - cap, min=0.0)
        if t <= hp["ov_hold"]:
            w = 0.0
        elif hp["ov_ramp"] <= 1e-9:
            w = 1.0
        else:
            w = min(1.0, (t - hp["ov_hold"]) / hp["ov_ramp"])
        de = lapsum_topk_mean(spread + w * overlap, 0.10, hp["tau_d"])
        opt.zero_grad()
        (lam_wl * P.wirelength(coord, gamma) + lam_d * de).backward()
        opt.step()
        with torch.no_grad():
            if gD0 > 0 and t < gfrac:               # independent global diffusion, high -> 0 by gfrac
                coord.add_(gD0 * csz * (1.0 - t / gfrac) * mob * torch.randn(coord.shape, generator=ngen, device=P.dev))
            coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
            if sT0 > 0 and t < sfrac and it % SWAP_EVERY == 0 and groups:   # SA swaps of adjacent same-size macros
                T = sT0 * (1.0 - t / sfrac)
                if T > 1e-9:
                    wl0 = float(P.wirelength(coord, gamma))
                    for _ in range(SWAP_ATTEMPTS):
                        grp = groups[rs.randint(len(groups))]
                        i = int(grp[rs.randint(len(grp))])
                        d2 = ((coord[grp] - coord[i]) ** 2).sum(1); d2[grp == i] = 1e18
                        j = int(grp[int(d2.argmin())])
                        if i == j:
                            continue
                        coord[[i, j]] = coord[[j, i]]
                        wl1 = float(P.wirelength(coord, gamma))
                        if wl1 <= wl0 or rs.rand() < np.exp(-(wl1 - wl0) / T):
                            wl0 = wl1                # accept
                        else:
                            coord[[i, j]] = coord[[j, i]]   # revert
    out = coord.detach()
    if not torch.isfinite(out).all():
        raise FloatingPointError("non-finite coords")
    return out.cpu().numpy()


def score_trial(hp, n_seeds):
    """GPU diffs for all (bench, seed); parallel greedy; best-of-n per bench -> ratios."""
    jobs = []
    for nm in BENCHES:
        P = ENV[nm][3]
        for s in range(n_seeds):
            raw = diff_place(P, s, hp)
            jobs.append((nm, raw, s, GREEDY_MOVES))
    results = list(POOL.map(greedy_worker, jobs))
    best = {}
    for nm, s, pr in results:
        best[nm] = min(pr, best.get(nm, 1e9))
    return [best[nm] / REF[nm] for nm in BENCHES]


def objective(trial):
    # Ranges informed by the 108 prior overflow/no-cong configs: FOUR params are locked to
    # a thin successful slab (lam_d_ratio, ramp_p, tau_d, target), so narrow them -> the
    # effective search is ~6-D and the random-startup phase covers it well. The THREE free
    # params (gamma, lam_wl0, lr) + the window carry the diversity best-of-N wants, so keep
    # them wide (lr widened past the prior 0.3 max). Narrowed ranges still contain 221 and
    # every window-study seed.
    p = dict(ov_hold=trial.suggest_float("ov_hold", 0.0, 0.5),          # free (new)
             ov_ramp=trial.suggest_float("ov_ramp", 0.0, 0.6),          # free (new)
             lam_d_end_o=trial.suggest_float("lam_d_end_o", 0.03, 0.20, log=True),
             lam_d_ratio=trial.suggest_float("lam_d_ratio", 0.02, 0.06, log=True),   # LOCKED ~0.035
             ramp_p=trial.suggest_float("ramp_p", 2.5, 5.0, log=True),            # LOCKED ~3.6
             target=trial.suggest_float("target", 0.64, 0.78),                    # LOCKED ~0.70
             tau_d=trial.suggest_float("tau_d", 0.05, 0.20, log=True),            # LOCKED ~0.10
             gamma=trial.suggest_float("gamma", 1.5, 30.0, log=True),   # free
             lam_wl0=trial.suggest_float("lam_wl0", 0.05, 1.0, log=True),  # free
             lr=trial.suggest_float("lr", 0.02, 0.5, log=True),         # free (diversity lever)
             # independent global diffusion (crossing-scale; >0.006 scatters) + its decay fraction
             gdiff_D0=trial.suggest_float("gdiff_D0", 0.0, 0.004),
             gdiff_frac=trial.suggest_float("gdiff_frac", 0.05, 0.6),
             # SA swaps of adjacent same-size macros: temperature (0=off) + active fraction
             swap_T0=trial.suggest_float("swap_T0", 0.0, SWAP_T0_MAX),
             swap_frac=trial.suggest_float("swap_frac", 0.05, 0.6))
    try:
        ratios = score_trial(hp_from(p), N_SEEDS)
    except Exception as e:
        print(f"trial {trial.number} FAILED: {type(e).__name__}: {e}", flush=True)
        return 10.0
    trial.set_user_attr("ratios", ratios)
    return float(np.mean(ratios))


def finalize(study, top_k=6, n_seeds=16):
    import optuna
    comp = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:top_k]
    print(f"\n=== FINALIZE (best-of-{n_seeds} post-greedy) on top {len(comp)} configs ===", flush=True)
    best = None
    for t in comp:
        ratios = score_trial(hp_from(t.params), n_seeds)
        rm = float(np.mean(ratios))
        print(f"  trial {t.number}: search={t.value:.4f} -> best-of-{n_seeds}={rm:.4f} "
              f"(hold={t.params['ov_hold']:.2f},ramp={t.params['ov_ramp']:.2f},lr={t.params['lr']:.3f}) "
              f"ratios={[round(x,3) for x in ratios]}", flush=True)
        if best is None or rm < best["ratio_mean"]:
            best = {"ratio_mean": rm, "per_bench_ratio": ratios, "params": t.params,
                    "trial": t.number, "n_seeds": n_seeds}
    json.dump({"benches": BENCHES, "reference": REF, "greedy_moves": GREEDY_MOVES, **best},
              open(f"{ROOT}/notes/bayes_swap_final.json", "w"), indent=2)
    print(f"FINAL wg best ratio-mean={best['ratio_mean']:.4f} "
          f"per-bench {[round(x,3) for x in best['per_bench_ratio']]}", flush=True)
    return best


def main():
    import optuna
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from placers.fast_eval import FastEval
    from placers.legalizer import legalize
    from placers.analytical import DifferentiablePlacer
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    global P0, POOL
    P0 = json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"]
    for nm in BENCHES:
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
        ENV[nm] = (b, plc, fe, P, sz)
        lg, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                            b.canvas_width, b.canvas_height, gap=0.01)
        REF[nm] = float(compute_proxy_cost(torch.tensor(lg, dtype=torch.float32), b, plc)["proxy_cost"])
    print(f"reference: { {k: round(v,4) for k,v in REF.items()} }  N_SEEDS={N_SEEDS} "
          f"GREEDY_MOVES={GREEDY_MOVES}", flush=True)

    workers = max(2, min(14, (os.cpu_count() or 4) - 2))
    POOL = mp.get_context("spawn").Pool(workers)

    if "--smoke" in sys.argv:                       # 2 quick trials to validate end-to-end
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(n_startup_trials=2, seed=0))
        st.optimize(objective, n_trials=2)
        print("SMOKE OK best", round(st.best_value, 4), flush=True); POOL.close(); return

    study = optuna.create_study(study_name="diffplace_swap", storage=DB, direction="minimize",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=48, seed=0,
                                                                    multivariate=True, group=True))
    if len(study.trials) == 0:                       # warm-start (mirror bayes_window)
        _enq = study.enqueue_trial
        def enq(params):                             # pin the new mechanisms OFF for known-good seeds unless set
            params = dict(params)
            for k, v in (("gdiff_D0", 0.0), ("gdiff_frac", 0.3), ("swap_T0", 0.0), ("swap_frac", 0.3)):
                params.setdefault(k, v)
            _enq(params)
        def cfg(src): return {k: src[k] for k in CL}
        prior = optuna.load_study(study_name="diffplace",
                                  storage="sqlite:////home/laz/partcl/my-macro-placer/notes/bayes.db")
        ok = [t for t in prior.trials if t.value is not None
              and t.params.get("dmode") == "overflow" and not t.params.get("use_cong")]
        clusters = [cfg(P0)]; seen = {tuple(round(P0[k], 6) for k in CL)}
        for t in sorted(ok, key=lambda t: t.value):
            key = tuple(round(t.params[k], 6) for k in CL)
            if key not in seen:
                seen.add(key); clusters.append(cfg(t.params))
        for c in clusters[:6]:
            enq({"ov_hold": 0.0, "ov_ramp": 0.0, **c})     # baseline anchors
        for h, r in [(0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.2, 0.4), (0.19, 0.58)]:
            enq({"ov_hold": h, "ov_ramp": r, **clusters[0]})  # window sweep on 221
        for c in clusters[1:4]:
            enq({"ov_hold": 0.19, "ov_ramp": 0.58, **c})   # window generalization
        # seed the WINDOW study's own top configs (identical 10-param space) so the greedy
        # objective directly re-evaluates the best single-seed-found windows (incl. finalize
        # winner 249) with their tuned cluster+lr, not just 221's cluster + a window shape.
        WK = ["ov_hold", "ov_ramp"] + CL
        try:
            wst = optuna.load_study(study_name="diffplace_window",
                                    storage="sqlite:////home/laz/partcl/my-macro-placer/notes/bayes_window.db")
            wtop = sorted([t for t in wst.trials if t.value is not None], key=lambda t: t.value)[:8]
            for t in wtop:
                p = {k: t.params[k] for k in WK}
                p["lr"] = min(p["lr"], 0.5)                    # clamp into the widened lr range
                enq(p)                                          # D0=0 (via enq default)
            # PAIRED SA-swap: same top-6 window configs, now with swap ON (T0 spread over a grid)
            # -> clean (config, swap-off) vs (config, swap-on) contrast for attribution.
            SG = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]
            for i, t in enumerate(wtop[:6]):
                p = {k: t.params[k] for k in WK}
                p["lr"] = min(p["lr"], 0.5); p["swap_T0"] = SG[i % len(SG)]; p["swap_frac"] = 0.3
                enq(p)
            print(f"seeded {len(wtop)} window configs + {min(6,len(wtop))} paired with SA-swap "
                  f"(best search={wtop[0].value:.4f} trial {wtop[0].number})", flush=True)
        except Exception as e:
            print(f"(window-study seeding skipped: {type(e).__name__}: {e})", flush=True)
        # cross-breed / diversity seeds: 221's locked manifold + widely varied FREE dims
        # (gamma, lam_wl0, lr) + window shapes -> anchors the diversity region best-of-N wants,
        # esp. high lr (outcome-variance lever the single-seed search never rewarded).
        base_lock = {k: P0[k] for k in ["lam_d_end_o", "lam_d_ratio", "ramp_p", "target", "tau_d"]}
        DIV = [dict(gamma=3.0,  lam_wl0=0.30, lr=0.40, ov_hold=0.19, ov_ramp=0.58),
               dict(gamma=20.0, lam_wl0=0.90, lr=0.40, ov_hold=0.10, ov_ramp=0.50),
               dict(gamma=6.0,  lam_wl0=0.20, lr=0.45, ov_hold=0.25, ov_ramp=0.55),
               dict(gamma=15.0, lam_wl0=0.60, lr=0.30, ov_hold=0.00, ov_ramp=0.60),
               dict(gamma=2.5,  lam_wl0=0.95, lr=0.20, ov_hold=0.30, ov_ramp=0.50),
               dict(gamma=10.0, lam_wl0=0.40, lr=0.35, ov_hold=0.15, ov_ramp=0.45)]
        for d in DIV:
            enq({**base_lock, **d})
        # mechanism anchors on 221 + window: SA-swap and global-diffusion ON, so TPE has
        # non-zero anchors for BOTH new levers.
        base_win = {**base_lock, "gamma": P0["gamma"], "lam_wl0": P0["lam_wl0"], "lr": 0.30,
                    "ov_hold": 0.19, "ov_ramp": 0.58}
        for sT0 in (0.001, 0.002, 0.003):
            enq({**base_win, "swap_T0": sT0, "swap_frac": 0.35})
        for gD0 in (0.001, 0.002):
            enq({**base_win, "gdiff_D0": gD0, "gdiff_frac": 0.25})

    if "--finalize" in sys.argv:
        finalize(study); POOL.close(); return
    deadline = time.time() + float(os.environ.get("HOURS", "12")) * 3600

    def cb(study, trial):
        if trial.number % 10 == 0:
            b = study.best_trial
            print(f"[{time.strftime('%H:%M:%S')}] trial {trial.number}: best={b.value:.4f} "
                  f"(hold={b.params.get('ov_hold'):.2f},ramp={b.params.get('ov_ramp'):.2f},"
                  f"lr={b.params.get('lr'):.3f}) per-bench={[round(x,3) for x in b.user_attrs.get('ratios') or []]}",
                  flush=True)
        if time.time() > deadline:
            study.stop()

    study.optimize(objective, callbacks=[cb], n_jobs=1)
    b = study.best_trial
    print(f"DONE search best={b.value:.4f} (hold={b.params['ov_hold']:.2f},ramp={b.params['ov_ramp']:.2f})", flush=True)
    finalize(study); POOL.close()


if __name__ == "__main__":
    main()
