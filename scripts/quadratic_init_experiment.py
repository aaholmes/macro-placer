"""Paired-seed comparison of the gradient stage's starting point (design doc
QUADRATIC_INIT_DESIGN_DOC.md, section 5, short-budget stages).

Arms: random (control), quadratic_b2b, quadratic_b2b_jitter at two sigmas,
corner_screen (M=12, N=200, K1=16, K2=4; its K2 seeds are all screened and the
best kept, screen-1/2 wall time recorded, plus the Spearman correlation between
quadratic-HPWL rank and exact-score rank among the K1 survivors).
Per benchmark x arm x seed: the contest-protocol gradient stage (bayes_swap
.diff_place, tuned t94_win+swap config, 15000 iters), scored raw and after
legalization, then a 60k-move CPU screen (the final-hour phase A screen).
Writes notes/quadratic_init/<tag>.json.

Usage: python scripts/quadratic_init_experiment.py [ibm01 ibm09 ...]
Env: SEEDS (default 4), SIGMAS (default "0.1,0.25"), ITERS (default 15000),
     SCREEN (default 60000), TAG (default "screen"),
     ARMS (comma list from random,quadratic_b2b,jitter,corner_screen; default all).
"""
import os, sys, json, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, multiprocessing as mp
import bayes_swap as bs

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
BENCH = sys.argv[1:] or ["ibm01", "ibm09", "ibm17", "ibm18"]
SEEDS = int(os.environ.get("SEEDS", "4"))
SIGMAS = [float(x) for x in os.environ.get("SIGMAS", "0.1,0.25").split(",")]
ITERS = int(os.environ.get("ITERS", "15000"))
SCREEN = int(os.environ.get("SCREEN", "60000"))
TAG = os.environ.get("TAG", "screen")
ARMS = os.environ.get("ARMS", "random,quadratic_b2b,jitter,corner_screen").split(",")
CORNER = dict(M=12, N=200, K1=16, K2=4)
KW = json.load(open(f"{ROOT}/notes/greedy_knobs_best.json"))["kw"]
_CACHE = {}


def _bench(nm):
    if nm not in _CACHE:
        from macro_place.loader import load_benchmark_from_dir
        from placers.fast_eval import FastEval
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        _CACHE[nm] = (b, plc, FastEval(b, plc), b.macro_sizes.numpy())
    return _CACHE[nm]


def score(fe, p):
    return float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))


def screen_worker(args):
    nm, lg, seed, moves, kw = args
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch as T; T.set_num_threads(1)
    from placers.local_search import optimize_fast
    b, plc, fe, sz = _bench(nm)
    p, _ = optimize_fast(fe, lg, iters=moves, seed=seed, move_hard=False, move_soft=True,
                         refresh=20000, log_every=10**9, logf=lambda *x: None, **({"T0": 0.0} | kw))
    return score(fe, p)


def main():
    from placers.legalizer import legalize
    from placers.analytical import DifferentiablePlacer
    from placers.quadratic import quadratic_place
    os.makedirs(f"{ROOT}/notes/quadratic_init", exist_ok=True)
    out_path = f"{ROOT}/notes/quadratic_init/{TAG}.json"
    pool_cfg = json.load(open(f"{ROOT}/notes/best100_pool.json"))
    dflt = dict(ov_hold=0.0, ov_ramp=0.0, swap_T0=0.0, swap_frac=0.3, gdiff_D0=0.0, gdiff_frac=0.3)
    cfg = next(p for p in pool_cfg if p["_label"] == "t94_win+swap")
    hp = {**bs.hp_from({**dflt, **cfg}), "iters": ITERS}
    arms = ([("random", None)] if "random" in ARMS else []) + \
           ([("quadratic_b2b", 0.0)] if "quadratic_b2b" in ARMS else []) + \
           ([(f"quadratic_b2b_jitter_{s}", s) for s in SIGMAS] if "jitter" in ARMS else [])
    diff_fn = lambda P, seed, hp, iters, deadline, x0: bs.diff_place(P, seed, {**hp, "iters": iters}, x0=x0)
    gp = mp.get_context("spawn").Pool(max(2, min(16, (os.cpu_count() or 4) - 2)))
    rows = []
    for nm in BENCH:
        b, plc, fe, sz = _bench(nm)
        P = DifferentiablePlacer(fe)
        t = time.time(); q0 = quadratic_place(fe); t_q = time.time() - t
        print(f"{nm}: N={b.num_macros} b2b solve {t_q:.2f}s  raw score {score(fe, q0):.4f}", flush=True)
        futs = []
        for arm, sig in arms:
            for seed in range(SEEDS):
                if sig is None:
                    x0 = None; t_init = 0.0
                else:
                    t = time.time()
                    x0 = quadratic_place(fe, jitter_sigma=sig, seed=seed); t_init = time.time() - t
                t = time.time(); raw = bs.diff_place(P, seed, hp, x0=x0); torch.cuda.synchronize() if P.dev == "cuda" else None
                t_diff = time.time() - t
                s_raw = score(fe, raw)
                lg, _, resid = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
                s_leg = score(fe, lg)
                row = dict(bench=nm, arm=arm, sigma=sig, seed=seed, t_init=t_init, t_diff=t_diff,
                           score_raw=s_raw, score_legal=s_leg, resid=int(resid))
                futs.append((row, gp.apply_async(screen_worker, ((nm, lg, seed, SCREEN, KW),))))
                print(f"  {arm:26s} seed {seed}: raw {s_raw:.4f} legal {s_leg:.4f} "
                      f"diff {t_diff:.0f}s init {t_init:.1f}s", flush=True)
        if "corner_screen" in ARMS:
            from placers.corner_seeds import corner_screen
            for seed in range(SEEDS):
                res = corner_screen(P, hp, ITERS, seed=seed, diff_fn=diff_fn, **CORNER)
                if P.dev == "cuda": torch.cuda.synchronize()
                sv = res.screen2
                rk_h = np.argsort(np.argsort([r["hpwl"] for r in sv]))
                rk_s = np.argsort(np.argsort([r["score"] for r in sv]))
                rho = float(np.corrcoef(rk_h, rk_s)[0, 1]) if len(sv) > 2 else float("nan")
                row = dict(bench=nm, arm="corner_screen", sigma=None, seed=seed,
                           t_init=res.t_screen1, t_diff=res.t_screen2,
                           score_raw=float("nan"), score_legal=sv[0]["score"], resid=0,
                           k2_legal=[r["score"] for r in sv[:CORNER["K2"]]],
                           k1_hpwl=[r["hpwl"] for r in sv], k1_legal=[r["score"] for r in sv],
                           spearman=rho,
                           winners=[[list(map(str, a)) for a in r["assignment"]] for r in sv[:CORNER["K2"]]])
                print(f"  corner_screen              seed {seed}: legal(best of K2) {row['score_legal']:.4f} "
                      f"spearman {rho:.2f} screen1 {res.t_screen1:.0f}s screen2 {res.t_screen2:.0f}s", flush=True)
                fs = [gp.apply_async(screen_worker, ((nm, r["pos"], seed, SCREEN, KW),)) for r in sv[:CORNER["K2"]]]
                futs.append((row, fs))
        for row, f in futs:
            if isinstance(f, list):
                row["k2_screen"] = [g.get() for g in f]; row["score_screen"] = min(row["k2_screen"])
            else:
                row["score_screen"] = f.get()
            rows.append(row)
            print(f"  {row['arm']:26s} seed {row['seed']}: screen {row['score_screen']:.4f}", flush=True)
        json.dump(dict(hp=hp, iters=ITERS, screen=SCREEN, rows=rows), open(out_path, "w"), indent=1)
    gp.close()
    summarize(rows)


def summarize(rows):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["bench"], r["arm"])].append(r)
    print(f"\n{'bench':7s} {'arm':26s} {'raw':>14s} {'legal':>14s} {'screen':>14s} {'t_diff':>7s}")
    for (nm, arm), rs in g.items():
        f = lambda k: f"{np.mean([r[k] for r in rs]):.4f}±{np.std([r[k] for r in rs]):.4f}"
        print(f"{nm:7s} {arm:26s} {f('score_raw'):>14s} {f('score_legal'):>14s} "
              f"{f('score_screen'):>14s} {np.mean([r['t_diff'] for r in rs]):7.0f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        summarize(json.load(open(sys.argv[2]))["rows"]); sys.exit()
    main()
