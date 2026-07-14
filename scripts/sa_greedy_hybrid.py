"""SA + greedy-until-N-accepts hybrid vs the full-homotopy pipeline, wall-time
budgeted (the natural unit for the parallel design).

Per chunk:
  - greedy runs until N ACCEPTED moves (so a chunk always makes real progress,
    not a rejected no-op); time it (t_g).
  - diff runs for t_g seconds at homotopy position t = elapsed/budget.
  - SA/Metropolis selection: take diff if it's better on the true proxy, else
    take the (worse) diff move with prob exp(-Delta/T) -- letting diff escape
    local minima early -- else take greedy. T anneals high->~0 over the budget.
  - keep-best on the true proxy.

Control = the actual pipeline: full diff homotopy, then greedy for the REST of
the same wall-time budget. Both get identical wall-time.

Usage: python sa_greedy_hybrid.py [--benches ...] [--budget 90] [--accepts 3] [--t0-frac 0.02]
"""
import os, sys, json, time, math, random, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"


def hp_from_params(p):
    dmode = p["dmode"]
    lam_d_end = p.get("lam_d_end_e") if dmode == "electrostatic" else p.get("lam_d_end_o")
    lam_c0 = lam_c1 = 0.0
    if p.get("use_cong"):
        lam_c1 = p["lam_c1"]; lam_c0 = lam_c1 * p["lam_c_startfrac"]
    return dict(dmode=dmode, lam_d_end=lam_d_end, lam_c0=lam_c0, lam_c1=lam_c1,
                lam_d_ratio=p["lam_d_ratio"], gamma=p["gamma"], tau_d=p["tau_d"], tau_c=p["tau_c"],
                lam_wl0=p["lam_wl0"], target=p["target"], lr=p["lr"], ramp_p=p["ramp_p"],
                lsteps=p["lsteps_n"], iters=p["iters"])


def run(nm, hp, budget, accepts, t0_frac, seed=0):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy(); nh = b.num_hard_macros
    loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"], tau_d=hp["tau_d"],
                      tau_c=hp["tau_c"], target=hp["target"], topk="lapsum", dmode=hp["dmode"],
                      lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"], legalize_steps=hp["lsteps"])
    rng = random.Random(seed)

    def leg(c):
        lg, _, _ = legalize(c, sz, nh, b.canvas_width, b.canvas_height, gap=0.01); return lg

    def fcost(pos):
        fe.init_incremental(pos); return float(fe.cost_current())

    def tcost(pos):
        return float(compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)["proxy_cost"])

    def diff_time(coord, secs, t):
        c = torch.tensor(np.asarray(coord, np.float32)[:, :2], device=P.dev, requires_grad=True)
        opt = torch.optim.Adam([c], lr=hp["lr"]); t0 = time.time()
        while time.time() - t0 < secs:
            for _ in range(20):
                opt.zero_grad(); l = loss(P, c, t); l.backward(); opt.step()
                with torch.no_grad():
                    c[:, 0].clamp_(P.hw, P.W - P.hw); c[:, 1].clamp_(P.hh, P.H - P.hh)
        out = np.asarray(coord, np.float64).copy(); out[:, :2] = c.detach().cpu().numpy(); return out

    def greedy_until(pos, n_acc, seed):
        return optimize_fast(fe, pos, iters=1_000_000, seed=seed, T0=0.0, move_hard=False,
                             move_soft=True, refresh=20000, log_every=2_000_000,
                             logf=lambda *a: None, max_accepts=n_acc)[0]

    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    start = np.zeros((b.num_macros, 2)); start[:, :2] = c0.numpy()

    # ---- SA + greedy-until-N hybrid ----
    W = start.copy(); best_pos = leg(W); best = fcost(best_pos); T0 = t0_frac * best
    diff_wins = 0; chunks = 0; t_start = time.time()
    while time.time() - t_start < budget:
        prog = min(0.999, (time.time() - t_start) / budget)
        T = T0 * (1e-3) ** prog
        tg0 = time.time(); gpos = greedy_until(leg(W), accepts, 1000 + chunks); tg = time.time() - tg0
        d = diff_time(W, max(0.05, tg), prog); d_lg = leg(d)
        g_cost = fcost(gpos); d_cost = fcost(d_lg)
        take_diff = (d_cost <= g_cost) or (T > 0 and rng.random() < math.exp(-(d_cost - g_cost) / T))
        if take_diff:
            W, cur_lg, cur = d, d_lg, d_cost; diff_wins += 1
        else:
            W, cur_lg, cur = gpos, gpos, g_cost
        if cur < best:
            best, best_pos = cur, cur_lg
        chunks += 1
    hybrid_final = tcost(best_pos)

    # ---- pipeline: full homotopy, then greedy for the rest of the same budget ----
    t_start = time.time()
    coord = P.place(loss, pos0=None, iters=hp["iters"], seed=seed, logf=None)
    pl = leg(coord); pbest_pos = pl; pbest = fcost(pl)
    while time.time() - t_start < budget:
        pl = greedy_until(pl, 50, 5000 + int((time.time() - t_start)))
        c = fcost(pl)
        if c < pbest:
            pbest, pbest_pos = c, pl.copy()
    pipeline_final = tcost(pbest_pos)

    return dict(benchmark=nm, pipeline=pipeline_final, hybrid=hybrid_final,
                gain=pipeline_final - hybrid_final, chunks=chunks, diff_wins=diff_wins, budget=budget)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="ibm06,ibm02")
    ap.add_argument("--budget", type=float, default=90.0)
    ap.add_argument("--accepts", type=int, default=3)
    ap.add_argument("--t0-frac", type=float, default=0.02)
    a = ap.parse_args()
    hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
    out = []
    for nm in a.benches.split(","):
        t0 = time.time()
        r = run(nm, hp, a.budget, a.accepts, a.t0_frac)
        out.append(r)
        json.dump(out, open(f"{ROOT}/notes/sa_greedy_hybrid.json", "w"), indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: pipeline={r['pipeline']:.4f} hybrid={r['hybrid']:.4f} "
              f"gain={r['gain']:+.4f} | hybrid: {r['chunks']} chunks, diff won {r['diff_wins']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    gains = [r["gain"] for r in out]
    print(f"\nmean gain (pipeline - hybrid): {np.mean(gains):+.4f} | "
          f"hybrid wins {sum(1 for g in gains if g>1e-4)}/{len(gains)}", flush=True)
