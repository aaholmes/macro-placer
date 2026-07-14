"""Adaptive SA hybrid: unify greedy detailed placement (CPU, true proxy) and the
differentiable placer (GPU, surrogate) into one annealed loop, instead of a fixed
schedule of bursts.

Each chunk, from the current state, generate TWO candidate next-states in parallel:
  - greedy  (CPU): a chunk of soft-macro moves -> descent on the TRUE proxy
  - diff    (GPU): a few gradient steps on the surrogate -> a directed perturbation
Both are scored on the true proxy (legalize + evaluate). Selection:
  - if diff <= greedy: take diff (it explored AND improved)
  - else: take the diff "hop" with prob exp(-(diff-greedy)/T), else take greedy
Greedy is descent-only (exploit); diff is the SA-gated explorer; keep-best is always
retained so exploration is safe. T is annealed high->~0 (explore early, exploit late).
This removes the arbitrary "N equally-spaced jumps".

For a fair read, each benchmark also reports a single-pass baseline that spends the
SAME total greedy budget on one deep descent.

Usage: python sa_hybrid.py [--benches ...] [--chunks 20] [--greedy-iters 6000]
       [--diff-steps 300] [--t0-frac 0.03] [--diff-t 0.9]
"""
import os, sys, json, time, math, random, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
from concurrent.futures import ThreadPoolExecutor
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


def run(nm, hp, chunks, greedy_iters, diff_steps, t0_frac, diff_t, seed=0):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    nh = b.num_hard_macros
    loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"], tau_d=hp["tau_d"],
                      tau_c=hp["tau_c"], target=hp["target"], topk="lapsum", dmode=hp["dmode"],
                      lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"], legalize_steps=hp["lsteps"])
    rng = random.Random(seed)

    def leg(coord):
        lg, _, _ = legalize(coord, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
        return lg

    def proxy(pos):
        c = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)
        return float(c["proxy_cost"]), int(c["overlap_count"])

    def greedy(pos, iters):
        best, _ = optimize_fast(fe, pos, iters=iters, seed=1, T0=0.0, move_hard=False,
                                move_soft=True, refresh=iters, log_every=iters + 1, logf=lambda *a: None)
        return best

    def diff_step(coord, n):
        c = torch.tensor(np.asarray(coord, np.float32)[:, :2], device=P.dev, requires_grad=True)
        opt = torch.optim.Adam([c], lr=hp["lr"])
        for _ in range(n):
            opt.zero_grad(); l = loss(P, c, diff_t); l.backward(); opt.step()
            with torch.no_grad():
                c[:, 0].clamp_(P.hw, P.W - P.hw); c[:, 1].clamp_(P.hh, P.H - P.hh)
        out = np.asarray(coord, np.float64).copy(); out[:, :2] = c.detach().cpu().numpy()
        return out

    # common warm start: initial global placement -> legalize
    coord = P.place(loss, pos0=None, iters=hp["iters"], seed=seed, logf=None)
    start = leg(coord)

    # ---- single-pass baseline: one deep descent, SAME total greedy budget ----
    total_greedy = chunks * greedy_iters
    sp = greedy(start, total_greedy); sp_cost, _ = proxy(sp)

    # ---- adaptive SA hybrid ----
    cur = start.copy(); cur_cost, _ = proxy(cur)
    best, best_cost = cur.copy(), cur_cost
    T0 = t0_frac * cur_cost
    traj = [best_cost]
    with ThreadPoolExecutor(max_workers=2) as ex:
        for k in range(chunks):
            T = T0 * (1e-3) ** (k / max(1, chunks - 1))          # geometric cool to ~0
            fg = ex.submit(greedy, cur, greedy_iters)             # CPU: descent from current
            fd = ex.submit(diff_step, cur, diff_steps)            # GPU: surrogate perturbation
            g_pos = fg.result(); d_coord = fd.result()
            g_cost, _ = proxy(g_pos)
            d_pos = leg(d_coord); d_cost, d_ov = proxy(d_pos)
            take_diff = False
            if d_ov == 0:
                if d_cost <= g_cost:
                    take_diff = True                              # explored and improved
                elif T > 0 and rng.random() < math.exp(-(d_cost - g_cost) / T):
                    take_diff = True                              # SA hop to a worse state
            if take_diff:
                cur, cur_cost = d_coord, d_cost                   # continue diff from continuous coord
                cur_legal = d_pos
            else:
                cur, cur_cost = g_pos, g_cost                     # take greedy descent
                cur_legal = g_pos
            if cur_cost < best_cost:
                best, best_cost = cur_legal.copy(), cur_cost
            traj.append(round(best_cost, 4))
    return dict(benchmark=nm, single_pass=sp_cost, sa_hybrid=best_cost,
                gain=sp_cost - best_cost, chunks=chunks, greedy_iters=greedy_iters,
                diff_steps=diff_steps, total_greedy=total_greedy, traj=traj)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="ibm06,ibm08,ibm04,ibm02,ibm07,ibm17,ibm09,ibm11,ibm01,ibm10")
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--greedy-iters", type=int, default=6000)
    ap.add_argument("--diff-steps", type=int, default=300)
    ap.add_argument("--t0-frac", type=float, default=0.03)
    ap.add_argument("--diff-t", type=float, default=0.9)
    a = ap.parse_args()
    hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
    out = []
    for nm in a.benches.split(","):
        t0 = time.time()
        r = run(nm, hp, a.chunks, a.greedy_iters, a.diff_steps, a.t0_frac, a.diff_t)
        out.append(r)
        json.dump(out, open(f"{ROOT}/notes/sa_hybrid_result.json", "w"), indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: single-pass={r['single_pass']:.4f} "
              f"sa-hybrid={r['sa_hybrid']:.4f} gain={r['gain']:+.4f} "
              f"traj={r['traj']} ({time.time()-t0:.0f}s)", flush=True)
    gains = [r["gain"] for r in out]
    print(f"\nmean gain (single-pass - sa-hybrid): {np.mean(gains):+.4f} | "
          f"sa-hybrid wins {sum(1 for g in gains if g>0)}/{len(gains)}", flush=True)
