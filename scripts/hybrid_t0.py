"""Test the T=0 'take-the-better' hybrid as specified: diff is the primary loop
running the homotopy; each chunk, consider BOTH N differentiable updates and a
greedy update, and advance the working state to whichever gives the lower TRUE
proxy (zero temperature, no stochastic hops). Baseline = pure diff (every chunk
diff), which at T=0-with-greedy-never-winning is identical.

Reports, per benchmark, the keep-best legalized proxy for:
  - diff-only     : the pure homotopy trajectory
  - hybrid-t0     : take-the-better each chunk
both with the same total wall-budget of chunks. A final greedy polish is applied
to each keep-best so the comparison reflects the deliverable.

Usage: python hybrid_t0.py [--benches ...] [--chunks 30] [--diff-steps 400]
       [--greedy-iters 8000]
"""
import os, sys, json, time, argparse
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


CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"


def run(nm, hp, chunks, diff_steps, greedy_iters, seed=0):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy(); nh = b.num_hard_macros
    loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"], tau_d=hp["tau_d"],
                      tau_c=hp["tau_c"], target=hp["target"], topk="lapsum", dmode=hp["dmode"],
                      lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"], legalize_steps=hp["lsteps"])

    def leg(coord):
        lg, _, _ = legalize(coord, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
        return lg

    def cost(pos):
        return float(compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)["proxy_cost"])

    def diff_steps_from(coord, n, t):
        c = torch.tensor(np.asarray(coord, np.float32)[:, :2], device=P.dev, requires_grad=True)
        opt = torch.optim.Adam([c], lr=hp["lr"])
        for _ in range(n):
            opt.zero_grad(); l = loss(P, c, t); l.backward(); opt.step()
            with torch.no_grad():
                c[:, 0].clamp_(P.hw, P.W - P.hw); c[:, 1].clamp_(P.hh, P.H - P.hh)
        out = np.asarray(coord, np.float64).copy(); out[:, :2] = c.detach().cpu().numpy()
        return out

    def greedy_from(pos, iters):
        best, _ = optimize_fast(fe, pos, iters=iters, seed=1, T0=0.0, move_hard=False,
                                move_soft=True, refresh=iters, log_every=iters + 1, logf=lambda *a: None)
        return best

    def polish(pos):
        return greedy_from(pos, 120000)

    # shared spread start
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    start = np.zeros((b.num_macros, 2)); start[:, :2] = c0.numpy()

    def trajectory(hybrid):
        W = start.copy()
        best_pos = leg(W); best = cost(best_pos)
        for k in range(chunks):
            t = (k + 1) / chunks                                  # advance the homotopy
            d = diff_steps_from(W, diff_steps, t); d_lg = leg(d); d_cost = cost(d_lg)
            if hybrid:
                gpos = greedy_from(leg(W), greedy_iters); g_cost = cost(gpos)
                if g_cost < d_cost:                               # T=0: take the better true loss
                    W = gpos; cur_lg, cur = gpos, g_cost
                else:
                    W = d; cur_lg, cur = d_lg, d_cost
            else:
                W = d; cur_lg, cur = d_lg, d_cost
            if cur < best:
                best, best_pos = cur, cur_lg
        pol = polish(best_pos)                                     # same final polish for both
        return min(best, cost(pol))

    diff_only = trajectory(False)
    hybrid_t0 = trajectory(True)
    return dict(benchmark=nm, diff_only=diff_only, hybrid_t0=hybrid_t0,
                gain=diff_only - hybrid_t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="ibm06,ibm08,ibm04,ibm02,ibm07,ibm09")
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--diff-steps", type=int, default=400)
    ap.add_argument("--greedy-iters", type=int, default=8000)
    a = ap.parse_args()
    hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
    out = []
    for nm in a.benches.split(","):
        t0 = time.time()
        r = run(nm, hp, a.chunks, a.diff_steps, a.greedy_iters)
        out.append(r)
        json.dump(out, open(f"{ROOT}/notes/hybrid_t0_result.json", "w"), indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: diff-only={r['diff_only']:.4f} "
              f"hybrid-t0={r['hybrid_t0']:.4f} gain={r['gain']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    gains = [r["gain"] for r in out]
    print(f"\nmean gain (diff-only - hybrid-t0): {np.mean(gains):+.4f} | "
          f"hybrid wins {sum(1 for g in gains if g>0)}/{len(gains)} | "
          f"hybrid loses {sum(1 for g in gains if g<-1e-4)}/{len(gains)}", flush=True)
