"""Concurrent (interleaved) diff+greedy prototype — iterated local search.

greedy = local descent on the TRUE proxy; a differentiable "burst" = a smart,
gradient-informed perturbation that kicks the layout out of greedy's local
optimum; keep-best on the true proxy makes it monotone. A burst reuses the tuned
loss but starts the anneal partway (t_start) so it reshapes rather than fully
re-spreading; t_start cools upward over rounds (big kicks early, small late).

Compares single-pass diff+greedy vs the interleaved loop on a couple benchmarks.
See notes/concurrent_design.md.

Usage: python concurrent.py [--benches ibm01,ibm17] [--rounds 8] [--burst 1500]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json, time, argparse
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
                lam_d_ratio=p["lam_d_ratio"], gamma=p["gamma"], tau_d=p["tau_d"],
                tau_c=p["tau_c"], lam_wl0=p["lam_wl0"], target=p["target"], lr=p["lr"],
                ramp_p=p["ramp_p"], lsteps=p["lsteps_n"], iters=p["iters"])


def build_loss(hp):
    return proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"],
                      tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"], topk="lapsum",
                      dmode=hp["dmode"], lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
                      legalize_steps=hp["lsteps"])


def burst(P, loss_fn, coord0, iters, lr, t_start, seed):
    """Optimize the surrogate from coord0 for `iters`, annealing t from t_start->1.
    t_start near 0 = big kick (re-spread then re-cluster); near 1 = gentle."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.tensor(np.asarray(coord0, np.float32)[:, :2], device=P.dev)
    c = c + 0.01 * torch.randn(c.shape, generator=g).to(P.dev)   # tiny symmetry-break
    c.requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    for it in range(iters):
        t = t_start + (1.0 - t_start) * (it / iters)
        opt.zero_grad(); loss = loss_fn(P, c, t); loss.backward(); opt.step()
        with torch.no_grad():
            c[:, 0].clamp_(P.hw, P.W - P.hw); c[:, 1].clamp_(P.hh, P.H - P.hh)
    out = np.asarray(coord0, np.float64).copy()
    out[:, :2] = c.detach().cpu().numpy()
    return out


def greedy(fe, b, plc, pos, iters=80000):
    best, _ = optimize_fast(fe, pos, iters=iters, seed=0, T0=0.0, move_hard=False,
                            move_soft=True, refresh=iters, log_every=iters + 1,
                            logf=lambda *a: None)
    c = compute_proxy_cost(torch.tensor(best, dtype=torch.float32), b, plc)
    return best, float(c["proxy_cost"]), int(c["overlap_count"])


def run(nm, hp, rounds, burst_iters):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    loss = build_loss(hp)

    def legal_greedy(coord):
        lg, _, _ = legalize(coord, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        return greedy(fe, b, plc, lg)

    # single-pass baseline
    coord = P.place(loss, pos0=None, iters=hp["iters"], seed=0, logf=None)
    best, base_cost, _ = legal_greedy(coord)
    best_cost = base_cost
    traj = [best_cost]
    # interleaved loop: cool t_start 0.2 -> 0.8 over rounds (big kicks early)
    for r in range(rounds):
        t_start = 0.2 + 0.6 * (r / max(1, rounds - 1))
        pk = burst(P, loss, best, burst_iters, hp["lr"], t_start, seed=r + 1)
        cand, cost, ov = legal_greedy(pk)
        if ov == 0 and cost < best_cost:
            best, best_cost = cand, cost
        traj.append(best_cost)
    return dict(benchmark=nm, single_pass=base_cost, interleaved=best_cost,
                gain=base_cost - best_cost, rounds=rounds, burst=burst_iters,
                traj=[round(x, 4) for x in traj])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="ibm01,ibm17")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--burst", type=int, default=1500)
    a = ap.parse_args()
    hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
    out = []
    for nm in a.benches.split(","):
        t0 = time.time()
        r = run(nm, hp, a.rounds, a.burst)
        out.append(r)
        json.dump(out, open(f"{ROOT}/notes/concurrent_result.json", "w"), indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: single-pass={r['single_pass']:.4f} "
              f"interleaved={r['interleaved']:.4f} gain={r['gain']:+.4f} "
              f"traj={r['traj']} ({time.time()-t0:.0f}s)", flush=True)
