"""Test the T=0 'take-the-better' hybrid, TIME-BALANCED and fast-evaluated.

Diff is the primary loop (homotopy). Each chunk, both operators get the SAME
wall-time budget (chunk-sec): the GPU does as many differentiable steps as fit,
the CPU does as many greedy iters as fit. Advance the working state to whichever
gives the lower TRUE proxy (zero temperature).

Fair control (matched EFFECTIVE work): the hybrid keeps only the winner each
chunk, so over `chunks` chunks it advances via D diff-chunks + G greedy-chunks
(D+G=chunks). The control is the OLD algorithm run with exactly D diff-chunks
then G greedy-chunks (classic order, same per-type counts) -- so the only
difference is adaptive interleaving vs diff-then-greedy, not the amount of work.

True-proxy comparisons in the loop use the validated fast incremental evaluator
(~1000x faster than the TILOS evaluator, exact in the search regime); the final
reported numbers use the TILOS compute_proxy_cost for accuracy. Per-operator
step counts are printed so the diff/greedy time balance is visible.

Usage: python hybrid_t0.py [--benches ...] [--chunks 25] [--chunk-sec 4]
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


def run(nm, hp, chunks, greedy_iters, seed=0):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy(); nh = b.num_hard_macros
    loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"], tau_d=hp["tau_d"],
                      tau_c=hp["tau_c"], target=hp["target"], topk="lapsum", dmode=hp["dmode"],
                      lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"], legalize_steps=hp["lsteps"])

    def leg(coord):
        lg, _, _ = legalize(coord, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
        return lg

    def fast_cost(pos):
        fe.init_incremental(pos)
        return float(fe.cost_current())

    def true_cost(pos):
        return float(compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)["proxy_cost"])

    def diff_time(coord, secs, t):
        c = torch.tensor(np.asarray(coord, np.float32)[:, :2], device=P.dev, requires_grad=True)
        opt = torch.optim.Adam([c], lr=hp["lr"]); n = 0; t0 = time.time()
        while time.time() - t0 < secs:
            for _ in range(25):
                opt.zero_grad(); l = loss(P, c, t); l.backward(); opt.step()
                with torch.no_grad():
                    c[:, 0].clamp_(P.hw, P.W - P.hw); c[:, 1].clamp_(P.hh, P.H - P.hh)
            n += 25
        out = np.asarray(coord, np.float64).copy(); out[:, :2] = c.detach().cpu().numpy()
        return out, n

    def greedy_fixed(pos, iters, seed=1):
        # seed MUST vary across chunks: optimize_fast is memoryless per move, so a
        # fixed seed re-proposes the identical move sequence every chunk (explores
        # ~iters distinct moves total instead of chunks*iters). That silently
        # cripples fine-grained chunking.
        best, _ = optimize_fast(fe, pos, iters=iters, seed=seed, T0=0.0, move_hard=False,
                                move_soft=True, refresh=iters, log_every=iters + 1, logf=lambda *a: None)
        return best

    # shared spread start
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    start = np.zeros((b.num_macros, 2)); start[:, :2] = c0.numpy()

    # calibrate: greedy_iters is the fixed knob; time it once, then the diff branch
    # fills that same wall-time each chunk (auto-balanced per problem, N set by hardware)
    t0 = time.time(); greedy_fixed(leg(start), greedy_iters); t_ref = max(0.2, time.time() - t0)

    def greedy_deep(pos, iters):
        pol, _ = optimize_fast(fe, pos, iters=iters, seed=7, T0=0.0, move_hard=False,
                               move_soft=True, refresh=iters, log_every=iters + 1, logf=lambda *a: None)
        return pol

    # ---- HYBRID: each chunk take the better of {diff burst, greedy} (T=0) ----
    W = start.copy(); best_pos = leg(W); best = fast_cost(best_pos); nd_tot = 0; t_diffs = []
    for k in range(chunks):
        t = (k + 1) / chunks
        d, nd = diff_time(W, t_ref, t); nd_tot += nd; d_lg = leg(d); d_cost = fast_cost(d_lg)
        gpos = greedy_fixed(leg(W), greedy_iters, seed=1000 + k); g_cost = fast_cost(gpos)
        if g_cost < d_cost:
            W, cur_lg, cur = gpos, gpos, g_cost
        else:
            W, cur_lg, cur = d, d_lg, d_cost; t_diffs.append(t)        # record diff-pick t
        if cur < best:
            best, best_pos = cur, cur_lg
    hybrid_final = true_cost(best_pos); D = len(t_diffs); G = chunks - D

    # ---- CONTROL: same D diff-chunks at the SAME t-values, classic order, then G greedy ----
    W2 = start.copy(); b2_pos = leg(W2); b2 = fast_cost(b2_pos)
    for t in t_diffs:
        d, _ = diff_time(W2, t_ref, t); d_lg = leg(d); W2 = d
        c = fast_cost(d_lg)
        if c < b2:
            b2, b2_pos = c, d_lg
    control_final = true_cost(b2_pos)
    if G > 0:
        control_final = min(control_final, true_cost(greedy_deep(b2_pos, G * greedy_iters)))

    # ---- REFERENCE: the actual pipeline (full diff homotopy + G greedy) ----
    coord = P.place(loss, pos0=None, iters=hp["iters"], seed=0, logf=None)
    pl_pos = leg(coord); pipeline_final = true_cost(pl_pos)
    if G > 0:
        pipeline_final = min(pipeline_final, true_cost(greedy_deep(pl_pos, G * greedy_iters)))

    return dict(benchmark=nm, control=control_final, hybrid=hybrid_final, pipeline=pipeline_final,
                gain=control_final - hybrid_final, t_ref=round(t_ref, 2),
                avg_diff_steps=nd_tot // chunks, greedy_iters=greedy_iters, D=D, G=G, chunks=chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="ibm06,ibm08,ibm04,ibm02,ibm07,ibm09")
    ap.add_argument("--chunks", type=int, default=25)
    ap.add_argument("--greedy-iters", type=int, default=8000)
    a = ap.parse_args()
    hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
    out = []
    for nm in a.benches.split(","):
        t0 = time.time()
        r = run(nm, hp, a.chunks, a.greedy_iters)
        out.append(r)
        json.dump(out, open(f"{ROOT}/notes/hybrid_t0_result.json", "w"), indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: control(matched-t)={r['control']:.4f} "
              f"hybrid={r['hybrid']:.4f} gain={r['gain']:+.4f} | pipeline(full-homotopy)={r['pipeline']:.4f} | "
              f"hybrid picked {r['D']} diff / {r['G']} greedy chunks ({time.time()-t0:.0f}s)", flush=True)
    gains = [r["gain"] for r in out]
    print(f"\nmean gain (control - hybrid): {np.mean(gains):+.4f} | "
          f"hybrid wins {sum(1 for g in gains if g>1e-4)}/{len(gains)} | "
          f"loses {sum(1 for g in gains if g<-1e-4)}/{len(gains)}", flush=True)
