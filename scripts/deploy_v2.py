"""Deployment best-of-N v2: the best100 pool + (optionally) the new greedy
operators + a diff-from-reference candidate family, saving winner placements.

Env: DEADLINE_MIN (GPU-diff budget/bench, default 45), OPS ("" | "ops" | "ops_sa"
-> greedy operator kwargs, chosen by the repolish gate), ITERS, MOVES.
-> notes/deploy_v2_report.json + notes/deploy_v2/{nm}_best.npz
Compare after_avg to 1.0137 (best100) / 1.0155 (original).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import sys, json, time
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import numpy as np, torch, multiprocessing as mp
import bayes_swap as bs
from placers.analytical import anneal, lapsum_topk_mean

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
if os.environ.get("BENCHES"): ALL = os.environ["BENCHES"].split(",")
DEADLINE_MIN = float(os.environ.get("DEADLINE_MIN", "45"))
ITERS = int(os.environ.get("ITERS", "15000"))
MOVES = int(os.environ.get("MOVES", "100000"))
OPS = os.environ.get("OPS", "")
TAG = os.environ.get("TAG", "v2")
if os.environ.get("OPS_JSON"):                       # tuned knob file overrides OPS presets
    OP_KW = json.load(open(os.environ["OPS_JSON"]))["kw"]
    OPS = "json:" + os.environ["OPS_JSON"]
else:
    OP_KW = {"": {}, "ops": dict(opt_frac=0.25, opt_flat=True, route_frac=0.2),
             "ops_sa": dict(opt_frac=0.25, opt_flat=True, route_frac=0.2, T0=3e-4, Tend=1e-6)}[OPS]

_CACHE = {}


def _bench(nm):
    if nm not in _CACHE:
        from macro_place.loader import load_benchmark_from_dir
        from placers.fast_eval import FastEval
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        _CACHE[nm] = (b, plc, FastEval(b, plc), b.macro_sizes.numpy())
    return _CACHE[nm]


def greedy_worker(args):
    nm, raw, seed, moves, op_kw = args
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch as T
    T.set_num_threads(1)
    from macro_place.objective import compute_proxy_cost
    from placers.legalizer import legalize
    from placers.local_search import optimize_fast
    b, plc, fe, sz = _bench(nm)
    lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
    pol, _ = optimize_fast(fe, lg, iters=moves, seed=seed, move_hard=False, move_soft=True,
                           refresh=20000, log_every=10**9, logf=lambda *x: None,
                           **({"T0": 0.0} | op_kw))
    c = compute_proxy_cost(T.tensor(pol, dtype=T.float32), b, plc)
    pr = float(c["proxy_cost"]) + (1.0 if c["overlap_count"] else 0.0)
    return (nm, seed, pr, pol)


def diff_from(P, seed, hp, x0=None):
    """bayes_swap.diff_place with an optional warm start (x0 [N,2] numpy)."""
    if x0 is None:
        return bs.diff_place(P, seed, hp)
    coord = torch.tensor(np.asarray(x0, np.float32), device=P.dev, requires_grad=True)
    opt = torch.optim.Adam([coord], lr=hp["lr"])
    LD0 = hp["lam_d_end"] * hp["lam_d_ratio"]; LD1 = hp["lam_d_end"]; cap = 1.0
    for it in range(hp["iters"]):
        t = it / hp["iters"]; te = t ** hp["ramp_p"]
        gamma = anneal(te, P.gw * hp["gamma"], P.gw, geometric=True)
        lam_d = anneal(te, LD0, LD1); lam_wl = anneal(te, max(hp["lam_wl0"], 1e-9), 1.0, geometric=True)
        rho = P.density_field(coord)
        spread = torch.clamp(torch.clamp(rho, max=cap) - hp["target"], min=0.0)
        overlap = torch.clamp(rho - cap, min=0.0)
        w = 0.0 if t <= hp["ov_hold"] else (1.0 if hp["ov_ramp"] <= 1e-9 else min(1.0, (t - hp["ov_hold"]) / hp["ov_ramp"]))
        de = lapsum_topk_mean(spread + w * overlap, 0.10, hp["tau_d"])
        opt.zero_grad(); (lam_wl * P.wirelength(coord, gamma) + lam_d * de).backward(); opt.step()
        with torch.no_grad():
            coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
    return coord.detach().cpu().numpy()


def main():
    from macro_place.loader import load_benchmark_from_dir
    from placers.fast_eval import FastEval
    from placers.legalizer import legalize
    from placers.analytical import DifferentiablePlacer
    os.makedirs(f"{ROOT}/notes/deploy_{TAG}", exist_ok=True)
    pool_cfg = json.load(open(f"{ROOT}/notes/best100_pool.json"))
    dflt = dict(ov_hold=0.0, ov_ramp=0.0, swap_T0=0.0, swap_frac=0.3, gdiff_D0=0.0, gdiff_frac=0.3)
    hps = [{**bs.hp_from({**dflt, **p}), "iters": ITERS} for p in pool_cfg]
    labels = [p["_label"] for p in pool_cfg]
    tuned = labels.index("t94_win+swap")
    print(f"deploy_{TAG}: OPS='{OPS}' kw={OP_KW}  DEADLINE_MIN={DEADLINE_MIN} pool={labels}", flush=True)
    gp = mp.get_context("spawn").Pool(max(2, min(30, (os.cpu_count() or 4) - 2)))
    results = {}; wins = {}
    for nm in ALL:
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        P = DifferentiablePlacer(FastEval(b, plc)); sz = b.macro_sizes.numpy()
        futs = []
        # extra family: tuned config warm-started from the legalized reference
        ref, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                             b.canvas_width, b.canvas_height, gap=0.01)
        raw = diff_from(P, 0, hps[tuned], x0=ref)
        futs.append(("diff_from_ref", gp.apply_async(greedy_worker, ((nm, raw, 0, MOVES, OP_KW),))))
        deadline = time.time() + DEADLINE_MIN * 60
        cnt = 0
        while time.time() < deadline:
            ci = cnt % len(hps); seed = cnt // len(hps)
            try:
                raw = diff_from(P, seed, hps[ci])
            except Exception as e:
                print(f"  {nm} cand{cnt} diff FAILED: {type(e).__name__}", flush=True); cnt += 1; continue
            futs.append((labels[ci], gp.apply_async(greedy_worker, ((nm, raw, cnt, MOVES, OP_KW),))))
            cnt += 1
        best = 1e9; bl = None; bpol = None
        for lbl, f in futs:
            try:
                _, _, pr, pol = f.get()
            except Exception:
                continue
            if pr < best:
                best, bl, bpol = pr, lbl, pol
        results[nm] = best; wins[bl] = wins.get(bl, 0) + 1
        np.savez(f"{ROOT}/notes/deploy_{TAG}/{nm}_best.npz", placement=bpol)
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: {cnt+1} cands, best={best:.4f} (won by {bl}) "
              f"| running avg={np.mean(list(results.values())):.4f}", flush=True)
    avg = float(np.mean(list(results.values())))
    json.dump({"per_bench": results, "after_avg": avg, "wins": wins, "ops": OPS,
               "deadline_min": DEADLINE_MIN}, open(f"{ROOT}/notes/deploy_{TAG}_report.json", "w"), indent=1)
    print(f"\nAFTER_AVG = {avg:.4f}   (best100 1.0137, original 1.0155)", flush=True)
    print(f"wins: {wins}", flush=True)
    gp.close()


if __name__ == "__main__":
    main()
