"""Three-arm comparison of jump strategies in the differentiable global stage.

For each (benchmark, config, seed) we run all three arms from the SAME seed, so both
jump strategies compare against the same baseline (cancels run-to-run GPU noise for
both deltas and gives a direct wl-macro vs net-collapse head-to-head):

  baseline      plain diff, no jumps
  wl-macro      teleport top-wirelength macros to their net-neighbour centroid
  net-collapse  collapse whole high-HPWL small nets (<=net-cap pins) to their center

Each arm: diff -> legalize -> greedy(same seed) -> true proxy. Checkpointed per
(bench, config, seed, arm) so it resumes and partial results are readable anytime.

Usage: python jump_experiment.py [ibm01 ibm09 ibm17] [--configs 221] [--n 6]
       [--jump-frac .5] [--jump-every 500] [--k-frac .05] [--noise .05]
       [--net-cap 20] [--nets-per-batch 8]   |   [--report]
"""
import os, sys, json, glob, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import numpy as np, torch, optuna
from collections import defaultdict
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss
from congestion_experiment import hp_from_params

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
CK = f"{ROOT}/notes/jump_ck"
ARMS = ["baseline", "wl-macro", "net-collapse"]
ap = argparse.ArgumentParser()
ap.add_argument("benches", nargs="*", default=["ibm01", "ibm09", "ibm17"])
ap.add_argument("--configs", default="221")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--iters", type=int, default=15000)
ap.add_argument("--jump-frac", type=float, default=0.5)
ap.add_argument("--jump-every", type=int, default=500)
ap.add_argument("--k-frac", type=float, default=0.05)
ap.add_argument("--noise", type=float, default=0.05)
ap.add_argument("--net-cap", type=int, default=20)
ap.add_argument("--nets-per-batch", type=int, default=8)
ap.add_argument("--report", action="store_true", help="just aggregate existing checkpoints")
a = ap.parse_args()
os.makedirs(CK, exist_ok=True)


def report():
    cells = defaultdict(dict)          # (bench,cfg,seed) -> {arm: proxy}
    for f in glob.glob(f"{CK}/*.json"):
        d = json.load(open(f))
        cells[(d["bench"], d["cfg"], d["seed"])][d["arm"]] = d["proxy"]
    per = defaultdict(lambda: {"wl": [], "nc": []})
    for (bench, cfg, seed), arms in cells.items():
        if not all(x in arms for x in ARMS):
            continue
        per[bench]["wl"].append(arms["wl-macro"] - arms["baseline"])
        per[bench]["nc"].append(arms["net-collapse"] - arms["baseline"])
        per["ALL"]["wl"].append(arms["wl-macro"] - arms["baseline"])
        per["ALL"]["nc"].append(arms["net-collapse"] - arms["baseline"])
    print(f"\n=== jump strategies vs baseline (paired; negative = jump better) ===", flush=True)
    for bench in sorted(per):
        wl, nc = np.array(per[bench]["wl"]), np.array(per[bench]["nc"])
        if not len(wl):
            continue
        print(f"  {bench:5s} (n={len(wl):2d}): wl-macro d={wl.mean():+.4f} (win {int((wl<0).sum())}/{len(wl)}) | "
              f"net-collapse d={nc.mean():+.4f} (win {int((nc<0).sum())}/{len(nc)})", flush=True)


if a.report:
    report(); sys.exit(0)

cfg_ids = [int(x) for x in a.configs.split(",")]
study = optuna.load_study(study_name="diffplace", storage=f"sqlite:///{ROOT}/notes/bayes.db")
byn = {t.number: t for t in study.trials}
HP = {c: hp_from_params(byn[c].params) for c in cfg_ids}


def stats(P, coord):
    dev = P.dev; owner = P.owner; net = P.net; off = P.off
    nn = int(P.nw.shape[0]); nm = P.b.num_macros
    mv = owner >= 0
    px = torch.where(mv, coord[owner.clamp(min=0), 0] + off[:, 0], off[:, 0])
    py = torch.where(mv, coord[owner.clamp(min=0), 1] + off[:, 1], off[:, 1])
    def nred(src, red, init):
        return torch.full((nn,), init, device=dev).scatter_reduce(0, net, src, reduce=red, include_self=False)
    hpwl = (nred(px, "amax", -1e30) - nred(px, "amin", 1e30)) + (nred(py, "amax", -1e30) - nred(py, "amin", 1e30))
    cnt = torch.zeros(nn, device=dev).index_add(0, net, torch.ones_like(px))
    cx = torch.zeros(nn, device=dev).index_add(0, net, px) / cnt.clamp(min=1)
    cy = torch.zeros(nn, device=dev).index_add(0, net, py) / cnt.clamp(min=1)
    net_ctr = torch.stack([cx, cy], 1)
    mo, mn = owner[mv], net[mv]
    wl_macro = torch.zeros(nm, device=dev).index_add(0, mo, hpwl[mn] * P.nw[mn])
    tc = torch.zeros(nm, device=dev).index_add(0, mo, torch.ones_like(mo, dtype=torch.float32))
    tx = torch.zeros(nm, device=dev).index_add(0, mo, cx[mn]) / tc.clamp(min=1)
    ty = torch.zeros(nm, device=dev).index_add(0, mo, cy[mn]) / tc.clamp(min=1)
    return wl_macro, torch.stack([tx, ty], 1), tc, hpwl, net_ctr, cnt


def build_net_members(P):
    o = P.owner.cpu().numpy(); nt = P.net.cpu().numpy()
    mm = defaultdict(set)
    for pin in range(len(o)):
        if o[pin] >= 0:
            mm[int(nt[pin])].add(int(o[pin]))
    return {n: torch.tensor(sorted(v), device=P.dev, dtype=torch.long) for n, v in mm.items() if len(v) >= 2}


def place(P, loss_fn, iters, lr, seed, mode, net_members):
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(P.b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    coord = c0.to(P.dev).requires_grad_(True)
    opt = torch.optim.Adam([coord], lr=lr)
    rng = torch.Generator(device=P.dev).manual_seed(seed + 7)
    nm = P.b.num_macros; k = max(1, int(a.k_frac * nm)); win = int(a.jump_frac * iters)
    nscale = a.noise * min(P.W, P.H)

    def teleport(idx, target, temp):
        noise = torch.randn(idx.numel(), 2, generator=rng, device=P.dev) * nscale * temp
        np_ = target + noise
        np_[:, 0].clamp_(P.hw[idx], P.W - P.hw[idx]); np_[:, 1].clamp_(P.hh[idx], P.H - P.hh[idx])
        coord[idx] = np_
        st = opt.state.get(coord)
        if st:
            st["exp_avg"][idx] = 0; st["exp_avg_sq"][idx] = 0

    for it in range(iters):
        t = it / iters
        opt.zero_grad(); loss = loss_fn(P, coord, t); loss.backward(); opt.step()
        with torch.no_grad():
            coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
            if mode != "baseline" and it < win and it % a.jump_every == 0:
                wl, tgt, tc, net_hpwl, net_ctr, net_cnt = stats(P, coord)
                temp = 1.0 - it / max(1, win)
                if mode == "wl-macro":
                    wl = wl.clone(); wl[tc == 0] = -1.0
                    idx = torch.topk(wl, k).indices
                    teleport(idx, tgt[idx], temp)
                else:
                    hp = net_hpwl.clone()
                    hp[(net_cnt < 2) | (net_cnt > a.net_cap)] = -1.0
                    for n in torch.topk(hp, a.nets_per_batch).indices.tolist():
                        mem = net_members.get(n)
                        if mem is not None and mem.numel():
                            teleport(mem, net_ctr[n].unsqueeze(0).expand(mem.numel(), 2), temp)
    return coord.detach().cpu().numpy()


for nm in a.benches:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    netmem = build_net_members(P)
    for cfg in cfg_ids:
        hp = HP[cfg]
        loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                          lam_d1=hp["lam_d_end"], lam_c0=0.0, lam_c1=0.0, tau_d=hp["tau_d"],
                          tau_c=hp["tau_c"], target=hp["target"], topk="lapsum", dmode=hp["dmode"],
                          lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"])
        for seed in range(a.n):
            for arm in ARMS:
                ck = f"{CK}/{nm}_c{cfg}_s{seed}_{arm}.json"
                if os.path.exists(ck):
                    continue
                raw = place(P, loss, a.iters, hp["lr"], seed, arm, netmem)
                lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
                pol, _ = optimize_fast(fe, lg, iters=100000, seed=seed, T0=0.0, move_hard=False,
                                       move_soft=True, refresh=20000, log_every=10**9, logf=lambda *x: None)
                c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
                json.dump({"bench": nm, "cfg": cfg, "seed": seed, "arm": arm,
                           "proxy": float(c["proxy_cost"]), "overlaps": int(c["overlap_count"])}, open(ck, "w"))
                print(f"  {nm} c{cfg} s{seed} {arm:12s}: proxy={float(c['proxy_cost']):.4f} "
                      f"ov={int(c['overlap_count'])}", flush=True)
        report()
print("DONE", flush=True)
