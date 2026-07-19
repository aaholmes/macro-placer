"""ibm17 gentleness sweep: does a less-aggressive net-collapse turn ibm17 from a
loss into neutral/win? Per seed we run one shared baseline + a ladder of gentler
net-collapse settings, so every setting is paired against the same baseline.
Checkpointed per (seed, label); resumable; --report aggregates.

Usage: python jump_sweep.py [--bench ibm17] [--n 4]  |  [--report]
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
    return None, None, None, hpwl, torch.stack([cx, cy], 1), cnt


def build_net_members(P):
    o = P.owner.cpu().numpy(); nt = P.net.cpu().numpy()
    mm = defaultdict(set)
    for pin in range(len(o)):
        if o[pin] >= 0:
            mm[int(nt[pin])].add(int(o[pin]))
    return {n: torch.tensor(sorted(v), device=P.dev, dtype=torch.long) for n, v in mm.items() if len(v) >= 2}

ROOT = "/home/laz/partcl/my-macro-placer"
CHAL = "/home/laz/partcl/macro-place-challenge-2026"
CK = f"{ROOT}/notes/jump_sweep_ck"
# gentleness ladder for net-collapse (noise = frac of canvas; nets = collapsed/batch;
# frac = jump window; every = batch period). Baseline first.
SETTINGS = [
    {"label": "baseline", "mode": "baseline"},
    {"label": "nc_cur",   "mode": "net-collapse", "noise": 0.05, "nets": 8, "frac": 0.50, "every": 500, "cap": 20},
    {"label": "nc_g1",    "mode": "net-collapse", "noise": 0.02, "nets": 4, "frac": 0.50, "every": 500, "cap": 20},
    {"label": "nc_g2",    "mode": "net-collapse", "noise": 0.01, "nets": 2, "frac": 0.50, "every": 500, "cap": 20},
    {"label": "nc_early", "mode": "net-collapse", "noise": 0.02, "nets": 4, "frac": 0.25, "every": 500, "cap": 20},
    {"label": "nc_rare",  "mode": "net-collapse", "noise": 0.02, "nets": 2, "frac": 0.35, "every": 1000, "cap": 12},
    # soft-only: collapse only the soft (standard-cell) members of a net, leave hard macros put
    {"label": "nc_cur_soft", "mode": "net-collapse", "noise": 0.05, "nets": 8, "frac": 0.50, "every": 500, "cap": 20, "soft": True},
    {"label": "nc_soft_g",   "mode": "net-collapse", "noise": 0.02, "nets": 4, "frac": 0.50, "every": 500, "cap": 20, "soft": True},
]
ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="ibm17")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--iters", type=int, default=15000)
ap.add_argument("--report", action="store_true")
a = ap.parse_args()
os.makedirs(CK, exist_ok=True)


def report():
    cells = defaultdict(dict)
    for f in glob.glob(f"{CK}/*.json"):
        d = json.load(open(f)); cells[d["seed"]][d["label"]] = d["proxy"]
    labels = [s["label"] for s in SETTINGS if s["label"] != "baseline"]
    agg = {lab: [] for lab in labels}
    for seed, r in cells.items():
        if "baseline" not in r:
            continue
        for lab in labels:
            if lab in r:
                agg[lab].append(r[lab] - r["baseline"])
    print(f"\n=== ibm17 net-collapse gentleness vs shared baseline (negative = better) ===", flush=True)
    for lab in labels:
        d = np.array(agg[lab])
        if len(d):
            print(f"  {lab:9s} (n={len(d)}): d={d.mean():+.4f}+-{d.std():.4f} (win {int((d<0).sum())}/{len(d)})", flush=True)


if a.report:
    report(); sys.exit(0)

study = optuna.load_study(study_name="diffplace", storage=f"sqlite:///{ROOT}/notes/bayes.db")
hp = hp_from_params({t.number: t for t in study.trials}[221].params)
b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{a.bench}")
fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
netmem = build_net_members(P)
NH = b.num_hard_macros                                    # soft macros are indices [NH, num_macros)
loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"], lam_d1=hp["lam_d_end"],
                  lam_c0=0.0, lam_c1=0.0, tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"],
                  topk="lapsum", dmode="overflow", lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"])


def place(seed, st):
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    coord = c0.to(P.dev).requires_grad_(True)
    opt = torch.optim.Adam([coord], lr=hp["lr"])
    rng = torch.Generator(device=P.dev).manual_seed(seed + 7)
    if st["mode"] != "baseline":
        win = int(st["frac"] * a.iters); nscale = st["noise"] * min(P.W, P.H)
    for it in range(a.iters):
        t = it / a.iters
        opt.zero_grad(); l = loss(P, coord, t); l.backward(); opt.step()
        with torch.no_grad():
            coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
            if st["mode"] != "baseline" and it < win and it % st["every"] == 0:
                _, _, _, net_hpwl, net_ctr, net_cnt = stats(P, coord)
                temp = 1.0 - it / max(1, win)
                hp2 = net_hpwl.clone(); hp2[(net_cnt < 2) | (net_cnt > st["cap"])] = -1.0
                for n in torch.topk(hp2, st["nets"]).indices.tolist():
                    mem = netmem.get(n)
                    if mem is not None and st.get("soft"):
                        mem = mem[mem >= NH]                # collapse only soft members
                    if mem is not None and mem.numel():
                        noise = torch.randn(mem.numel(), 2, generator=rng, device=P.dev) * nscale * temp
                        np_ = net_ctr[n].unsqueeze(0) + noise
                        np_[:, 0].clamp_(P.hw[mem], P.W - P.hw[mem]); np_[:, 1].clamp_(P.hh[mem], P.H - P.hh[mem])
                        coord[mem] = np_
                        sta = opt.state.get(coord)
                        if sta:
                            sta["exp_avg"][mem] = 0; sta["exp_avg_sq"][mem] = 0
    return coord.detach().cpu().numpy()


for seed in range(a.n):
    for st in SETTINGS:
        ck = f"{CK}/{a.bench}_s{seed}_{st['label']}.json"
        if os.path.exists(ck):
            continue
        raw = place(seed, st)
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        pol, _ = optimize_fast(fe, lg, iters=100000, seed=seed, T0=0.0, move_hard=False,
                               move_soft=True, refresh=20000, log_every=10**9, logf=lambda *x: None)
        c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
        json.dump({"bench": a.bench, "seed": seed, "label": st["label"],
                   "proxy": float(c["proxy_cost"]), "overlaps": int(c["overlap_count"])}, open(ck, "w"))
        print(f"  {a.bench} s{seed} {st['label']:9s}: proxy={float(c['proxy_cost']):.4f} ov={int(c['overlap_count'])}", flush=True)
    report()
print("DONE", flush=True)
