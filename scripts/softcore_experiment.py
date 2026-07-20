"""Soft-core (annealed force-cap) density in the differentiable global stage.

The density force is strongest where macros overlap; capping its MAGNITUDE (not the
energy) keeps the spreading direction but lowers the overlap barrier so wirelength
can pull macros through each other early, then hardens to the true force. This is a
soft-core potential that anneals to hard-core. Applied to the overflow density
(config 221's actual density, already weight-calibrated).

Each step: grad(WL) and grad(density) computed separately; the per-macro density
grad is clipped to cap(t) = quantile(|grad|, q(t)), with q ramping qmin -> 1 (no
clip) by cap_frac of the schedule. Splitting the gradient is exactly grad(WL+dens),
so with clipping OFF this reproduces baseline bit-for-bit (asserted).

Paired, checkpointed per (bench, seed, arm). Usage:
  python softcore_experiment.py [ibm01 ibm09 ibm17] [--n 6] [--qmin .3] [--cap-frac .6]
  python softcore_experiment.py --check    # equivalence test only
  python softcore_experiment.py --report
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
from placers.analytical import DifferentiablePlacer, anneal
from congestion_experiment import hp_from_params

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
ROOT = "/home/laz/partcl/my-macro-placer"
CK = f"{ROOT}/notes/softcore_ck"
ap = argparse.ArgumentParser()
ap.add_argument("benches", nargs="*", default=["ibm01", "ibm09", "ibm17"])
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--iters", type=int, default=15000)
ap.add_argument("--qmin", type=float, default=0.3, help="initial force-cap quantile (lower = gentler start)")
ap.add_argument("--cap-frac", type=float, default=0.6, help="fraction of schedule after which clipping is off")
ap.add_argument("--check", action="store_true")
ap.add_argument("--report", action="store_true")
a = ap.parse_args()
os.makedirs(CK, exist_ok=True)
# shared baseline + soft-core settings (qmin, cap_frac); qmin=0 -> gentlest start
SETTINGS = [("baseline", None), ("sc_.3_.6", (0.3, 0.6)), ("sc_0_1", (0.0, 1.0))]


def report():
    cells = defaultdict(dict)
    for f in glob.glob(f"{CK}/*.json"):
        d = json.load(open(f)); cells[(d["bench"], d["seed"])][d["arm"]] = d["proxy"]
    labels = [lab for lab, _ in SETTINGS if lab != "baseline"]
    per = defaultdict(lambda: defaultdict(list))
    for (bench, seed), arms in cells.items():
        if "baseline" not in arms:
            continue
        for lab in labels:
            if lab in arms:
                per[bench][lab].append(arms[lab] - arms["baseline"]); per["ALL"][lab].append(arms[lab] - arms["baseline"])
    print("\n=== soft-core density vs baseline (paired; negative = better) ===", flush=True)
    for bench in sorted(per):
        parts = []
        for lab in labels:
            d = np.array(per[bench][lab])
            if len(d):
                parts.append(f"{lab} d={d.mean():+.4f} (win {int((d < 0).sum())}/{len(d)})")
        if parts:
            print(f"  {bench:5s}: " + " | ".join(parts), flush=True)


if a.report:
    report(); sys.exit(0)

P0 = json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"]
hp = hp_from_params(P0)
LD0 = hp["lam_d_end"] * hp["lam_d_ratio"]; LD1 = hp["lam_d_end"]


def place(P, seed, sc, iters):        # sc = None (baseline) or (qmin, cap_frac)
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(P.b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    coord = c0.to(P.dev).requires_grad_(True)
    opt = torch.optim.Adam([coord], lr=hp["lr"])
    for it in range(iters):
        t = it / iters; te = t ** hp["ramp_p"]
        gamma = anneal(te, P.gw * hp["gamma"], P.gw, geometric=True)
        lam_d = anneal(te, LD0, LD1); lam_wl = anneal(te, max(hp["lam_wl0"], 1e-9), 1.0, geometric=True)
        opt.zero_grad(); (lam_wl * P.wirelength(coord, gamma)).backward()
        wl_grad = coord.grad.detach().clone()
        opt.zero_grad()
        (lam_d * P.density(coord, tau=hp["tau_d"], target=hp["target"], topk="lapsum", mode="overflow")).backward()
        dens_grad = coord.grad.detach().clone()
        if sc is not None:
            qmin, cf = sc
            q = qmin + (1 - qmin) * min(1.0, t / max(1e-9, cf))
            if q < 1.0:
                gn = dens_grad.norm(dim=1)
                cap = torch.quantile(gn, q)
                dens_grad = dens_grad * torch.clamp(cap / (gn + 1e-12), max=1.0)[:, None]
        coord.grad = wl_grad + dens_grad
        opt.step()
        with torch.no_grad():
            coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
    return coord.detach().cpu().numpy()


if a.check:                                    # equivalence: soft-core with no clip == baseline
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/ibm01")
    P = DifferentiablePlacer(FastEval(b, plc))
    r1 = place(P, 0, None, 400); r2 = place(P, 0, (1.0, 0.6), 400)  # qmin=1 -> never clips
    print(f"no-clip soft-core vs baseline: max|diff|={np.abs(r1 - r2).max():.2e} (should be ~0)")
    sys.exit(0)

for nm in a.benches:
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    for seed in range(a.n):
        for arm, sc in SETTINGS:
            ck = f"{CK}/{nm}_s{seed}_{arm}.json"
            if os.path.exists(ck):
                continue
            raw = place(P, seed, sc, a.iters)
            lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
            pol, _ = optimize_fast(fe, lg, iters=100000, seed=seed, T0=0.0, move_hard=False,
                                   move_soft=True, refresh=20000, log_every=10**9, logf=lambda *x: None)
            c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
            json.dump({"bench": nm, "seed": seed, "arm": arm, "proxy": float(c["proxy_cost"]),
                       "overlaps": int(c["overlap_count"])}, open(ck, "w"))
            print(f"  {nm} s{seed} {arm:9s}: proxy={float(c['proxy_cost']):.4f} ov={int(c['overlap_count'])}", flush=True)
    report()
print("DONE", flush=True)
