"""Does the L-route congestion surrogate, added to the differentiable loss,
reduce the TRUE final proxy on the congestion-bound benchmarks?

For each hard benchmark and each congestion weight lam_c, build the loss as
(tuned WL+density, RUDY off) + lam_c * ramp(t) * congestion_lroute (normalized to
the reference magnitude so lam_c is comparable to the density weight). Run
best-of-N diff -> legalize -> greedy polish; record the true proxy and its
congestion component. lam_c=0 is the baseline (our current pipeline).

Checkpointed per (bench, lam_c) to notes/cong_exp/.
Usage: python congestion_experiment.py [--benches ...] [--lam-cs 0,0.05,0.1,0.2,0.4]
       [--iters 15000] [--seeds 2]
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
OUT = f"{ROOT}/notes/cong_exp"


def hp_from_params(p):
    dmode = p["dmode"]
    lam_d_end = p.get("lam_d_end_e") if dmode == "electrostatic" else p.get("lam_d_end_o")
    return dict(dmode=dmode, lam_d_end=lam_d_end, lam_d_ratio=p["lam_d_ratio"], gamma=p["gamma"],
                tau_d=p["tau_d"], tau_c=p["tau_c"], lam_wl0=p["lam_wl0"], target=p["target"],
                lr=p["lr"], ramp_p=p["ramp_p"], lsteps=p["lsteps_n"], iters=p["iters"])


def run_cell(nm, hp, lam_c, iters, seeds):
    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy()
    base = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                      lam_d1=hp["lam_d_end"], lam_c0=0.0, lam_c1=0.0, tau_d=hp["tau_d"],
                      tau_c=hp["tau_c"], target=hp["target"], topk="lapsum", dmode=hp["dmode"],
                      lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"], legalize_steps=hp["lsteps"])
    # normalize congestion term to the reference placement's magnitude
    ref = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                   b.canvas_width, b.canvas_height, gap=0.01)[0]
    with torch.no_grad():
        norm = max(1e-6, float(P.congestion_lroute(torch.tensor(ref[:, :2], dtype=torch.float32, device=P.dev))))

    def loss(Pl, coord, t):
        L = base(Pl, coord, t)
        if lam_c > 0:
            # te-ramp: congestion on the density schedule (off during spread, on when clustered)
            L = L + lam_c * (t ** hp["ramp_p"]) * Pl.congestion_lroute(coord) / norm
        return L

    best = None
    for s in range(seeds):
        raw = P.place(loss, pos0=None, iters=iters, lr=hp["lr"], seed=s, logf=None)
        lg, _, _ = legalize(raw, sz, b.num_hard_macros, b.canvas_width, b.canvas_height, gap=0.01)
        pol, _ = optimize_fast(fe, lg, iters=100000, seed=0, T0=0.0, move_hard=False,
                               move_soft=True, refresh=100000, log_every=200000, logf=lambda *a: None)
        c = compute_proxy_cost(torch.tensor(pol, dtype=torch.float32), b, plc)
        r = dict(proxy=float(c["proxy_cost"]), cong=0.5 * float(c["congestion_cost"]),
                 overlaps=int(c["overlap_count"]))
        if best is None or r["proxy"] < best["proxy"]:
            best = r
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="ibm06,ibm08,ibm12,ibm14,ibm15,ibm17,ibm18")
    ap.add_argument("--lam-cs", default="0,0.05,0.1,0.2,0.4")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
    lam_cs = [float(x) for x in a.lam_cs.split(",")]
    for nm in a.benches.split(","):
        row = {"benchmark": nm}
        for lc in lam_cs:
            ck = f"{OUT}/{nm}_lc{lc}.json"
            if os.path.exists(ck):
                r = json.load(open(ck))
            else:
                t0 = time.time(); r = run_cell(nm, hp, lc, a.iters, a.seeds); r["seconds"] = time.time() - t0
                json.dump(r, open(ck, "w"))
            row[f"lc{lc}"] = r
            print(f"[{time.strftime('%H:%M:%S')}] {nm} lam_c={lc}: proxy={r['proxy']:.4f} "
                  f"cong={r['cong']:.4f} ov={r['overlaps']}", flush=True)
        base_p = row[f"lc0"]["proxy"]
        best_lc = min(lam_cs, key=lambda lc: row[f"lc{lc}"]["proxy"])
        best_p = row[f"lc{best_lc}"]["proxy"]
        print(f"  => {nm}: baseline(lc0)={base_p:.4f}  best(lc={best_lc})={best_p:.4f}  "
              f"gain={base_p-best_p:+.4f}", flush=True)
