"""Full end-to-end with the congestion-aware diff loss: all 17 benchmarks,
best-of-N diff (te-ramp congestion at the given lam_c / formulation) + greedy,
per-benchmark best, elementwise-min with greedy-from-reference. Reports the abs
average vs the prior personal best 1.0376.

Usage: python cong_endtoend.py --lam-c 0.003 [--log] [--seeds 6]
"""
import os, sys, json, glob, time, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np
from congestion_experiment import run_cell, hp_from_params

ROOT = "/home/laz/partcl/my-macro-placer"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
OUT = f"{ROOT}/notes/cong_e2e"
PRIOR = 1.0376

ap = argparse.ArgumentParser()
ap.add_argument("--lam-c", type=float, required=True)
ap.add_argument("--log", action="store_true")
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--iters", type=int, default=15000)
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
hp = hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])

gref = {}
for f in glob.glob(f"{ROOT}/notes/results/*.json"):
    d = json.load(open(f))
    if d.get("method", "greedy") == "greedy" and d.get("overlaps", 1) == 0:
        gref[d["benchmark"]] = d["tilos"]

tag = "log" if a.log else "raw"
rows = []
for nm in ALL:
    ck = f"{OUT}/{nm}_{tag}_lc{a.lam_c}.json"
    if os.path.exists(ck):
        r = json.load(open(ck))
    else:
        t0 = time.time()
        r = run_cell(nm, hp, a.lam_c, a.iters, a.seeds, use_log=a.log)
        r["seconds"] = time.time() - t0
        json.dump(r, open(ck, "w"))
    cong_final = min([x for x in [r["proxy"], gref.get(nm)] if x is not None])
    rows.append({"benchmark": nm, "cong_proxy": r["proxy"], "greedy_ref": gref.get(nm),
                 "best": cong_final, "cong": r["cong"]})
    print(f"[{time.strftime('%H:%M:%S')}] {nm}: cong-loss proxy={r['proxy']:.4f} "
          f"(cong={r['cong']:.4f})  greedy-ref={gref.get(nm)}  BEST={cong_final:.4f}", flush=True)

avg = float(np.mean([r["best"] for r in rows]))
avg_cong_only = float(np.mean([r["cong_proxy"] for r in rows]))
print(f"\n=== END-TO-END (lam_c={a.lam_c} {tag}, best-of-{a.seeds}) ===")
print(f"  congestion-loss pipeline avg: {avg_cong_only:.4f}")
print(f"  + greedy-ref (min) => {avg:.4f}")
print(f"  prior personal best {PRIOR:.4f}  ({100*(PRIOR-avg)/PRIOR:+.1f}% vs prior)")
json.dump({"lam_c": a.lam_c, "tag": tag, "avg": avg, "avg_cong_only": avg_cong_only,
           "rows": rows, "prior": PRIOR}, open(f"{ROOT}/notes/cong_e2e_report.json", "w"), indent=2)
print("saved notes/cong_e2e_report.json", flush=True)
