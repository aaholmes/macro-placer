"""Full-run stage C (report): per benchmark, pick the best POST-greedy candidate
from the pool (top-K configs x many seeds). Reports:
  - best-of-N AFTER greedy (new) vs BEFORE greedy (old: polish only best diff),
  - the winning config per benchmark,
  - the new personal best (elementwise min with greedy-from-reference) vs the
    prior 1.0376 and leaderboard #1 (0.9507).
"""
import sys, json, glob, os, re
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np

ROOT = "/home/laz/partcl/my-macro-placer"
OUT = os.environ.get("FULLRUN_DIR", f"{ROOT}/notes/fullrun")
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
PRIOR = 1.0376
LB1 = 0.9507

gref = {}
for f in glob.glob(f"{ROOT}/notes/results/*.json"):
    d = json.load(open(f))
    if d.get("method", "greedy") == "greedy" and d.get("overlaps", 1) == 0:
        gref[d["benchmark"]] = d["tilos"]

rows = []
after_sel, before_sel, agree = [], [], 0
for nm in ALL:
    mj = f"{OUT}/{nm}_meta.json"
    if not os.path.exists(mj):
        continue
    meta = {m["idx"]: m for m in json.load(open(mj))}       # idx -> {diff, trial, ...}
    finals = {}
    for f in glob.glob(f"{OUT}/{nm}_p*_final.json"):
        d = json.load(open(f)); finals[d["idx"]] = d["final"]
    common = [i for i in meta if i in finals]
    if not common:
        continue
    best_idx = min(common, key=lambda i: finals[i])          # select on FINAL
    best_after = finals[best_idx]
    best_diff_idx = min(common, key=lambda i: meta[i]["diff"])  # old: select on diff
    before = finals[best_diff_idx]
    after_sel.append(best_after); before_sel.append(before)
    if best_idx == best_diff_idx:
        agree += 1
    pb = min([x for x in [best_after, gref.get(nm)] if x is not None])
    rows.append({"benchmark": nm, "best_after": best_after, "before_sel": before,
                 "greedy_ref": gref.get(nm), "personal_best": pb,
                 "win_trial": meta[best_idx]["trial"], "n_cand": len(common)})

if rows:
    hdr = f"{'bench':>6} {'after':>8} {'before':>8} {'greedy-ref':>11} {'BEST':>8} {'win-cfg':>8} {'#cand':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        gr = f"{r['greedy_ref']:.4f}" if r["greedy_ref"] else "--"
        print(f"{r['benchmark']:>6} {r['best_after']:>8.4f} {r['before_sel']:>8.4f} {gr:>11} "
              f"{r['personal_best']:>8.4f} {('t'+str(r['win_trial'])):>8} {r['n_cand']:>6}")
    print("-" * len(hdr))
    n = len(rows)
    pb_avg = float(np.mean([r["personal_best"] for r in rows]))
    aft = float(np.mean(after_sel)); bef = float(np.mean(before_sel))
    print(f"\nabs avg over {n}/17 (lower=better):")
    print(f"  best-of-N BEFORE greedy (old) : {bef:.4f}")
    print(f"  best-of-N AFTER  greedy (new) : {aft:.4f}   (gain {bef-aft:+.4f})")
    print(f"  + greedy-from-ref (min)  => PERSONAL BEST: {pb_avg:.4f}")
    print(f"  prior personal best {PRIOR:.4f}  ({100*(PRIOR-pb_avg)/PRIOR:+.1f}% vs prior)")
    print(f"  leaderboard #1 {LB1:.4f}  ({100*(pb_avg-LB1)/LB1:+.1f}% vs #1)")
    print(f"  after==before on {agree}/{n} benches (does after-greedy selection change the pick?)")
    if n < 17:
        print(f"  NOTE: {n}/17 benchmarks present (partial).")
    json.dump({"rows": rows, "pb_avg": pb_avg, "after_avg": aft, "before_avg": bef,
               "n": n, "prior": PRIOR, "lb1": LB1},
              open(f"{OUT}_report.json", "w"), indent=2)
    print(f"\nsaved {OUT}_report.json")
else:
    print("no results yet")
