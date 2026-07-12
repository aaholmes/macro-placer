"""Full-run stage C (report): per benchmark, pick the best POST-greedy seed.
Compares three things:
  - best-of-N chosen AFTER greedy (what we now do)  vs
  - best-of-N chosen BEFORE greedy (old: polish only the best diff seed)
to quantify whether after-greedy selection actually helps, and reports the new
personal best (elementwise min with greedy-from-reference) vs the prior 1.0376
and leaderboard #1 (0.9507).
"""
import sys, json, glob, os
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np

ROOT = "/home/laz/partcl/my-macro-placer"
OUT = f"{ROOT}/notes/fullrun"
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
    dj = f"{OUT}/{nm}_diff.json"
    if not os.path.exists(dj):
        continue
    diffs = json.load(open(dj))["seeds"]                       # {seed: diff_proxy}
    ref = json.load(open(dj))["reference"]
    finals = {}
    for f in glob.glob(f"{OUT}/{nm}_s*_final.json"):
        d = json.load(open(f)); finals[str(d["seed"])] = d["final"]
    common = [k for k in diffs if k in finals]
    if not common:
        continue
    best_after = min(finals[k] for k in common)                # select on final
    best_diff_seed = min(common, key=lambda k: diffs[k])       # old: select on diff
    final_if_before = finals[best_diff_seed]
    after_sel.append(best_after); before_sel.append(final_if_before)
    if abs(final_if_before - best_after) < 1e-9:
        agree += 1
    pb = min([x for x in [best_after, gref.get(nm)] if x is not None])
    rows.append({"benchmark": nm, "ref": ref, "best_after": best_after,
                 "before_sel": final_if_before, "greedy_ref": gref.get(nm),
                 "personal_best": pb, "ratio": pb / ref, "n_seeds": len(common)})

if rows:
    hdr = f"{'bench':>6} {'ref':>7} {'after':>8} {'before':>8} {'greedy-ref':>11} {'BEST':>8} {'ratio':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        gr = f"{r['greedy_ref']:.4f}" if r["greedy_ref"] else "--"
        print(f"{r['benchmark']:>6} {r['ref']:>7.3f} {r['best_after']:>8.4f} {r['before_sel']:>8.4f} "
              f"{gr:>11} {r['personal_best']:>8.4f} {r['ratio']:>6.3f}")
    print("-" * len(hdr))
    n = len(rows)
    pb_avg = np.mean([r["personal_best"] for r in rows])
    after_avg = np.mean(after_sel); before_avg = np.mean(before_sel)
    print(f"\nabs avg over {n}/17 (lower=better):")
    print(f"  best-of-N BEFORE greedy (old) : {before_avg:.4f}")
    print(f"  best-of-N AFTER  greedy (new) : {after_avg:.4f}   (gain {before_avg-after_avg:+.4f})")
    print(f"  + greedy-from-ref (elementwise min) => PERSONAL BEST: {pb_avg:.4f}")
    print(f"  prior personal best: {PRIOR:.4f}   ({100*(PRIOR-pb_avg)/PRIOR:+.1f}% vs prior)")
    print(f"  leaderboard #1: {LB1:.4f}   ({100*(pb_avg-LB1)/LB1:+.1f}% vs #1)")
    print(f"  after==before on {agree}/{n} benches (diff-rank vs final-rank agreement)")
    if n < 17:
        print(f"  NOTE: {n}/17 benchmarks present (partial).")
    json.dump({"rows": rows, "pb_avg": float(pb_avg), "after_avg": float(after_avg),
               "before_avg": float(before_avg), "n": n, "prior": PRIOR, "lb1": LB1},
              open(f"{ROOT}/notes/fullrun_report.json", "w"), indent=2)
    print("\nsaved notes/fullrun_report.json")
else:
    print("no results yet")
