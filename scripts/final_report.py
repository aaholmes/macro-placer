"""Stage 4: overnight final report. Combines the three methods per benchmark --
diff-alone (notes/validation), diff+greedy (notes/polish), and greedy-from-
reference (notes/results, if present) -- and reports the elementwise minimum as
the personal-best submission, with the absolute average proxy over all 17 (the
leaderboard's ranking metric) alongside #1 Archgen 0.9507.
"""
import sys, json, glob, os
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np

ROOT = "/home/laz/partcl/my-macro-placer"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
LB1 = 0.9507


def load(pat, key):
    d = {}
    for f in glob.glob(pat):
        j = json.load(open(f)); d[j["benchmark"]] = j.get(key)
    return d

diff = load(f"{ROOT}/notes/validation/*.json", "proxy")
pol = load(f"{ROOT}/notes/polish/*.json", "polished_proxy")
# greedy-from-reference: only zero-overlap results count
gref = {}
for f in glob.glob(f"{ROOT}/notes/results/*.json"):
    j = json.load(open(f))
    if j.get("method", "greedy") == "greedy" and j.get("overlaps", 1) == 0:
        gref[j["benchmark"]] = j["tilos"]

hdr = f"{'bench':>6} {'diff':>9} {'diff+greedy':>12} {'greedy-ref':>11} {'BEST':>9}"
print(hdr); print("-" * len(hdr))
best_rows = {}
for nm in ALL:
    cands = {"diff": diff.get(nm), "diff+greedy": pol.get(nm), "greedy-ref": gref.get(nm)}
    vals = {k: v for k, v in cands.items() if v is not None}
    bv = min(vals.values()) if vals else None
    best_rows[nm] = bv
    def fmt(x): return f"{x:>9.4f}" if isinstance(x, float) else f"{'--':>9}"
    print(f"{nm:>6} {fmt(diff.get(nm))} {fmt(pol.get(nm)):>12} {fmt(gref.get(nm)):>11} "
          f"{fmt(bv)}" + ("" if bv is None else f"  <- {min(vals, key=vals.get)}"))
print("-" * len(hdr))

have = [nm for nm in ALL if best_rows[nm] is not None]
if have:
    best_avg = float(np.mean([best_rows[nm] for nm in have]))
    diff_avg = float(np.mean([diff[nm] for nm in have if diff.get(nm) is not None]))
    print(f"\nabsolute avg proxy over {len(have)}/17 benches (leaderboard metric, lower=better):")
    print(f"  diff-alone     : {diff_avg:.4f}")
    print(f"  PERSONAL BEST  : {best_avg:.4f}   (elementwise min over methods)")
    print(f"  leaderboard #1 : {LB1:.4f} 'Archgen'    #2: 0.9522")
    delta = 100 * (best_avg - LB1) / LB1
    verdict = f"BEAT #1 by {-delta:.1f}%" if best_avg < LB1 else f"{delta:.1f}% above #1"
    print(f"  => personal best is {verdict}")
    if len(have) < 17:
        print(f"  NOTE: only {len(have)}/17 benchmarks present -- partial result.")
    json.dump({"per_bench_best": best_rows, "best_avg": best_avg, "diff_avg": diff_avg,
               "n": len(have), "leaderboard_1": LB1},
              open(f"{ROOT}/notes/final_report.json", "w"), indent=2)
    print("\nsaved notes/final_report.json")
else:
    print("no results yet")
