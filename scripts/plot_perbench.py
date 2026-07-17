"""Per-benchmark bar chart for the write-up: our final proxy vs the reference,
read straight from notes/fullrun_report.json (the 1.0155 result) and the reference
abs from notes/validate_all.json. Saves notes/fig_perbench.png.
"""
import os, sys, json
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/laz/partcl/my-macro-placer"
rep = json.load(open(f"{ROOT}/notes/fullrun_report.json"))
ref = {r["benchmark"]: r["reference"] for r in json.load(open(f"{ROOT}/notes/validate_all.json"))["rows"]}
order = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
pb = {r["benchmark"]: min(r["best_after"], r["greedy_ref"]) for r in rep["rows"]}

ours = [pb[nm] for nm in order]
refs = [ref[nm] for nm in order]
avg_ours = float(np.mean(ours)); avg_ref = float(np.mean(refs)); lb1 = 0.9507
x = np.arange(len(order)); w = 0.4

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(x - w / 2, refs, w, label=f"reference (avg {avg_ref:.3f})", color="#cbd5e1")
ax.bar(x + w / 2, ours, w, label=f"ours (avg {avg_ours:.4f})", color="#16a34a")
ax.axhline(lb1, ls="--", lw=1, color="#b91c1c", label=f"leaderboard #1 avg ({lb1})")
ax.set_xticks(x); ax.set_xticklabels([nm.replace("ibm", "") for nm in order])
ax.set_xlabel("IBM benchmark"); ax.set_ylabel("proxy cost (lower is better)")
ax.set_title(f"Final proxy per benchmark — avg {avg_ours:.4f} (−{100*(1-avg_ours/avg_ref):.0f}% vs reference, zero overlaps)")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.margins(x=0.01); fig.tight_layout()
fig.savefig(f"{ROOT}/notes/fig_perbench.png", dpi=115, bbox_inches="tight")
print(f"saved fig_perbench.png  avg_ours={avg_ours:.4f} avg_ref={avg_ref:.3f}")
