"""Complete comparison: our best result vs SA and RePlAce baselines across all
17 IBM benchmarks + average. Also overlays greedy vs best-of."""
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "/home/laz/partcl/my-macro-placer/notes/results"
RP = {'ibm01':0.9976,'ibm02':1.8370,'ibm03':1.3222,'ibm04':1.3024,'ibm06':1.6187,
'ibm07':1.4633,'ibm08':1.4285,'ibm09':1.1194,'ibm10':1.5009,'ibm11':1.1774,
'ibm12':1.7261,'ibm13':1.3355,'ibm14':1.5436,'ibm15':1.5159,'ibm16':1.4780,
'ibm17':1.6446,'ibm18':1.7722}
SA = {'ibm01':1.3166,'ibm02':1.9072,'ibm03':1.7401,'ibm04':1.5037,'ibm06':2.5057,
'ibm07':2.0229,'ibm08':1.9239,'ibm09':1.3875,'ibm10':2.1108,'ibm11':1.7111,
'ibm12':2.8261,'ibm13':1.9141,'ibm14':2.2750,'ibm15':2.3000,'ibm16':2.2337,
'ibm17':3.6726,'ibm18':2.7755}
data = {}
for f in glob.glob(f"{RES}/*.json"):
    d = json.load(open(f)); data[(d["benchmark"], d.get("method", "greedy"))] = d
B = list(RP)
greedy = [data[(b, "greedy")]["tilos"] for b in B]
bestof = [data[(b, "bestof")]["tilos"] for b in B]
ours = [min(g, o) for g, o in zip(greedy, bestof)]
import statistics as st
def avg(x): return st.mean(x)
B2 = B + ["AVG"]
sa = [SA[b] for b in B] + [avg([SA[b] for b in B])]
rp = [RP[b] for b in B] + [avg([RP[b] for b in B])]
ourb = ours + [avg(ours)]

# ---- Figure 1: grouped bars, ours vs baselines ----
x = np.arange(len(B2)); w = 0.27
fig, ax = plt.subplots(figsize=(16, 7))
ax.bar(x - w, sa, w, label=f"SA (avg {sa[-1]:.3f})", color="#c0c0c0")
ax.bar(x, rp, w, label=f"RePlAce (avg {rp[-1]:.3f})", color="#f0a35e")
ax.bar(x + w, ourb, w, label=f"ours (avg {ourb[-1]:.3f})", color="#2563eb")
for i, v in enumerate(ourb):
    ax.text(x[i] + w, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, rotation=90, color="#2563eb")
ax.axvline(len(B) - 0.5, color="k", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(B2, rotation=45)
ax.set_ylabel("proxy cost (lower is better)")
ax.set_title("Macro placement: our best vs SA / RePlAce baselines (17 IBM benchmarks)\n"
             f"average {ourb[-1]:.3f}  vs RePlAce {rp[-1]:.3f} (+{100*(rp[-1]-ourb[-1])/rp[-1]:.1f}%)  "
             f"and SA {sa[-1]:.3f} (+{100*(sa[-1]-ourb[-1])/sa[-1]:.1f}%)")
ax.legend(); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig("/home/laz/partcl/my-macro-placer/notes/final_comparison.png", dpi=110)
print("saved final_comparison.png")

# ---- Figure 2: greedy vs best-of A/B ----
fig2, ax2 = plt.subplots(figsize=(16, 6))
ax2.bar(x[:-1] - 0.2, greedy, 0.4, label=f"greedy (avg {avg(greedy):.4f})", color="#2563eb")
ax2.bar(x[:-1] + 0.2, bestof, 0.4, label=f"best-of (avg {avg(bestof):.4f})", color="#16a34a")
ax2.set_xticks(x[:-1]); ax2.set_xticklabels(B, rotation=45)
ax2.set_ylabel("proxy cost"); ax2.set_ylim(min(greedy+bestof) - 0.05, max(greedy+bestof) + 0.05)
ax2.set_title("A/B: greedy (deep) vs best-of-operators (fresh) — essentially tied")
ax2.legend(); ax2.grid(axis="y", alpha=0.25)
fig2.tight_layout(); fig2.savefig("/home/laz/partcl/my-macro-placer/notes/greedy_bestof_ab.png", dpi=110)
print("saved greedy_bestof_ab.png")
print(f"ours avg {ourb[-1]:.4f}  greedy {avg(greedy):.4f}  bestof {avg(bestof):.4f}")
