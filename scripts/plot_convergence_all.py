"""Parse notes/overnight.log and plot proxy-vs-iterations convergence for all 17
IBM benchmarks (one panel each), with the per-benchmark RePlAce baseline as a
dotted line. No benchmark reloading — reads the log only."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/laz/partcl/my-macro-placer/notes/overnight.log"
OUT = "/home/laz/partcl/my-macro-placer/notes/convergence_all.png"
RP = {'ibm01':0.9976,'ibm02':1.8370,'ibm03':1.3222,'ibm04':1.3024,'ibm06':1.6187,
'ibm07':1.4633,'ibm08':1.4285,'ibm09':1.1194,'ibm10':1.5009,'ibm11':1.1774,
'ibm12':1.7261,'ibm13':1.3355,'ibm14':1.5436,'ibm15':1.5159,'ibm16':1.4780,
'ibm17':1.6446,'ibm18':1.7722}
ORDER = list(RP)

pat = re.compile(r"(ibm\d\d): (\d+) iters, fast=([\d.]+) tilos=([\d.]+) ov=(\d+)")
series = {b: [] for b in ORDER}
for ln in open(LOG):
    m = pat.search(ln)
    if m:
        b, it, fast, tl, ov = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5))
        series[b].append((it, fast, ov))

# keep only the last monotonic-increasing segment (drops ibm06's stale pass)
def last_segment(pts):
    seg = []
    for p in pts:
        if seg and p[0] < seg[-1][0]:
            seg = []
        seg.append(p)
    return seg

fig, axes = plt.subplots(5, 4, figsize=(18, 20))
axes = axes.ravel()
for k, b in enumerate(ORDER):
    ax = axes[k]
    seg = last_segment(series[b])
    if seg:
        its = [p[0] for p in seg]; fs = [p[1] for p in seg]
        ax.plot(its, fs, color="#2563eb", lw=2)
        ax.scatter([its[-1]], [fs[-1]], color="#2563eb", s=25, zorder=5)
        ax.annotate(f"{fs[-1]:.3f}", (its[-1], fs[-1]), textcoords="offset points",
                    xytext=(-4, 6), ha="right", fontsize=9, color="#2563eb")
    ax.axhline(RP[b], ls=":", color="#16a34a", lw=1.5)
    ax.annotate(f"RePlAce {RP[b]:.2f}", (0, RP[b]), textcoords="offset points",
                xytext=(4, 3), fontsize=8, color="#16a34a")
    ax.set_title(b, fontweight="bold")
    ax.set_xlabel("iterations"); ax.set_ylabel("proxy")
    ax.grid(alpha=0.2)
for k in range(len(ORDER), len(axes)):
    axes[k].axis("off")
fig.suptitle("IBM benchmarks: local-search convergence (blue) vs RePlAce baseline (green dotted)",
             fontsize=15, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT, dpi=90)
print(f"saved {OUT}")
for b in ORDER:
    seg = last_segment(series[b])
    if seg:
        print(f"{b}: {seg[-1][0]} iters -> {seg[-1][1]:.4f} (ov={seg[-1][2]})")
