"""Aggregate per-benchmark result JSONs into a table + average vs baselines.
Reports each method (greedy / bestof) as its own column for A/B comparison."""
import json, glob, statistics as st
RES = "/home/laz/partcl/my-macro-placer/notes/results"
RP = {'ibm01':0.9976,'ibm02':1.8370,'ibm03':1.3222,'ibm04':1.3024,'ibm06':1.6187,
'ibm07':1.4633,'ibm08':1.4285,'ibm09':1.1194,'ibm10':1.5009,'ibm11':1.1774,
'ibm12':1.7261,'ibm13':1.3355,'ibm14':1.5436,'ibm15':1.5159,'ibm16':1.4780,
'ibm17':1.6446,'ibm18':1.7722}
data = {}  # (benchmark, method) -> row
for f in glob.glob(f"{RES}/*.json"):
    d = json.load(open(f)); data[(d["benchmark"], d.get("method", "greedy"))] = d
methods = sorted({m for (_, m) in data})
hdr = f"{'bench':>6} {'RePlAce':>8}" + "".join(f" {m:>10}" for m in methods)
print(hdr); print("-" * len(hdr))
avg = {m: [] for m in methods}
for b in RP:
    line = f"{b:>6} {RP[b]:>8.4f}"
    for m in methods:
        d = data.get((b, m))
        if d and d["overlaps"] == 0:
            line += f" {d['tilos']:>10.4f}"; avg[m].append(d["tilos"])
        elif d:
            line += f" {'ov='+str(d['overlaps']):>10}"
        else:
            line += f" {'--':>10}"
    print(line)
print("-" * len(hdr))
line = f"{'AVG':>6} {st.mean(RP.values()):>8.4f}"
for m in methods:
    line += f" {st.mean(avg[m]):>10.4f}" if avg[m] else f" {'--':>10}"
print(line + f"   (RePlAce 1.4578)")
for m in methods:
    if avg[m]:
        print(f"  {m}: avg {st.mean(avg[m]):.4f} over {len(avg[m])}/17, "
              f"{100*(1.4578-st.mean(avg[m]))/1.4578:+.1f}% vs RePlAce")
