"""Aggregate per-benchmark result JSONs into a table + average vs baselines."""
import json, glob, os, statistics as st
RES = "/home/laz/partcl/my-macro-placer/notes/results"
RP = {'ibm01':0.9976,'ibm02':1.8370,'ibm03':1.3222,'ibm04':1.3024,'ibm06':1.6187,
'ibm07':1.4633,'ibm08':1.4285,'ibm09':1.1194,'ibm10':1.5009,'ibm11':1.1774,
'ibm12':1.7261,'ibm13':1.3355,'ibm14':1.5436,'ibm15':1.5159,'ibm16':1.4780,
'ibm17':1.6446,'ibm18':1.7722}
rows = {}
for f in glob.glob(f"{RES}/*.json"):
    d = json.load(open(f)); rows[d["benchmark"]] = d
print(f"{'bench':>6} {'ours':>7} {'RePlAce':>8} {'vsRP':>7} {'ov':>3} {'iters':>8} {'sec':>6}")
ours = []; rp = []
for b in RP:
    if b not in rows:
        print(f"{b:>6}   (pending)"); continue
    o = rows[b]["tilos"]; ov = rows[b]["overlaps"]; it = rows[b]["iters"]; s = rows[b]["seconds"]
    ours.append(o); rp.append(RP[b])
    print(f"{b:>6} {o:7.4f} {RP[b]:8.4f} {100*(RP[b]-o)/RP[b]:+6.1f}% {ov:>3} {it:>8} {s:>6.0f}")
if ours:
    print("-" * 52)
    print(f"{'AVG':>6} {st.mean(ours):7.4f} {st.mean(rp):8.4f} "
          f"{100*(st.mean(rp)-st.mean(ours))/st.mean(rp):+6.1f}%   (n={len(ours)}/17)")
    print(f"\nRePlAce avg 1.4578 | leaderboard: #1 0.9507, #11 1.0121, #24 1.0815")
