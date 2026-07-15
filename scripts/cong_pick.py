"""Read all congestion sweep cells (notes/cong_exp/) and print the best config as
'<tag> <lam_c>' (e.g. 'raw 0.002'), chosen by the lowest mean FINAL proxy across
the benchmarks tested. Used by the orchestrator to pick the end-to-end config.
Falls back to 'raw 0.0' (baseline) if nothing usable.
"""
import glob, json, re, os
from collections import defaultdict

OUT = "/home/laz/partcl/my-macro-placer/notes/cong_exp"
pat = re.compile(r"(ibm\d+)_(?:(log|raw)_)?lc([\d.]+)\.json$")
groups = defaultdict(dict)   # (tag, lc) -> {bench: proxy}
for f in glob.glob(f"{OUT}/*.json"):
    m = pat.search(os.path.basename(f))
    if not m:
        continue
    nm, tag, lc = m.group(1), (m.group(2) or "raw"), float(m.group(3))
    try:
        groups[(tag, lc)][nm] = json.load(open(f))["proxy"]
    except Exception:
        pass

best = None
for (tag, lc), by_bench in groups.items():
    if not by_bench:
        continue
    mean = sum(by_bench.values()) / len(by_bench)
    if best is None or mean < best[0]:
        best = (mean, tag, lc, len(by_bench))

if best is None:
    print("raw 0.0")
else:
    import sys
    sys.stderr.write(f"best: {best[1]} lc={best[2]} mean-proxy={best[0]:.4f} over {best[3]} benches\n")
    print(f"{best[1]} {best[2]}")
