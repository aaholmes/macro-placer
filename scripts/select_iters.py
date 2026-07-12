"""Read notes/iters_sweep.json and print the chosen differentiable iteration
count: the smallest iters whose mean diff+greedy proxy is within 1% of the best
(the knee) -- so we spend the time budget on more seeds rather than iterations
past the point of diminishing returns. Falls back to 15000 if data is missing.
"""
import json, sys
from collections import defaultdict

try:
    rows = json.load(open("/home/laz/partcl/my-macro-placer/notes/iters_sweep.json"))
    by = defaultdict(list)
    for r in rows:
        by[r["iters"]].append(r["diff_greedy"])
    # only iters values evaluated on the same full set of benchmarks are comparable
    nb = max(len(v) for v in by.values())
    means = {it: sum(v) / len(v) for it, v in by.items() if len(v) == nb}
    best = min(means.values())
    chosen = min(it for it, m in means.items() if m <= best * 1.01)
    print(int(chosen))
except Exception as e:
    sys.stderr.write(f"select_iters fallback (15000): {e}\n")
    print(15000)
