"""Paired comparison of the two gate arms (windowed config 249 vs 221 alone),
best-of-N + greedy per benchmark. Both arms use a single config, so candidate idx
== seed and the arms pair by idx. Reports the EQUAL-N best (min over the shared idx
set) so neither arm is favored by extra draws, plus the all-candidate best for
reference. Negative diff => the window helps end-to-end. -> stdout only."""
import sys, json, glob, os
import numpy as np
ROOT = "/home/laz/partcl/my-macro-placer"
ALL = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
A = f"{ROOT}/notes/gate_window"; B = f"{ROOT}/notes/gate_221"


def finals(d, nm):
    """idx -> post-greedy proxy for overlap-free candidates of benchmark nm."""
    out = {}
    for f in glob.glob(f"{d}/{nm}_p*_final.json"):
        r = json.load(open(f))
        if int(r.get("overlaps", 1)) == 0:
            out[int(r["idx"])] = float(r["final"])
    return out


rows = []
eqA, eqB = [], []
for nm in ALL:
    fa, fb = finals(A, nm), finals(B, nm)
    if not fa or not fb:
        rows.append((nm, None, None, None, len(fa), len(fb), 0)); continue
    common = sorted(set(fa) & set(fb))
    if not common:
        rows.append((nm, min(fa.values()), min(fb.values()), None, len(fa), len(fb), 0)); continue
    ba = min(fa[i] for i in common); bb = min(fb[i] for i in common)
    rows.append((nm, ba, bb, ba - bb, len(fa), len(fb), len(common)))
    eqA.append(ba); eqB.append(bb)

print(f"{'bench':7s} {'window':>8s} {'221':>8s} {'diff':>8s}  {'nW':>3s} {'n221':>4s} {'ncmn':>4s}")
for nm, ba, bb, d, na, nb, nc in rows:
    if ba is None:
        print(f"{nm:7s} {'--':>8s} {'--':>8s} {'--':>8s}  {na:3d} {nb:4d} {nc:4d}  (incomplete)"); continue
    ds = f"{d:+.4f}" if d is not None else "n/a"
    mark = ""
    if d is not None:
        mark = "  <- window better" if d < -1e-4 else ("  <- 221 better" if d > 1e-4 else "  ~tie")
    print(f"{nm:7s} {ba:8.4f} {bb:8.4f} {ds:>8s}  {na:3d} {nb:4d} {nc:4d}{mark}")

if eqA:
    mA, mB = float(np.mean(eqA)), float(np.mean(eqB))
    nwin = sum(1 for _, ba, bb, d, *_ in rows if d is not None and d < -1e-4)
    n221 = sum(1 for _, ba, bb, d, *_ in rows if d is not None and d > 1e-4)
    print(f"\nEQUAL-N avg over {len(eqA)} benches:  window={mA:.4f}  221={mB:.4f}  "
          f"diff={mA-mB:+.4f}  (window better on {nwin}, 221 better on {n221})")
    print(f"reference: deployed 5-config pool = 1.0155 (after_avg in notes/fullrun_report.json)")
