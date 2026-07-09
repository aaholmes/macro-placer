"""Task 6: does moving HARD macros help beyond soft-only optimization?

Phase A: soft-only greedy (hard frozen)                    -> S_A
Phase B: from A, hard-macro swap+correction (soft frozen)  -> S_B
hard-only contribution = S_A - S_B   (the Tier-2-relevant number)

Also runs a control: swap+correction directly from the legalized reference
(no soft opt) to see if hard moves help on their own. TILOS-certifies results.
"""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize
from placers.moves import propose_swap, cell_potential

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
soft_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
swap_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 400

bench, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{name}")
fe = FastEval(bench, plc)
nh = bench.num_hard_macros
sz = bench.macro_sizes.numpy()

def proxy(p):
    return fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p)

def tilos(p):
    c = compute_proxy_cost(torch.tensor(p, dtype=torch.float32), bench, plc)
    return c["proxy_cost"], c["overlap_count"]

def hard_disp(a, b):
    d = np.linalg.norm(a[:nh] - b[:nh], axis=1)
    return d.mean(), int((d > 0.5).sum())


def run_swaps(pos0, iters, seed, refresh=25):
    """Greedy hard-macro swap+correction; soft frozen. Returns (best, naccept)."""
    rng = np.random.default_rng(seed)
    pos = pos0.copy()
    cur = proxy(pos)
    pot, gc, gr, gw, gh = cell_potential(fe, pos)
    naccept = 0
    for it in range(iters):
        if it and it % refresh == 0:
            pot, gc, gr, gw, gh = cell_potential(fe, pos)
        new, moved = propose_swap(fe, pos, rng, pot, gc, gr, gw, gh)
        if new is None:
            continue
        cand = proxy(new)
        if cand < cur:
            pos = new; cur = cand; naccept += 1
    return pos, cur, naccept


base = bench.macro_positions.numpy().astype(np.float64)
legal, _, _ = legalize(base, sz, nh, bench.canvas_width, bench.canvas_height, gap=0.01)
S0 = proxy(legal)
print(f"{name}: legalized reference proxy = {S0:.4f}")

# ---- Phase A: soft-only ----
t = time.time()
posA, _, _ = optimize(fe, legal, iters=soft_iters, seed=1, T0=0.0,
                      jitter_sigma=1.0, move_hard=False, move_soft=True,
                      log_every=10**9, logf=lambda *a: None)
SA = proxy(posA)
print(f"Phase A soft-only:            proxy = {SA:.4f}  ({time.time()-t:.0f}s)")

# ---- Phase B: hard swap+correction from A ----
t = time.time()
posB, SB, nacc = run_swaps(posA, swap_iters, seed=2)
mdisp, nmoved = hard_disp(posA, posB)
print(f"Phase B hard swaps from A:    proxy = {SB:.4f}  ({time.time()-t:.0f}s, "
      f"{nacc} accepted, {nmoved} hard moved >0.5um, mean {mdisp:.2f}um)")
print(f"  >> HARD-ONLY contribution (S_A - S_B) = {SA - SB:+.4f}")

# ---- Control: hard swaps directly from legalized reference ----
t = time.time()
posC, SC, naccC = run_swaps(legal, swap_iters, seed=3)
mdispC, nmovedC = hard_disp(legal, posC)
print(f"Control hard swaps from ref:  proxy = {SC:.4f}  ({time.time()-t:.0f}s, "
      f"{naccC} accepted, {nmovedC} hard moved >0.5um)")

# ---- certify final (Phase B) with TILOS ----
tB, ovB = tilos(posB)
print(f"\nTILOS  Phase B: proxy={tB:.4f} overlaps={ovB}")
np.save(f"/home/laz/partcl/my-macro-placer/notes/{name}_swapB.npy", posB)
