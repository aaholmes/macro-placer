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
from placers.moves import (propose_swap, propose_cluster, cell_potential,
                           build_hard_adjacency)

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


def run_moves(pos0, iters, seed, kind="swap", adj=None, refresh=25):
    """Greedy hard-macro moves (swap or cluster); soft frozen."""
    rng = np.random.default_rng(seed)
    pos = pos0.copy()
    cur = proxy(pos)
    pot, gc, gr, gw, gh = cell_potential(fe, pos)
    naccept = 0
    for it in range(iters):
        if it and it % refresh == 0:
            pot, gc, gr, gw, gh = cell_potential(fe, pos)
        if kind == "swap":
            new, _ = propose_swap(fe, pos, rng, pot, gc, gr, gw, gh)
        else:
            new, _ = propose_cluster(fe, pos, rng, pot, gc, gr, gw, gh, adj)
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

adj, _ = build_hard_adjacency(fe)

# ---- Phase B: single-swap hard moves from A ----
t = time.time()
posB, SB, nacc = run_moves(posA, swap_iters, seed=2, kind="swap")
mdisp, nmoved = hard_disp(posA, posB)
print(f"Phase B single-swaps from A:  proxy = {SB:.4f}  ({time.time()-t:.0f}s, "
      f"{nacc} accepted, {nmoved} hard moved >0.5um)")
print(f"  >> single-swap HARD-ONLY contribution (S_A - S_B) = {SA - SB:+.4f}")

# ---- Phase C: CLUSTER hard moves from A (same start as B) ----
t = time.time()
posC, SC, naccC = run_moves(posA, swap_iters, seed=2, kind="cluster", adj=adj)
mdispC, nmovedC = hard_disp(posA, posC)
print(f"Phase C cluster moves from A: proxy = {SC:.4f}  ({time.time()-t:.0f}s, "
      f"{naccC} accepted, {nmovedC} hard moved >0.5um, mean {mdispC:.2f}um)")
print(f"  >> cluster HARD-ONLY contribution (S_A - S_C) = {SA - SC:+.4f}")

# ---- certify best of B/C with TILOS ----
best_pos = posC if SC < SB else posB
tB, ovB = tilos(best_pos)
print(f"\nTILOS  best hard result: proxy={tB:.4f} overlaps={ovB}")
np.save(f"/home/laz/partcl/my-macro-placer/notes/{name}_hardbest.npy", best_pos)
