"""Discrete swap + correction move for HARD macros.

Motivation: random single hard-macro relocations almost never land in a legal
empty spot (canvas ~80% full), so the optimizer left hard macros untouched.
A density/congestion-guided swap instead:
  1. pick a hard macro sitting in a HOT cell (high density+congestion),
  2. move it to a COLD cell (low density+congestion) center,
  3. correct: legalize hard macros so displaced neighbours are pushed aside
     (a swap is only overlap-free for equal sizes; with 33x size variation we
     need the correction step).

Acceptance is on the true proxy delta (greedy). We track how much of the total
improvement is attributable to HARD moves (the Tier-2-relevant number).
"""
from __future__ import annotations
import numpy as np
from placers.legalizer import legalize


def cell_potential(fe, pos):
    """Per-cell hotness = density(normalized) + congestion(V+H), both already
    the quantities that enter the proxy. Returns (pot[gr*gc], gc, gr, gw, gh)."""
    gc = fe.plc.grid_col; gr = fe.plc.grid_row
    gw = fe.W / gc; gh = fe.H / gr
    dens = fe.build_density(pos) / fe.grid_area          # ~occupancy per cell
    fe.build_congestion(pos)
    Vs = fe._smooth(fe.Vraw / fe.grid_v_routes, "V") + fe.Vmac / fe.grid_v_routes
    Hs = fe._smooth(fe.Hraw / fe.grid_h_routes, "H") + fe.Hmac / fe.grid_h_routes
    pot = 0.5 * dens + 0.5 * (Vs + Hs)
    return pot, gc, gr, gw, gh


def _cell_of(x, y, gc, gr, gw, gh):
    return min(int(y // gh), gr - 1) * gc + min(int(x // gw), gc - 1)


def neighbor_centroid(fe, pos, i):
    """Mean position of the pins connected to hard macro i (excluding i's own
    pins) — the point wirelength wants macro i near. None if i has no nets."""
    nets = fe.macro_nets.get(i)
    if nets is None or len(nets) == 0:
        return None
    mask = np.isin(fe.pin_net, nets) & (fe.pin_owner != i)
    if not mask.any():
        return None
    owner = fe.pin_owner[mask]
    off = fe.pin_off[mask]
    x = np.where(owner >= 0, pos[np.clip(owner, 0, None), 0] + off[:, 0], off[:, 0])
    y = np.where(owner >= 0, pos[np.clip(owner, 0, None), 1] + off[:, 1], off[:, 1])
    return float(x.mean()), float(y.mean())


def build_hard_adjacency(fe):
    """Macro-macro connectivity among HARD macros: adj[i] = {j: shared_net_cnt}.
    Also returns net_hard_owners: net id -> array of hard macro indices on it."""
    nh = fe.b.num_hard_macros
    hard_mask = (fe.pin_owner >= 0) & (fe.pin_owner < nh)
    nets = fe.pin_net[hard_mask]; owners = fe.pin_owner[hard_mask]
    net_owners = {}
    for n, o in zip(nets, owners):
        net_owners.setdefault(int(n), set()).add(int(o))
    adj = {}
    for n, os in net_owners.items():
        os = list(os)
        for a in range(len(os)):
            for b in range(a + 1, len(os)):
                i, j = os[a], os[b]
                adj.setdefault(i, {}); adj.setdefault(j, {})
                adj[i][j] = adj[i].get(j, 0) + 1
                adj[j][i] = adj[j].get(i, 0) + 1
    return adj, {n: np.array(sorted(os)) for n, os in net_owners.items()}


def grow_cluster(adj, seed, k, rng):
    """Grow a connected cluster of up to k hard macros from seed, greedily
    adding the strongest-connected frontier macro (with a little randomness)."""
    cluster = {seed}
    while len(cluster) < k:
        # frontier: neighbours of cluster not yet in it, scored by edge weight
        cand = {}
        for m in cluster:
            for j, w in adj.get(m, {}).items():
                if j not in cluster:
                    cand[j] = cand.get(j, 0) + w
        if not cand:
            break
        js = np.array(list(cand)); ws = np.array([cand[j] for j in js], float)
        cluster.add(int(rng.choice(js, p=ws / ws.sum())))
    return np.array(sorted(cluster))


def cluster_external_centroid(fe, pos, cluster):
    """Mean position of pins connected to the cluster but owned OUTSIDE it
    (where wirelength wants the block). None if fully internal."""
    cset = set(int(c) for c in cluster)
    nets = set()
    for i in cluster:
        nn = fe.macro_nets.get(int(i))
        if nn is not None:
            nets.update(int(n) for n in nn)
    if not nets:
        return None
    net_arr = np.fromiter(nets, dtype=np.int64)
    mask = np.isin(fe.pin_net, net_arr)
    owner = fe.pin_owner[mask]; off = fe.pin_off[mask]
    inside = np.isin(owner, list(cset))
    keep = ~inside
    if not keep.any():
        return None
    owner = owner[keep]; off = off[keep]
    x = np.where(owner >= 0, pos[np.clip(owner, 0, None), 0] + off[:, 0], off[:, 0])
    y = np.where(owner >= 0, pos[np.clip(owner, 0, None), 1] + off[:, 1], off[:, 1])
    return float(x.mean()), float(y.mean())


def propose_cluster(fe, pos, rng, pot, gc, gr, gw, gh, adj, gap=1e-3,
                    max_cluster=6, legal_max_iters=600, wl_sigma=5.0):
    """Rigidly translate a connected hard-macro cluster from a hot region toward
    a cold cell near its external-connection centroid, then legalize. Internal
    nets are preserved (rigid move); only external nets change length."""
    nh = fe.b.num_hard_macros
    sz = fe.b.macro_sizes.numpy()
    hw = sz[:, 0] / 2.0; hh = sz[:, 1] / 2.0

    hard_hot = np.array([pot[_cell_of(pos[i, 0], pos[i, 1], gc, gr, gw, gh)]
                         for i in range(nh)])
    hard_hot = np.clip(hard_hot, 1e-9, None)
    seed = int(rng.choice(nh, p=hard_hot / hard_hot.sum()))
    k = int(rng.integers(2, max_cluster + 1))
    cluster = grow_cluster(adj, seed, k, rng)
    if len(cluster) < 2:
        return None, cluster

    cen = cluster_external_centroid(fe, pos, cluster)
    cold = pot.max() - pot + 1e-6
    if cen is not None:
        ci = np.arange(len(pot))
        cxs = (ci % gc + 0.5) * gw; cys = (ci // gc + 0.5) * gh
        d2 = (cxs - cen[0]) ** 2 + (cys - cen[1]) ** 2
        weight = cold * np.exp(-d2 / (2 * wl_sigma ** 2))
    else:
        weight = cold
    weight = np.clip(weight, 1e-12, None)
    cidx = int(rng.choice(len(pot), p=weight / weight.sum()))
    dest = np.array([(cidx % gc + 0.5) * gw, (cidx // gc + 0.5) * gh])

    ccentroid = pos[cluster].mean(axis=0)
    shift = dest - ccentroid
    new = pos.copy()
    new[cluster] += shift
    # keep each moved macro in-bounds (may slightly distort rigidity)
    new[cluster, 0] = np.clip(new[cluster, 0], hw[cluster], fe.W - hw[cluster])
    new[cluster, 1] = np.clip(new[cluster, 1], hh[cluster], fe.H - hh[cluster])

    legal, _, resid = legalize(new, sz, nh, fe.W, fe.H, gap=gap,
                               max_iters=legal_max_iters)
    if resid > 0:
        return None, cluster
    return legal, cluster


def propose_swap(fe, pos, rng, pot, gc, gr, gw, gh, gap=1e-3,
                 legal_max_iters=400, wl_sigma=4.0):
    """Return (new_pos, moved_macro) or (None, -1). Moves one hard macro from a
    hot cell to a cold cell that is ALSO near its net-centroid (WL-aware), then
    legalizes. new_pos is overlap-free."""
    nh = fe.b.num_hard_macros
    sz = fe.b.macro_sizes.numpy()
    hw = sz[:, 0] / 2.0; hh = sz[:, 1] / 2.0

    # hotness of each hard macro's current cell -> pick a hot macro
    hard_hot = np.array([pot[_cell_of(pos[i, 0], pos[i, 1], gc, gr, gw, gh)]
                         for i in range(nh)])
    hard_hot = np.clip(hard_hot, 1e-9, None)
    i = int(rng.choice(nh, p=hard_hot / hard_hot.sum()))

    # target weight = coldness x proximity-to-net-centroid (WL-aware). Without
    # the proximity term the macro gets yanked far from its nets and WL kills it.
    cold = pot.max() - pot + 1e-6
    cen = neighbor_centroid(fe, pos, i)
    if cen is not None:
        ci = np.arange(len(pot))
        cxs = (ci % gc + 0.5) * gw; cys = (ci // gc + 0.5) * gh
        d2 = (cxs - cen[0]) ** 2 + (cys - cen[1]) ** 2
        weight = cold * np.exp(-d2 / (2 * wl_sigma ** 2))
    else:
        weight = cold
    weight = np.clip(weight, 1e-12, None)
    cidx = int(rng.choice(len(pot), p=weight / weight.sum()))
    cx = (cidx % gc + 0.5) * gw
    cy = (cidx // gc + 0.5) * gh
    cx = min(max(cx, hw[i]), fe.W - hw[i]); cy = min(max(cy, hh[i]), fe.H - hh[i])

    new = pos.copy()
    new[i, 0] = cx; new[i, 1] = cy
    legal, _, resid = legalize(new, sz, nh, fe.W, fe.H, gap=gap,
                               max_iters=legal_max_iters)
    if resid > 0:
        return None, -1
    return legal, i
