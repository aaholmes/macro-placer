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
