"""Continuous force-field (gradient-descent) placement for SOFT macros.

Unlike single-move greedy, this moves ALL soft macros together each step, down a
smooth surrogate of the proxy gradient:
  - WL attraction: star-model spring toward each net's pin centroid
    (smooth surrogate of HPWL's gradient).
  - Density repulsion: -grad of the occupancy grid, sampled per macro cell.
  - Congestion repulsion: -grad of the (V+H) congestion grid.
Combined with the proxy weights (1, 0.5, 0.5). Steps are accepted only if the
ACTUAL proxy (fast_eval) drops (backtracking line search + best tracking), so
force-scale mis-tuning can't diverge.
"""
from __future__ import annotations
import numpy as np


def _wl_force(fe, pos):
    """Star-model WL force per macro: sum over its pins of (net_centroid - pin).
    Vectorized over all pins. Returns [num_macros, 2]."""
    owner = fe.pin_owner; off = fe.pin_off; net = fe.pin_net
    oc = np.clip(owner, 0, None)
    px = np.where(owner >= 0, pos[oc, 0] + off[:, 0], off[:, 0])
    py = np.where(owner >= 0, pos[oc, 1] + off[:, 1], off[:, 1])
    n = fe.num_nets
    cnt = np.bincount(net, minlength=n).astype(np.float64)
    sx = np.bincount(net, weights=px, minlength=n)
    sy = np.bincount(net, weights=py, minlength=n)
    cx = sx / np.maximum(cnt, 1); cy = sy / np.maximum(cnt, 1)
    fx = cx[net] - px; fy = cy[net] - py            # per-pin pull to net centroid
    F = np.zeros((fe.b.num_macros, 2))
    movable = owner >= 0
    np.add.at(F[:, 0], owner[movable], fx[movable])
    np.add.at(F[:, 1], owner[movable], fy[movable])
    return F


def _grid_grad(field, gc, gr, gw, gh):
    """Central-difference gradient of a flat [gr*gc] field. Returns gx,gy grids."""
    A = field.reshape(gr, gc)
    gx = np.zeros_like(A); gy = np.zeros_like(A)
    gx[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2 * gw)
    gy[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * gh)
    return gx, gy


def _field_force(fe, pos, which):
    """Repulsion force per macro = -grad(field) at its cell. which in {dens,cong}."""
    gc = fe.plc.grid_col; gr = fe.plc.grid_row
    gw = fe.W / gc; gh = fe.H / gr
    if which == "dens":
        field = fe.build_density(pos) / fe.grid_area
    else:
        fe.build_congestion(pos)
        Vs = fe._smooth(fe.Vraw / fe.grid_v_routes, "V") + fe.Vmac / fe.grid_v_routes
        Hs = fe._smooth(fe.Hraw / fe.grid_h_routes, "H") + fe.Hmac / fe.grid_h_routes
        field = Vs + Hs
    gx, gy = _grid_grad(field, gc, gr, gw, gh)
    nm = fe.b.num_macros
    cols = np.clip((pos[:nm, 0] // gw).astype(int), 0, gc - 1)
    rows = np.clip((pos[:nm, 1] // gh).astype(int), 0, gr - 1)
    F = np.zeros((nm, 2))
    F[:, 0] = -gx[rows, cols]
    F[:, 1] = -gy[rows, cols]
    return F


def optimize_gradient(fe, pos0, iters=400, step0=0.4, w_wl=0.15, w_dens=1.0,
                      w_cong=0.0, max_move=1.0, seed=0, log_every=50, logf=print):
    """Gradient descent on soft macros (hard frozen). Returns (best_pos, hist)
    where hist is [(elapsed_evals, proxy)]."""
    import time
    b = fe.b; nh = b.num_hard_macros; nm = b.num_macros
    sz = b.macro_sizes.numpy(); hw = sz[:, 0] / 2; hh = sz[:, 1] / 2
    W, H = fe.W, fe.H
    pos = np.asarray(pos0, np.float64).copy()

    def proxy(p):
        return fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p)

    cur = proxy(pos); best = cur; best_pos = pos.copy()
    hist = [(0.0, best)]; t0 = time.time()
    step = step0
    for it in range(iters):
        Fd = _field_force(fe, pos, "dens")
        Fw = _wl_force(fe, pos)
        F = w_wl * Fw + w_dens * Fd
        if w_cong:
            F = F + w_cong * _field_force(fe, pos, "cong")
        F[:nh] = 0.0                                   # freeze hard macros
        # per-macro displacement, clipped so no single macro flies off
        if np.abs(F).max() < 1e-12:
            break

        # backtracking line search on the ACTUAL proxy (scale the whole field)
        accepted = False
        s = step
        for _ in range(6):
            disp = s * F
            d = np.linalg.norm(disp, axis=1)
            over = d > max_move
            disp[over] *= (max_move / d[over])[:, None]
            trial = pos.copy()
            trial[nh:] += disp[nh:]
            trial[nh:, 0] = np.clip(trial[nh:, 0], hw[nh:], W - hw[nh:])
            trial[nh:, 1] = np.clip(trial[nh:, 1], hh[nh:], H - hh[nh:])
            pc = proxy(trial)
            if pc < cur:
                pos = trial; cur = pc; accepted = True
                if pc < best:
                    best = pc; best_pos = pos.copy()
                break
            s *= 0.5
        step = min(step * 1.2, step0) if accepted else max(s, 1e-4)
        if it % log_every == 0:
            hist.append((time.time() - t0, best))
            logf(f"  it={it:4d} step={step:.3f} cur={cur:.4f} best={best:.4f}")
    hist.append((time.time() - t0, best))
    logf(f"  done: {iters} steps, best={best:.4f}")
    return best_pos, hist
