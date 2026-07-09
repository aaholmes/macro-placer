"""Task 2: single-macro-move local search over the true proxy cost.

Strategy (agreed):
- start from the legalized reference (zero overlaps),
- PROPOSE moves biased toward hard macros sitting in the hottest
  congestion/density cells (spend effort where the cost is),
- keep every placement legal by REJECTING any proposal that creates a
  hard-macro overlap or leaves the canvas (cheap O(n) check *before* the
  expensive cost eval),
- ACCEPT on the true proxy delta from fast_eval (greedy, with an optional
  small SA-style tolerance so the cluster can spread),
- verify the final placement with the ground-truth TILOS evaluator.

Full-recompute fast_eval (~50 ms/move) is fast enough for a first demo; the
incremental delta layer (task 4) will scale this up later.
"""
from __future__ import annotations
import numpy as np


def _overlaps(pos, i, xy, hw, hh, nh, gap):
    """True if hard macro i placed at xy would overlap any other hard macro."""
    dx = np.abs(xy[0] - pos[:nh, 0])
    dy = np.abs(xy[1] - pos[:nh, 1])
    need_x = hw[i] + hw[:nh] + gap
    need_y = hh[i] + hh[:nh] + gap
    hit = (dx < need_x) & (dy < need_y)
    hit[i] = False
    return bool(hit.any())


def _hot_macros(fe, pos, nm):
    """Rank all macros (hard+soft) by the congestion+density of their cell."""
    fe.build_congestion(pos)
    Vs = fe._smooth(fe.Vraw / fe.grid_v_routes, "V") + fe.Vmac / fe.grid_v_routes
    Hs = fe._smooth(fe.Hraw / fe.grid_h_routes, "H") + fe.Hmac / fe.grid_h_routes
    cong = Vs + Hs
    dens = fe.build_density(pos) / fe.grid_area
    gc = fe.plc.grid_col; gr = fe.plc.grid_row
    gw = fe.W / gc; gh = fe.H / gr
    hot = np.empty(nm)
    for i in range(nm):
        c = min(int(pos[i, 0] // gw), gc - 1)
        r = min(int(pos[i, 1] // gh), gr - 1)
        hot[i] = cong[r * gc + c] + dens[r * gc + c]
    return np.clip(hot, 1e-9, None)


def optimize(fe, pos0, iters=1500, seed=0, jump_frac=0.5, jitter_sigma=1.5,
             hot_bias=0.7, T0=0.0, Tend=1e-5, gap=1e-3, log_every=100, logf=print):
    """Run local search. Returns (best_pos, history) where history is a list of
    (eval_count, best_cost)."""
    rng = np.random.default_rng(seed)
    b = fe.b
    nh = b.num_hard_macros
    nm = b.num_macros
    sz = b.macro_sizes.numpy()
    hw = sz[:, 0] / 2.0; hh = sz[:, 1] / 2.0
    W, H = fe.W, fe.H

    pos = np.asarray(pos0, np.float64).copy()

    def proxy(p):
        return fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p)

    cur = proxy(pos)
    best = cur; best_pos = pos.copy()
    hist = [(0, best)]
    hot = _hot_macros(fe, pos, nm)
    evals = 0
    accepts = 0

    for it in range(iters):
        # refresh hotspots occasionally (placement drifts as it spreads)
        if it and it % 300 == 0:
            hot = _hot_macros(fe, pos, nm)

        # pick a macro (hard or soft): hot-biased or uniform
        if rng.random() < hot_bias:
            i = int(rng.choice(nm, p=hot / hot.sum()))
        else:
            i = int(rng.integers(nm))
        is_hard = i < nh

        # propose a target. Random full jumps only for hard macros (into empty
        # space); soft macros only jitter locally (random soft jumps are noise).
        if is_hard and rng.random() < jump_frac:
            x = rng.uniform(hw[i], W - hw[i]); y = rng.uniform(hh[i], H - hh[i])
        else:
            x = pos[i, 0] + rng.normal(0, jitter_sigma)
            y = pos[i, 1] + rng.normal(0, jitter_sigma)
            x = min(max(x, hw[i]), W - hw[i]); y = min(max(y, hh[i]), H - hh[i])

        # hard macros must stay overlap-free; soft macros may overlap freely
        if is_hard and _overlaps(pos, i, (x, y), hw, hh, nh, gap):
            continue  # keep placement legal; no expensive eval wasted

        old = pos[i].copy()
        pos[i, 0] = x; pos[i, 1] = y
        cand = proxy(pos)
        evals += 1
        delta = cand - cur
        T = T0 * (Tend / T0) ** (it / iters) if T0 > 0 else 0.0  # geometric cooling
        accept = delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T))
        if accept:
            cur = cand; accepts += 1
            if cur < best:
                best = cur; best_pos = pos.copy()
        else:
            pos[i] = old  # revert

        if evals and evals % log_every == 0:
            hist.append((evals, best))
            logf(f"  it={it:5d} evals={evals:5d} accepts={accepts:5d} "
                 f"cur={cur:.4f} best={best:.4f}")

    hist.append((evals, best))
    logf(f"  done: {evals} evals, {accepts} accepts, best={best:.4f}")
    return best_pos, hist
