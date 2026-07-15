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


def _cell_field(fe, pos):
    """Per-grid-cell congestion+density field (length grid_col*grid_row), the
    same quantity `_hot_macros` gathers per macro. Used both to bias *which* cell
    to move (hot) and *where to send it* (cold)."""
    fe.build_congestion(pos)
    Vs = fe._smooth(fe.Vraw / fe.grid_v_routes, "V") + fe.Vmac / fe.grid_v_routes
    Hs = fe._smooth(fe.Hraw / fe.grid_h_routes, "H") + fe.Hmac / fe.grid_h_routes
    dens = fe.build_density(pos) / fe.grid_area
    return np.clip(Vs + Hs + dens, 0.0, None)


def _hot_from_field(field, fe, pos, nm):
    gc = fe.plc.grid_col; gr = fe.plc.grid_row
    gw = fe.W / gc; gh = fe.H / gr
    hot = np.empty(nm)
    for i in range(nm):
        c = min(int(pos[i, 0] // gw), gc - 1)
        r = min(int(pos[i, 1] // gh), gr - 1)
        hot[i] = field[r * gc + c]
    return np.clip(hot, 1e-9, None)


def _hot_macros(fe, pos, nm):
    """Rank all macros (hard+soft) by the congestion+density of their cell."""
    return _hot_from_field(_cell_field(fe, pos), fe, pos, nm)


def _directed_target(field, gc, gr, gw, gh, hw_i, hh_i, W, H, rng):
    """Sample a destination in a COLD grid cell (low field), then a uniform point
    within it, clipped to the macro's legal band. This is the discrete analogue of
    following the congestion gradient toward cold space -- the piece the global
    placer had and the detailed placer lacked."""
    w = field.max() - field                 # coldest cells -> largest weight
    w = np.clip(w, 1e-9, None)
    t = int(rng.choice(field.size, p=w / w.sum()))
    r, c = divmod(t, gc)
    x = (c + rng.random()) * gw
    y = (r + rng.random()) * gh
    x = min(max(x, hw_i), W - hw_i)
    y = min(max(y, hh_i), H - hh_i)
    return x, y


def optimize_portfolio(fe, pos0, iters=200000, seed=0, jump_frac=0.3,
                       jitter_sigma=1.0, hot_bias=0.7,
                       w_move=0.6, w_flip=0.2, w_swap=0.2,
                       refresh=2000, log_every=20000, logf=print):
    """Unified greedy over an operator portfolio (all overlap-safe):
      - soft-macro move (jitter/jump, hot-biased)
      - hard-macro orientation flip (Klein-4)
      - congruent hard-macro swap (assignment)
    Each iteration picks one operator, proposes a move, accepts iff the true
    proxy improves. Jointly optimizes soft placement + hard orientation +
    hard assignment throughout the descent."""
    from collections import defaultdict
    from placers.moves import neighbor_centroid
    rng = np.random.default_rng(seed)
    b = fe.b; nh = b.num_hard_macros; nm = b.num_macros
    sz = b.macro_sizes.numpy(); hw = sz[:, 0] / 2; hh = sz[:, 1] / 2
    W, H = fe.W, fe.H
    groups = defaultdict(list)
    for i in range(nh):
        groups[tuple(np.round(sz[i], 3))].append(i)
    glist = [np.array(v) for v in groups.values() if len(v) >= 2]

    cur = fe.init_incremental(pos0)
    best = cur; best_pos = fe.ipos.copy()
    hist = [(0, best)]
    hot = _hot_macros(fe, fe.ipos, nm)
    acc = dict(move=0, flip=0, swap=0)
    ops = ["move", "flip", "swap"]; wts = np.array([w_move, w_flip, w_swap]); wts = wts / wts.sum()

    for it in range(iters):
        if it and it % refresh == 0:
            hot = _hot_macros(fe, fe.ipos, nm)
        op = ops[rng.choice(3, p=wts)]
        if op == "flip" and nh:
            i = int(rng.integers(nh)); o = int(rng.integers(4))
            u = fe.apply_flip(i, o); c = fe.cost_current()
            if c < cur - 1e-12:
                cur = c; acc["flip"] += 1
            else:
                fe.undo_flip(u)
        elif op == "swap" and glist:
            g = glist[rng.integers(len(glist))]; i = int(rng.choice(g))
            cen = neighbor_centroid(fe, fe.ipos, i)
            if cen is None:
                continue
            j = int(g[np.argmin(np.linalg.norm(fe.ipos[g] - np.array(cen), axis=1))])
            if j == i:
                continue
            pi = fe.ipos[i].copy(); pj = fe.ipos[j].copy()
            ui = fe.apply_move(i, pj[0], pj[1]); uj = fe.apply_move(j, pi[0], pi[1])
            c = fe.cost_current()
            if c < cur - 1e-12:
                cur = c; acc["swap"] += 1
            else:
                fe.undo_move(uj); fe.undo_move(ui)
        else:  # soft-macro move (hot-biased)
            if rng.random() < hot_bias:
                w = hot[nh:nm]; i = nh + int(rng.choice(nm - nh, p=w / w.sum()))
            else:
                i = int(rng.integers(nh, nm))
            x = min(max(fe.ipos[i, 0] + rng.normal(0, jitter_sigma), hw[i]), W - hw[i])
            y = min(max(fe.ipos[i, 1] + rng.normal(0, jitter_sigma), hh[i]), H - hh[i])
            u = fe.apply_move(i, x, y); c = fe.cost_current()
            if c < cur - 1e-12:
                cur = c; acc["move"] += 1
            else:
                fe.undo_move(u)
        if cur < best:
            best = cur; best_pos = fe.ipos.copy()
        if it and it % log_every == 0:
            logf(f"  it={it:6d} acc={acc} best={best:.4f}")
    logf(f"  done: acc={acc} best={best:.4f}")
    return best_pos, hist + [(iters, best)]


def optimize_best_of(fe, pos0, iters=200000, seed=0, jitter_sigma=1.0, jump_frac=0.3,
                     hot_bias=0.7, refresh=2000, log_every=20000, logf=print):
    """BEST-OF-ALL-operators greedy: each iteration proposes ONE candidate of
    every operator type (soft move, orientation flip, congruent swap), evaluates
    all, and commits the single best-improving one. Because the soft-move
    candidate is always considered, each step is >= a soft-only greedy step, and
    the reached local minimum is over the full (richer) move set."""
    from collections import defaultdict
    from placers.moves import neighbor_centroid
    rng = np.random.default_rng(seed)
    b = fe.b; nh = b.num_hard_macros; nm = b.num_macros
    sz = b.macro_sizes.numpy(); hw = sz[:, 0] / 2; hh = sz[:, 1] / 2
    W, H = fe.W, fe.H
    groups = defaultdict(list)
    for i in range(nh):
        groups[tuple(np.round(sz[i], 3))].append(i)
    glist = [np.array(v) for v in groups.values() if len(v) >= 2]

    cur = fe.init_incremental(pos0)
    best = cur; best_pos = fe.ipos.copy()
    hist = [(0, best)]
    hot = _hot_macros(fe, fe.ipos, nm)
    acc = dict(move=0, flip=0, swap=0)

    for it in range(iters):
        if it and it % refresh == 0:
            hot = _hot_macros(fe, fe.ipos, nm)
        cands = []  # (delta, kind, apply_thunk)  -- evaluated by apply/undo

        # --- soft-move candidate ---
        if rng.random() < hot_bias:
            w = hot[nh:nm]; si = nh + int(rng.choice(nm - nh, p=w / w.sum()))
        else:
            si = int(rng.integers(nh, nm))
        sx = min(max(fe.ipos[si, 0] + rng.normal(0, jitter_sigma), hw[si]), W - hw[si])
        sy = min(max(fe.ipos[si, 1] + rng.normal(0, jitter_sigma), hh[si]), H - hh[si])
        u = fe.apply_move(si, sx, sy); c = fe.cost_current(); fe.undo_move(u)
        cands.append((c - cur, "move", ("move", si, sx, sy)))

        # --- flip candidate (best of the 3 non-current orientations of a macro) ---
        if nh:
            fi = int(rng.integers(nh)); base = int(fe.macro_orient[fi])
            bo, bc = base, cur
            for o in range(4):
                if o == base:
                    continue
                uf = fe.apply_flip(fi, o); c = fe.cost_current(); fe.undo_flip(uf)
                if c < bc:
                    bc, bo = c, o
            if bo != base:
                cands.append((bc - cur, "flip", ("flip", fi, bo)))

        # --- congruent-swap candidate ---
        if glist:
            g = glist[rng.integers(len(glist))]; i = int(rng.choice(g))
            cen = neighbor_centroid(fe, fe.ipos, i)
            if cen is not None:
                j = int(g[np.argmin(np.linalg.norm(fe.ipos[g] - np.array(cen), axis=1))])
                if j != i:
                    pi = fe.ipos[i].copy(); pj = fe.ipos[j].copy()
                    ui = fe.apply_move(i, pj[0], pj[1]); uj = fe.apply_move(j, pi[0], pi[1])
                    c = fe.cost_current(); fe.undo_move(uj); fe.undo_move(ui)
                    cands.append((c - cur, "swap", ("swap", i, j, pi, pj)))

        # --- commit the single best-improving candidate ---
        cands.sort(key=lambda t: t[0])
        d, kind, act = cands[0]
        if d < -1e-12:
            if act[0] == "move":
                fe.apply_move(act[1], act[2], act[3])
            elif act[0] == "flip":
                fe.apply_flip(act[1], act[2])
            else:
                fe.apply_move(act[1], act[4][0], act[4][1]); fe.apply_move(act[2], act[3][0], act[3][1])
            cur += d; acc[kind] += 1
            if cur < best:
                best = cur; best_pos = fe.ipos.copy()
        if it and it % log_every == 0:
            hist.append((it, best)); logf(f"  it={it:6d} acc={acc} best={best:.4f}")
    logf(f"  done: acc={acc} best={best:.4f}")
    return best_pos, hist + [(iters, best)]


def optimize_congruent_swaps(fe, pos0, iters=4000, seed=0, logf=print):
    """Swap pairs of CONGRUENT (same-size) hard macros -> always overlap-free.
    Heuristic: move a macro toward its net-centroid by swapping it with the
    congruent macro currently nearest that centroid (assignment optimization).
    Accept on true proxy. Assumes fe.init_incremental already called."""
    from collections import defaultdict
    from placers.moves import neighbor_centroid
    nh = fe.b.num_hard_macros; sz = fe.b.macro_sizes.numpy()
    groups = defaultdict(list)
    for i in range(nh):
        groups[tuple(np.round(sz[i], 3))].append(i)
    glist = [np.array(v) for v in groups.values() if len(v) >= 2]
    rng = np.random.default_rng(seed)
    cur = fe.cost_current(); accepts = 0
    for it in range(iters):
        g = glist[rng.integers(len(glist))]
        i = int(rng.choice(g))
        cen = neighbor_centroid(fe, fe.ipos, i)
        if cen is None:
            continue
        d = np.linalg.norm(fe.ipos[g] - np.array(cen), axis=1)
        j = int(g[np.argmin(d)])
        if j == i:
            continue
        pi = fe.ipos[i].copy(); pj = fe.ipos[j].copy()
        ui = fe.apply_move(i, pj[0], pj[1]); uj = fe.apply_move(j, pi[0], pi[1])
        c = fe.cost_current()
        if c < cur - 1e-12:
            cur = c; accepts += 1
        else:
            fe.undo_move(uj); fe.undo_move(ui)
    logf(f"  congruent swaps: {accepts} accepted, cost {cur:.4f}")
    return fe.ipos.copy(), cur, accepts


def optimize_flips(fe, pos0, sweeps=4, logf=print):
    """Greedy Klein-4 orientation optimization over hard macros: for each macro
    try all 4 orientations, keep the best. Footprint unchanged -> overlaps stay
    at zero. Returns (pos, orient, best_cost)."""
    fe.init_incremental(pos0)
    nh = fe.b.num_hard_macros
    cur = fe.cost_current()
    start = cur
    for sweep in range(sweeps):
        improved = 0
        for i in range(nh):
            base = int(fe.macro_orient[i])
            best_o, best_c = base, cur
            for o in range(4):
                if o == base:
                    continue
                u = fe.apply_flip(i, o)
                c = fe.cost_current()
                fe.undo_flip(u)
                if c < best_c - 1e-12:
                    best_c, best_o = c, o
            if best_o != base:
                fe.apply_flip(i, best_o); cur = best_c; improved += 1
        logf(f"  flip sweep {sweep}: {improved} flips, cost {cur:.4f}")
        if improved == 0:
            break
    return fe.ipos.copy(), fe.macro_orient.copy(), cur


def optimize_fast(fe, pos0, iters=20000, seed=0, jump_frac=0.3, jitter_sigma=1.0,
                  hot_bias=0.7, T0=0.0, Tend=1e-5, gap=1e-3, move_hard=True,
                  move_soft=True, refresh=2000, log_every=2000, logf=print, max_accepts=None,
                  cong_directed=0.0):
    """Greedy/SA single-move local search using the INCREMENTAL evaluator
    (~55x faster than full recompute). Returns (best_pos, hist).

    cong_directed in [0,1]: fraction of proposals that send a hot-selected cell to
    a COLD grid cell (sampled from the congestion+density field) instead of a local
    jitter/random jump. Acceptance is unchanged (true proxy delta), so this only
    reshapes the proposal distribution toward congestion-relieving moves -- and it
    lets *soft* cells make the long directed jumps that plain jitter forbids. 0.0
    reproduces the original behaviour exactly."""
    rng = np.random.default_rng(seed)
    b = fe.b
    nh = b.num_hard_macros; nm = b.num_macros
    sz = b.macro_sizes.numpy()
    hw = sz[:, 0] / 2.0; hh = sz[:, 1] / 2.0
    W, H = fe.W, fe.H
    gc = fe.plc.grid_col; gr = fe.plc.grid_row
    gw = W / gc; gh = H / gr

    cur = fe.init_incremental(pos0)
    best = cur; best_pos = fe.ipos.copy()
    hist = [(0, best)]
    field = _cell_field(fe, fe.ipos)     # rebuilds grid/cong at committed pos
    hot = _hot_from_field(field, fe, fe.ipos, nm)
    accepts = 0

    for it in range(iters):
        if it and it % refresh == 0:
            field = _cell_field(fe, fe.ipos)
            hot = _hot_from_field(field, fe, fe.ipos, nm)
        lo = 0 if move_hard else nh
        hi = nm if move_soft else nh
        if rng.random() < hot_bias:
            w = hot[lo:hi]; i = lo + int(rng.choice(hi - lo, p=w / w.sum()))
        else:
            i = int(rng.integers(lo, hi))
        is_hard = i < nh

        if cong_directed > 0 and rng.random() < cong_directed:
            x, y = _directed_target(field, gc, gr, gw, gh, hw[i], hh[i], W, H, rng)
        elif is_hard and rng.random() < jump_frac:
            x = rng.uniform(hw[i], W - hw[i]); y = rng.uniform(hh[i], H - hh[i])
        else:
            x = min(max(fe.ipos[i, 0] + rng.normal(0, jitter_sigma), hw[i]), W - hw[i])
            y = min(max(fe.ipos[i, 1] + rng.normal(0, jitter_sigma), hh[i]), H - hh[i])

        if is_hard and _overlaps(fe.ipos, i, (x, y), hw, hh, nh, gap):
            continue

        undo = fe.apply_move(i, x, y)
        cand = fe.cost_current()
        delta = cand - cur
        T = T0 * (Tend / T0) ** (it / iters) if T0 > 0 else 0.0
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            cur = cand; accepts += 1
            if cur < best:
                best = cur; best_pos = fe.ipos.copy()
            if max_accepts and accepts >= max_accepts:   # stop after N accepted moves
                break
        else:
            fe.undo_move(undo)

        if it and it % log_every == 0:
            hist.append((it, best))
            logf(f"  it={it:6d} accepts={accepts:6d} cur={cur:.4f} best={best:.4f}")

    hist.append((iters, best))
    logf(f"  done: {iters} iters, {accepts} accepts, best={best:.4f}")
    return best_pos, hist


def optimize(fe, pos0, iters=1500, seed=0, jump_frac=0.5, jitter_sigma=1.5,
             hot_bias=0.7, T0=0.0, Tend=1e-5, gap=1e-3, move_hard=True,
             move_soft=True, log_every=100, logf=print):
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

    def components(p):
        wl = fe.wirelength_cost(p); d = fe.density_cost(p); c = fe.congestion_cost(p)
        return wl, d, c, wl + 0.5 * d + 0.5 * c

    cwl, cd, cc, cur = components(pos)
    best = cur; best_pos = pos.copy()
    hist = [(0, best)]
    comp_hist = [(0, cwl, 0.5 * cd, 0.5 * cc)]  # weighted contributions of `cur`
    hot = _hot_macros(fe, pos, nm)
    evals = 0
    accepts = 0
    # diagnosis counters
    stat = dict(hard_prop=0, hard_rej_overlap=0, hard_rej_cost=0, hard_accept=0,
                soft_prop=0, soft_rej_cost=0, soft_accept=0)

    for it in range(iters):
        # refresh hotspots occasionally (placement drifts as it spreads)
        if it and it % 300 == 0:
            hot = _hot_macros(fe, pos, nm)

        # candidate index range restricted by move_hard/move_soft
        lo = 0 if move_hard else nh
        hi = nm if move_soft else nh
        if lo >= hi:
            break
        if rng.random() < hot_bias:
            w = hot[lo:hi]
            i = lo + int(rng.choice(hi - lo, p=w / w.sum()))
        else:
            i = int(rng.integers(lo, hi))
        is_hard = i < nh
        stat["hard_prop" if is_hard else "soft_prop"] += 1

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
            stat["hard_rej_overlap"] += 1
            continue  # keep placement legal; no expensive eval wasted

        old = pos[i].copy()
        pos[i, 0] = x; pos[i, 1] = y
        wl, d, c, cand = components(pos)
        evals += 1
        delta = cand - cur
        T = T0 * (Tend / T0) ** (it / iters) if T0 > 0 else 0.0  # geometric cooling
        accept = delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T))
        if accept:
            cur = cand; cwl, cd, cc = wl, d, c; accepts += 1
            stat["hard_accept" if is_hard else "soft_accept"] += 1
            if cur < best:
                best = cur; best_pos = pos.copy()
        else:
            pos[i] = old  # revert
            stat["hard_rej_cost" if is_hard else "soft_rej_cost"] += 1

        if evals and evals % log_every == 0:
            hist.append((evals, best))
            comp_hist.append((evals, cwl, 0.5 * cd, 0.5 * cc))
            logf(f"  it={it:5d} evals={evals:5d} accepts={accepts:5d} "
                 f"cur={cur:.4f} best={best:.4f}")

    hist.append((evals, best))
    comp_hist.append((evals, cwl, 0.5 * cd, 0.5 * cc))
    logf(f"  done: {evals} evals, {accepts} accepts, best={best:.4f}")
    info = dict(stat=stat, comp_hist=comp_hist)
    return best_pos, hist, info
