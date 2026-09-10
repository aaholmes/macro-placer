"""Bound-to-bound (B2B) quadratic initial placement.

Minimizes sum_e w_e * d_e^2 over pin pairs chosen by the B2B net model: every
pin of a net connects to the net's two extreme pins on that axis, with weight
w_net / ((k-1) * |d|), so the quadratic equals HPWL at the linearization point
(Spindler, Schlichtmann, Johannes, "Kraftwerk2", 2008). The problem is convex
and separable per axis; each axis solves a sparse SPD system L x = b with
Jacobi-preconditioned conjugate gradient (torch, GPU when available), and the
model is re-linearized a few rounds.

Fixed terminals anchor the system. Blocks with no path to any terminal have a
translation-invariant energy; they are held at the core center by a weak
spring. Output is positions only: overlap is expected and large. The
analytical stage's density term does the spreading.
"""
from __future__ import annotations

import numpy as np
import torch


ORIENT_SIGN = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])


def _pin_coord(owner, off, pos, axis):
    return np.where(owner >= 0, pos[np.clip(owner, 0, None), axis] + off[:, axis],
                    off[:, axis])


def effective_pins(fe, fixed_pos=None, orient=None):
    """(owner, off) with Klein-4 orientation signs applied to pin offsets and
    the blocks in fixed_pos {idx: (x, y)} turned into terminals (owner -1,
    off = absolute pin position)."""
    owner = fe.pin_owner.copy(); off = fe.pin_off.copy()
    if orient is not None:
        mv = owner >= 0
        off[mv] = off[mv] * ORIENT_SIGN[np.asarray(orient)[owner[mv]]]
    if fixed_pos:
        for i, (x, y) in fixed_pos.items():
            m = owner == i
            off[m, 0] += x; off[m, 1] += y
            owner[m] = -1
    return owner, off


def hpwl(owner, off, pin_net, net_weight, num_nets, pos):
    """Weighted HPWL for the given pin arrays (same convention as FastEval)."""
    total = 0.0
    for axis in (0, 1):
        c = _pin_coord(owner, off, pos, axis)
        hi = np.full(num_nets, -np.inf); lo = np.full(num_nets, np.inf)
        np.maximum.at(hi, pin_net, c); np.minimum.at(lo, pin_net, c)
        ok = np.isfinite(hi)
        total += float(np.sum(net_weight[ok] * (hi[ok] - lo[ok])))
    return total


def b2b_weights(pin_net, net_weight, pin, num_nets, dist_floor=1e-9, uniform=False):
    """B2B edge list at the pin coordinates `pin` (one axis).

    Returns (a, b, w): pin indices of each edge and its weight. With
    uniform=False the weights are w_net / ((k-1) |pin[a]-pin[b]|) so that
    sum w (pin[a]-pin[b])^2 equals the weighted HPWL. With uniform=True the
    distance factor is dropped (a plain quadratic on the B2B topology), used
    for the first linearization where all blocks coincide.
    """
    order = np.lexsort((pin, pin_net))              # by net, then coordinate
    net_sorted = pin_net[order]
    start = np.searchsorted(net_sorted, np.arange(num_nets), side="left")
    end = np.searchsorted(net_sorted, np.arange(num_nets), side="right")
    k = end - start
    ok = k >= 2
    lo = order[start[ok]]                            # min pin per net
    hi = order[end[ok] - 1]                          # max pin per net
    net_ids = np.nonzero(ok)[0]
    # (min, max) edge for every net
    a = [lo]; b = [hi]; wn = [net_weight[net_ids] / (k[ok] - 1)]
    # interior pins -> both extremes
    interior = np.ones(len(pin), bool)
    interior[lo] = False; interior[hi] = False
    interior &= ok[pin_net]
    idx = np.nonzero(interior)[0]
    if len(idx):
        nid = pin_net[idx]
        pos_in_net = np.searchsorted(net_ids, nid)
        wi = net_weight[nid] / (k[nid] - 1)
        a += [idx, idx]; b += [lo[pos_in_net], hi[pos_in_net]]; wn += [wi, wi]
    a = np.concatenate(a); b = np.concatenate(b); w = np.concatenate(wn)
    if not uniform:
        w = w / np.maximum(np.abs(pin[a] - pin[b]), dist_floor)
    return a, b, w


def _floating_mask(owner, net, n, m):
    """True for movable blocks with no net path to a fixed terminal."""
    ANCH = n + m                                     # node ids: blocks, nets, anchor
    lab = np.arange(n + m + 1)
    node = np.where(owner >= 0, owner, ANCH)
    for _ in range(n + m + 2):
        old = lab.copy()
        nl = lab[n:n + m].copy(); np.minimum.at(nl, net, lab[node]); lab[n:n + m] = nl
        bl = lab.copy(); np.minimum.at(bl, node, lab[n + net]); lab = np.minimum(lab, bl)
        if np.array_equal(lab, old):
            break
    return lab[:n] != lab[ANCH]


def _pcg(A, b, x0, tol, max_iter):
    """Jacobi-preconditioned CG for SPD sparse A (torch CSR)."""
    diag = _csr_diag(A)
    Minv = 1.0 / diag.clamp(min=1e-30)
    x = x0.clone()
    r = b - A @ x
    bnorm = float(torch.linalg.norm(b)) + 1e-30
    if float(torch.linalg.norm(r)) / bnorm <= tol:
        return x
    z = Minv * r; p = z.clone(); rz = float(r @ z)
    for _ in range(max_iter):
        Ap = A @ p
        alpha = rz / (float(p @ Ap) + 1e-300)
        x = x + alpha * p; r = r - alpha * Ap
        if float(torch.linalg.norm(r)) / bnorm <= tol:
            break
        z = Minv * r; rz_new = float(r @ z)
        p = z + (rz_new / rz) * p; rz = rz_new
    return x


def _csr_diag(A):
    n = A.shape[0]
    ptr = A.crow_indices(); col = A.col_indices(); val = A.values()
    row = torch.repeat_interleave(torch.arange(n, device=val.device), ptr[1:] - ptr[:-1])
    d = torch.zeros(n, dtype=val.dtype, device=val.device)
    d.index_add_(0, row[row == col], val[row == col])
    return d


def _assemble(n, owner, off, a, b, w, spring_k, spring_c, dev, held=None):
    """Laplacian and rhs for one axis from pin-pair edges (a, b, w).
    Pin p is at x[owner[p]] + off[p] (movable) or off[p] (fixed)."""
    oa, ob = owner[a], owner[b]
    ca, cb = off[a], off[b]
    both_fixed = (oa < 0) & (ob < 0)
    same = (oa == ob) & (oa >= 0)
    keep = ~(both_fixed | same)
    oa, ob, ca, cb, w = oa[keep], ob[keep], ca[keep], cb[keep], w[keep]
    ma, mb = oa >= 0, ob >= 0
    rows = []; cols = []; vals = []
    rhs = np.zeros(n)
    # movable a: diag += w, rhs += w*(cb - ca) (+ w*x_b coupling if b movable)
    rows.append(oa[ma]); cols.append(oa[ma]); vals.append(w[ma])
    np.add.at(rhs, oa[ma], w[ma] * (cb[ma] - ca[ma]))
    rows.append(ob[mb]); cols.append(ob[mb]); vals.append(w[mb])
    np.add.at(rhs, ob[mb], w[mb] * (ca[mb] - cb[mb]))
    mm = ma & mb
    rows += [oa[mm], ob[mm]]; cols += [ob[mm], oa[mm]]; vals += [-w[mm], -w[mm]]
    # weak spring for floating blocks
    sp = np.nonzero(spring_k > 0)[0]
    rows.append(sp); cols.append(sp); vals.append(spring_k[sp])
    rhs[sp] += spring_k[sp] * spring_c
    if held is not None and held.any():                # decoupled rows for held blocks
        hidx = np.nonzero(held)[0]
        rows.append(hidx); cols.append(hidx); vals.append(np.ones(len(hidx)))
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=dev)
    A = torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float64, device=dev),
                                (n, n)).coalesce().to_sparse_csr()
    return A, torch.tensor(rhs, dtype=torch.float64, device=dev)


def quadratic_place(fe, rounds=3, tol=1e-4, max_iter=200, hpwl_tol=0.005,
                    jitter_sigma=0.0, seed=0, device=None, logf=None,
                    fixed_pos=None, orient=None, return_hpwl=False):
    """B2B quadratic placement of every block of `fe` (a FastEval).

    rounds: number of B2B re-linearizations after the initial plain-quadratic
    solve; stops early once HPWL changes by less than hpwl_tol.
    tol, max_iter: PCG relative residual and iteration cap.
    jitter_sigma: if > 0, add N(0, (jitter_sigma * median block size)^2) noise
    per coordinate (generator seeded by `seed`) and clamp to the core.
    fixed_pos: {block idx: (x, y)} blocks held at given centers (extra anchors).
    orient: [N] Klein-4 orientation per block (0 N, 1 FN, 2 FS, 3 S), applied
    to pin offsets for the solve only.
    Returns [N, 2] float64 centers (and the weighted HPWL of the un-jittered
    solution if return_hpwl). Deterministic for fixed inputs.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = fe.b.num_macros
    owner, off = effective_pins(fe, fixed_pos, orient)
    W, H = float(fe.W), float(fe.H)
    sizes = fe.b.macro_sizes.numpy().astype(np.float64)
    fixed = owner < 0
    center = np.array([W / 2, H / 2])
    start = off[fixed].mean(0) if fixed.any() else center
    pos = np.tile(start, (n, 1)).astype(np.float64)
    if fixed_pos:
        for i, xy in fixed_pos.items():
            pos[i] = xy

    floating = _floating_mask(owner, fe.pin_net, n, fe.num_nets)
    spring_k = np.zeros(n)
    mean_w = float(np.mean(fe.net_weight)) if fe.num_nets else 1.0
    spring_k[floating] = 1e-4 * mean_w
    if not fixed.any():
        spring_k[:] = 1e-4 * mean_w
    held = np.zeros(n, bool)
    if fixed_pos:
        held[list(fixed_pos)] = True
        spring_k[held] = 0.0

    # floor on |d| so nearly-coincident pins do not blow up the weights
    dist_floor = 1e-3 * float(np.median(np.sqrt(sizes[:, 0] * sizes[:, 1])))
    prev_hpwl = None
    for rd in range(rounds + 1):
        new = pos.copy()
        for axis in (0, 1):
            pin = _pin_coord(owner, off, pos, axis)
            a, b, w = b2b_weights(fe.pin_net, fe.net_weight, pin, fe.num_nets,
                                  dist_floor=dist_floor, uniform=(rd == 0))
            A, rhs = _assemble(n, owner, off[:, axis], a, b, w, spring_k, center[axis], dev, held)
            x0 = torch.tensor(pos[:, axis], dtype=torch.float64, device=dev)
            new[:, axis] = _pcg(A, rhs, x0, tol, max_iter).cpu().numpy()
        if fixed_pos:
            for i, xy in fixed_pos.items():
                new[i] = xy
        pos = new
        cur = hpwl(owner, off, fe.pin_net, fe.net_weight, fe.num_nets, pos)
        if logf:
            logf(f"  b2b round {rd}: hpwl={cur:.1f}")
        if prev_hpwl is not None and abs(prev_hpwl - cur) <= hpwl_tol * max(prev_hpwl, 1e-30):
            break
        prev_hpwl = cur
    q_hpwl = hpwl(owner, off, fe.pin_net, fe.net_weight, fe.num_nets, pos)

    if jitter_sigma > 0:
        med = float(np.median(np.sqrt(sizes[:, 0] * sizes[:, 1])))
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(n, 2, generator=g, dtype=torch.float64).numpy()
        pos = pos + jitter_sigma * med * noise
    hw = sizes[:, 0] / 2; hh = sizes[:, 1] / 2
    pos[:, 0] = np.clip(pos[:, 0], np.minimum(hw, W / 2), np.maximum(W - hw, W / 2))
    pos[:, 1] = np.clip(pos[:, 1], np.minimum(hh, H / 2), np.maximum(H - hh, H / 2))
    return (pos, q_hpwl) if return_hpwl else pos
