"""Overlap legalizer for hard macros.

Takes an arbitrary placement (possibly with overlaps) and returns one with zero
hard-macro overlaps under the evaluator's strict (>0) test, staying within the
canvas. Uses iterative pairwise push-apart: each overlapping pair is separated
along its axis of least penetration by half the overlap plus a safety gap, then
positions are clamped into bounds. Repeats until clean or max_iters.

Soft macros are left untouched (they are allowed to overlap).

Design goals:
- Idempotent on already-legal input (only injects the gap once).
- Minimal displacement: pushes along the *shorter* overlap axis.
- Deterministic.
"""
from __future__ import annotations
import numpy as np


def legalize(positions, sizes, num_hard, canvas_w, canvas_h,
             gap: float = 0.01, max_iters: int = 2000, cleanup_rounds: int = 300,
             seed: int = 0, verbose: bool = False):
    """positions, sizes: [N,2] arrays/tensors (centers, w/h). Returns np.float64 [N,2] copy."""
    pos = np.asarray(positions, dtype=np.float64).copy()
    sz = np.asarray(sizes, dtype=np.float64)
    hw = sz[:num_hard, 0] / 2.0
    hh = sz[:num_hard, 1] / 2.0

    # clamp helper: keep each hard macro fully inside the canvas
    def clamp():
        np.clip(pos[:num_hard, 0], hw, canvas_w - hw, out=pos[:num_hard, 0])
        np.clip(pos[:num_hard, 1], hh, canvas_h - hh, out=pos[:num_hard, 1])

    clamp()
    for it in range(max_iters):
        x = pos[:num_hard, 0]; y = pos[:num_hard, 1]
        # pairwise required separations and actual gaps (vectorized upper triangle)
        dx = np.abs(x[:, None] - x[None, :])
        dy = np.abs(y[:, None] - y[None, :])
        req_x = hw[:, None] + hw[None, :] + gap
        req_y = hh[:, None] + hh[None, :] + gap
        ox = req_x - dx  # >0 means overlapping-with-gap in x
        oy = req_y - dy
        overlap = (ox > 0) & (oy > 0)
        iu = np.triu_indices(num_hard, k=1)
        pairs = np.where(overlap[iu])[0]
        if pairs.size == 0:
            if verbose:
                print(f"  legalized in {it} iters")
            return pos, it, 0
        # accumulate displacement per macro, then apply (Jacobi-style, stable)
        disp = np.zeros((num_hard, 2))
        I = iu[0][pairs]; J = iu[1][pairs]
        for i, j in zip(I, J):
            pex, pey = ox[i, j], oy[i, j]
            if pex <= pey:  # separate along x (least penetration)
                shift = pex / 2.0 + 1e-9
                s = 1.0 if x[i] >= x[j] else -1.0
                disp[i, 0] += s * shift; disp[j, 0] -= s * shift
            else:
                shift = pey / 2.0 + 1e-9
                s = 1.0 if y[i] >= y[j] else -1.0
                disp[i, 1] += s * shift; disp[j, 1] -= s * shift
        pos[:num_hard] += disp
        clamp()

    # ---- Gauss-Seidel cleanup for STRICT residual overlaps only ----
    # Jacobi leaves some pairs sub-gap-close (dist < gap but not overlapping);
    # those are fine for the evaluator (strict >0 test). We only need to clear
    # true overlaps, and must never make the strict count worse.
    area = sz[:num_hard, 0] * sz[:num_hard, 1]
    iu = np.triu_indices(num_hard, k=1)
    MARGIN = 1e-3   # target clearance for cleared pairs (float32-safe)

    def strict_pairs():
        x = pos[:num_hard, 0]; y = pos[:num_hard, 1]
        ox = hw[:, None] + hw[None, :] - np.abs(x[:, None] - x[None, :])
        oy = hh[:, None] + hh[None, :] - np.abs(y[:, None] - y[None, :])
        m = ((ox > 0) & (oy > 0))[iu]
        return iu[0][m], iu[1][m]

    I, J = strict_pairs()
    if len(I) == 0:
        return pos, it, 0                         # ibm08/ibm17: sub-gap only, fine

    best_pos = pos.copy(); best_n = len(I)
    for rnd in range(cleanup_rounds):
        I, J = strict_pairs()
        if len(I) == 0:
            return pos, it + rnd, 0
        for i, j in zip(I, J):
            pex = hw[i] + hw[j] - abs(pos[i, 0] - pos[j, 0])
            pey = hh[i] + hh[j] - abs(pos[i, 1] - pos[j, 1])
            if pex <= 0 or pey <= 0:
                continue
            a = 0 if pex <= pey else 1             # least-penetration axis
            pen = (pex if a == 0 else pey) + MARGIN
            half = hw if a == 0 else hh
            lim = canvas_w if a == 0 else canvas_h
            mover, other = (i, j) if area[i] <= area[j] else (j, i)
            dirn = 1.0 if pos[mover, a] >= pos[other, a] else -1.0
            newp = pos[mover, a] + dirn * pen
            if half[mover] <= newp <= lim - half[mover]:
                pos[mover, a] = newp
            else:                                  # mover pinned -> move the other
                mover = other
                newp = pos[mover, a] - dirn * pen
                pos[mover, a] = min(max(newp, half[mover]), lim - half[mover])
        n = len(strict_pairs()[0])
        if n < best_n:
            best_pos = pos.copy(); best_n = n

    # ---- guaranteed finisher: relocate remaining stuck macros to empty space
    # (utilization << 100%, so a free spot always exists). ----
    pos = best_pos
    step = max(min(hw.min(), hh.min()), 0.1)

    def free_spot(k):
        others = np.ones(num_hard, bool); others[k] = False
        oc = pos[:num_hard][others]
        ohw = hw[others] + hw[k] + MARGIN; ohh = hh[others] + hh[k] + MARGIN
        xs = np.arange(hw[k], canvas_w - hw[k] + 1e-9, step)
        ys = np.arange(hh[k], canvas_h - hh[k] + 1e-9, step)
        gx, gy = np.meshgrid(xs, ys)
        cand = np.column_stack([gx.ravel(), gy.ravel()])
        d = np.linalg.norm(cand - pos[k], axis=1)
        for c in cand[np.argsort(d)]:
            if not (((np.abs(c[0] - oc[:, 0]) < ohw) &
                     (np.abs(c[1] - oc[:, 1]) < ohh)).any()):
                return c
        return None

    for _ in range(best_n + 5):
        I, J = strict_pairs()
        if len(I) == 0:
            break
        i, j = I[0], J[0]
        k = i if area[i] <= area[j] else j        # relocate the smaller macro
        spot = free_spot(k)
        if spot is None:
            break
        pos[k] = spot
    resid = len(strict_pairs()[0])
    if verbose:
        print(f"  cleanup+relocate: strict residual {resid}")
    return pos, it + cleanup_rounds, resid
