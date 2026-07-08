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
             gap: float = 0.01, max_iters: int = 2000, verbose: bool = False):
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
    # ran out of iters — report residual
    x = pos[:num_hard, 0]; y = pos[:num_hard, 1]
    dx = np.abs(x[:, None] - x[None, :]); dy = np.abs(y[:, None] - y[None, :])
    resid = int((((hw[:, None] + hw[None, :] - dx) > 0) &
                 ((hh[:, None] + hh[None, :] - dy) > 0))[np.triu_indices(num_hard, 1)].sum())
    if verbose:
        print(f"  hit max_iters={max_iters}, residual overlaps={resid}")
    return pos, max_iters, resid
