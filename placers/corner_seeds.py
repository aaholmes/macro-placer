"""Corner-assignment screening: the B2B quadratic solve as a fast screen over
where the largest hard macros go.

Slots are the four core corners and the four edge midpoints. An assignment
fixes four large macros at four distinct slots, oriented so each macro's pin
centroid faces the core interior, and solves the B2B quadratic for everything
else (placers.quadratic). Screen 1 ranks assignments by quadratic HPWL and
keeps K1; screen 2 runs the gradient stage plus legalization from each
survivor (macros unfixed) and ranks by the exact contest score; the top K2
placements are returned as seeds for the deepen stage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from placers.quadratic import quadratic_place

# bl, br, tl, tr corners; bm, tm, lm, rm edge midpoints. (ux, uy) in {-1,0,1}
# locates the slot on the core: -1 low edge, +1 high edge, 0 mid.
SLOTS = {"bl": (-1, -1), "br": (1, -1), "tl": (-1, 1), "tr": (1, 1),
         "bm": (0, -1), "tm": (0, 1), "lm": (-1, 0), "rm": (1, 0)}


def slot_center(fe, i, slot):
    """Center of block i when flush inside the core at `slot`."""
    ux, uy = SLOTS[slot]
    w, h = (float(v) for v in fe.b.macro_sizes[i])
    W, H = float(fe.W), float(fe.H)
    x = {-1: w / 2, 0: W / 2, 1: W - w / 2}[ux]
    y = {-1: h / 2, 0: H / 2, 1: H - h / 2}[uy]
    return x, y


def inward_orientation(fe, i, slot):
    """Klein-4 orientation (0 N, 1 FN mirror-x, 2 FS mirror-y, 3 S) that turns
    block i's pin centroid toward the core interior at `slot`. Along an axis
    where the slot is mid-core, or the centroid offset is zero, no flip."""
    pins = fe.pin_owner == i
    if not pins.any():
        return 0
    cx, cy = fe.pin_off[pins].mean(0)
    ux, uy = SLOTS[slot]
    flip_x = ux != 0 and cx * (-ux) < 0            # interior direction is -ux
    flip_y = uy != 0 and cy * (-uy) < 0
    return int(flip_x) + 2 * int(flip_y)


def sample_assignments(fe, M=12, N=200, seed=0, per=4):
    """N assignments of `per` macros (from the top-M hard macros by area,
    drawn without replacement with probability proportional to area) to `per`
    distinct slots; deduplicated. Returns a list of tuples of (macro, slot)."""
    sizes = fe.b.macro_sizes.numpy().astype(np.float64)
    nh = fe.b.num_hard_macros
    area = sizes[:nh, 0] * sizes[:nh, 1]
    top = np.argsort(-area)[:min(M, nh)]
    per = min(per, len(top), len(SLOTS))
    if per == 0:
        return []
    prob = area[top] / area[top].sum()
    rng = np.random.default_rng(seed)
    names = list(SLOTS)
    seen = set(); out = []
    for _ in range(N):
        macros = rng.choice(top, size=per, replace=False, p=prob)
        slots = rng.choice(len(names), size=per, replace=False)
        a = tuple(sorted((int(m), names[s]) for m, s in zip(macros, slots)))
        if a not in seen:
            seen.add(a); out.append(a)
    return out


@dataclass
class CornerScreenResult:
    seeds: list                       # K2 legalized placements, best first
    screen1: list = field(default_factory=list)   # dicts: assignment, hpwl
    screen2: list = field(default_factory=list)   # + b2b_pos, score, pos
    t_screen1: float = 0.0
    t_screen2: float = 0.0


def corner_screen(P, hp, iters, M=12, N=200, K1=16, K2=4, seed=0,
                  deadline=None, logf=None, b2b_rounds=1, b2b_tol=1e-3, b2b_device="cpu",
                  diff_fn=None):
    """Run both screens on DifferentiablePlacer P. hp/iters configure the
    gradient stage (placers.api._diff_stage). The B2B solves run on the CPU by
    default: at a few thousand unknowns they are launch-overhead bound on a GPU
    and 4-12x slower there. diff_fn(P, seed, hp, iters, deadline, x0) -> raw
    positions replaces the api gradient stage (e.g. the contest-protocol one).
    Returns CornerScreenResult."""
    from placers.api import _diff_stage
    if diff_fn is None:
        diff_fn = lambda P, seed, hp, iters, deadline, x0: _diff_stage(
            P, seed, hp, iters, deadline, init="given", x0=x0)
    from placers.legalizer import legalize
    fe = P.fe
    sizes = fe.b.macro_sizes.numpy().astype(np.float64)
    nh = fe.b.num_hard_macros
    t0 = time.monotonic()
    rows = []
    for a in sample_assignments(fe, M=M, N=N, seed=seed):
        fixed = {m: slot_center(fe, m, s) for m, s in a}
        orient = np.zeros(fe.b.num_macros, np.int64)
        for m, s in a:
            orient[m] = inward_orientation(fe, m, s)
        pos, h = quadratic_place(fe, rounds=b2b_rounds, tol=b2b_tol, device=b2b_device,
                                 fixed_pos=fixed, orient=orient, return_hpwl=True)
        rows.append(dict(assignment=a, hpwl=h, b2b_pos=pos, orient=orient))
    rows.sort(key=lambda r: r["hpwl"])
    t1 = time.monotonic()
    if logf:
        logf(f"  corner screen 1: {len(rows)} assignments in {t1 - t0:.2f}s")
    surv = rows[:K1]
    for k, r in enumerate(surv):
        raw = diff_fn(P, seed, hp, iters, deadline, r["b2b_pos"])
        lg, _, resid = legalize(raw, sizes, nh, fe.W, fe.H, seed=seed)
        sc = float(fe.wirelength_cost(lg) + 0.5 * fe.density_cost(lg) + 0.5 * fe.congestion_cost(lg))
        r["score"] = sc + (1.0 if resid else 0.0); r["pos"] = lg
        if logf:
            logf(f"  corner screen 2 [{k}]: hpwl={r['hpwl']:.0f} score={sc:.4f} {r['assignment']}")
    surv.sort(key=lambda r: r["score"])
    t2 = time.monotonic()
    return CornerScreenResult(seeds=[r["pos"] for r in surv[:K2]],
                              screen1=[dict(assignment=r["assignment"], hpwl=r["hpwl"]) for r in rows],
                              screen2=surv, t_screen1=t1 - t0, t_screen2=t2 - t1)
