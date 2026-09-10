"""Tests for the corner-assignment screen (placers.corner_seeds)."""
import numpy as np
import pytest

from placers.api import PlacementInput, build_fast_eval, place
from placers.corner_seeds import (SLOTS, corner_screen, inward_orientation,
                                  sample_assignments, slot_center)
from test_quadratic import random_netlist


def big_macro_problem(seed=0, n_big=6):
    """random_netlist plus n_big large hard macros (indices 0..n_big-1) wired
    to terminals and to a few small blocks, each with off-center pins."""
    p, _ = random_netlist(seed)
    n_small = len(p.sizes)
    big = np.full((n_big, 2), 40.0)
    sizes = np.vstack([big, p.sizes])
    nets = [(w, [(o + n_big if o >= 0 else o, a, b) for o, a, b in pins]) for w, pins in p.nets]
    rng = np.random.default_rng(seed)
    for i in range(n_big):
        for _ in range(3):
            nets.append((1.0, [(i, 12.0, -7.0), (int(rng.integers(n_big, n_big + n_small)), 0.0, 0.0),
                               (-1, float(rng.uniform(0, 200)), float(rng.uniform(0, 200)))]))
    return PlacementInput(width=200.0, height=200.0, sizes=sizes,
                          num_hard=n_big + p.num_hard, nets=nets)


def test_inward_orientation_rule():
    fe = build_fast_eval(big_macro_problem())
    # macro 0's pin centroid is at (+12, -7) relative to its center.
    # bottom-left corner: interior is +x,+y -> keep x, mirror y (FS = 2)
    assert inward_orientation(fe, 0, "bl") == 2
    # top-right corner: interior is -x,-y -> mirror x, keep y (FN = 1)
    assert inward_orientation(fe, 0, "tr") == 1
    # bottom edge midpoint: only y matters (interior +y) -> mirror y
    assert inward_orientation(fe, 0, "bm") == 2
    # left edge midpoint: only x matters (interior +x) -> keep
    assert inward_orientation(fe, 0, "lm") == 0


def test_slot_centers_inside_core():
    p = big_macro_problem(); fe = build_fast_eval(p)
    for s in SLOTS:
        x, y = slot_center(fe, 0, s)
        assert 20 - 1e-9 <= x <= 180 + 1e-9 and 20 - 1e-9 <= y <= 180 + 1e-9


def test_assignments_are_deduplicated_and_valid():
    p = big_macro_problem(); fe = build_fast_eval(p)
    asg = sample_assignments(fe, M=6, N=200, seed=0)
    keys = [tuple(sorted(a)) for a in asg]
    assert len(keys) == len(set(keys))
    for a in asg:
        macros = [m for m, _ in a]; slots = [s for _, s in a]
        assert len(a) == 4 and len(set(macros)) == 4 and len(set(slots)) == 4
        assert all(m < 6 for m in macros) and all(s in SLOTS for s in slots)
    assert len(asg) > 20


def test_corner_screen_end_to_end():
    p = big_macro_problem(); fe = build_fast_eval(p)
    from placers.analytical import DifferentiablePlacer
    from placers.api import DEFAULT_HP
    P = DifferentiablePlacer(fe)
    res = corner_screen(P, hp=DEFAULT_HP, iters=50, N=20, K1=4, K2=2, M=6, seed=0)
    assert len(res.seeds) == 2
    for s in res.seeds:
        assert s.shape == (len(p.sizes), 2) and np.all(np.isfinite(s))
    assert len(res.screen1) >= 4 and len(res.screen2) == 4
    # screen-2 survivors are the K1 lowest quadratic-HPWL assignments
    q = sorted(r["hpwl"] for r in res.screen1)[:4]
    assert sorted(r["hpwl"] for r in res.screen2) == pytest.approx(q)
    # fixed macros sit at their slots in the screen-1 solution
    r = res.screen2[0]
    for m, s in r["assignment"]:
        np.testing.assert_allclose(r["b2b_pos"][m], slot_center(fe, m, s), atol=1e-6)
    assert res.screen2 == sorted(res.screen2, key=lambda r: r["score"])


def test_place_with_corner_screen_init():
    p = big_macro_problem()
    r = place(p, budget_s=5.0, seed=1, iters=50, ls_iters=200, init="corner_screen",
              corner=dict(N=20, K1=4, K2=2, M=6))
    nh = p.num_hard; pos = r.positions
    hw, hh = p.sizes[:nh, 0] / 2, p.sizes[:nh, 1] / 2
    assert np.all(np.isfinite(pos))
    for i in range(nh):
        for j in range(i + 1, nh):
            dx = abs(pos[i, 0] - pos[j, 0]); dy = abs(pos[i, 1] - pos[j, 1])
            assert (dx >= hw[i] + hw[j] - 1e-6) or (dy >= hh[i] + hh[j] - 1e-6), (i, j)
    assert r.meta["init"] == "corner_screen" and r.meta["corner"]["K2"] == 2
