"""Tests for the bound-to-bound quadratic initial placement (placers.quadratic)."""
import numpy as np
import pytest

from placers.api import PlacementInput, build_fast_eval, place
from placers.quadratic import b2b_weights, quadratic_place


def chain_problem(offset=0.0):
    """Terminals at (0,0) and (100,100); blocks A-B-C chained between them with
    2-pin unit-weight nets. A's pin sits `offset` right of and above its center."""
    sizes = np.array([[4, 4], [4, 4], [4, 4]], dtype=np.float64)
    nets = [
        (1.0, [(-1, 0.0, 0.0), (0, offset, offset)]),
        (1.0, [(0, offset, offset), (1, 0.0, 0.0)]),
        (1.0, [(1, 0.0, 0.0), (2, 0.0, 0.0)]),
        (1.0, [(2, 0.0, 0.0), (-1, 100.0, 100.0)]),
    ]
    return PlacementInput(width=100.0, height=100.0, sizes=sizes, num_hard=3, nets=nets)


def test_chain_matches_hand_computed_minimum():
    fe = build_fast_eval(chain_problem())
    pos = quadratic_place(fe, tol=1e-12)
    # equal spacing of the four unit-weight segments: 25, 50, 75 on each axis
    np.testing.assert_allclose(pos, [[25, 25], [50, 50], [75, 75]], atol=1e-6)


def test_pin_offsets_pull_by_pin_not_center():
    fe = build_fast_eval(chain_problem(offset=5.0))
    pos = quadratic_place(fe, tol=1e-12)
    # A's PIN sits at 25, so A's center sits at 20
    np.testing.assert_allclose(pos, [[20, 20], [50, 50], [75, 75]], atol=1e-6)


def random_netlist(seed, n_blocks=30, n_nets=40, n_fixed=6):
    rng = np.random.default_rng(seed)
    sizes = rng.uniform(2, 10, (n_blocks, 2))
    nets = []
    for _ in range(n_nets):
        k = int(rng.integers(2, 8))
        pins = []
        for _ in range(k):
            if rng.random() < 0.15:
                pins.append((-1, float(rng.uniform(0, 200)), float(rng.uniform(0, 200))))
            else:
                pins.append((int(rng.integers(0, n_blocks)),
                             float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))))
        nets.append((float(rng.uniform(0.5, 3)), pins))
    p = PlacementInput(width=200.0, height=200.0, sizes=sizes, num_hard=n_blocks // 2, nets=nets)
    pos = rng.uniform(0, 200, (n_blocks, 2))
    return p, pos


def test_b2b_quadratic_equals_hpwl_at_linearization_point():
    p, pos = random_netlist(1)
    fe = build_fast_eval(p)
    total = 0.0
    for axis in (0, 1):
        pin = np.where(fe.pin_owner >= 0,
                       pos[np.clip(fe.pin_owner, 0, None), axis] + fe.pin_off[:, axis],
                       fe.pin_off[:, axis])
        a, b, w = b2b_weights(fe.pin_net, fe.net_weight, pin, fe.num_nets)
        total += float(np.sum(w * (pin[a] - pin[b]) ** 2))
    assert total == pytest.approx(fe.wirelength_sum(pos), rel=1e-6)


def test_deterministic():
    p, _ = random_netlist(2)
    fe = build_fast_eval(p)
    a = quadratic_place(fe)
    b = quadratic_place(fe)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a))


def test_floating_component_is_finite_and_inside_core():
    p, _ = random_netlist(3)
    # two extra blocks connected only to each other: no path to any terminal
    p.sizes = np.vstack([p.sizes, [[5, 5], [5, 5]]])
    p.nets.append((1.0, [(30, 0.0, 0.0), (31, 1.0, 0.0)]))
    fe = build_fast_eval(p)
    pos = quadratic_place(fe, tol=1e-10)
    assert np.all(np.isfinite(pos))
    assert np.all(pos >= 0) and np.all(pos[:, 0] <= 200) and np.all(pos[:, 1] <= 200)
    # the floating pair sits at the core center, its pins coincident
    assert pos[30, 0] + 0.0 == pytest.approx(pos[31, 0] + 1.0, abs=1e-3)
    assert pos[30, 1] == pytest.approx(100.0, abs=1e-3)


def test_no_terminals_at_all_is_finite():
    p, _ = random_netlist(4, n_fixed=0)
    p.nets = [(w, [pin for pin in pins if pin[0] >= 0]) for w, pins in p.nets]
    p.nets = [(w, pins) for w, pins in p.nets if len(pins) >= 2]
    fe = build_fast_eval(p)
    pos = quadratic_place(fe)
    assert np.all(np.isfinite(pos))


def test_jitter_is_seeded_and_bounded():
    p, _ = random_netlist(5)
    fe = build_fast_eval(p)
    base = quadratic_place(fe)
    j1 = quadratic_place(fe, jitter_sigma=0.5, seed=7)
    j2 = quadratic_place(fe, jitter_sigma=0.5, seed=7)
    j3 = quadratic_place(fe, jitter_sigma=0.5, seed=8)
    np.testing.assert_array_equal(j1, j2)
    assert not np.array_equal(j1, j3)
    assert not np.array_equal(j1, base)
    med = float(np.median(np.sqrt(p.sizes[:, 0] * p.sizes[:, 1])))
    assert np.abs(j1 - base).max() < 6 * 0.5 * med


def test_pipeline_smoke_quadratic_init():
    p, _ = random_netlist(6)
    for init in ("quadratic_b2b", "quadratic_b2b_jitter"):
        r = place(p, budget_s=5.0, seed=0, iters=100, ls_iters=200, init=init)
        pos = r.positions
        nh = p.num_hard
        hw, hh = p.sizes[:nh, 0] / 2, p.sizes[:nh, 1] / 2
        assert np.all(np.isfinite(pos))
        assert np.all(pos[:nh, 0] >= hw - 1e-6) and np.all(pos[:nh, 0] <= 200 - hw + 1e-6)
        for i in range(nh):
            for j in range(i + 1, nh):
                dx = abs(pos[i, 0] - pos[j, 0]); dy = abs(pos[i, 1] - pos[j, 1])
                assert (dx >= hw[i] + hw[j] - 1e-6) or (dy >= hh[i] + hh[j] - 1e-6), (init, i, j)
        assert r.meta["init"] == init


def test_random_init_path_is_unchanged():
    """init='random' must reproduce the pre-change trajectory bit for bit."""
    p, _ = random_netlist(7)
    a = place(p, budget_s=5.0, seed=1, iters=100, ls_iters=200)
    b = place(p, budget_s=5.0, seed=1, iters=100, ls_iters=200, init="random")
    np.testing.assert_array_equal(a.positions, b.positions)
