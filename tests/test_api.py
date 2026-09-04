"""Tests for the in-memory entry point (placers.api).

The fixture is a tiny synthetic problem, so these run in seconds on CPU.
The parity test versus the challenge loader is opt-in (needs macro_place).
"""
import numpy as np
import pytest

from placers.api import PlacementInput, place, build_fast_eval


def tiny_problem():
    """4 hard + 2 soft macros, 3 nets, 100x100 canvas."""
    sizes = np.array([[10, 10], [10, 10], [8, 12], [12, 8], [6, 6], [6, 6]], dtype=np.float64)
    # each net: (weight, [pins]); pin = (macro_idx, ox, oy) or (-1, x, y) fixed port
    nets = [
        (1.0, [(0, 0.0, 0.0), (1, 0.0, 0.0), (4, 0.0, 0.0)]),
        (2.0, [(2, 1.0, -1.0), (3, 0.0, 0.0)]),
        (1.0, [(1, 0.0, 0.0), (5, 0.0, 0.0), (-1, 0.0, 50.0)]),
    ]
    return PlacementInput(width=100.0, height=100.0, sizes=sizes, num_hard=4, nets=nets)


def test_fast_eval_wirelength_matches_hand_computation():
    p = tiny_problem()
    fe = build_fast_eval(p)
    pos = np.array([[20, 20], [80, 20], [20, 80], [80, 80], [50, 50], [50, 20]], dtype=np.float64)
    # net0: pins at (20,20),(80,20),(50,50) -> hpwl 60+30 = 90, w=1
    # net1: pins at (21,79),(80,80) -> 59+1 = 60, w=2
    # net2: pins at (80,20),(50,20),(0,50) -> 80+30 = 110, w=1
    assert fe.wirelength_sum(pos) == pytest.approx(90 + 2 * 60 + 110)


def test_place_returns_legal_in_bounds_hard_macros():
    p = tiny_problem()
    r = place(p, budget_s=10.0, seed=0)
    pos = r.positions
    assert pos.shape == (6, 2)
    hw = p.sizes[:4, 0] / 2
    hh = p.sizes[:4, 1] / 2
    assert np.all(pos[:4, 0] >= hw - 1e-6) and np.all(pos[:4, 0] <= 100 - hw + 1e-6)
    assert np.all(pos[:4, 1] >= hh - 1e-6) and np.all(pos[:4, 1] <= 100 - hh + 1e-6)
    # pairwise hard-macro overlap
    for i in range(4):
        for j in range(i + 1, 4):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            assert (dx >= hw[i] + hw[j] - 1e-6) or (dy >= hh[i] + hh[j] - 1e-6), (i, j)
    assert r.seed == 0


def test_place_is_deterministic_per_seed():
    p = tiny_problem()
    a = place(p, budget_s=5.0, seed=3, iters=200, ls_iters=500)
    b = place(p, budget_s=5.0, seed=3, iters=200, ls_iters=500)
    c = place(p, budget_s=5.0, seed=4, iters=200, ls_iters=500)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions, c.positions)


CHAL = "/home/laz/partcl/macro-place-challenge-2026"


@pytest.mark.parity
def test_parity_with_challenge_loader_ibm01():
    """Shim FastEval must score identically to the challenge-object FastEval."""
    macro_place = pytest.importorskip("macro_place")
    from macro_place.loader import load_benchmark_from_dir
    from placers.fast_eval import FastEval
    from placers.api import input_from_challenge

    b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/ibm01")
    fe_ref = FastEval(b, plc)
    p = input_from_challenge(b, plc)
    fe_shim = build_fast_eval(p)
    rng = np.random.default_rng(0)
    n = b.num_macros
    pos = rng.uniform(0, 1, (n, 2)) * [plc.width, plc.height]
    assert fe_shim.wirelength_sum(pos) == pytest.approx(fe_ref.wirelength_sum(pos), rel=1e-12)
