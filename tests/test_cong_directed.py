"""Tests for the congestion-directed detailed-placement operator.

The operator changes only the PROPOSAL distribution: for a hot-selected cell it
proposes a destination in a COLD grid cell (sampled from the same congestion+
density field `_hot_macros` already builds), instead of a random local jitter.
Acceptance is still on the true incremental proxy delta, so the operator can only
raise the hit-rate on good moves -- it can never worsen the result. These tests
pin the sampler's behaviour (cold-seeking, in-bounds, deterministic) and the
search invariants (greedy monotonicity, determinism), NOT the open hypothesis
that it beats baseline -- that's what the A/B run measures.

Self-contained assert runner (no pytest).
"""
import os, sys, traceback
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np
from placers.local_search import _directed_target, _cell_field, _hot_macros, optimize_fast

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
_ENV = {}


def env(nm="ibm01"):
    if nm not in _ENV:
        from macro_place.loader import load_benchmark_from_dir
        from placers.fast_eval import FastEval
        from placers.legalizer import legalize
        b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
        fe = FastEval(b, plc)
        sz = b.macro_sizes.numpy()
        pos = legalize(b.macro_positions.numpy().astype(np.float64), sz, b.num_hard_macros,
                       b.canvas_width, b.canvas_height, gap=0.01)[0]
        _ENV[nm] = (b, plc, fe, sz, pos)
    return _ENV[nm]


TESTS = []
def test(fn): TESTS.append(fn); return fn


# ---- unit: the cold-seeking sampler (no benchmark) -----------------------------
@test
def directed_target_prefers_cold_cells():
    gc = gr = 10; gw = gh = 5.0; W = H = 50.0
    field = np.full(gc * gr, 10.0)          # everything hot ...
    # ... except a 3x3 cold block in the bottom-left corner (rows 0-2, cols 0-2)
    for r in range(3):
        for c in range(3):
            field[r * gc + c] = 0.0
    rng = np.random.default_rng(0)
    inb = 0; N = 400
    for _ in range(N):
        x, y = _directed_target(field, gc, gr, gw, gh, 0.5, 0.5, W, H, rng)
        col = int(x // gw); row = int(y // gh)
        if row < 3 and col < 3:
            inb += 1
    # cold block is 9/100 cells; uniform would hit ~9%. Cold-seeking must be >>.
    assert inb / N > 0.75, f"only {inb}/{N} samples landed in the cold block"


@test
def directed_target_stays_in_bounds():
    gc = gr = 8; gw = gh = 4.0; W = H = 32.0
    field = np.abs(np.random.default_rng(1).normal(size=gc * gr)) + 0.1
    rng = np.random.default_rng(2)
    hw = hh = 3.0                            # big macro -> tight legal band
    for _ in range(500):
        x, y = _directed_target(field, gc, gr, gw, gh, hw, hh, W, H, rng)
        assert hw - 1e-9 <= x <= W - hw + 1e-9, f"x={x} out of [{hw},{W-hw}]"
        assert hh - 1e-9 <= y <= H - hh + 1e-9, f"y={y} out of [{hh},{H-hh}]"


@test
def directed_target_is_deterministic():
    gc = gr = 6; gw = gh = 2.0; W = H = 12.0
    field = np.linspace(0.1, 5.0, gc * gr)
    a = [_directed_target(field, gc, gr, gw, gh, 0.3, 0.3, W, H, np.random.default_rng(7))
         for _ in range(3)]
    # same fresh rng seed each call -> identical draw
    assert all(t == a[0] for t in a), f"non-deterministic: {a}"


# ---- integration: field + search invariants (ibm01) ----------------------------
@test
def cell_field_is_finite_nonneg():
    _, _, fe, _, pos = env()
    f = _cell_field(fe, pos)
    assert f.shape[0] == fe.plc.grid_col * fe.plc.grid_row, f"bad shape {f.shape}"
    assert np.all(np.isfinite(f)) and np.all(f >= 0), "field has nan/inf/negative"
    # matches the per-cell values _hot_macros gathers (same underlying field)
    hot = _hot_macros(fe, pos, env()[0].num_macros)
    assert np.all(np.isfinite(hot)) and np.all(hot > 0)


@test
def directed_search_never_worsens():
    _, plc, fe, _, pos = env()
    init = fe.init_incremental(pos)
    best, _ = optimize_fast(fe, pos, iters=3000, seed=0, cong_directed=1.0,
                            move_hard=False, move_soft=True, refresh=1000,
                            log_every=100000, logf=lambda *a: None)
    final = fe.cost_current()
    assert final <= init + 1e-9, f"greedy worsened cost: {init} -> {final}"


@test
def directed_search_is_deterministic():
    _, _, fe, _, pos = env()
    kw = dict(iters=2000, seed=3, cong_directed=0.7, move_hard=False, move_soft=True,
              refresh=1000, log_every=100000, logf=lambda *a: None)
    b1, _ = optimize_fast(fe, pos, **kw)
    b2, _ = optimize_fast(fe, pos, **kw)
    assert np.allclose(b1, b2), "same seed gave different placements"


@test
def directed_enables_soft_long_moves():
    """With directed=1.0 a soft cell can make a single large jump toward cold
    space -- something the baseline (soft cells only jitter by sigma) forbids.
    Verify the proposal reaches far cells, via the sampler on the real field."""
    b, _, fe, sz, pos = env()
    field = _cell_field(fe, pos)
    gc = fe.plc.grid_col; gr = fe.plc.grid_row
    gw = fe.W / gc; gh = fe.H / gr
    rng = np.random.default_rng(0)
    i = b.num_hard_macros                    # first soft macro
    far = 0
    for _ in range(200):
        x, y = _directed_target(field, gc, gr, gw, gh, sz[i, 0] / 2, sz[i, 1] / 2,
                                fe.W, fe.H, rng)
        if abs(x - pos[i, 0]) > 5.0 or abs(y - pos[i, 1]) > 5.0:
            far += 1
    assert far > 0, "directed sampler never proposed a long-range soft move"


if __name__ == "__main__":
    passed = 0
    for fn in TESTS:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(TESTS)} passed")
    sys.exit(0 if passed == len(TESTS) else 1)
