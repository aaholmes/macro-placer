"""Trajectory recording and gif rendering (contest-free path)."""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from placers.api import PlacementInput, place
from placers.viz import TrajectoryRecorder, render_frame


def tiny_problem():
    sizes = np.array([[10, 10], [10, 10], [8, 12], [6, 6]], dtype=np.float64)
    nets = [
        (1.0, [(0, 0.0, 0.0), (1, 0.0, 0.0)]),
        (1.0, [(2, 0.0, 0.0), (3, 0.0, 0.0), (-1, 0.0, 50.0)]),
    ]
    return PlacementInput(width=100.0, height=100.0, sizes=sizes,
                          num_hard=3, nets=nets)


def test_render_frame_returns_image():
    p = tiny_problem()
    pos = np.array([[20, 20], [80, 20], [50, 80], [50, 50]], dtype=float)
    img = render_frame(pos, p.sizes, p.num_hard, p.width, p.height,
                       curve=[(0, 1.0), (10, 0.8)], title="t")
    assert img.size[0] > 100 and img.size[1] > 100


def test_place_invokes_on_frame_and_gif_saves(tmp_path):
    p = tiny_problem()
    rec = TrajectoryRecorder(p)
    r = place(p, budget_s=5.0, seed=0, iters=120, ls_iters=300,
              on_frame=rec, frame_every=40)
    stages = {f.stage for f in rec.frames}
    assert "diff" in stages and "final" in stages
    assert len(rec.frames) >= 4
    assert rec.frames[-1].pos.shape == r.positions.shape
    out = tmp_path / "traj.gif"
    rec.save_gif(str(out), fps=4)
    assert out.stat().st_size > 1000


def test_on_frame_none_is_unchanged():
    p = tiny_problem()
    a = place(p, budget_s=5.0, seed=1, iters=100, ls_iters=200)
    rec = TrajectoryRecorder(p)
    b = place(p, budget_s=5.0, seed=1, iters=100, ls_iters=200,
              on_frame=rec, frame_every=25)
    np.testing.assert_array_equal(a.positions, b.positions)
