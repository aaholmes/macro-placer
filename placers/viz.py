"""Contest-free trajectory visualization: record frames from placers.api.place
via its on_frame callback and render them as a gif (macros moving, score
curve alongside). The contest-tied wrapper lives in scripts/contest_gif.py;
integration repos import this module directly.

Optional dependencies: matplotlib, pillow (the "viz" extra).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    stage: str        # "diff" | "legal" | "final"
    it: int
    pos: np.ndarray   # [N,2] centers
    score: float


class TrajectoryRecorder:
    """Pass as on_frame= to placers.api.place. Recording never alters the
    optimization; it only copies positions and scores them with FastEval."""

    def __init__(self, problem, score_fn=None):
        from placers.api import build_fast_eval
        self.problem = problem
        self.fe = build_fast_eval(problem)
        self.score_fn = score_fn or (lambda pos: self.fe.wirelength_cost(pos))
        self.frames: list[Frame] = []

    def __call__(self, stage: str, it: int, pos: np.ndarray):
        pos = np.asarray(pos, np.float64).copy()
        self.frames.append(Frame(stage, it, pos, float(self.score_fn(pos))))

    def save_gif(self, path: str, fps: int = 8, dpi: int = 80,
                 hold_last: int = 8):
        from PIL import Image
        p = self.problem
        curve = [(i, f.score) for i, f in enumerate(self.frames)]
        imgs = []
        for i, f in enumerate(self.frames):
            imgs.append(render_frame(
                f.pos, p.sizes, p.num_hard, p.width, p.height,
                curve=curve[: i + 1], curve_total=len(self.frames),
                title=f"{f.stage} it={f.it}  score={f.score:.4f}", dpi=dpi))
        imgs += [imgs[-1]] * hold_last
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)


def render_frame(pos, sizes, num_hard, width, height, curve=(),
                 curve_total=None, title="", dpi=80):
    """One frame: layout on the left, score curve on the right. Returns a
    PIL image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    fig, (ax, axc) = plt.subplots(
        1, 2, figsize=(9, 4.5), dpi=dpi,
        gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(len(pos)):
        w, h = sizes[i]
        hard = i < num_hard
        ax.add_patch(Rectangle(
            (pos[i, 0] - w / 2, pos[i, 1] - h / 2), w, h,
            facecolor="#c44e52" if hard else "#4c72b0",
            alpha=0.9 if hard else 0.25,
            edgecolor="black" if hard else "none", linewidth=0.5))
    ax.set_title(title, fontsize=9)

    if curve:
        xs, ys = zip(*curve)
        axc.plot(xs, ys, color="#4c72b0")
        axc.scatter([xs[-1]], [ys[-1]], color="#c44e52", zorder=3, s=12)
    if curve_total:
        axc.set_xlim(0, max(1, curve_total - 1))
    axc.set_xlabel("frame")
    axc.set_ylabel("wirelength cost")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P")
