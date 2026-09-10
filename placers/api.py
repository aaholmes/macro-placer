"""In-memory entry point: place a problem given as plain arrays.

The core modules (fast_eval, analytical, legalizer, local_search) duck-type
the challenge repo's benchmark/plc objects. This module builds lightweight
stand-ins for those two objects from arrays, so integration code can call
the placer without the `macro_place` package or bookshelf files on disk.

Nothing in the core modules changes; the stand-ins expose exactly the
attributes and methods FastEval reads.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# Tuned configuration wo221 from the final protocol's pool (see notes/writeup).
DEFAULT_HP = dict(
    lam_wl0=0.577661641519922,
    lr=0.26629024654116507,
    tau_d=0.10785220922976244,
    lam_d_end=0.07795314107256225,
    lam_d_ratio=0.040916599877077596,
    ramp_p=3.7919570781529326,
    target=0.7067529176926369,
    gamma=9.680919613962839,
    ov_hold=0.0,
    ov_ramp=0.0,
)


@dataclass
class PlacementInput:
    """A placement problem as arrays. Lengths in one consistent unit (the
    caller's choice); the placer is unit-agnostic.

    nets: list of (weight, pins); pin = (macro_idx, off_x, off_y) with offsets
    relative to the macro CENTER, or (-1, abs_x, abs_y) for a fixed terminal.
    The first pin of each net is its source. Macros 0..num_hard-1 are hard
    (zero overlap); the rest are soft.
    """
    width: float
    height: float
    sizes: np.ndarray            # [N, 2] (w, h)
    num_hard: int
    nets: list
    grid_col: int = 128
    grid_row: int = 128
    # Routing-model attributes; only the congestion surrogate reads them.
    vroutes_per_micron: float = 10.0
    hroutes_per_micron: float = 10.0
    vrouting_alloc: float = 1.0
    hrouting_alloc: float = 1.0
    smooth_range: int = 2


@dataclass
class PlaceResult:
    positions: np.ndarray        # [N, 2] float64 macro centers
    num_hard: int
    seed: int
    meta: dict = field(default_factory=dict)


class _Mod:
    """Stand-in for one challenge module (macro, macro pin, or port)."""
    __slots__ = ("_type", "_pos", "_off", "_weight", "_ref", "_sink")

    def __init__(self, type_, pos=(0.0, 0.0), off=(0.0, 0.0), weight=1.0,
                 ref=-1, sink=None):
        self._type = type_
        self._pos = pos
        self._off = off
        self._weight = weight
        self._ref = ref
        self._sink = sink or {}

    def get_type(self):
        return self._type

    def get_pos(self):
        return self._pos

    def get_offset(self):
        return self._off

    def get_weight(self):
        return self._weight

    def get_sink(self):
        return self._sink


class _Plc:
    """Stand-in for the challenge plc: only what FastEval reads."""

    def __init__(self, p: PlacementInput):
        self.width = float(p.width)
        self.height = float(p.height)
        self.net_cnt = float(len(p.nets))
        self.grid_col = int(p.grid_col)
        self.grid_row = int(p.grid_row)
        self.vroutes_per_micron = p.vroutes_per_micron
        self.hroutes_per_micron = p.hroutes_per_micron
        self.vrouting_alloc = p.vrouting_alloc
        self.hrouting_alloc = p.hrouting_alloc
        self.smooth_range = p.smooth_range

        n = len(p.sizes)
        self.modules_w_pins = [_Mod("MACRO") for _ in range(n)]
        self.mod_name_to_indices = {}
        self.nets = {}
        for k, (weight, pins) in enumerate(p.nets):
            names = []
            for j, (owner, a, b) in enumerate(pins):
                name = f"n{k}p{j}"
                w = float(weight) if j == 0 else 1.0
                if owner < 0:
                    m = _Mod("PORT", pos=(float(a), float(b)), weight=w)
                else:
                    m = _Mod("MACRO_PIN", off=(float(a), float(b)), weight=w,
                             ref=int(owner))
                self.mod_name_to_indices[name] = len(self.modules_w_pins)
                self.modules_w_pins.append(m)
                names.append(name)
            self.modules_w_pins[self.mod_name_to_indices[names[0]]]._sink = \
                {"sinks": names[1:]}
            self.nets[names[0]] = names[1:]

    def get_ref_node_id(self, idx):
        return self.modules_w_pins[idx]._ref


class _Benchmark:
    def __init__(self, p: PlacementInput):
        import torch
        n = len(p.sizes)
        self.num_hard_macros = int(p.num_hard)
        self.num_macros = n
        self.hard_macro_indices = list(range(p.num_hard))
        self.soft_macro_indices = list(range(p.num_hard, n))
        self.macro_sizes = torch.tensor(np.asarray(p.sizes, np.float64),
                                        dtype=torch.float32)


def build_fast_eval(p: PlacementInput):
    from placers.fast_eval import FastEval
    return FastEval(_Benchmark(p), _Plc(p))


def input_from_challenge(b, plc) -> PlacementInput:
    """Convert loaded challenge objects to a PlacementInput (for parity tests
    and as the file-based path's wrapper)."""
    from placers.fast_eval import _pin_position

    plc_to_tensor = {pidx: t for t, pidx in enumerate(b.hard_macro_indices)}
    for k, pidx in enumerate(b.soft_macro_indices):
        plc_to_tensor[pidx] = b.num_hard_macros + k

    nets = []
    for driver_name in plc.nets.keys():
        driver_idx = plc.mod_name_to_indices[driver_name]
        weight = float(plc.modules_w_pins[driver_idx].get_weight())
        members = [driver_idx] + [plc.mod_name_to_indices[s]
                                  for s in plc.nets[driver_name]]
        pins = []
        for pidx in members:
            owner, a, bb = _pin_position(plc, pidx)
            pins.append((-1 if owner is None else plc_to_tensor[owner], a, bb))
        nets.append((weight, pins))

    sizes = b.macro_sizes.numpy().astype(np.float64)
    return PlacementInput(
        width=float(plc.width), height=float(plc.height), sizes=sizes,
        num_hard=int(b.num_hard_macros), nets=nets,
        grid_col=int(plc.grid_col), grid_row=int(plc.grid_row),
        vroutes_per_micron=float(plc.vroutes_per_micron),
        hroutes_per_micron=float(plc.hroutes_per_micron),
        vrouting_alloc=float(plc.vrouting_alloc),
        hrouting_alloc=float(plc.hrouting_alloc),
        smooth_range=int(plc.smooth_range),
    )


INITS = ("random", "quadratic_b2b", "quadratic_b2b_jitter", "corner_screen")

# Corner-assignment screen defaults (placers.corner_seeds.corner_screen).
DEFAULT_CORNER = dict(M=12, N=200, K1=16, K2=4)


def initial_positions(P, seed, init="random", jitter_sigma=0.25, x0=None):
    """Starting coordinates [N,2] float32 for the gradient stage.

    random: uniform in the core, seeded (the original start).
    quadratic_b2b: the bound-to-bound quadratic minimum (placers.quadratic);
        deterministic, so every seed starts from the same point.
    quadratic_b2b_jitter: the same plus seeded Gaussian noise with standard
        deviation jitter_sigma * median block size, so multi-start keeps its
        diversity while every start sits near the wirelength minimum.
    """
    import torch
    if init == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        c0 = torch.rand(P.b.num_macros, 2, generator=g)
        c0[:, 0] = c0[:, 0] * (P.W - 2) + 1
        c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
        return c0
    if init == "given":
        return torch.tensor(np.asarray(x0, np.float32)[:, :2])
    if init not in ("quadratic_b2b", "quadratic_b2b_jitter"):
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    from placers.quadratic import quadratic_place
    sig = jitter_sigma if init == "quadratic_b2b_jitter" else 0.0
    pos = quadratic_place(P.fe, jitter_sigma=sig, seed=seed, device=P.dev)
    return torch.tensor(pos, dtype=torch.float32)


def _diff_stage(P, seed, hp, iters, deadline, on_frame=None, frame_every=500,
                init="random", jitter_sigma=0.25, x0=None):
    """Gradient global placement (the diff_place core loop, no swap/diffusion).
    init="given" starts from x0 [N,2]."""
    import torch
    from placers.analytical import anneal, lapsum_topk_mean

    c0 = initial_positions(P, seed, init, jitter_sigma, x0)
    coord = c0.to(P.dev).requires_grad_(True)
    opt = torch.optim.Adam([coord], lr=hp["lr"])
    LD0 = hp["lam_d_end"] * hp["lam_d_ratio"]
    LD1 = hp["lam_d_end"]
    for it in range(iters):
        if deadline is not None and it % 100 == 0 and time.monotonic() > deadline:
            break
        if on_frame is not None and it % frame_every == 0:
            on_frame("diff", it, coord.detach().cpu().numpy())
        t = it / iters
        te = t ** hp["ramp_p"]
        gamma = anneal(te, P.gw * hp["gamma"], P.gw, geometric=True)
        lam_d = anneal(te, LD0, LD1)
        lam_wl = anneal(te, max(hp["lam_wl0"], 1e-9), 1.0, geometric=True)
        rho = P.density_field(coord)
        spread = torch.clamp(torch.clamp(rho, max=1.0) - hp["target"], min=0.0)
        overlap = torch.clamp(rho - 1.0, min=0.0)
        de = lapsum_topk_mean(spread + overlap, 0.10, hp["tau_d"])
        opt.zero_grad()
        (lam_wl * P.wirelength(coord, gamma) + lam_d * de).backward()
        opt.step()
        with torch.no_grad():
            coord[:, 0].clamp_(P.hw, P.W - P.hw)
            coord[:, 1].clamp_(P.hh, P.H - P.hh)
    return coord.detach().cpu().numpy().astype(np.float64)


def place(p: PlacementInput, budget_s: float, seed: int,
          iters: int | None = None, ls_iters: int | None = None,
          hp: dict | None = None, device=None,
          on_frame=None, frame_every: int = 500,
          init: str = "random", jitter_sigma: float = 0.25,
          corner: dict | None = None) -> PlaceResult:
    """Run the full pipeline: gradient placement -> legalize -> local search.

    init selects the gradient stage's starting positions (see
    initial_positions): "random" (default), "quadratic_b2b", or
    "quadratic_b2b_jitter" with noise jitter_sigma * median block size.
    init="corner_screen" instead runs the corner-assignment screen
    (placers.corner_seeds, parameters from `corner` over DEFAULT_CORNER):
    screen 2 already performs the gradient stage and legalization on each
    survivor, so the pipeline continues from the (seed mod K2)-th best of its
    K2 seeds. Screen time is charged to budget_s.

    budget_s splits ~40/60 between the gradient and local-search stages.
    Passing explicit iters/ls_iters disables the wall-clock cutoffs, which
    makes the result deterministic for a given seed and device.

    on_frame(stage, it, pos), if given, is called every frame_every gradient
    iterations and once after legalization and after local search; it is
    read-only and never changes the trajectory (placers.viz.TrajectoryRecorder
    is the intended consumer).
    """
    import torch
    from placers.analytical import DifferentiablePlacer
    from placers.legalizer import legalize
    from placers.local_search import optimize_fast

    t0 = time.monotonic()
    torch.manual_seed(seed)
    hp = {**DEFAULT_HP, **(hp or {})}
    fe = build_fast_eval(p)
    P = DifferentiablePlacer(fe, device=device)

    fixed_counts = iters is not None or ls_iters is not None
    diff_deadline = None if fixed_counts else t0 + 0.4 * budget_s
    corner_cfg = None
    if init == "corner_screen":
        from placers.corner_seeds import corner_screen
        corner_cfg = {**DEFAULT_CORNER, **(corner or {})}
        res = corner_screen(P, hp, iters or 5000, seed=seed, deadline=diff_deadline,
                            **corner_cfg)
        pos = res.seeds[seed % len(res.seeds)]
        if on_frame is not None:
            on_frame("diff", 0, pos)
        corner_cfg["t_screen1"] = res.t_screen1; corner_cfg["t_screen2"] = res.t_screen2
        corner_cfg["screen2"] = [dict(assignment=r["assignment"], hpwl=r["hpwl"],
                                      score=r["score"]) for r in res.screen2]
    elif init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    else:
        pos = _diff_stage(P, seed, hp, iters or 5000, diff_deadline,
                          on_frame=on_frame, frame_every=frame_every,
                          init=init, jitter_sigma=jitter_sigma)

    sizes = np.asarray(p.sizes, np.float64)
    pos, _, _ = legalize(pos, sizes, p.num_hard, p.width, p.height, seed=seed)
    if on_frame is not None:
        on_frame("legal", 0, pos)

    logf = lambda *a, **k: None
    if ls_iters is not None:
        pos, _ = optimize_fast(fe, pos, iters=ls_iters, seed=seed, logf=logf)
    else:
        remaining = max(1.0, budget_s - (time.monotonic() - t0))
        pos, _ = optimize_fast(fe, pos, iters=10 ** 9, seed=seed,
                               time_budget_s=remaining, logf=logf)
    pos, _, _ = legalize(pos, sizes, p.num_hard, p.width, p.height, seed=seed)
    if on_frame is not None:
        on_frame("final", 0, pos)

    return PlaceResult(positions=np.asarray(pos, np.float64),
                       num_hard=p.num_hard, seed=seed,
                       meta=dict(hp=hp, budget_s=budget_s, init=init,
                                 jitter_sigma=jitter_sigma, corner=corner_cfg,
                                 wall_s=time.monotonic() - t0))
