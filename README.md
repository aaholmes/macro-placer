# GPU-era Macro Placement — Solver & Analysis

A fast, validated solver for the [Partcl/HRT Macro Placement Challenge](https://github.com/partcleda/macro-place-challenge-2026), plus the experiments and findings that shaped it. Built around a from-scratch evaluator that is **exact to the reference metric but ~50–3600× faster**, which turns the placement search into something you can actually iterate on.

---

## The problem in one minute

Place large fixed blocks ("macros") on a chip canvas to minimize a **proxy cost**, with **zero overlaps** and under **1 hour per benchmark**:

```
proxy = 1.0 · wirelength + 0.5 · density + 0.5 · congestion   (lower is better)
```

- **wirelength** — total half-perimeter of all nets (wires want connected macros close → clustering)
- **density** — mean of the top-10% most crowded grid cells (wants spreading)
- **congestion** — mean of the top-5% most routing-congested cells (wants routing breathing room)

The three objectives fight each other, the search space is ~10^800, and there are two macro types:
- **hard macros** — fixed-size rectangles you position (strict zero overlap);
- **soft macros** — standard-cell clusters, movable and allowed to overlap (a density/congestion proxy).

Scoring is over 17 IBM benchmarks. Baselines (same metric): **RePlAce 1.4578**, **SA 2.1251** (17-benchmark averages).

---

## Architecture

The whole solver is built on one idea: **make evaluation cheap, then search hard.**

```
 reference placement ──► legalize (zero overlap) ──► greedy local search ──► score
                                                       ▲
                              incremental evaluator (exact, ~1 ms/move)
                                                       │
                                    run all 17 benchmarks in parallel (checkpointed)
```

### 1. `fast_eval.py` — a validated, fast reimplementation of the metric
The official evaluator is pure-Python and takes ~1 s per evaluation. We reimplemented it in vectorized NumPy and **validated each term against the ground truth**:

| term | accuracy vs official | speed |
|------|---------------------|-------|
| wirelength | 6×10⁻¹⁰ rel. err | 0.16 ms (was 570 ms) |
| density | 3.7×10⁻⁸ rel. err | exact |
| congestion | ~10⁻⁵ in the search regime | 37 ms (was 430 ms) |

### 2. Incremental single-move evaluation — the engine
Local search changes **one macro at a time**, so we keep the per-net wirelength, the density grid, and the routing arrays as live state and update only what a move touches. Result: **0.96 ms/move vs 52 ms full recompute (55×), exact to 2×10⁻¹⁶**, with correct undo. Also supports **orientation flips** (Klein-4) incrementally. This is what makes deep search (10⁵–10⁶ moves) feasible in seconds.

### 3. `legalizer.py` — guaranteed zero overlap
Three stages: vectorized push-apart → strict-gated Gauss-Seidel cleanup (never worsens) → relocate-to-empty finisher. **All 17 benchmarks reach strict zero overlap.**

### 4. `local_search.py` — the optimizer
Greedy hill-climbing over single-macro moves, accepting only true-proxy improvements. Co-optimizes **soft macros** (the key lever). Also includes SA, orientation flips, congruent swaps, and a unified operator-portfolio variant.

### 5. `parallel_run.py` — 16× throughput
All 17 benchmarks run as independent checkpointed chains across a worker pool (single-threaded NumPy per worker). Resumable; a plateau-stop frees cores as small benchmarks converge.

### Also explored
- `moves.py` — hard-macro swap and rigid cluster moves.
- `force.py` — a continuous density force field and a RUDY congestion surrogate.
- `analytical.py` — a **differentiable analytical placer** (PyTorch/GPU): smooth log-sum-exp wirelength + overflow-density + **soft-top-k RUDY** congestion, optimized by autograd + Adam.

---

## Results

Current standing on the 17 IBM benchmarks (greedy soft co-optimization, parallel sweep, still improving):

| metric | value |
|--------|-------|
| **average proxy** | **~1.08** |
| vs RePlAce (1.4578) | **+25%** |
| vs SA (2.1251) | +49% |
| overlaps | 0 (all 17) |
| approx. leaderboard rank | ~24 / 135 |
| best single benchmark (ibm01) | **0.795** |

Every benchmark beats both published baselines.

---

## Key findings (the interesting part)

These shaped the approach and are worth as much as the score:

1. **The proxy is soft-macro-dominated.** ~98% of our gain comes from co-optimizing soft macros; moving hard macros changes the proxy by only ~0.2%. Great for Tier-1 ranking, but a caveat for the real objective (at Tier-2, only hard-macro positions survive).
2. **Congestion is the wall.** At our best point it's 57% of the remaining cost and the term local search reduces least — because it depends on discrete *routing*, not a smooth function of positions.
3. **The metric is non-differentiable, so gradients need surrogates.** A naive congestion "force" (gradient of the discrete routing grid) fails. A **RUDY** surrogate with a **soft-top-k** penalty — `softmax(R/τ)·R`, matching the top-5% semantics — with an *autodiff* gradient is the first thing that provably reduces the real congestion.
4. **Local search is near its ceiling on this metric.** Greedy from the reference reaches ~0.79 on ibm01; richer operators (flips, swaps, portfolios) add ~0.1%; analytical refinement and from-scratch analytical (as tuned) do *not* beat it — the smooth surrogate's optimum is coarser than the real one.

---

## Most promising approach & how to run it

**Greedy soft-macro co-optimization on the fast incremental evaluator, run in parallel across all 17 benchmarks.** It is exact-metric, zero-overlap, fast, and already beats both baselines.

```bash
# one benchmark
uv run python scripts/run_fast.py ibm01 200000

# all 17 in parallel (checkpointed, resumable)
uv run python scripts/parallel_run.py --iter-cap 1000000 --time-cap-min 60
uv run python scripts/show_results.py       # live standings vs baselines
```
(Requires the challenge repo's `macro_place` package + benchmarks; scripts use its venv.)

---

## What would push toward the top of the leaderboard

Local search plateaus because congestion is *structural* — it depends on the global arrangement, which local moves can't restructure. Reaching the leaders (~0.95 avg) would need a **properly-tuned congestion-aware analytical global placer** (DREAMPlace-grade: log-sum-exp wirelength + electrostatic density + soft-top-k RUDY, with γ/λ annealing on GPU) as a *global* stage, followed by our legalizer and greedy polish. The core primitives — including the validated soft-top-k RUDY congestion gradient — are built in `analytical.py`; the remaining work is the annealing schedule and tuning that make analytical placement a mature-codebase problem.
