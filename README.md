# GPU-era Macro Placement — Solver & Analysis

A solver for the [Partcl/HRT Macro Placement Challenge](https://github.com/partcleda/macro-place-challenge-2026), plus the experiments and findings that shaped it. A **differentiable global placer** (any smooth loss) produces a good layout, a legalizer removes all overlap, and **greedy detailed placement** polishes against the exact metric. Built on a from-scratch evaluator that is **exact to the reference metric but ~50–3600× faster**, which turns the search into something you can actually iterate on.

**Result: average proxy 1.038 across all 17 IBM benchmarks (zero overlaps) — 29% better than the reference placement, and a new personal best.** Full write-up with plots: [`notes/solution.html`](notes/solution.html).

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

Two ideas: **make evaluation cheap so you can search hard**, and **combine a differentiable global stage with an exact-metric detailed stage.**

```
                     ┌─ differentiable global placer (GPU, any smooth loss) ─┐
   random inits ──►  │  log-sum-exp WL + overflow density, back-loaded ramp   │
   (best of N)       └───────────────────────────────────────────────────────┘
                                          │
                     legalize (strict zero overlap)
                                          │
                     greedy detailed placement ──► score
                          ▲
        incremental evaluator (exact, ~1 ms/move), all 17 benchmarks in parallel

   hyperparameters tuned once by Bayesian search (TPE) on 3 designs;
   winner picked under best-of-16 multi-start; applied unchanged to all 17.
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

### 4. `analytical.py` — the differentiable global placer (GPU)
Macro centers are free coordinates optimized by Adam against a **smooth surrogate** of the proxy: per-net **log-sum-exp** wirelength and an **overflow density** penalty (plus optional soft-top-k RUDY congestion and an electrostatic-FFT density mode). A **back-loaded homotopy schedule** — start spread and weakly constrained, ramp the weights up sharply late — avoids the local minima a fully-on objective falls into. The loss is pluggable, so the same engine optimizes any differentiable objective. `place_multistart` keeps the best of N random inits.

### 5. `local_search.py` — exact-metric detailed placement
Greedy hill-climbing over single-macro moves, accepting only true-proxy improvements. Warm-started from the global placement, it closes the gap between the smooth surrogate (plus legalization) and the real metric. Co-optimizes **soft macros** (the key lever) and preserves hard-macro legality. Also includes SA, orientation flips, congruent swaps, and an operator-portfolio variant.

### 6. Tuning & validation pipeline
- `bayes_search.py` — Bayesian (TPE) search over the loss weights and schedule on 3 representative designs, minimizing mean ratio-to-reference; a `--finalize` stage re-scores the top configs under **best-of-16 multi-start** to pick the winner above the seed-noise floor.
- `validate_all.py` — applies the frozen winner to **all 17** benchmarks (held-out generalization check).
- `greedy_polish.py` + `final_report.py` — detailed-placement polish and the three-way comparison (global / global+greedy / greedy-from-reference), reporting the per-benchmark best.
- `parallel_run.py` — 16× throughput: all 17 as independent checkpointed chains across a worker pool.

### Also explored
- `moves.py` — hard-macro swap and rigid cluster moves.
- `force.py` — a continuous density force field and a RUDY congestion surrogate.

---

## Results

Differentiable global placement → legalize → greedy polish, applied to all 17 IBM benchmarks (per-benchmark best over the pipeline's methods):

| metric | value |
|--------|-------|
| **average proxy (abs, over 17)** | **1.038** |
| vs reference placement (1.477) | **−29%** (mean ratio 0.706) |
| vs RePlAce (1.4578) | **+29%** |
| vs SA (2.1251) | +51% |
| overlaps | 0 (all 17) |
| approx. leaderboard rank | ~14 / 129 |
| best single benchmark (ibm01) | **0.794** |

Every benchmark beats the reference; on the easiest designs (ibm01 0.79, ibm09 0.81) we dip below the current leaderboard leader's *average* (0.9507). The 14 held-out designs generalize as well as the 3 used for tuning. Previous approach (greedy from the reference alone) averaged ~1.08; the differentiable global stage supplies a ~4% better basin on top of that.

---

## Key findings (the interesting part)

These shaped the approach and are worth as much as the score:

1. **The proxy is soft-macro-dominated.** ~98% of our gain comes from co-optimizing soft macros; moving hard macros changes the proxy by only ~0.2%. Great for Tier-1 ranking, but a caveat for the real objective (at Tier-2, only hard-macro positions survive).
2. **Congestion is the wall.** At our best point it's 57% of the remaining cost and the term local search reduces least — because it depends on discrete *routing*, not a smooth function of positions.
3. **The metric is non-differentiable, so gradients need surrogates.** A naive congestion "force" (gradient of the discrete routing grid) fails. A **RUDY** surrogate with a **soft-top-k** penalty — `softmax(R/τ)·R`, matching the top-5% semantics — with an *autodiff* gradient is the first thing that provably reduces the real congestion.
4. **A differentiable global stage + greedy detail beats either alone.** Greedy from the reference averages ~1.08; the differentiable global placer supplies a better starting basin, and global+greedy reaches 1.038 (global wins 14/17 designs). The global stage optimizes a *surrogate* and is then legalized, so greedy — working on the *exact* metric — does most of the absolute-quality cleanup (~23% on top of the global output).
5. **Multi-start matters more than tuning precision.** Single-seed scores carry ~2% noise, larger than the gap between good configs, so single-seed rankings are unreliable — the search's apparent-best config was a lucky seed. Re-scoring the top configs at best-of-16 reshuffled the order (a config ranked 5th won) and lowered the score ~6%. The final pick is always made under multi-seed evaluation.
6. **The tuned config generalizes.** Bayesian search on 3 designs, applied unchanged to the other 14, does equally well on the held-out set — no per-benchmark overfitting.

---

## How to run it

The full pipeline: tune once, finalize the winner under multi-start, validate on all 17, then greedy-polish.

```bash
# 1. Bayesian search for the loss weights/schedule (3 designs), then finalize
uv run python scripts/bayes_search.py                 # search
uv run python scripts/bayes_search.py --finalize      # best-of-16 pick over the top configs

# 2. Apply the frozen winner to all 17 (held-out validation)
uv run python scripts/validate_all.py --seeds 8

# 3. Greedy detailed-placement polish + three-way report
uv run python scripts/greedy_polish.py
uv run python scripts/final_report.py                 # per-benchmark best vs leaderboard
```
(Requires the challenge repo's `macro_place` package + benchmarks; scripts use its venv.)

---

## What would push toward the top of the leaderboard

We are ~9% above the leaders (~0.95 avg). The gap is on the **hard, congestion-bound designs** (ibm08/17/18), where the global arrangement — not local moves — sets the floor. Closing it would need a stronger *detailed*-placement stage (congestion is structural and our greedy soft-only moves plateau against it) or a better congestion model in the global loss (the RUDY surrogate is gameable and is currently off). The primitives are in `analytical.py`; the remaining work is a detailed placer that can restructure the global layout, not more hyperparameter tuning.
