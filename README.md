# GPU-era Macro Placement — Solver & Analysis

A solver for the [Partcl/HRT Macro Placement Challenge](https://github.com/partcleda/macro-place-challenge-2026), plus the experiments and findings that shaped it. A **differentiable global placer** (any smooth loss) produces a good layout, a legalizer removes all overlap, and **greedy detailed placement** polishes against the exact metric. Built on a from-scratch evaluator that is **exact to the reference metric but ~50–3600× faster**, which turns the search into something you can actually iterate on.

**Result: average proxy 1.0155 across all 17 IBM benchmarks (zero overlaps) — 31% below the reference placement, and a new personal best.** Full write-up with plots: [`notes/solution.html`](notes/solution.html).

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

## Formulation

**The scored metric.** Wirelength is half-perimeter (HPWL) summed over nets; density and congestion are top-fraction means over an *m×n* grid:

```
WL_true = (1/Z) · Σ_e  w_e [ (max_{i∈e} x_i − min_{i∈e} x_i) + (max_{i∈e} y_i − min_{i∈e} y_i) ]
```

Congestion deposits, per net, one unit of routing demand along an **L-route skeleton** — a horizontal segment on the *source row* and a vertical segment on the *sink column* — into capacity-normalized H/V grids, box-blurs them, and reports `abu₅` = mean of the top 5% of cells. Density is the analogous `abu₁₀` over cell occupancy `ρ_b`. Both top-*k* means and the L-route routing are non-differentiable in the macro coordinates.

**The smooth surrogate (global stage).** `max`/`min` are replaced by log-sum-exp,

```
max_i x_i ≈  γ · log Σ_i e^{ x_i/γ},   min_i x_i ≈ −γ · log Σ_i e^{−x_i/γ}      (exact as γ→0)
```

density by a differentiable **overflow** penalty `Σ_b max(0, ρ_b − ρ*)²` over soft cell-assignments, and the objective follows a **back-loaded homotopy**: with normalized time `t∈[0,1]` and effective progress `τ(t)=t^p` (`p≈3.8`), `γ` and the term weights anneal from smooth/weak to sharp/strong on `τ` — start spread and forgiving, tighten late. Adam optimizes the free macro centers; `place_multistart` keeps the best of *N* random seeds. The loss is pluggable — any differentiable objective drops into the same engine (this is what let us test the congestion surrogates in the appendix).

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
| **average proxy (abs, over 17)** | **1.0155** |
| vs reference (1.4772) | **−31%** (mean per-benchmark ratio 0.692) |
| vs RePlAce (1.4578) | **−30%** |
| vs SA (2.1251) | **−52%** |
| vs leaderboard #1 (0.9507) | +6.8% |
| overlaps | 0 (all 17) |
| best single benchmark (ibm01) | **0.784** |

Every benchmark beats the reference; on the easiest designs (ibm01 0.78, ibm09 0.81) we dip below the current leaderboard leader's *average* (0.9507). The 14 held-out designs generalize as well as the 3 used for tuning. Greedy from the reference alone averages **1.083**; the differentiable global stage supplies a **6.2%** better basin on top of that. The score comes from a best-of-*N*-after-greedy pool over the top-5 tuned configs (each wins some benchmarks: trial 221 wins 6, 87 wins 5, 95 wins 3, 222 wins 2, 97 wins 1) plus the greedy-from-reference floor.

### Placement layouts

Hard macros as rectangles (color = area); soft cell-clusters as light dots. The reference scatters macros with dead space; ours packs them into a compact, connected arrangement at the same zero-overlap legality.

![ibm01 reference vs ours](notes/fig_layout_ibm01.png)

The differentiable optimization in motion — random spread → clustered under the back-loaded schedule → legalized → greedy-polished. Each replays the actual winning run (config 221, winning seed) and ends on the saved final placement. Left/top **ibm01** (small, proxy 0.784); right/bottom **ibm17** (large, congestion-bound, 1.205).

![ibm01 optimization](notes/opt_ibm01.gif)
![ibm17 optimization](notes/opt_ibm17.gif)

---

## Key findings (the interesting part)

These shaped the approach and are worth as much as the score:

1. **The proxy is soft-macro-dominated.** ~98% of our gain comes from co-optimizing soft macros; moving hard macros changes the proxy by only ~0.2%. Great for Tier-1 ranking, but a caveat for the real objective (at Tier-2, only hard-macro positions survive).
2. **Congestion is the wall.** At our best point it's 57% of the remaining cost and the term local search reduces least — because it depends on discrete *routing*, not a smooth function of positions.
3. **Congestion is best optimized implicitly, not as an explicit loss term.** The metric is non-differentiable, so we built two congestion surrogates — a RUDY penalty and a faithful, non-gameable **L-route** surrogate that correlates 0.94–0.99 with the true metric (appendix). *Neither improves the final score when added to the global loss.* Across four schedules — persistent, late, congestion-first, and a low-congestion warm-start — the paired effect on the post-greedy proxy is neutral-to-negative. Greedy detailed placement already minimizes the *true* congestion directly, and an explicit global term mostly trades away wirelength for an arrangement greedy would have reached anyway. Congestion falls out of good wirelength + spreading + exact-metric polish; forcing it earlier hurts.
4. **A differentiable global stage + greedy detail beats either alone.** Greedy from the reference averages 1.083; the differentiable global placer supplies a better starting basin, and global+greedy reaches **1.0155** (a 6.2% better basin than greedy-from-reference). The global stage optimizes a *surrogate* and is then legalized, so greedy — working on the *exact* metric — does most of the absolute-quality cleanup on top of the global output.
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

# 3. Best-of-N pipeline for the final score: generate candidates from the top-5
#    finalize-verified configs, greedy-polish every candidate, report per-benchmark best
uv run python scripts/fullrun_place.py  --topk 5 --deadline-min 300   # diff candidates (GPU)
uv run python scripts/fullrun_polish.py --time-cap-min 15             # polish each (CPU, parallel)
uv run python scripts/fullrun_report.py                               # per-benchmark best vs leaderboard
```
(Requires the challenge repo's `macro_place` package + benchmarks; scripts use its venv.)

---

## What would push toward the top of the leaderboard

We sit ~7% above the leaders (~0.95 avg), and the gap concentrates on the hard, congestion-bound designs (ibm08/17). Two intuitive levers we can now **rule out** (appendix): a better congestion term in the global loss, and a congestion-directed detailed placer — both are neutral-to-negative in paired tests, because the proxy is soft-cell-dominated and greedy already owns congestion.

What remains is structural. Single-cell greedy at zero temperature plateaus against congestion because relieving a hot region requires moving *groups* of cells coherently, not one point at a time — any single move that vacates a congested cell lengthens its own nets and is rejected. The lever is therefore a detailed placer that makes **coordinated, multi-cell moves** (or runs under a temperature that accepts transient wirelength increases), so it can restructure the global arrangement rather than locally polish it. That — not more hyperparameter tuning or more congestion modeling — is where the remaining gap lives.

---

## Appendix: what we tried that isn't in the final pipeline

The negative results constrain the design as much as the positive ones. Each was evaluated with **paired** experiments (identical seeds across arms, differencing per seed) to cancel the ~2–6% seed noise that otherwise swamps these effects.

**Congestion surrogates in the global loss.**
- *RUDY.* Routing demand modeled as a net's bounding-box footprint spread over its cells. Differentiable and cheap, but **gameable**: deposited demand scales as `1/bbox-area`, so the optimizer lowers the surrogate by *spreading* nets while the true L-route congestion rises. It diverges from the metric (correlation ≈ 0); abandoned.
- *L-route surrogate.* A faithful, non-gameable differentiable version of the true metric: a partition-of-unity soft row/column assignment (`softrow`) and a sigmoid-product segment occupancy (`softseg`) deposit demand on the L-route skeleton; the top-5% mean is a bisection **soft-top-k**. It correlates **0.94–0.99** with the true congestion. Added to the global loss under four schedules — **persistent**, **late** (ramped on `τ`), **congestion-first** (congestion at `t=0`, WL/density ramped in afterward), and as a **low-congestion warm-start** (a congestion+spread pre-phase before the standard schedule). Paired isolation tests on the hard designs came back **neutral-to-negative** on the post-greedy proxy every time (e.g. the low-congestion warm-start: +0.09 proxy, 0/4 seeds — the warm-started descent lands in a *worse* basin). Mechanism: greedy optimizes true congestion directly, so the global term adds no capability it lacks while corrupting the wirelength basin it starts from. A "scatter" (anti-clustered) warm-start leaned slightly positive but sat inside the n=4 noise band.

**Congestion-directed detailed placement.** A move operator that sends hot-cell macros toward cold grid cells sampled from the live congestion field — the discrete analogue of a congestion gradient — with acceptance still on the exact proxy. Paired against baseline greedy it was **negative**: teleport-to-cold moves lengthen the moved cell's own nets and get rejected, so they waste the fixed move budget; zero-temperature single-cell descent cannot express the coordinated moves congestion relief actually needs (this is what motivates the "coordinated moves" conclusion above).

**Concurrent / interleaved global+detailed.** Alternating chunks of Adam steps and greedy moves — every confound-controlled variant, including SA-style acceptance and matched compute — never beats the phase-separated global-then-greedy pipeline. The two stages are genuinely separated: the global stage sets the coarse arrangement, greedy does the exact-metric detail.

**Other.** SA and SA/greedy hybrids (greedy-until-*N*-accepts) — no gain over pure greedy at matched compute. Electrostatic-FFT density mode — equivalent to the overflow penalty, not better. Multi-start initialization variants (density-only, low-congestion, scatter) — see the warm-start row above.
