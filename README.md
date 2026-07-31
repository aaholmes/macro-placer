# GPU-era Macro Placement — Solver & Analysis

**Macro placement** is an early step in chip design: positioning the large fixed blocks — memories and other prebuilt components — on the die before the millions of small standard cells are placed around them. Where those blocks go largely determines wirelength, routability, and congestion for everything that follows.

This is a solver for the [Partcl/HRT Macro Placement Challenge](https://github.com/partcleda/macro-place-challenge-2026), which ranks placers on 17 IBM/ICCAD04 benchmarks by a single cost (wirelength + density + congestion) with zero block overlap, one hour per benchmark. A **differentiable global placer** produces a layout from a random start, a legalizer removes all overlap, and **annealed local search with computed proposals** polishes against the exact metric — orchestrated by a one-hour schedule that spends most of its time deeply polishing the best placement. All of it is built on a from-scratch evaluator that is **exact to the official metric but ~50–3600× faster**, which turns the search into something you can iterate on.

**Result: average score 0.9758 across all 17 benchmarks — 34% below the provided reference placements — with zero overlaps, verified on the official scorer, under a hard 60-minute wall clock per benchmark on hardware slower than the challenge allows (so the budget claim is conservative).** An earlier version of this pipeline reached 1.0155; this one improves on it by 3.9%. Full write-up with plots and animations: [`notes/solution.html`](notes/solution.html).

---

## The problem in one minute

Place blocks on a canvas to minimize

```
score = 1.0 · wirelength + 0.5 · density + 0.5 · congestion   (lower is better)
```

- **wirelength** — total bounding-box perimeter of all nets (pulls connected blocks together)
- **density** — mean occupancy of the top-10% most crowded grid cells (pushes blocks apart)
- **congestion** — mean routing demand of the top-5% most congested cells (wants routing room)

Two block types: **hard macros** (fixed rectangles, strict zero overlap) and **soft macros** (standard-cell clusters, movable, may overlap). The objectives fight each other and none of them is differentiable as officially defined: wirelength uses hard min/max, density and congestion are top-fraction means, and congestion routes each net along a discrete L-shaped path. Baselines on the same metric: reference placements 1.477 avg, RePlAce 1.458, simulated annealing 2.125.

---

## Architecture

Two ideas: **make evaluation cheap so you can search hard**, and **combine a smooth global stage with an exact-metric local stage**.

```
                     ┌─ differentiable global placer (GPU, any smooth loss) ─┐
   random inits ──►  │  smoothed wirelength + density overflow, scheduled     │
   + ref warm-start  └───────────────────────────────────────────────────────┘
                                          │
                     legalize (strict zero overlap)
                                          │
                     annealed local search, computed proposals ──► score
                          ▲
        incremental evaluator (exact, ~1 ms/move)

   the hour, per benchmark: generate + screen ~8 placements → deep-polish top 2
   → parallel polish rounds (12 annealed copies, keep best) until the wall
```

- **`fast_eval.py`** — the official scorer reimplemented in vectorized NumPy and validated term-by-term against it (wirelength to 10⁻¹⁰, congestion to ~10⁻⁵). Then made **incremental**: moving one block updates only the wires, grid cells, and routes it touches — ~1 ms per move instead of ~1 s, exact, with undo. This is what makes million-move searches feasible.
- **`analytical.py`** — the global placer. Every block center is a free variable; gradient descent (GPU) minimizes a smooth stand-in for the score: softened wirelength plus a grid-overflow density penalty. Weights follow a schedule — loose early so connected blocks pull together, tightened late — which avoids the traps a full-strength objective falls into from step one. The loss is pluggable: any differentiable objective drops in unchanged.
- **`legalizer.py`** — pushes overlapping blocks apart along the axis of least penetration, then a cleanup pass; strict zero overlap on all 17 benchmarks.
- **`local_search.py`** — the exact-metric polish. Single-block moves under a small annealing temperature, with **computed proposals** instead of random ones: 25% place a cell at its mathematically optimal spot given its wires (sampled inside the interval where wirelength provably doesn't change, so density and congestion improve for free); 20% are **route-shifts** — nudge a net's endpoint one grid cell so its whole route leaves a congested row or column; the rest are small random steps.
- **`final_hour.py`** — the protocol: one process per benchmark under a hard 60-minute wall (loading included). Generate and screen ~8 global placements (the tuned configuration, variants, and one warm-started from the reference), deep-polish the best two, then run 12 parallel annealed polishing copies of the current best, keep the winner, and repeat until time runs out.
- **`bayes_search.py`, `bayes_swap.py`, `bayes_greedy_knobs.py`** — Bayesian tuning of all of the above on 3 designs (the other 14 held out), always comparing configurations on identical random seeds.

---

## Results

| metric | value |
|--------|-------|
| **average score (17 benchmarks)** | **0.9758** |
| vs reference placements (1.477) | **−34%** |
| vs RePlAce (1.458) | −33% |
| vs simulated annealing (2.125) | −54% |
| vs leaderboard #1 (0.9507) | +2.6% |
| vs the previous version of this pipeline (1.0155) | −3.9% |
| overlaps | 0 on all 17 (official scorer) |
| best single benchmark (ibm09) | 0.768 |

Every benchmark beats its reference; nine of seventeen individually score below the leaderboard leader's *average*. The configuration was tuned on 3 designs and transfers to the 14 held-out ones with only a mild (~4-point) penalty. The hour's allocation matters more than the placement count: after ~8 screened candidates, everything goes into 8–23 rounds of parallel annealed polish of the single best placement.

### The optimization in motion

Three panels per benchmark: the reference placement, my placer running through one complete cycle — spread, legalize, polish — and the true score per frame diving under the reference line. Spanning the design space: ibm01 (small), ibm09 (medium), ibm17 (large, congestion-heavy), ibm18 (large).

![ibm01 optimization](notes/opt_ibm01.gif)
![ibm09 optimization](notes/opt_ibm09.gif)
![ibm17 optimization](notes/opt_ibm17.gif)
![ibm18 optimization](notes/opt_ibm18.gif)

---

## Key findings

1. **The score is dominated by the movable cell clusters.** Co-optimizing them drives ~98% of the gain; moving the big fixed blocks changes the score only ~0.2%.
2. **Congestion was the wall — route-shifts and annealing cracked it.** The congestion in a hot cell mostly comes from wires passing *through* it, so evicting the resident block does nothing. Relocating a passing wire — a one-cell endpoint nudge moves the whole route — under a temperature that lets the disruption survive until nearby cells re-settle, produced 87% of the final improvement.
3. **Congestion belongs in the local stage, not the global loss.** Even a differentiable congestion model that tracks the true metric closely (correlation 0.94–0.99) doesn't help as a global-stage term: the local search already optimizes true congestion, and the global term only trades away wirelength.
4. **Smooth global + exact local beats either alone.** The global stage supplies a ~6% better starting arrangement than polishing the reference; the local stage, working on the exact score, does most of the absolute-quality work.
5. **Compare on identical seeds; spend the budget on depth.** Run-to-run noise (~2%) exceeds the gap between good configurations, so single-run rankings measure luck. And collecting placements saturates after a few dozen — deep repeated polish of the single best placement is where the final margin came from.
6. **An apparent ceiling was a budget artifact, and a "failed" method revived.** Scores stalled near 1.015 until it turned out the local search simply hadn't converged — deeper passes kept paying. Likewise annealing lost to plain greedy while its proposals were random, and became the engine of the method once proposals were computed. Negative results are only binding while their conditions hold.

---

## How to run it

```bash
# 1. Tune (Bayesian optimization on 3 designs; compares configs on identical seeds)
uv run python scripts/bayes_search.py && uv run python scripts/bayes_swap.py

# 2. Validate the frozen configuration on all 17 (held-out check)
uv run python scripts/validate_all.py --seeds 8

# 3. The final protocol: one hard-walled hour per benchmark
for nm in ibm01 ibm02 ... ibm18; do BUDGET_S=3600 uv run python scripts/final_hour.py $nm; done
#    -> notes/final_hour/{nm}.json (avg of "best" = the headline score)
```
(Requires the challenge repo's `macro_place` package and benchmarks.)

---

## What remains

The ~2.6% gap to the leaderboard leaders sits on the most congestion-heavy designs (ibm08/17/18), where the local search has converged — fresh polish rounds now return under 0.05%. Closing it would need a qualitatively different global arrangement on those designs; how to find one is an open question. Things I can rule out from paired experiments (identical seeds, both arms): congestion terms in the global loss, evicting blocks from hot cells, interleaving the global and local stages, and a family of reordering mechanisms in the global stage — each improved the average run but never the best-of-many, because they shrink exactly the run-to-run spread that picking-the-best feeds on. Details and two more pages of negative results: [`notes/solution.html`](notes/solution.html).
