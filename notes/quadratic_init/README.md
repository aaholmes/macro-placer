# Quadratic (B2B) initial placement: measurements

What was built and what it measured. Design: `QUADRATIC_INIT_DESIGN_DOC.md`.
Reproduce with `scripts/quadratic_init_experiment.py`; analyze with
`scripts/quadratic_init_report.py`.

## Setup

Gradient stage: the contest-protocol one (`scripts/bayes_swap.py:diff_place`,
configuration `t94_win+swap`, 15000 iterations), which is what the final-hour
protocol runs. Each run is scored raw, after push-apart legalization, and after
a 60000-move local-search screen (the final-hour phase-A screen) using the
exact contest score `wirelength + 0.5 density + 0.5 congestion`.

Four benchmarks, four seeds per arm, seeds paired across arms. Note the
pairing is weak for the quadratic arms: the B2B solution is deterministic, so a
seed only changes the diffusion and swap noise inside the gradient stage, not
the starting point. Arms: `random` (control), `quadratic_b2b`,
`quadratic_b2b_jitter` at sigma 0.1 and 0.25 of the median block size.

Screen budget is short. These numbers do not settle what happens at the full
one-hour budget, where the local search has 20 to 60 times more moves.

## Result: no gain that survives the local search

Mean paired difference from the control, 95% bootstrap interval over seeds,
negative favors the arm. Score after the 60000-move screen.

| benchmark | control | quadratic_b2b | jitter 0.1 | jitter 0.25 |
|---|---|---|---|---|
| ibm01 | 0.8106 | +0.0044 [-0.0048, +0.0135] | +0.0044 [-0.0041, +0.0130] | +0.0021 [-0.0042, +0.0074] |
| ibm09 | 0.8128 | -0.0123 [-0.0340, +0.0018] | -0.0165 [-0.0404, -0.0017] | -0.0158 [-0.0469, +0.0024] |
| ibm17 | 1.2893 | +0.0021 [-0.0653, +0.0392] | +0.0321 [-0.0180, +0.0822] | +0.0028 [-0.0541, +0.0598] |
| ibm18 | 1.2275 | -0.0043 [-0.0438, +0.0248] | -0.0126 [-0.0268, -0.0010] | -0.0003 [-0.0368, +0.0362] |

Pooled over the four benchmarks (mean of per-benchmark mean differences):

| stage | quadratic_b2b | jitter 0.1 | jitter 0.25 |
|---|---|---|---|
| after legalization | -0.0197 [-0.0361, -0.0033] | -0.0126 [-0.0480, +0.0241] | -0.0095 [-0.0232, -0.0004] |
| after 60k screen | -0.0025 [-0.0087, +0.0032] | +0.0019 [-0.0146, +0.0209] | -0.0028 [-0.0113, +0.0025] |

Only ibm09 shows an effect that holds up: about 3.4% after legalization and
1.5 to 2% after the screen, in the arm's favor. Elsewhere the intervals
straddle zero at both stages, and on ibm01 and ibm17 the point estimates favor
the control. Pooled after the screen, every arm is within +/-0.003 of the
control on scores near 0.8 to 1.3.

## The predictions, and what happened

Predictions were recorded in the design doc before running.

- **"5 to 15% better score after the analytical stage, seed variance down by
  more than half."** Not observed. After legalization the pooled gain is 1.2 to
  2.0%, and per benchmark only ibm09 clears its interval. Seed spread is not
  systematically lower: on ibm01 the control's spread after legalization is
  0.0057 and quadratic_b2b's is 0.0118.
- **"Final score improves 0 to 3%, largest on ibm17."** The 0% end is right,
  and ibm17 is the benchmark where the arms look worst rather than best.
- **"Wall time of the analytical stage drops by at least a third."** No. It is
  85 to 87 seconds in every arm, because the schedule is a fixed iteration
  count with no early exit. The design doc anticipated an early exit rule that
  this pipeline does not have. The B2B solve itself is cheap: 0.16 to 0.38
  seconds.
- **"Small sigma matches, large sigma approaches the control."** No ordering in
  sigma is visible above the noise.

## Why the cell-placer result did not transfer: wirelength is 8% of the score

The quadratic solve minimizes half-perimeter wirelength (HPWL) exactly. HPWL is
a small part of what is being scored. Decomposing the contest score on the
final-hour placements (`notes/final_hour/{nm}_best.npz`):

| benchmark | HPWL | 0.5 x density | 0.5 x congestion | HPWL share |
|---|---|---|---|---|
| ibm01 | 0.0810 | 0.2550 | 0.4504 | 10.3% |
| ibm09 | 0.0659 | 0.2518 | 0.4506 | 8.6% |
| ibm17 | 0.0727 | 0.2655 | 0.8333 | 6.2% |
| ibm18 | 0.0798 | 0.2867 | 0.7453 | 7.2% |

Mean HPWL share 8.1%; congestion carries 57 to 71%. A start that is optimal in
HPWL is optimal in 8% of the objective and says nothing about the other 92%.
That is sufficient to explain a null result of the size measured here, and it
would have predicted the null before the experiment ran rather than after.

This also bounds what any wirelength-only initialization can be worth,
including the corner screen, whose screen-1 ranking is by quadratic HPWL. It is
consistent with the measured screen-1 rank correlation of -0.01: ranking
candidates by 8% of the objective does not order them by the whole of it.

An earlier draft of this note blamed the gradient stage's tuned overlap window
instead, arguing that it re-sorts macro ordering and so does the work a good
start would have done. That was speculation and it was not tested; the
decomposition above is measured and needs no such mechanism. The overlap-window
account is not ruled out and would be a second-order effect at most.

### What the congestion term actually models

Not Steiner minimal trees, and not HPWL. Reading the official evaluator
(`plc_client_os.py`, `get_routing`): 2-pin nets are L-routed, horizontally
along the source row and vertically along the sink column; 3-pin nets have
their own case; nets above 3 pins are split into a star of 2-pin L-routes from
the source pin. Demand accumulates per grid cell, is normalized by that cell's
routing capacity, is smoothed over `smooth_range` cells, and the cost comes
from the peak cells rather than the total.

So the dominant term is path-based, spatially resolved, and peak-driven.
Wirelength-optimal placement does not target any of those three properties.
Anything aimed at closing the remaining gap should be built for this term, not
for HPWL.

## Corner-assignment screen

Two separable results: the ranking mechanism does nothing, and the arm still
wins, which means the win comes from somewhere other than the ranking.

### Screen 1 does not predict screen 2

Spearman correlation between screen-1 rank (quadratic HPWL) and screen-2 rank
(exact score after gradient stage and legalization) among the 16 survivors:

| benchmark | per-seed values | mean |
|---|---|---|
| ibm01 | +0.22, +0.51, +0.22, -0.15 | +0.20 |
| ibm09 | -0.25, -0.39, -0.01, -0.10 | -0.19 |
| ibm17 | -0.10, -0.10, +0.07, -0.24 | -0.09 |
| ibm18 | -0.45, +0.04, +0.35, +0.16 | +0.03 |

Pooled over all 16 runs: -0.01, 95% bootstrap interval [-0.14, +0.11]. The
design doc set +0.50 as the threshold below which screen 1 is uninformative.
The measured value is indistinguishable from zero, and the interval excludes
+0.50 comfortably. Ordering assignments by quadratic wirelength carries no
information about which will score well. The doc's stated remedy, growing K1
toward N, rescues a weak signal but not an absent one; it would just convert
the method into brute force over assignments.

### At matched compute the arm is indistinguishable from plain multi-start

One corner run costs 16 gradient stages (screen 2 over K1=16 survivors). The
cost-matched control is therefore the best of 16 random seeds, measured
separately in `random16.json`. Score after the 60000-move screen:

| benchmark | corner, 4 runs | corner mean | random best-of-16 | difference |
|---|---|---|---|---|
| ibm01 | 0.8054, 0.8031, 0.8041, 0.8058 | 0.8046 | 0.8030 | +0.0016 |
| ibm09 | 0.7896, 0.7884, 0.7900, 0.7830 | 0.7878 | 0.7860 | +0.0017 |
| ibm17 | 1.2383, 1.2245, 1.2314, 1.2130 | 1.2268 | 1.2339 | -0.0071 |
| ibm18 | 1.1680, 1.1910, 1.1837, 1.1751 | 1.1794 | 1.1852 | -0.0058 |

Pooled over the 16 corner runs: -0.0024, 95% bootstrap interval [-0.0064,
+0.0011]. The interval includes zero. Wall time is comparable and slightly
favors the corner arm on the larger benchmarks: on ibm17, 1520 seconds versus
2187 seconds for 16 random seeds.

Earlier drafts of this note compared the corner arm's best over four
independent runs (64 gradient stages) with the best of four random seeds (4
stages), a 16-fold compute mismatch, and reported a 1.3% win. That comparison
was wrong. At matched compute the effect disappears.

The sign splits by problem size: the corner arm loses on the two smaller
benchmarks and wins on the two larger ones, where the local-search budget is
tightest relative to the number of macros. With four benchmarks this is a
pattern worth testing, not a result. Testing it means more large benchmarks,
and separating the pinning from the ranking would mean running the same pinning
with assignments drawn at random rather than ranked, since the ranking itself
is measurably uninformative.

### Cost model corrections

One B2B solve with four macros fixed takes 33 milliseconds on ibm01 and 171
milliseconds on ibm17, on CPU, not the milliseconds the design doc assumed. On
a GPU the same solve takes 0.4 to 1.2 seconds, because at a few thousand
unknowns the conjugate-gradient iterations are launch bound, so screen 1 runs
on CPU by default. Screen 1 over 200 assignments costs 5 to 40 seconds, which
is small. Screen 2 is the entire cost.

## Seed spread: real, but unhelpful under this protocol

The corner arm does tighten the seed distribution. Standard deviation of the
screen score, corner (n=4 runs) versus random (n=16), with a bootstrap interval
on the ratio:

| benchmark | sd random | sd corner | ratio | 95% CI |
|---|---|---|---|---|
| ibm01 | 0.0135 | 0.0012 | 11.0x | [5.7x, 64.4x] |
| ibm09 | 0.0092 | 0.0032 | 2.9x | [1.4x, 46.8x] |
| ibm17 | 0.0327 | 0.0108 | 3.0x | [2.0x, 10.7x] |
| ibm18 | 0.0140 | 0.0100 | 1.4x | [0.7x, 4.5x] |

Three of four exclude 1, so the reduction is real, but the magnitude is poorly
determined off 4 corner runs and the headline 11x is the most favorable of the
four benchmarks. The mean run improves 2.24% while the best of many improves
0.88% with an interval including zero.

Under this protocol that is not a benefit. The hour selects the best of many
candidates, and best-of-many is fed by the upside tail, so compressing the
distribution removes the tail it feeds on. This is the mechanism already
recorded in the write-up for several earlier methods that improved the average
run without improving the best. Low variance would be worth having only if a
single unselected run had to be submitted, which is not the case here.

## Status

`random` remains the default. Nothing measured here justifies changing it: at
matched compute every arm's interval includes zero.

Nor would it change the standing if it held. The average over 17 benchmarks is
0.9758 and the leaderboard leader is 0.9507. A uniform 0.88% gain gives 0.9672,
and even applying ibm17's 1.69%, the best single-benchmark figure, gives
0.9593. Both remain behind.

The `init` option is in place so the question can be re-opened at the full
one-hour budget or on larger benchmarks. The more promising direction is not a
better wirelength start at all: the score is 8% HPWL and roughly 60 to 70% peak
routing congestion, so a method aimed at the congestion term is where the
remaining gap lives.
