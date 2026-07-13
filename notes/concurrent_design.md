# Concurrent diff+greedy — design note

## Idea
Instead of running the differentiable global placer once and greedy detailed
placement once, **interleave** them so each escapes the other's failure mode.
The clean framing is **iterated local search (basin-hopping)**:

- **greedy** = local descent on the *true* proxy (what we're scored on)
- **diff** = a smart, gradient-informed *perturbation operator* that kicks the
  placement out of greedy's local optimum
- **keep-best** on the true proxy = safety net that makes it monotonic

(Not block-coordinate-descent by macro type: hard-macro placement barely moves
the proxy (~0.2%), so both stages really optimize soft placement — it's
perturb-and-descend on one soft-dominated objective, not a clean variable split.)

## Loop
```
coord ← diff global placement (spread→cluster); legalize
best  ← greedy(coord)                       # first true-proxy local optimum
repeat until budget:
    coord ← best (as continuous)
    diff burst from coord, "reheated" (raise density/WL weight or add noise),
        run until surrogate plateaus         # trigger: <ε over last N iters
    legalize(coord)
    cand ← greedy(coord)                      # descend on true proxy
    if proxy(cand) < proxy(best): best ← cand # keep-best
    # else optionally accept-worse w/ small prob (anneal), else revert
return best
```
The plateau trigger (greedy takes over when the last N diff iters improve < ε) is
the handoff; after greedy, re-perturb.

## Core tension → keep-best is non-negotiable
diff optimizes the **surrogate**, greedy the **truth**. A diff burst can *raise*
the true proxy and undo greedy's gains. Mitigations:
1. **keep-best on the true proxy** — never lose the best legalized state; the loop
   can wander safely. Essential (turns oscillation into monotone improvement).
2. **controlled bursts + displacement penalty** — short bursts and/or a term
   penalizing movement from `best`, so the kick reshapes locally instead of
   drifting to the surrogate's (worse-on-truth) optimum. Burst length is the
   exploration↔safety knob; it should **cool** (big kicks early, small late).

## Honest ceiling: the surrogate is congestion-blind
The diff loss has **congestion off** (RUDY is gameable), and **congestion is the
wall** on the hard benchmarks (ibm08/17/18). So diff perturbations move along a
WL+density gradient that ignores congestion — they can't steer toward
lower-congestion layouts, which is where the ~9% to the leaderboard lives.
- WL/density-limited benchmarks (easy): interleaving can escape greedy plateaus.
- Congestion-limited benchmarks (hard): won't break the wall **unless** we also
  put a **non-gameable congestion term** back in the diff loss.

So the real structural play is **concurrent diff+greedy WITH a congestion-aware
surrogate**. Concurrent-alone likely dips just under 1.0; concurrent + real
congestion is what could contend for the top.

## Cheap test (also a diagnosis)
Prototype on **ibm01** (WL/density-limited) and **ibm17** (congestion-limited);
compare final proxy vs single-pass diff+greedy, measure plateau-escape on each:
- ibm01 improves, ibm17 flat → confirms congestion-blindness; congestion model is
  the required next piece.
- both improve → surrogate more aligned than expected; scale interleaving alone.

## Implementation
Reuse `place()` (diff burst + displacement-penalty term + weight reheat),
`legalize()`, `optimize_fast()` (greedy), wrapped in a controller doing keep-best
on `compute_proxy_cost`. Working prototype ≈ 1 day.
