# Key findings

## Proxy is ~insensitive to hard-macro placement (ibm01)
- legalized reference: 1.0385
- soft-only greedy (hard frozen): 0.9388
- + WL-aware hard swaps: 0.9370  (hard-only contribution +0.0019, ~0.2%)
- hard swaps from reference alone: +0.0032 (~0.3%)
- disabling hard moves BEAT mixed hard+soft (0.9388 vs 0.9517) — hard proposals waste eval budget
- TILOS-certified 0.9372, 0 overlaps

Implication: Tier-1 proxy leverage = scaling soft optimization (task 4).
Tier-2 depends only on hard positions but proxy barely rewards them (a real tension).

## Cluster moves tested — did NOT unlock hard headroom (ibm01)
Cluster move preserves internal nets (removes the single-swap WL confound), yet:
- 14/60 proposals fail legalization; of 46 evaluated, only 4% improve
- median delta +0.0044 (raises cost), best -0.0004
Reason: a well-placed cluster's EXTERNAL connections point back to its origin
region (reference is WL-optimized), so relocating it lengthens those nets more
than it relieves density/congestion. Conclusion: hard-macro proxy headroom from
this reference is small even with the WL confound removed. Proxy headroom is in
soft co-optimization. (Not ironclad: a steepest-descent cluster selection or a
from-scratch hard placement could still be tried.)
