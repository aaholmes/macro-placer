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
