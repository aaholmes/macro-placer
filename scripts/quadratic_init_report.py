"""Paired analysis of the init-arm experiment.

For each benchmark and arm, the difference from the random control on the SAME
seed, with a bootstrap interval on the mean difference (resampling seeds).
Negative favors the arm. Also reports seed spread per arm and, for
corner_screen, the Spearman correlation between quadratic-HPWL rank and
exact-score rank among the screen-2 survivors.

Usage: python scripts/quadratic_init_report.py notes/quadratic_init/*.json
"""
import sys, json, collections
import numpy as np

STAGES = [("score_legal", "legal"), ("score_screen", "screen")]


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), float("nan"), float("nan")
    m = rng.choice(d, (n, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(paths):
    rows = []
    for p in paths:
        rows += json.load(open(p))["rows"]
    by = {(r["bench"], r["arm"], r["seed"]): r for r in rows}
    benches = sorted({r["bench"] for r in rows})
    arms = list(dict.fromkeys(r["arm"] for r in rows if r["arm"] != "random"))
    for key, label in STAGES:
        print(f"\n=== {label}: mean paired difference vs random, "
              f"95% bootstrap interval (negative favors the arm) ===")
        print(f"{'bench':7s} {'arm':26s} {'control':>9s} {'diff':>9s} {'ci_lo':>9s} {'ci_hi':>9s} {'sd_arm':>7s}")
        for nm in benches:
            for arm in arms:
                d = []; vals = []
                for s in sorted({r["seed"] for r in rows if r["bench"] == nm}):
                    a = by.get((nm, arm, s)); c = by.get((nm, "random", s))
                    if a is None or c is None or not np.isfinite(a.get(key, np.nan)):
                        continue
                    d.append(a[key] - c[key]); vals.append(a[key])
                if not d:
                    continue
                ctrl = np.mean([by[(nm, "random", s)][key]
                                for s in sorted({r["seed"] for r in rows if r["bench"] == nm})
                                if (nm, "random", s) in by])
                m, lo, hi = boot(d)
                print(f"{nm:7s} {arm:26s} {ctrl:9.4f} {m:+9.4f} {lo:+9.4f} {hi:+9.4f} {np.std(vals):7.4f}")
    # pooled over benchmarks: mean of per-benchmark mean differences
    print(f"\n=== pooled over benchmarks (mean of per-benchmark paired differences) ===")
    for key, label in STAGES:
        for arm in arms:
            per = []
            for nm in benches:
                d = [by[(nm, arm, s)][key] - by[(nm, "random", s)][key]
                     for s in sorted({r["seed"] for r in rows if r["bench"] == nm})
                     if (nm, arm, s) in by and (nm, "random", s) in by
                     and np.isfinite(by[(nm, arm, s)].get(key, np.nan))]
                if d:
                    per.append(np.mean(d))
            if per:
                m, lo, hi = boot(per)
                print(f"{label:7s} {arm:26s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n_bench={len(per)}")
    sp = [r["spearman"] for r in rows if r.get("spearman") is not None
          and np.isfinite(r.get("spearman", np.nan))]
    if sp:
        print(f"\nscreen-1 vs screen-2 rank correlation among survivors: "
              f"mean {np.mean(sp):+.2f} over {len(sp)} runs, per run "
              f"{', '.join(f'{v:+.2f}' for v in sp)}")
        t1 = [r["t_init"] for r in rows if r["arm"] == "corner_screen"]
        t2 = [r["t_diff"] for r in rows if r["arm"] == "corner_screen"]
        print(f"corner screen wall time: screen1 {np.mean(t1):.0f}s, screen2 {np.mean(t2):.0f}s per run")


if __name__ == "__main__":
    main(sys.argv[1:])
