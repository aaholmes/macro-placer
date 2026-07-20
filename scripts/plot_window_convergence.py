"""Convergence plot for the overlap-window TPE study (diffplace_window): running-best
ratio-mean vs trial, all trials scattered, with the warm-start-seed / random-startup /
TPE-model phases marked and the baseline (window-off) band shown for reference. Reads
the Optuna DB only -- no benchmark reloading. -> notes/bayes_window_trials.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

DB = "sqlite:////home/laz/partcl/my-macro-placer/notes/bayes_window.db"
OUT = "/home/laz/partcl/my-macro-placer/notes/bayes_window_trials.png"
N_SEED = 14          # structured warm-start seeds enqueued first
N_STARTUP = 45       # TPESampler n_startup_trials (seeds + random before model kicks in)

s = optuna.load_study(study_name="diffplace_window", storage=DB)
trials = sorted([t for t in s.trials if t.state == TrialState.COMPLETE and t.value is not None],
                key=lambda t: t.number)
xs = np.array([t.number for t in trials])
ys = np.array([t.value for t in trials])
# window-off (baseline) trials: hold==0 and ramp==0
off = np.array([t.params["ov_hold"] == 0 and t.params["ov_ramp"] == 0 for t in trials])
run_best = np.minimum.accumulate(ys)
bi = int(np.argmin(ys)); bt = trials[bi]

fig, ax = plt.subplots(figsize=(9, 5.2))
# baseline (window-off) reference band
if off.any():
    b_lo = ys[off].min()
    ax.axhspan(b_lo, ys[off].max(), color="#9ca3af", alpha=0.15, zorder=0)
    ax.axhline(b_lo, ls=":", color="#6b7280", lw=1.2, zorder=1)
    ax.annotate(f"best window-OFF (baseline) = {b_lo:.4f}", (xs.min(), b_lo),
                xytext=(6, 6), textcoords="offset points", ha="left", fontsize=8.5, color="#6b7280")
# all trials
ax.scatter(xs[~off], ys[~off], s=16, color="#2563eb", alpha=0.45, label="windowed trial", zorder=2)
ax.scatter(xs[off], ys[off], s=26, color="#6b7280", alpha=0.8, marker="s", label="window-off (baseline)", zorder=3)
# running best
ax.plot(xs, run_best, color="#dc2626", lw=2, label="running best", zorder=4)
# best marker
ax.scatter([bt.number], [bt.value], s=70, facecolor="none", edgecolor="#dc2626", lw=2, zorder=5)
ax.annotate(f"best #{bt.number} = {bt.value:.4f}\nhold={bt.params['ov_hold']:.2f}, ramp={bt.params['ov_ramp']:.2f}",
            (bt.number, bt.value), xytext=(-14, -42), textcoords="offset points", fontsize=9,
            ha="right", color="#dc2626", arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))
# phase boundaries
for x, lab in [(N_SEED - 0.5, f"warm-start seeds\n(0–{N_SEED-1})"),
               (N_STARTUP - 0.5, f"random startup\n({N_SEED}–{N_STARTUP-1})")]:
    ax.axvline(x, ls="--", color="#111827", lw=0.8, alpha=0.5)
ytop = ax.get_ylim()[1]
ax.text(N_SEED / 2, ytop, "seed", ha="center", va="top", rotation=90, fontsize=7.5, color="#374151")
ax.text((N_SEED + N_STARTUP) / 2, ytop, "random startup", ha="center", va="top", fontsize=8, color="#374151")
ax.text((N_STARTUP + xs.max()) / 2, ytop, "TPE model", ha="center", va="top", fontsize=8, color="#374151")

ax.set_xlabel("trial")
ax.set_ylabel("ratio-mean vs reference  (ibm01/13/17, single-seed; lower = better)")
ax.set_title("Overlap-window search: tuning the (hold, ramp) window + density/schedule cluster")
ax.grid(alpha=0.2); ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print(f"saved {OUT}")
print(f"trials={len(trials)}  best #{bt.number}={bt.value:.4f} (hold={bt.params['ov_hold']:.2f},"
      f"ramp={bt.params['ov_ramp']:.2f})  baseline-off best={ys[off].min():.4f}")
