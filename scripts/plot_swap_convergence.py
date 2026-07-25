"""Convergence plot for the deployment-metric search (diffplace_swap): objective is
best-of-3 seeds AFTER a short greedy polish (not single-seed) — the metric we actually
deploy. Running-best vs trial, mechanism-off (window/swap/diffusion all off) vs on, with
the winner marked. Reads the Optuna DB only. -> notes/bayes_swap_trials.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

DB = "sqlite:////home/laz/partcl/my-macro-placer/notes/bayes_swap.db"
OUT = "/home/laz/partcl/my-macro-placer/notes/bayes_swap_trials.png"
N_STARTUP = 48       # seeds + random startup before the TPE model takes over

s = optuna.load_study(study_name="diffplace_swap", storage=DB)
trials = sorted([t for t in s.trials if t.state == TrialState.COMPLETE and t.value is not None],
                key=lambda t: t.number)
xs = np.array([t.number for t in trials]); ys = np.array([t.value for t in trials])
off = np.array([t.params["swap_T0"] == 0 and t.params["gdiff_D0"] == 0 for t in trials])
run_best = np.minimum.accumulate(ys)
bt = trials[int(np.argmin(ys))]

fig, ax = plt.subplots(figsize=(9, 5.2))
if off.any():
    b_lo = ys[off].min()
    ax.axhline(b_lo, ls=":", color="#6b7280", lw=1.2, zorder=1)
    ax.annotate(f"best with no mechanism = {b_lo:.4f}", (xs.min(), b_lo),
                xytext=(6, 6), textcoords="offset points", ha="left", fontsize=8.5, color="#6b7280")
ax.scatter(xs[~off], ys[~off], s=16, color="#2563eb", alpha=0.45, label="window / swap / diffusion on", zorder=2)
ax.scatter(xs[off], ys[off], s=26, color="#6b7280", alpha=0.8, marker="s", label="all mechanisms off", zorder=3)
ax.plot(xs, run_best, color="#dc2626", lw=2, label="running best", zorder=4)
ax.scatter([bt.number], [bt.value], s=70, facecolor="none", edgecolor="#dc2626", lw=2, zorder=5)
ax.annotate(f"winner #{bt.number} = {bt.value:.4f}\nwindow ramp={bt.params['ov_ramp']:.2f}, swap on",
            (bt.number, bt.value), xytext=(-14, 30), textcoords="offset points", fontsize=9,
            ha="right", color="#dc2626", arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))
ax.axvline(N_STARTUP - 0.5, ls="--", color="#111827", lw=0.8, alpha=0.5)
ytop = ax.get_ylim()[1]
ax.text(N_STARTUP / 2, ytop, "warm-start + random startup", ha="center", va="top", fontsize=8, color="#374151")
ax.text((N_STARTUP + xs.max()) / 2, ytop, "TPE model", ha="center", va="top", fontsize=8, color="#374151")
ax.set_xlabel("trial")
ax.set_ylabel("best-of-3 after greedy — ratio vs reference  (lower = better)")
ax.set_title("Deployment-metric search: tuning the overlap window + swaps on best-of-N-after-greedy")
ax.grid(alpha=0.2); ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
fig.tight_layout(); fig.savefig(OUT, dpi=120)
print(f"saved {OUT}  trials={len(trials)} winner #{bt.number}={bt.value:.4f} no-mech best={ys[off].min():.4f}")
