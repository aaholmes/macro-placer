"""Animate the actual winning run for one benchmark, side-by-side with a live
proxy-score curve whose dot tracks the placement.

Left panel: the differentiable global-placement trajectory (spread -> clustered)
replayed from the winning candidate's exact init (config + seed that produced the
reported score), then the legalization snap and the SAVED greedy-polished final.
Right panel: the true proxy cost at every frame, with a moving dot.

Saves notes/opt_<nm>.gif. Usage: python make_gif.py [ibm01]
"""
import sys, os, json, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm, matplotlib.colors as mcolors
import numpy as np, torch, optuna
from PIL import Image
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
NOTES = "/home/laz/partcl/my-macro-placer/notes"
FR = f"{NOTES}/fullrun"
nm = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
WIN_ITERS = 15000     # iters used by the full run (fullrun_place --iters 15000)


def hp_from_params(p):
    dmode = p["dmode"]
    lam_d_end = p.get("lam_d_end_e") if dmode == "electrostatic" else p.get("lam_d_end_o")
    lam_c0 = lam_c1 = 0.0
    if p.get("use_cong"):
        lam_c1 = p["lam_c1"]; lam_c0 = lam_c1 * p["lam_c_startfrac"]
    return dict(dmode=dmode, lam_d_end=lam_d_end, lam_c0=lam_c0, lam_c1=lam_c1,
                lam_d_ratio=p["lam_d_ratio"], gamma=p["gamma"], tau_d=p["tau_d"],
                tau_c=p["tau_c"], lam_wl0=p["lam_wl0"], target=p["target"], lr=p["lr"],
                iters=p["iters"], ramp_p=p["ramp_p"], lsteps=p["lsteps_n"])


b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy(); nh = b.num_hard_macros


def proxy_of(pos):
    # FastEval proxy (validated exact to ~1e-5 vs compute_proxy_cost) — ~ms/frame so
    # the per-frame score curve is cheap even on the large designs.
    p = pos[:, :2] if pos.shape[1] > 2 else pos
    return float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))


# --- discover the winning candidate and load ITS config (may not be bayes_final) ---
cands = [(json.load(open(f))["final"], json.load(open(f))["idx"])
         for f in glob.glob(f"{FR}/{nm}_p*_final.json")
         if os.path.exists(f.replace("_final.json", "_polished.npz"))]
win_final, win_idx = min(cands)
wmeta = {e["idx"]: e for e in json.load(open(f"{FR}/{nm}_meta.json"))}[win_idx]
win_trial = wmeta["trial"]
init_seed = wmeta["seed"] * 101 + wmeta["config_i"]   # exactly what fullrun_place passed to P.place
study = optuna.load_study(study_name="diffplace", storage=f"sqlite:///{NOTES}/bayes.db")
wp = {t.number: t for t in study.trials}[win_trial].params
hp = hp_from_params(wp)
loss_fn = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                     lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"],
                     tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"], topk="lapsum",
                     dmode=hp["dmode"], lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
                     legalize_steps=hp["lsteps"])
print(f"{nm}: winning idx={win_idx} trial={win_trial} seed={wmeta['seed']} "
      f"init_seed={init_seed} final={win_final:.4f}", flush=True)

# --- replay the winning global-placement trajectory from its exact init ---
iters = WIN_ITERS
STRIDE = max(80, iters // 55)     # ~55 trajectory frames regardless of iters
g = torch.Generator(device="cpu").manual_seed(init_seed)
c0 = torch.rand(b.num_macros, 2, generator=g)
c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
coord = c0.to(P.dev).requires_grad_(True)
opt = torch.optim.Adam([coord], lr=hp["lr"])
frames = []   # (positions, label, proxy)
for it in range(iters):
    t = it / iters
    opt.zero_grad(); loss = loss_fn(P, coord, t); loss.backward(); opt.step()
    with torch.no_grad():
        coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
    if it % STRIDE == 0:
        pos = coord.detach().cpu().numpy().copy()
        frames.append((pos, "global placement", proxy_of(pos)))

# --- closing frames from the SAVED winning artifacts (the exact placement scoring win_final) ---
diff_cand = np.load(f"{FR}/{nm}_p{win_idx}.npz")["placement"].astype(np.float64)
frames.append((diff_cand.copy(), "global placement (final)", proxy_of(diff_cand)))
legal, _, _ = legalize(diff_cand, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
frames.append((legal.copy(), "legalized (zero overlap)", proxy_of(legal)))
polished = np.load(f"{FR}/{nm}_p{win_idx}_polished.npz")["placement"].astype(np.float64)
frames.append((polished.copy(), "greedy-polished (final)", win_final))

# --- reference (shipped) placement, shown static in the first column + as a line in the curve ---
ref_pos, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, nh,
                         b.canvas_width, b.canvas_height, gap=0.01)
ref_proxy = proxy_of(ref_pos)

# --- render three-panel frames: reference | animating placement | live proxy curve ---
areas = sz[:nh, 0] * sz[:nh, 1]
cnorm = mcolors.LogNorm(vmin=areas.min(), vmax=areas.max())
ys = [f[2] for f in frames]
xs = list(range(len(frames)))
n_detail = len(frames) - 2        # index where the detailed stage (legalize+polish) begins
ylo = min(ys) * 0.9; yhi = max(max(ys), ref_proxy) * 1.1


def draw_place(ax, pos, title):
    ax.add_patch(Rectangle((0, 0), b.canvas_width, b.canvas_height, fill=False, ec="black", lw=1.2))
    ax.scatter(pos[nh:, 0], pos[nh:, 1], s=2, c="0.7", alpha=0.35)
    for k in range(nh):
        w, h = sz[k]; x, y = pos[k, 0], pos[k, 1]
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                               facecolor=cm.viridis(cnorm(areas[k])), alpha=0.85, ec="none"))
    ax.set_xlim(-1, b.canvas_width + 1); ax.set_ylim(-1, b.canvas_height + 1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10.5)


imgs = []
for i, (pos, label, py) in enumerate(frames):
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(14.2, 4.9),
                                        gridspec_kw={"width_ratios": [1, 1, 1.2]}, constrained_layout=True)
    draw_place(axL, ref_pos, f"reference  (score {ref_proxy:.3f})")
    draw_place(axM, pos, f"{nm}: {label}   ·   score {py:.3f}")
    # right: score curve (log-y) with moving dot + reference line
    axR.axvspan(n_detail - 0.5, len(frames) - 0.5, color="#fde68a", alpha=0.5, lw=0)
    axR.axhline(ref_proxy, ls="--", lw=1.2, color="#b91c1c", zorder=1)
    axR.text(len(frames) - 1, ref_proxy, "reference ", color="#b91c1c", fontsize=8.5, va="bottom", ha="right")
    axR.plot(xs, ys, color="0.75", lw=1.3, zorder=2)
    axR.plot(xs[:i + 1], ys[:i + 1], color="#2563eb", lw=1.9, zorder=3)
    axR.scatter([i], [py], s=60, color="#dc2626", zorder=4, ec="white", lw=0.8)
    axR.set_yscale("log"); axR.set_xlim(-0.5, len(frames) - 0.5); axR.set_ylim(ylo, yhi)
    axR.set_xlabel("frame  (global placement → legalize + greedy)", fontsize=9)
    axR.set_ylabel("score (log)", fontsize=10)
    axR.set_title(f"score = {py:.3f}", fontsize=11)
    axR.grid(alpha=0.25, which="both")
    fig.canvas.draw()
    imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)

durations = [110] * len(imgs)
durations[0] = 700; durations[-1] = 2500; durations[-2] = 900; durations[-3] = 900
imgs[0].save(f"{NOTES}/opt_{nm}.gif", save_all=True, append_images=imgs[1:],
             duration=durations, loop=0)
print(f"saved notes/opt_{nm}.gif  ({len(imgs)} frames, final proxy {win_final:.3f})")
