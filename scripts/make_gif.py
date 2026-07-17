"""Animate the personal-best optimization for one benchmark: the differentiable
global placement trajectory (spread -> clustered) using the tuned config, then
the legalization snap and the greedy-polished final as closing frames.

Saves notes/opt_<nm>.gif. Usage: python make_gif.py [ibm01]
"""
import sys, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm, matplotlib.colors as mcolors
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
NOTES = "/home/laz/partcl/my-macro-placer/notes"
nm = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
STRIDE = 80          # snapshot every STRIDE gradient steps


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
hp = hp_from_params(json.load(open(f"{NOTES}/bayes_final.json"))["params"])
loss_fn = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                     lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"],
                     tau_d=hp["tau_d"], tau_c=hp["tau_c"], target=hp["target"], topk="lapsum",
                     dmode=hp["dmode"], lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"],
                     legalize_steps=hp["lsteps"])

# --- discover the ACTUAL winning candidate for this benchmark, so the animation
# reproduces the run that produced the reported score (winning config + init seed)
# and closes on the SAVED winning placement rather than an unrelated re-polish ---
import glob
FR = f"{NOTES}/fullrun"
WIN_ITERS = 15000     # iters used by the full run (fullrun_place --iters 15000)
cands = [(json.load(open(f))["final"], json.load(open(f))["idx"])
         for f in glob.glob(f"{FR}/{nm}_p*_final.json")
         if os.path.exists(f.replace("_final.json", "_polished.npz"))]
win_final, win_idx = min(cands)
wmeta = {e["idx"]: e for e in json.load(open(f"{FR}/{nm}_meta.json"))}[win_idx]
assert wmeta["trial"] == json.load(open(f"{NOTES}/bayes_final.json"))["trial"], \
    f"{nm} winner is trial {wmeta['trial']}, not the bayes_final config — load that config instead"
init_seed = wmeta["seed"] * 101 + wmeta["config_i"]   # exactly what fullrun_place passed to P.place
iters = WIN_ITERS
STRIDE = max(80, iters // 60)     # ~60 trajectory frames regardless of iters

# --- replay the winning global-placement trajectory from its exact init, snapshotting ---
g = torch.Generator(device="cpu").manual_seed(init_seed)
c0 = torch.rand(b.num_macros, 2, generator=g)
c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
coord = c0.to(P.dev).requires_grad_(True)
opt = torch.optim.Adam([coord], lr=hp["lr"])
frames = []   # (positions, label)
for it in range(iters):
    t = it / iters
    opt.zero_grad(); loss = loss_fn(P, coord, t); loss.backward(); opt.step()
    with torch.no_grad():
        coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
    if it % STRIDE == 0:
        frames.append((coord.detach().cpu().numpy().copy(), f"global placement — step {it}/{iters}"))

# --- closing frames from the SAVED winning artifacts (the exact placement scoring win_final) ---
diff_cand = np.load(f"{FR}/{nm}_p{win_idx}.npz")["placement"].astype(np.float64)
frames.append((diff_cand.copy(), f"global placement — step {iters}/{iters}"))
legal, _, _ = legalize(diff_cand, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
frames.append((legal.copy(), "legalized (zero overlap)"))
polished = np.load(f"{FR}/{nm}_p{win_idx}_polished.npz")["placement"].astype(np.float64)
frames.append((polished.copy(), f"greedy-polished — proxy {win_final:.3f}"))
print(f"{nm}: winning idx={win_idx} trial={wmeta['trial']} seed={wmeta['seed']} "
      f"init_seed={init_seed} final={win_final:.4f}", flush=True)

# --- render frames to a GIF ---
areas = sz[:nh, 0] * sz[:nh, 1]
norm = mcolors.LogNorm(vmin=areas.min(), vmax=areas.max())
imgs = []
for pos, label in frames:
    fig, ax = plt.subplots(figsize=(5, 5.2), constrained_layout=True)
    ax.add_patch(Rectangle((0, 0), b.canvas_width, b.canvas_height, fill=False, ec="black", lw=1.2))
    ax.scatter(pos[nh:, 0], pos[nh:, 1], s=2, c="0.7", alpha=0.35)
    for i in range(nh):
        w, h = sz[i]; x, y = pos[i]
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                               facecolor=cm.viridis(norm(areas[i])), alpha=0.85, ec="none"))
    ax.set_xlim(-1, b.canvas_width + 1); ax.set_ylim(-1, b.canvas_height + 1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{nm}: {label}", fontsize=11)
    fig.canvas.draw()
    imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)

# hold the first and last frames longer
durations = [90] * len(imgs)
durations[0] = 700; durations[-1] = 2500; durations[-2] = 900
imgs[0].save(f"{NOTES}/opt_{nm}.gif", save_all=True, append_images=imgs[1:],
             duration=durations, loop=0)
print(f"saved notes/opt_{nm}.gif  ({len(imgs)} frames)")
