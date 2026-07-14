"""Capture single-pass vs interleaved optimization for one benchmark as GIFs.
Continuous differentiable phases render normally; discrete greedy phases (the
"jumps") get a RED BORDER so you can see when the layout is being shuffled by
the true-proxy local search. Produces notes/opt_<nm>_singlepass.gif and
notes/opt_<nm>_interleaved.gif.

Usage: python interleaved_gif.py [--bench ibm06] [--rounds 8] [--burst 1500]
"""
import os, sys, json, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm, matplotlib.colors as mcolors
import numpy as np, torch
from PIL import Image, ImageOps
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, proxy_loss

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
NOTES = "/home/laz/partcl/my-macro-placer/notes"


def hp_from_params(p):
    dmode = p["dmode"]
    lam_d_end = p.get("lam_d_end_e") if dmode == "electrostatic" else p.get("lam_d_end_o")
    lam_c0 = lam_c1 = 0.0
    if p.get("use_cong"):
        lam_c1 = p["lam_c1"]; lam_c0 = lam_c1 * p["lam_c_startfrac"]
    return dict(dmode=dmode, lam_d_end=lam_d_end, lam_c0=lam_c0, lam_c1=lam_c1,
                lam_d_ratio=p["lam_d_ratio"], gamma=p["gamma"], tau_d=p["tau_d"], tau_c=p["tau_c"],
                lam_wl0=p["lam_wl0"], target=p["target"], lr=p["lr"], ramp_p=p["ramp_p"],
                lsteps=p["lsteps_n"], iters=p["iters"])


ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="ibm06")
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--burst", type=int, default=1500)
a = ap.parse_args()
nm = a.bench

b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy(); nh = b.num_hard_macros
hp = hp_from_params(json.load(open(f"{NOTES}/bayes_final.json"))["params"])
loss = proxy_loss(gamma_hi_mult=hp["gamma"], lam_d0=hp["lam_d_end"] * hp["lam_d_ratio"],
                  lam_d1=hp["lam_d_end"], lam_c0=hp["lam_c0"], lam_c1=hp["lam_c1"], tau_d=hp["tau_d"],
                  tau_c=hp["tau_c"], target=hp["target"], topk="lapsum", dmode=hp["dmode"],
                  lam_wl0=hp["lam_wl0"], lam_wl1=1.0, ramp_p=hp["ramp_p"], legalize_steps=hp["lsteps"])


def proxy(pos):
    return float(compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), b, plc)["proxy_cost"])


def diff_capture(coord0, iters, t_start, seed, stride, frames, tag):
    """Adam on the surrogate from coord0, snapshotting every `stride` steps."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if coord0 is None:
        c = torch.rand(b.num_macros, 2, generator=g)
        c[:, 0] = c[:, 0] * (P.W - 2) + 1; c[:, 1] = c[:, 1] * (P.H - 2) + 1
        c = c.to(P.dev)
    else:
        c = torch.tensor(np.asarray(coord0, np.float32)[:, :2], device=P.dev)
        c = c + 0.01 * torch.randn(c.shape, generator=g).to(P.dev)
    c.requires_grad_(True)
    opt = torch.optim.Adam([c], lr=hp["lr"])
    for it in range(iters):
        t = t_start + (1.0 - t_start) * (it / iters)
        opt.zero_grad(); l = loss(P, c, t); l.backward(); opt.step()
        with torch.no_grad():
            c[:, 0].clamp_(P.hw, P.W - P.hw); c[:, 1].clamp_(P.hh, P.H - P.hh)
        if it % stride == 0:
            pos = np.zeros((b.num_macros, 2)); pos[:, :2] = c.detach().cpu().numpy()
            frames.append((pos.copy(), tag, False))
    out = np.zeros((b.num_macros, 2)); out[:, :2] = c.detach().cpu().numpy()
    return out


def greedy_capture(pos0, total, chunk, frames, tag):
    """Greedy in chunks, snapshotting each (RED border = discrete jumps)."""
    pos = pos0.copy(); done = 0
    while done < total:
        pos, _ = optimize_fast(fe, pos, iters=chunk, seed=1000 + done, T0=0.0, move_hard=False,
                               move_soft=True, refresh=chunk, log_every=chunk + 1, logf=lambda *a: None)
        done += chunk
        frames.append((pos.copy(), tag, True))
    return pos


def build(interleaved):
    frames = []
    coord = diff_capture(None, hp["iters"], 0.0, 0, max(1, hp["iters"] // 40), frames, "diff global")
    lg, _, _ = legalize(coord, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
    frames.append((lg.copy(), "legalized", False))
    best = greedy_capture(lg, 80000, 20000, frames, "greedy (discrete jumps)")
    best_cost = proxy(best)
    if interleaved:
        for r in range(a.rounds):
            t_start = 0.2 + 0.6 * (r / max(1, a.rounds - 1))
            pk = diff_capture(best, a.burst, t_start, r + 1, max(1, a.burst // 15),
                              frames, f"round {r+1}: diff burst")
            lg, _, _ = legalize(pk, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
            cand = greedy_capture(lg, 80000, 20000, frames, f"round {r+1}: greedy")
            c = proxy(cand)
            if c < best_cost:
                best, best_cost = cand, c
        frames.append((best.copy(), f"BEST interleaved ({best_cost:.3f})", False))
    else:
        frames.append((best.copy(), f"single-pass final ({best_cost:.3f})", False))
    return frames, best_cost


def render(frames, path, title):
    areas = sz[:nh, 0] * sz[:nh, 1]
    norm = mcolors.LogNorm(vmin=areas.min(), vmax=areas.max())
    imgs = []
    for pos, label, red in frames:
        fig, ax = plt.subplots(figsize=(5, 5.2), constrained_layout=True)
        ax.add_patch(Rectangle((0, 0), b.canvas_width, b.canvas_height, fill=False, ec="black", lw=1.2))
        ax.scatter(pos[nh:, 0], pos[nh:, 1], s=2, c="0.7", alpha=0.35)
        for i in range(nh):
            w, h = sz[i]; x, y = pos[i]
            ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                                   facecolor=cm.viridis(norm(areas[i])), alpha=0.85, ec="none"))
        ax.set_xlim(-1, b.canvas_width + 1); ax.set_ylim(-1, b.canvas_height + 1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}: {label}", fontsize=10, color=("#dc2626" if red else "#111"))
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        img = ImageOps.expand(img, border=10, fill=(220, 38, 38) if red else (255, 255, 255))
        imgs.append(img); plt.close(fig)
    dur = [120] * len(imgs); dur[0] = 500; dur[-1] = 2500
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0)
    print(f"saved {path} ({len(imgs)} frames)", flush=True)


fr_sp, c_sp = build(False)
render(fr_sp, f"{NOTES}/opt_{nm}_singlepass.gif", "single-pass")
fr_il, c_il = build(True)
render(fr_il, f"{NOTES}/opt_{nm}_interleaved.gif", "interleaved")
print(f"{nm}: single-pass={c_sp:.4f}  interleaved={c_il:.4f}  gain={c_sp-c_il:+.4f}", flush=True)
