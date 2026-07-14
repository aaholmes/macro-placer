"""Capture single-pass vs interleaved optimization for one benchmark as GIFs.
Each frame is two panels: LEFT = the placement (discrete greedy "jumps" get a red
border/title); RIGHT = the true-proxy score curve with a dot tracing the current
frame, a keep-best envelope, and the other method's curve as a faint reference.
Produces notes/opt_<nm>_singlepass.gif and notes/opt_<nm>_interleaved.gif.

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


def proxy_of(pos, need_legal):
    p = pos
    if need_legal:
        p, _, _ = legalize(pos, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
    return float(compute_proxy_cost(torch.tensor(p, dtype=torch.float32), b, plc)["proxy_cost"])


def diff_capture(coord0, iters, t_start, seed, stride, frames, tag):
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
            frames.append((pos.copy(), tag, False, True))     # (pos, label, red, need_legal)
    out = np.zeros((b.num_macros, 2)); out[:, :2] = c.detach().cpu().numpy()
    return out


def greedy_capture(pos0, total, chunk, frames, tag):
    pos = pos0.copy(); done = 0
    while done < total:
        pos, _ = optimize_fast(fe, pos, iters=chunk, seed=1000 + done, T0=0.0, move_hard=False,
                               move_soft=True, refresh=chunk, log_every=chunk + 1, logf=lambda *a: None)
        done += chunk
        frames.append((pos.copy(), tag, True, False))
    return pos


def build(interleaved):
    frames = []
    coord = diff_capture(None, hp["iters"], 0.0, 0, max(1, hp["iters"] // 40), frames, "diff global")
    lg, _, _ = legalize(coord, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
    frames.append((lg.copy(), "legalized", False, False))
    best = greedy_capture(lg, 80000, 20000, frames, "greedy (discrete jumps)")
    best_cost = proxy_of(best, False)
    if interleaved:
        for r in range(a.rounds):
            t_start = 0.2 + 0.6 * (r / max(1, a.rounds - 1))
            pk = diff_capture(best, a.burst, t_start, r + 1, max(1, a.burst // 15),
                              frames, f"round {r+1}: diff burst")
            lg, _, _ = legalize(pk, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
            cand = greedy_capture(lg, 80000, 20000, frames, f"round {r+1}: greedy")
            c = proxy_of(cand, False)
            if c < best_cost:
                best, best_cost = cand, c
        frames.append((best.copy(), f"BEST ({best_cost:.3f})", False, False))
    else:
        frames.append((best.copy(), f"final ({best_cost:.3f})", False, False))
    scores = [proxy_of(pos, nl) for (pos, _, _, nl) in frames]
    return frames, scores, best_cost


def render(frames, scores, other_scores, path, title, ylim):
    areas = sz[:nh, 0] * sz[:nh, 1]
    norm = mcolors.LogNorm(vmin=areas.min(), vmax=areas.max())
    N = len(scores); x = np.arange(N) / max(1, N - 1)
    keep = np.minimum.accumulate(scores)
    xo = np.arange(len(other_scores)) / max(1, len(other_scores) - 1)
    keep_o = np.minimum.accumulate(other_scores)
    imgs = []
    for i, (pos, label, red, _) in enumerate(frames):
        fig, (axp, axs) = plt.subplots(1, 2, figsize=(10.5, 5.4), constrained_layout=True)
        # left: placement
        axp.add_patch(Rectangle((0, 0), b.canvas_width, b.canvas_height, fill=False, ec="black", lw=1.2))
        axp.scatter(pos[nh:, 0], pos[nh:, 1], s=2, c="0.7", alpha=0.35)
        for k in range(nh):
            w, h = sz[k]; xx, yy = pos[k]
            axp.add_patch(Rectangle((xx - w / 2, yy - h / 2), w, h,
                                    facecolor=cm.viridis(norm(areas[k])), alpha=0.85, ec="none"))
        axp.set_xlim(-1, b.canvas_width + 1); axp.set_ylim(-1, b.canvas_height + 1)
        axp.set_aspect("equal"); axp.set_xticks([]); axp.set_yticks([])
        axp.set_title(f"{title}: {label}", fontsize=10, color=("#dc2626" if red else "#111"))
        # right: score curves
        axs.plot(xo, other_scores, color="#cbd5e1", lw=1.2, label="other (per-frame)")
        axs.plot(x, scores, color="#93c5fd", lw=1.0, label="this (per-frame)")
        axs.plot(x, keep, color="#2563eb", lw=2.0, label="this (keep-best)")
        axs.plot(xo, keep_o, color="#94a3b8", lw=1.3, ls="--", label="other (keep-best)")
        axs.scatter([x[i]], [scores[i]], s=80, color=("#dc2626" if red else "#2563eb"), zorder=5)
        axs.set_ylim(ylim); axs.set_xlim(-0.02, 1.02)
        axs.set_xlabel("optimization progress"); axs.set_ylabel("proxy (lower better)")
        axs.set_title("score", fontsize=10); axs.grid(alpha=0.25); axs.legend(fontsize=7, loc="upper right")
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        img = ImageOps.expand(img, border=10, fill=(220, 38, 38) if red else (255, 255, 255))
        imgs.append(img); plt.close(fig)
    dur = [130] * len(imgs); dur[0] = 500; dur[-1] = 2500
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0)
    print(f"saved {path} ({len(imgs)} frames)", flush=True)


fr_sp, sc_sp, c_sp = build(False)
fr_il, sc_il, c_il = build(True)
allv = np.array(sc_sp + sc_il)
ylim = (allv.min() * 0.99, np.percentile(allv, 88))          # clip early diff-spread spikes
render(fr_sp, sc_sp, sc_il, f"{NOTES}/opt_{nm}_singlepass.gif", "single-pass", ylim)
render(fr_il, sc_il, sc_sp, f"{NOTES}/opt_{nm}_interleaved.gif", "interleaved", ylim)
print(f"{nm}: single-pass={c_sp:.4f}  interleaved={c_il:.4f}  gain={c_sp-c_il:+.4f}", flush=True)
