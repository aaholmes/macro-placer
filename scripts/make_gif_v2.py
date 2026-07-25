"""Regenerate opt_<nm>.gif from the best-of-100 run's ACTUAL winning config + seed,
faithful to the overlap window + SA-swap (reimplements bayes_swap.diff_place's loop with
frame capture). deploy_best100 used seed = cnt // n_configs, so each config's winning seed
is in 0..NSEED-1; we sweep those, pick the best (matching the reported per-bench score), and
animate that run: diff trajectory -> legalize -> greedy-polished final, with a live score
curve. Usage: python make_gif_v2.py [ibm01]
"""
import sys, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm, matplotlib.colors as mcolors
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/home/laz/partcl/my-macro-placer"); sys.path.insert(0, "/home/laz/partcl/my-macro-placer/scripts")
import bayes_swap as bs
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, anneal, lapsum_topk_mean

CHAL = "/home/laz/partcl/macro-place-challenge-2026"
NOTES = "/home/laz/partcl/my-macro-placer/notes"
nm = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
WINNER = {"ibm01": "t94_win+swap", "ibm09": "t94_win+swap", "ibm17": "wo97", "ibm18": "t94_win+swap"}
ITERS = 15000; NSEED = 7; MOVES = 100000

pool = json.load(open(f"{NOTES}/best100_pool.json"))
cfg = next(p for p in pool if p["_label"] == WINNER[nm])
dflt = dict(ov_hold=0.0, ov_ramp=0.0, swap_T0=0.0, swap_frac=0.3, gdiff_D0=0.0, gdiff_frac=0.3)
hp = {**bs.hp_from({**dflt, **cfg}), "iters": ITERS}

b, plc = load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
fe = FastEval(b, plc); P = DifferentiablePlacer(fe); sz = b.macro_sizes.numpy(); nh = b.num_hard_macros


def proxy_of(pos):
    p = pos[:, :2] if pos.shape[1] > 2 else pos
    return float(fe.wirelength_cost(p) + 0.5 * fe.density_cost(p) + 0.5 * fe.congestion_cost(p))


def run_diff(seed, capture=False):
    """Faithful copy of bayes_swap.diff_place, optionally capturing trajectory frames."""
    cap = 1.0
    gD0 = hp["gdiff_D0"]; gfrac = max(1e-6, hp["gdiff_frac"]); sT0 = hp["swap_T0"]; sfrac = max(1e-6, hp["swap_frac"])
    csz = float(min(P.W, P.H))
    ar = getattr(P, "area", None)
    mob = ((ar.median() / ar).sqrt().clamp(max=5.0).unsqueeze(1) if ar is not None
           else torch.ones(P.b.num_macros, 1, device=P.dev))
    ngen = torch.Generator(device=P.dev).manual_seed(seed * 1000 + 7)
    rs = np.random.RandomState(seed * 7 + 3)
    groups = bs._swap_groups(P) if sT0 > 0 else []
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(P.b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    coord = c0.to(P.dev).requires_grad_(True)
    opt = torch.optim.Adam([coord], lr=hp["lr"])
    LD0 = hp["lam_d_end"] * hp["lam_d_ratio"]; LD1 = hp["lam_d_end"]
    STRIDE = max(80, ITERS // 55)
    frames = []
    for it in range(ITERS):
        t = it / ITERS; te = t ** hp["ramp_p"]
        gamma = anneal(te, P.gw * hp["gamma"], P.gw, geometric=True)
        lam_d = anneal(te, LD0, LD1); lam_wl = anneal(te, max(hp["lam_wl0"], 1e-9), 1.0, geometric=True)
        rho = P.density_field(coord)
        spread = torch.clamp(torch.clamp(rho, max=cap) - hp["target"], min=0.0)
        overlap = torch.clamp(rho - cap, min=0.0)
        w = 0.0 if t <= hp["ov_hold"] else (1.0 if hp["ov_ramp"] <= 1e-9 else min(1.0, (t - hp["ov_hold"]) / hp["ov_ramp"]))
        de = lapsum_topk_mean(spread + w * overlap, 0.10, hp["tau_d"])
        opt.zero_grad(); (lam_wl * P.wirelength(coord, gamma) + lam_d * de).backward(); opt.step()
        with torch.no_grad():
            if gD0 > 0 and t < gfrac:
                coord.add_(gD0 * csz * (1.0 - t / gfrac) * mob * torch.randn(coord.shape, generator=ngen, device=P.dev))
            coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
            if sT0 > 0 and t < sfrac and it % bs.SWAP_EVERY == 0 and groups:
                T = sT0 * (1.0 - t / sfrac)
                if T > 1e-9:
                    wl0 = float(P.wirelength(coord, gamma))
                    for _ in range(bs.SWAP_ATTEMPTS):
                        grp = groups[rs.randint(len(groups))]; i = int(grp[rs.randint(len(grp))])
                        d2 = ((coord[grp] - coord[i]) ** 2).sum(1); d2[grp == i] = 1e18; j = int(grp[int(d2.argmin())])
                        if i == j: continue
                        coord[[i, j]] = coord[[j, i]]; wl1 = float(P.wirelength(coord, gamma))
                        if wl1 <= wl0 or rs.rand() < np.exp(-(wl1 - wl0) / T): wl0 = wl1
                        else: coord[[i, j]] = coord[[j, i]]
        if capture and it % STRIDE == 0:
            pos = coord.detach().cpu().numpy().copy()
            frames.append((pos, "global placement", proxy_of(pos)))
    return coord.detach().cpu().numpy(), frames


# --- find the winning seed (best of 0..NSEED-1, full pipeline) ---
best = (1e9, None, None, None)
for s in range(NSEED):
    raw, _ = run_diff(s)
    lg, _, _ = legalize(raw, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
    pol, _ = optimize_fast(fe, lg, iters=MOVES, seed=s, T0=0.0, move_hard=False, move_soft=True,
                           refresh=20000, log_every=10**9, logf=lambda *x: None)
    pr = proxy_of(pol)
    print(f"{nm} {WINNER[nm]} seed {s}: {pr:.4f}", flush=True)
    if pr < best[0]: best = (pr, s, raw, pol)
win_proxy, win_seed, win_raw, win_pol = best
print(f"{nm}: winning seed={win_seed} score={win_proxy:.4f}", flush=True)

# --- replay winning seed with frame capture + closing frames ---
diff_cand, frames = run_diff(win_seed, capture=True)
frames.append((diff_cand.copy(), "global placement (final)", proxy_of(diff_cand)))
legal, _, _ = legalize(diff_cand, sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
frames.append((legal.copy(), "legalized (zero overlap)", proxy_of(legal)))
frames.append((win_pol.copy(), "greedy-polished (final)", win_proxy))

ref_pos, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, nh, b.canvas_width, b.canvas_height, gap=0.01)
ref_proxy = proxy_of(ref_pos)
areas = sz[:nh, 0] * sz[:nh, 1]; cnorm = mcolors.LogNorm(vmin=areas.min(), vmax=areas.max())
ys = [f[2] for f in frames]; xs = list(range(len(frames))); n_detail = len(frames) - 2
ylo = min(ys) * 0.9; yhi = max(max(ys), ref_proxy) * 1.1


def draw_place(ax, pos, title):
    ax.add_patch(Rectangle((0, 0), b.canvas_width, b.canvas_height, fill=False, ec="black", lw=1.2))
    ax.scatter(pos[nh:, 0], pos[nh:, 1], s=2, c="0.7", alpha=0.35)
    for k in range(nh):
        wd, ht = sz[k]; x, y = pos[k, 0], pos[k, 1]
        ax.add_patch(Rectangle((x - wd / 2, y - ht / 2), wd, ht, facecolor=cm.viridis(cnorm(areas[k])), alpha=0.85, ec="none"))
    ax.set_xlim(-1, b.canvas_width + 1); ax.set_ylim(-1, b.canvas_height + 1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10.5)


imgs = []
for i, (pos, label, py) in enumerate(frames):
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(14.2, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.2]}, constrained_layout=True)
    draw_place(axL, ref_pos, f"reference  (score {ref_proxy:.3f})")
    draw_place(axM, pos, f"{nm}: {label}   ·   score {py:.3f}")
    axR.axvspan(n_detail - 0.5, len(frames) - 0.5, color="#fde68a", alpha=0.5, lw=0)
    axR.axhline(ref_proxy, ls="--", lw=1.2, color="#b91c1c", zorder=1)
    axR.text(len(frames) - 1, ref_proxy, "reference ", color="#b91c1c", fontsize=8.5, va="bottom", ha="right")
    axR.plot(xs, ys, color="0.75", lw=1.3, zorder=2)
    axR.plot(xs[:i + 1], ys[:i + 1], color="#2563eb", lw=1.9, zorder=3)
    axR.scatter([i], [py], s=60, color="#dc2626", zorder=4, ec="white", lw=0.8)
    axR.set_yscale("log"); axR.set_xlim(-0.5, len(frames) - 0.5); axR.set_ylim(ylo, yhi)
    axR.set_xlabel("frame  (global placement → legalize + greedy)", fontsize=9)
    axR.set_ylabel("score (log)", fontsize=10); axR.set_title(f"score = {py:.3f}", fontsize=11)
    axR.grid(alpha=0.25, which="both")
    fig.canvas.draw(); imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); plt.close(fig)

durations = [110] * len(imgs); durations[0] = 700; durations[-1] = 2500; durations[-2] = 900; durations[-3] = 900
imgs[0].save(f"{NOTES}/opt_{nm}.gif", save_all=True, append_images=imgs[1:], duration=durations, loop=0)
print(f"saved notes/opt_{nm}.gif ({len(imgs)} frames, final {win_proxy:.3f}, seed {win_seed})", flush=True)
