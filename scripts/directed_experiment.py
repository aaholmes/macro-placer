"""Compare random vs directed proposals, and best-of variants incl. a combined
(soft-move + swap) move, on ibm01 from the reference. Plots proxy vs iteration."""
import sys, re, time
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import _hot_macros
from placers.moves import neighbor_centroid
from placers.force import _field_force

b, plc = load_benchmark_from_dir(
    "/home/laz/partcl/macro-place-challenge-2026/external/MacroPlacement/Testcases/ICCAD04/ibm01")
fe = FastEval(b, plc); sz = b.macro_sizes.numpy()
nh = b.num_hard_macros; nm = b.num_macros
hw = sz[:, 0] / 2; hh = sz[:, 1] / 2; W, H = fe.W, fe.H
legal, _, _ = legalize(b.macro_positions.numpy().astype(np.float64), sz, nh, W, H, gap=0.01)
from collections import defaultdict
groups = defaultdict(list)
for i in range(nh):
    groups[tuple(np.round(sz[i], 3))].append(i)
GLIST = [np.array(v) for v in groups.values() if len(v) >= 2]


def clampxy(i, x, y):
    return (min(max(x, hw[i]), W - hw[i]), min(max(y, hh[i]), H - hh[i]))


def run(variant, iters=80000, seed=1, sigma=1.0, refresh=2000):
    rng = np.random.default_rng(seed)
    cur = fe.init_incremental(legal)
    best = cur; hot = _hot_macros(fe, fe.ipos, nm)
    Ffield = None
    if "dir" in variant:
        Ffield = _field_force(fe, fe.ipos, "dens") + _field_force(fe, fe.ipos, "cong")
    hist = [(0, best)]

    def soft_move(directed):
        w = hot[nh:nm]; i = nh + int(rng.choice(nm - nh, p=w / w.sum()))
        if directed and Ffield is not None:
            d = Ffield[i]; n = np.hypot(*d)
            if n > 1e-9:
                s = abs(rng.normal(0, sigma))
                x, y = clampxy(i, fe.ipos[i, 0] + s * d[0] / n, fe.ipos[i, 1] + s * d[1] / n)
            else:
                x, y = clampxy(i, fe.ipos[i, 0] + rng.normal(0, sigma), fe.ipos[i, 1] + rng.normal(0, sigma))
        else:
            x, y = clampxy(i, fe.ipos[i, 0] + rng.normal(0, sigma), fe.ipos[i, 1] + rng.normal(0, sigma))
        return ("move", i, x, y)

    def swap_move(directed):
        if not GLIST:
            return None
        if directed:
            # sample a few hard macros, pick the one most misplaced vs its centroid
            cand = rng.integers(0, nh, size=8); bi, bm = None, -1
            for i in cand:
                cen = neighbor_centroid(fe, fe.ipos, int(i))
                if cen is None:
                    continue
                m = np.hypot(fe.ipos[i, 0] - cen[0], fe.ipos[i, 1] - cen[1])
                if m > bm:
                    bm, bi, bcen = m, int(i), cen
            if bi is None:
                return None
            grp = next((g for g in GLIST if bi in g), None)
            if grp is None:
                return None
            j = int(grp[np.argmin(np.linalg.norm(fe.ipos[grp] - np.array(bcen), axis=1))])
            i = bi
        else:
            g = GLIST[rng.integers(len(GLIST))]; i = int(rng.choice(g))
            cen = neighbor_centroid(fe, fe.ipos, i)
            if cen is None:
                return None
            j = int(g[np.argmin(np.linalg.norm(fe.ipos[g] - np.array(cen), axis=1))])
        if i == j:
            return None
        return ("swap", i, j, fe.ipos[i].copy(), fe.ipos[j].copy())

    def eval_action(a):
        if a is None:
            return 1e9, None, []
        if a[0] == "move":
            u = fe.apply_move(a[1], a[2], a[3]); c = fe.cost_current(); fe.undo_move(u)
            return c - cur, a, [a]
        else:
            _, i, j, pi, pj = a
            ui = fe.apply_move(i, pj[0], pj[1]); uj = fe.apply_move(j, pi[0], pi[1])
            c = fe.cost_current(); fe.undo_move(uj); fe.undo_move(ui)
            return c - cur, a, [a]

    def do(a):
        if a[0] == "move":
            fe.apply_move(a[1], a[2], a[3])
        else:
            _, i, j, pi, pj = a
            fe.apply_move(i, pj[0], pj[1]); fe.apply_move(j, pi[0], pi[1])

    for it in range(iters):
        if it and it % refresh == 0:
            hot = _hot_macros(fe, fe.ipos, nm)
            if "dir" in variant:
                Ffield = _field_force(fe, fe.ipos, "dens") + _field_force(fe, fe.ipos, "cong")
        cands = []
        if variant == "rand_soft":
            cands.append(eval_action(soft_move(False)))
        elif variant == "dir_soft":
            cands.append(eval_action(soft_move(True)))
        elif variant in ("bestof", "bestof_combined"):
            cands.append(eval_action(soft_move(True)))
        elif variant == "bestof_hedged":
            cands.append(eval_action(soft_move(False)))   # random (safety)
            cands.append(eval_action(soft_move(True)))    # directed
        if variant in ("bestof", "bestof_combined", "bestof_hedged"):
            cands.append(eval_action(swap_move(True)))
        if variant in ("bestof_combined", "bestof_hedged"):
            sm = soft_move(True); sw = swap_move(True)
            if sm and sw and sm[1] != sw[1] and sm[1] != sw[2]:
                u1 = fe.apply_move(sm[1], sm[2], sm[3])
                _, i, j, pi, pj = sw
                u2 = fe.apply_move(i, pj[0], pj[1]); u3 = fe.apply_move(j, pi[0], pi[1])
                c = fe.cost_current()
                fe.undo_move(u3); fe.undo_move(u2); fe.undo_move(u1)
                cands.append((c - cur, ("combo", sm, sw), [sm, sw]))
        cands = [c for c in cands if c[1] is not None]
        if not cands:
            continue
        cands.sort(key=lambda t: t[0])
        d, a, applylist = cands[0]
        if d < -1e-12:
            for act in applylist:
                do(act)
            cur += d
            if cur < best:
                best = cur
        if it and it % 4000 == 0:
            hist.append((it, best))
    hist.append((iters, best))
    return hist


VARIANTS = [("rand_soft", "#2563eb", "random soft-move"),
            ("dir_soft", "#16a34a", "directed soft-move"),
            ("bestof_combined", "#d97706", "best-of {dir soft, swap, combined}"),
            ("bestof_hedged", "#dc2626", "best-of {rand+dir soft, swap, combined} (hedged)")]
fig, ax = plt.subplots(figsize=(10, 6))
for v, col, lab in VARIANTS:
    t = time.time(); h = run(v)
    print(f"{v}: final {h[-1][1]:.4f} in {time.time()-t:.0f}s", flush=True)
    ax.plot([x[0] for x in h], [x[1] for x in h], color=col, lw=2, label=f"{lab} ({h[-1][1]:.4f})")
ax.set_xlabel("iterations"); ax.set_ylabel("proxy cost (fast_eval)")
ax.set_title("ibm01: random vs directed proposals (from reference)")
ax.grid(alpha=0.25); ax.legend()
fig.tight_layout(); fig.savefig("/home/laz/partcl/my-macro-placer/notes/directed_compare.png", dpi=100)
print("saved", flush=True)
