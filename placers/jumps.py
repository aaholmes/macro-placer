"""Targeted basin-hopping for the differentiable global stage (reusable).

Continuous Adam can't relocate a macro across the chip to the neighbours it's wired
to, so the random init's ordering survives. During the early phase these jumps
periodically collapse the most spread-out nets (high HPWL, small enough to not be a
hub) to their center + annealed noise, then let Adam relax the layout around them.

Empirically this HELPS wirelength-ish / hard-anchored designs (e.g. ibm09) and HURTS
soft-dominated / congestion-bound ones (e.g. ibm17), so it is meant to be used as an
*extra candidate source* in a best-of-N pool (selection gates it per benchmark), not
as a base-pipeline replacement.
"""
import numpy as np, torch
from collections import defaultdict

# net-collapse config that gave the clean ibm09 win (used as the pool default)
DEFAULT = dict(noise=0.05, nets=8, frac=0.50, every=500, cap=20, soft=False)


def _net_stats(P, coord):
    dev = P.dev; owner = P.owner; net = P.net; off = P.off
    nn = int(P.nw.shape[0])
    mv = owner >= 0
    px = torch.where(mv, coord[owner.clamp(min=0), 0] + off[:, 0], off[:, 0])
    py = torch.where(mv, coord[owner.clamp(min=0), 1] + off[:, 1], off[:, 1])
    def nred(src, red, init):
        return torch.full((nn,), init, device=dev).scatter_reduce(0, net, src, reduce=red, include_self=False)
    hpwl = (nred(px, "amax", -1e30) - nred(px, "amin", 1e30)) + (nred(py, "amax", -1e30) - nred(py, "amin", 1e30))
    cnt = torch.zeros(nn, device=dev).index_add(0, net, torch.ones_like(px))
    cx = torch.zeros(nn, device=dev).index_add(0, net, px) / cnt.clamp(min=1)
    cy = torch.zeros(nn, device=dev).index_add(0, net, py) / cnt.clamp(min=1)
    return hpwl, torch.stack([cx, cy], 1), cnt


def build_net_members(P):
    """net id -> tensor of movable member macro ids (built once per benchmark)."""
    o = P.owner.cpu().numpy(); nt = P.net.cpu().numpy()
    mm = defaultdict(set)
    for pin in range(len(o)):
        if o[pin] >= 0:
            mm[int(nt[pin])].add(int(o[pin]))
    return {n: torch.tensor(sorted(v), device=P.dev, dtype=torch.long) for n, v in mm.items() if len(v) >= 2}


def place_with_jumps(P, loss_fn, iters, lr, seed, jump=None, net_members=None):
    """Adam on loss_fn(P, coord, t) with periodic net-collapse jumps in the early
    window. jump=None reproduces plain P.place. Returns [N,2] positions (numpy)."""
    j = None if jump is None else {**DEFAULT, **jump}
    if j is not None and net_members is None:
        net_members = build_net_members(P)
    g = torch.Generator(device="cpu").manual_seed(seed)
    c0 = torch.rand(P.b.num_macros, 2, generator=g)
    c0[:, 0] = c0[:, 0] * (P.W - 2) + 1; c0[:, 1] = c0[:, 1] * (P.H - 2) + 1
    coord = c0.to(P.dev).requires_grad_(True)
    opt = torch.optim.Adam([coord], lr=lr)
    NH = P.b.num_hard_macros
    if j is not None:
        rng = torch.Generator(device=P.dev).manual_seed(seed + 7)
        win = int(j["frac"] * iters); nscale = j["noise"] * min(P.W, P.H)
    for it in range(iters):
        t = it / iters
        opt.zero_grad(); loss = loss_fn(P, coord, t); loss.backward(); opt.step()
        with torch.no_grad():
            coord[:, 0].clamp_(P.hw, P.W - P.hw); coord[:, 1].clamp_(P.hh, P.H - P.hh)
            if j is not None and it < win and it % j["every"] == 0:
                hpwl, ctr, cnt = _net_stats(P, coord)
                temp = 1.0 - it / max(1, win)
                hp = hpwl.clone(); hp[(cnt < 2) | (cnt > j["cap"])] = -1.0
                for n in torch.topk(hp, j["nets"]).indices.tolist():
                    mem = net_members.get(n)
                    if mem is not None and j["soft"]:
                        mem = mem[mem >= NH]
                    if mem is None or not mem.numel():
                        continue
                    noise = torch.randn(mem.numel(), 2, generator=rng, device=P.dev) * nscale * temp
                    np_ = ctr[n].unsqueeze(0) + noise
                    np_[:, 0].clamp_(P.hw[mem], P.W - P.hw[mem]); np_[:, 1].clamp_(P.hh[mem], P.H - P.hh[mem])
                    coord[mem] = np_
                    st = opt.state.get(coord)
                    if st:
                        st["exp_avg"][mem] = 0; st["exp_avg_sq"][mem] = 0
    return coord.detach().cpu().numpy()
