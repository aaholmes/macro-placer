"""Differentiable, non-gameable L-route congestion surrogate.

Mirrors the TILOS metric's structure (the reason it is non-gameable, unlike RUDY):
  - per net, deposit demand on the L-route SKELETON (source row + sink column),
    a CONSTANT weight per net (not ~1/bbox-area) -> spreading a net crosses more
    cells and RAISES congestion, exactly as in the real metric;
  - hard macros add routing blockage on the cells they cover;
  - score the PEAK cells (soft top-k), not the mean.

All functions are pure torch and differentiable w.r.t. the geometry inputs.
"""
import torch


def softrow(centers, val, cell):
    """Partition-of-unity soft one-hot over `centers` (length K) at `val` (length
    N). Returns [N, K] summing to ~1 per net -> total deposited demand conserved."""
    d = centers[None, :] - val[:, None]
    g = torch.exp(-(d ** 2) / (2.0 * cell ** 2))
    return g / (g.sum(1, keepdim=True) + 1e-9)


def softseg(centers, lo, hi, cell):
    """Soft indicator ~1 for centers (K) inside [lo, hi] (each length N). [N, K].
    A wider [lo, hi] turns on more cells -> larger sum (this is the non-gameable
    ingredient: longer routes deposit more total demand)."""
    return (torch.sigmoid((centers[None, :] - lo[:, None]) / cell) *
            torch.sigmoid((hi[:, None] - centers[None, :]) / cell))


def lroute_field(xmin, xmax, ymin, ymax, nw, cxs, cys, gw, gh, tau_seg=1.0):
    """L-route demand for each net's soft bbox. Returns (H, V) each [R, C].
    H: horizontal segment on the source row (ymin) spanning cols [xmin, xmax].
    V: vertical segment on the sink column (xmax) spanning rows [ymin, ymax]."""
    rowk = softrow(cys, ymin, gh)                     # [N, R] source row
    colseg = softseg(cxs, xmin, xmax, gw * tau_seg)   # [N, C] col span
    H = torch.einsum("n,nr,nc->rc", nw, rowk, colseg)
    colk = softrow(cxs, xmax, gw)                     # [N, C] sink column
    rowseg = softseg(cys, ymin, ymax, gh * tau_seg)   # [N, R] row span
    V = torch.einsum("n,nc,nr->rc", nw, colk, rowseg)
    return H, V


def macro_blockage_field(cx, cy, hw, hh, weight, cxs, cys, gw, gh):
    """Routing blockage from hard macros: demand on the cells each macro covers.
    cx,cy,hw,hh,weight: [M]. Returns [R, C]. (Where hard-macro position bites.)"""
    covx = softseg(cxs, cx - hw, cx + hw, gw)         # [M, C]
    covy = softseg(cys, cy - hh, cy + hh, gh)         # [M, R]
    return torch.einsum("m,mr,mc->rc", weight, covy, covx)


def topk_mean(field, frac, tau, steps=30):
    """Differentiable mean of the top `frac` fraction of `field` (peak congestion).
    Threshold found by bisection so the soft mask sums to k=frac*N; gradient flows
    through the mask and values (threshold treated as a constant, as in a stop-grad)."""
    x = field.flatten()
    n = x.numel(); k = max(1.0, frac * n)
    lo = float(x.min().detach()); hi = float(x.max().detach())
    if hi <= lo:
        return x.mean()
    for _ in range(steps):
        thr = 0.5 * (lo + hi)
        if float(torch.sigmoid((x - thr) / tau).sum().detach()) > k:
            lo = thr
        else:
            hi = thr
    thr = 0.5 * (lo + hi)
    m = torch.sigmoid((x - thr) / tau)
    return (m * x).sum() / (m.sum() + 1e-9)
