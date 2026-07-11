"""Differentiable analytical global placement (PyTorch / GPU).

The placer optimizes module coordinates by autograd + Adam against ANY
differentiable loss function of those coordinates -- the objective is a
pluggable `loss_fn(placer, coord, t)`. It ships differentiable primitives that
compose the standard placement objective, but nothing about the optimizer is
specific to them:

  primitives (all differentiable functions of `coord`):
    wirelength(coord, gamma)      smooth HPWL via segment log-sum-exp
    density(coord, tau, target)   Gaussian area splat -> soft-top-k overflow
    congestion(coord, tau)        RUDY over smooth net bboxes -> soft-top-k

  soft-top-k(field) = sum(softmax(field/tau) * field): a smooth "mean of the
  hottest cells", matching top-k metrics (as tau->0 -> max; larger -> average).

Global placement anneals two schedule parameters (exposed to the loss via `t`
in [0,1]): gamma (wirelength smoothing, large->small to go from a smooth global
gradient to sharp HPWL) and the density weight (small->large to cluster first,
then spread). See `proxy_loss` for the default objective; swap in any other
differentiable loss to place for a different metric.
"""
from __future__ import annotations
import math
import numpy as np
import torch


def softtopk(field, tau):
    """Crude soft-top-k: softmax-weighted mean (temperature tau conflates with k)."""
    f = field.reshape(-1)
    return (torch.softmax(f / tau, dim=0) * f).sum()


def _lap_cdf(u):
    # overflow-safe Laplace CDF: 0.5*exp(u) (u<0) / 1-0.5*exp(-u) (u>=0), written
    # with only exp(-|u|) so the torch.where double-branch can't produce inf grads.
    a = torch.exp(-torch.abs(u))
    return 0.5 * (1.0 + torch.sign(u) * (1.0 - a))


def lapsum_topk_mean(field, frac, tau, steps=40):
    """Differentiable mean of the top-(frac*n) values via LapSum (arXiv:2503.06242).
    Mask m_i = F_Laplace((x_i - t)/tau) with threshold t solved so sum(m)=k=frac*n;
    returns sum(m*x)/k (as tau->0, the exact top-k mean). The threshold is solved
    without grad (Newton), then x-dependence is reattached via the implicit
    function theorem so autograd carries dt/dx = -g_x/g_t cleanly."""
    x = field.reshape(-1); n = x.numel(); k = frac * n
    with torch.no_grad():
        t = torch.quantile(x, 1.0 - frac)
        for _ in range(steps):
            u = (x - t) / tau
            S = _lap_cdf(u).sum()
            gt = -(0.5 * torch.exp(-torch.abs(u))).sum() / tau   # dS/dt
            t = t - (S - k) / (gt - 1e-12)
        gt = -(0.5 * torch.exp(-torch.abs((x - t) / tau))).sum() / tau
    g = _lap_cdf((x - t) / tau).sum() - k          # ~0 in value, differentiable in x
    t_diff = t - g / (gt - 1e-12)                  # implicit reattach: dt/dx = -g_x/g_t
    m = _lap_cdf((x - t_diff) / tau)
    return (m * x).sum() / k


class DifferentiablePlacer:
    def __init__(self, fe, device=None):
        self.fe = fe; self.b = fe.b
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(fe, "cong_nets"):
            fe._build_congestion_ir()
        d = self.dev
        self.owner = torch.tensor(fe.pin_owner, device=d, dtype=torch.long)
        self.off = torch.tensor(fe.pin_off, device=d, dtype=torch.float32)
        self.net = torch.tensor(fe.pin_net, device=d, dtype=torch.long)
        self.nw = torch.tensor(fe.net_weight, device=d, dtype=torch.float32)
        self.nnet = fe.num_nets
        self.W = float(fe.W); self.H = float(fe.H); self.net_cnt = float(fe.net_cnt)
        self.gc = fe.plc.grid_col; self.gr = fe.plc.grid_row
        self.gw = self.W / self.gc; self.gh = self.H / self.gr
        sz = self.b.macro_sizes.numpy()
        self.area = torch.tensor(sz[:, 0] * sz[:, 1], device=d, dtype=torch.float32)
        self.hw = torch.tensor(sz[:, 0] / 2, device=d, dtype=torch.float32)
        self.hh = torch.tensor(sz[:, 1] / 2, device=d, dtype=torch.float32)
        self.nh = self.b.num_hard_macros
        self.util = float((sz[:, 0] * sz[:, 1]).sum() / (self.W * self.H))
        self.cxs = torch.arange(self.gc, device=d, dtype=torch.float32) * self.gw + self.gw / 2
        self.cys = torch.arange(self.gr, device=d, dtype=torch.float32) * self.gh + self.gh / 2

    # ---- pin positions & smooth per-net bbox ----
    def _pin_xy(self, coord):
        oc = self.owner.clamp(min=0)
        mv = (self.owner >= 0).float().unsqueeze(1)
        return coord[oc] * mv + self.off

    def _seg_smoothmax(self, val, gamma):
        m = torch.full((self.nnet,), -1e30, device=self.dev).scatter_reduce(
            0, self.net, val, reduce="amax", include_self=True)
        e = torch.exp((val - m[self.net]) / gamma)
        s = torch.zeros(self.nnet, device=self.dev).scatter_add(0, self.net, e)
        return m + gamma * torch.log(s + 1e-30)

    def _bbox(self, coord, gamma):
        p = self._pin_xy(coord); x, y = p[:, 0], p[:, 1]
        xmax = self._seg_smoothmax(x, gamma); xmin = -self._seg_smoothmax(-x, gamma)
        ymax = self._seg_smoothmax(y, gamma); ymin = -self._seg_smoothmax(-y, gamma)
        return xmin, xmax, ymin, ymax

    # ---- differentiable primitives ----
    def wirelength(self, coord, gamma):
        xmin, xmax, ymin, ymax = self._bbox(coord, gamma)
        return (self.nw * ((xmax - xmin) + (ymax - ymin))).sum() / ((self.W + self.H) * self.net_cnt)

    def density_field(self, coord):
        """Per-cell density rho (Gaussian area splat), differentiable."""
        sx = torch.clamp(self.hw, min=self.gw); sy = torch.clamp(self.hh, min=self.gh)
        gx = torch.exp(-((coord[:, 0:1] - self.cxs) ** 2) / (2 * sx[:, None] ** 2))
        gy = torch.exp(-((coord[:, 1:2] - self.cys) ** 2) / (2 * sy[:, None] ** 2))
        gx = gx / (gx.sum(1, keepdim=True) + 1e-9); gy = gy / (gy.sum(1, keepdim=True) + 1e-9)
        return torch.einsum("m,mr,mc->rc", self.area, gy, gx) / (self.gw * self.gh)

    def rudy_field(self, coord):
        """Per-cell RUDY congestion estimate, differentiable."""
        xmin, xmax, ymin, ymax = self._bbox(coord, 0.5 * self.gw)
        Wn = torch.clamp(xmax - xmin, min=self.gw); Hn = torch.clamp(ymax - ymin, min=self.gh)
        dnet = (1.0 / Wn + 1.0 / Hn) * self.nw
        covx = torch.sigmoid((self.cxs[None] - xmin[:, None]) / self.gw) * \
               torch.sigmoid((xmax[:, None] - self.cxs[None]) / self.gw)
        covy = torch.sigmoid((self.cys[None] - ymin[:, None]) / self.gh) * \
               torch.sigmoid((ymax[:, None] - self.cys[None]) / self.gh)
        return torch.einsum("n,nr,nc->rc", dnet, covy, covx)

    def hard_overlap(self, coord):
        """Sum of squared PENETRATION DEPTHS over hard-hard macro pairs (the
        min-axis separation a legalizer would translate). Differentiable; zero
        iff the hard placement is legal. Soft macros excluded (may overlap)."""
        nh = self.nh
        x = coord[:nh, 0]; y = coord[:nh, 1]
        dx = (x[:, None] - x[None, :]).abs()
        dy = (y[:, None] - y[None, :]).abs()
        penx = torch.clamp((self.hw[:nh, None] + self.hw[None, :nh]) - dx, min=0.0)
        peny = torch.clamp((self.hh[:nh, None] + self.hh[None, :nh]) - dy, min=0.0)
        pen = torch.minimum(penx, peny)                 # penetration depth
        mask = torch.triu(torch.ones(nh, nh, device=self.dev), 1)
        return ((pen ** 2) * mask).sum()

    def hard_overlap_count(self, coord):
        """(non-diff) number of strictly-overlapping hard pairs, for reporting."""
        nh = self.nh
        with torch.no_grad():
            x = coord[:nh, 0]; y = coord[:nh, 1]
            ox = (self.hw[:nh, None] + self.hw[None, :nh]) - (x[:, None] - x[None, :]).abs()
            oy = (self.hh[:nh, None] + self.hh[None, :nh]) - (y[:, None] - y[None, :]).abs()
            m = ((ox > 0) & (oy > 0)) & torch.triu(torch.ones(nh, nh, device=self.dev), 1).bool()
            return int(m.sum())

    def density(self, coord, tau=0.05, target=None, topk="softmax"):
        target = self.util if target is None else target
        over = torch.clamp(self.density_field(coord) - target, min=0.0)
        return lapsum_topk_mean(over, 0.10, tau) if topk == "lapsum" else softtopk(over, tau)

    def congestion(self, coord, tau=0.02, topk="softmax"):
        R = self.rudy_field(coord)
        return lapsum_topk_mean(R, 0.05, tau) if topk == "lapsum" else softtopk(R, tau)

    # ---- generic optimizer over any differentiable loss ----
    def place(self, loss_fn, pos0=None, iters=500, lr=0.05, logf=print):
        """Adam gradient descent on loss_fn(self, coord, t), t in [0,1].
        pos0=None -> spread start (from-scratch). Returns [N,2] positions."""
        if pos0 is None:
            g = torch.Generator(device="cpu").manual_seed(0)
            c0 = torch.rand(self.b.num_macros, 2, generator=g)
            c0[:, 0] = c0[:, 0] * (self.W - 2) + 1; c0[:, 1] = c0[:, 1] * (self.H - 2) + 1
            coord = c0.to(self.dev).requires_grad_(True)
        else:
            coord = torch.tensor(np.asarray(pos0, np.float32)[:, :2], device=self.dev,
                                 requires_grad=True)
        opt = torch.optim.Adam([coord], lr=lr)
        for it in range(iters):
            t = it / iters
            opt.zero_grad()
            loss = loss_fn(self, coord, t)
            loss.backward(); opt.step()
            with torch.no_grad():
                coord[:, 0].clamp_(self.hw, self.W - self.hw)
                coord[:, 1].clamp_(self.hh, self.H - self.hh)
            if logf and it % max(1, iters // 10) == 0:
                logf(f"  it={it:4d} t={t:.2f} loss={float(loss):.4f}")
        out = (pos0.copy() if pos0 is not None
               else np.zeros((self.b.num_macros, 2), np.float64))
        out = np.asarray(out, np.float64)
        if out.shape[1] < 2:
            out = np.zeros((self.b.num_macros, 2))
        out[:, :2] = coord.detach().cpu().numpy()
        return out


def anneal(t, lo, hi, geometric=False):
    return lo * (hi / lo) ** t if geometric else lo + (hi - lo) * t


def proxy_loss(gamma_hi_mult=10.0, lam_d0=0.0025, lam_d1=0.05, lam_c=0.0,
               lam_ov0=0.0, lam_ov1=0.0, tau_d=0.05, tau_c=0.02, target=0.8,
               topk="softmax"):
    """Default objective = challenge proxy, with annealing over t:
      gamma  (WL smoothing) large->small,  lam_d (density) small->large,
      lam_ov (hard-overlap penalty, Form A) small->large so macros pass through
             early and become legal by the end (legalizer-free).
    `topk` selects the soft-top-k operator ('softmax' | 'lapsum')."""
    def loss(P, coord, t):
        gamma = anneal(t, P.gw * gamma_hi_mult, P.gw, geometric=True)
        lam_d = anneal(t, lam_d0, lam_d1)
        wl = P.wirelength(coord, gamma)
        de = P.density(coord, tau=tau_d, target=target, topk=topk)
        L = wl + lam_d * de
        if lam_c:
            L = L + lam_c * P.congestion(coord, tau=tau_c, topk=topk)
        if lam_ov1 or lam_ov0:
            lam_ov = anneal(t, max(lam_ov0, 1e-9), max(lam_ov1, 1e-9), geometric=True)
            L = L + lam_ov * P.hard_overlap(coord)
        return L
    return loss
