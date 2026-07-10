"""Differentiable analytical global placement (torch / GPU).

Objective (all smooth functions of module coordinates):
  - wirelength: per-net smooth HPWL via log-sum-exp (segment ops)
  - density:    Gaussian area splat onto a grid, penalized by SOFT-TOP-K
  - congestion: RUDY (each net spreads 1/W+1/H over its smooth bbox), SOFT-TOP-K

soft-top-k(field) = sum(softmax(field/tau) * field) -- a smooth "mean of the
hottest cells", matching the TILOS top-10%/top-5% semantics (as tau->0 it is
the max; larger tau averages more). Autograd gives the exact gradient.

Optimize all modules with Adam (overlap allowed), anneal the density weight,
then legalize hard macros and score the REAL proxy externally.
"""
from __future__ import annotations
import numpy as np
import torch


class AnalyticalPlacer:
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
        self.cxs = torch.arange(self.gc, device=d, dtype=torch.float32) * self.gw + self.gw / 2
        self.cys = torch.arange(self.gr, device=d, dtype=torch.float32) * self.gh + self.gh / 2

    def _pin_xy(self, coord):
        oc = self.owner.clamp(min=0)
        mv = (self.owner >= 0).float().unsqueeze(1)
        base = coord[oc] * mv                       # movable: coord; fixed: 0
        return base + self.off                      # off is offset (mv) or abs (fixed)

    def _seg_smoothmax(self, val, gamma):
        """per-net gamma*log sum exp(val/gamma), numerically stable."""
        m = torch.full((self.nnet,), -1e30, device=self.dev)
        m = m.scatter_reduce(0, self.net, val, reduce="amax", include_self=True)
        e = torch.exp((val - m[self.net]) / gamma)
        s = torch.zeros(self.nnet, device=self.dev).scatter_add(0, self.net, e)
        return m + gamma * torch.log(s + 1e-30)

    def _net_bbox(self, pin_x, pin_y, gamma):
        xmax = self._seg_smoothmax(pin_x, gamma); xmin = -self._seg_smoothmax(-pin_x, gamma)
        ymax = self._seg_smoothmax(pin_y, gamma); ymin = -self._seg_smoothmax(-pin_y, gamma)
        return xmin, xmax, ymin, ymax

    def _softtopk(self, field, tau):
        f = field.reshape(-1)
        w = torch.softmax(f / tau, dim=0)
        return (w * f).sum()

    def loss(self, coord, gamma, lam_d, lam_c, tau_d, tau_c):
        px = self._pin_xy(coord)
        pin_x, pin_y = px[:, 0], px[:, 1]
        xmin, xmax, ymin, ymax = self._net_bbox(pin_x, pin_y, gamma)
        # wirelength (normalized like the real proxy)
        wl = (self.nw * ((xmax - xmin) + (ymax - ymin))).sum() / ((self.W + self.H) * self.net_cnt)
        # density: separable Gaussian area splat -> grid
        sx = torch.clamp(self.hw, min=self.gw); sy = torch.clamp(self.hh, min=self.gh)
        dxc = coord[:, 0].unsqueeze(1) - self.cxs                     # [M, gc]
        dyc = coord[:, 1].unsqueeze(1) - self.cys                     # [M, gr]
        gx = torch.exp(-(dxc ** 2) / (2 * sx.unsqueeze(1) ** 2))
        gy = torch.exp(-(dyc ** 2) / (2 * sy.unsqueeze(1) ** 2))
        gx = gx / (gx.sum(1, keepdim=True) + 1e-9)
        gy = gy / (gy.sum(1, keepdim=True) + 1e-9)
        rho = torch.einsum("m,mr,mc->rc", self.area, gy, gx) / (self.gw * self.gh)
        # penalize only OVERFLOW above target density (else it over-spreads and
        # fights wirelength in already-fine regions).
        dens = self._softtopk(torch.clamp(rho - self.rho_target, min=0.0), tau_d)
        # congestion: RUDY over smooth bbox, soft-top-k
        Wn = torch.clamp(xmax - xmin, min=self.gw); Hn = torch.clamp(ymax - ymin, min=self.gh)
        dnet = (1.0 / Wn + 1.0 / Hn) * self.nw
        s = self.gw
        covx = torch.sigmoid((self.cxs.unsqueeze(0) - xmin.unsqueeze(1)) / s) * \
               torch.sigmoid((xmax.unsqueeze(1) - self.cxs.unsqueeze(0)) / s)   # [nets, gc]
        covy = torch.sigmoid((self.cys.unsqueeze(0) - ymin.unsqueeze(1)) / self.gh) * \
               torch.sigmoid((ymax.unsqueeze(1) - self.cys.unsqueeze(0)) / self.gh)  # [nets, gr]
        R = torch.einsum("n,nr,nc->rc", dnet, covy, covx)
        cong = self._softtopk(R, tau_c)
        total = wl + lam_d * dens + lam_c * cong
        return total, wl.item(), dens.item(), cong.item()

    def place(self, pos0, iters=300, lr=0.3, gamma=None, lam_d0=1.0, lam_d1=8.0,
              lam_c=0.0, tau_d=0.05, tau_c=0.02, rho_target=1.0, logf=print):
        gamma = gamma or self.gw
        self.rho_target = rho_target
        coord = torch.tensor(np.asarray(pos0, np.float32)[:, :2], device=self.dev,
                             requires_grad=True)
        opt = torch.optim.Adam([coord], lr=lr)
        for it in range(iters):
            lam_d = lam_d0 + (lam_d1 - lam_d0) * (it / iters)    # anneal spreading up
            opt.zero_grad()
            total, wl, dn, cg = self.loss(coord, gamma, lam_d, lam_c, tau_d, tau_c)
            total.backward()
            opt.step()
            with torch.no_grad():                                 # keep in-canvas
                coord[:, 0].clamp_(self.hw, self.W - self.hw)
                coord[:, 1].clamp_(self.hh, self.H - self.hh)
            if it % max(1, iters // 10) == 0:
                logf(f"  it={it:4d} lam_d={lam_d:.2f} loss={total.item():.4f} "
                     f"wl={wl:.4f} dens={dn:.3f} cong={cg:.3f}")
        out = pos0.copy()
        out[:, :2] = coord.detach().cpu().numpy()
        return out
