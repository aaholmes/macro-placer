"""Fast, incremental reimplementation of the TILOS proxy cost.

The TILOS evaluator (plc_client_os.py) recomputes every net and grid cell in
pure-Python triple loops (~1 s/eval on ibm01). Local search needs 10^4-10^6
evals, so we extract a static array-based IR from `plc` once, then evaluate
wirelength / density / congestion with numpy — and update only the parts that a
single-macro move changes.

This file is validated component-by-component against the TILOS ground truth
(see scripts/validate_fast_eval.py). Layer 1 here is wirelength.
"""
from __future__ import annotations
import numpy as np

_EMPTY = np.array([], dtype=np.int64)


def _pin_position(plc, pin_idx):
    """Mirror plc.__get_pin_position: PORT = own pos; MACRO_PIN = ref pos + offset."""
    m = plc.modules_w_pins[pin_idx]
    if m.get_type() == "PORT":
        x, y = m.get_pos()
        return None, float(x), float(y)  # owner None = fixed
    ref = plc.get_ref_node_id(pin_idx)
    ox, oy = m.get_offset()
    return ref, float(ox), float(oy)  # owner plc-idx, offset


class FastEval:
    def __init__(self, benchmark, plc):
        self.b = benchmark
        self.plc = plc
        self.W = float(plc.width)
        self.H = float(plc.height)
        self.net_cnt = float(plc.net_cnt)

        # ---- map plc module index -> tensor index (0..num_macros-1) ----
        plc_to_tensor = {}
        for tidx, pidx in enumerate(benchmark.hard_macro_indices):
            plc_to_tensor[pidx] = tidx
        for k, pidx in enumerate(benchmark.soft_macro_indices):
            plc_to_tensor[pidx] = benchmark.num_hard_macros + k
        self.plc_to_tensor = plc_to_tensor

        self._build_wirelength_ir()

    # ------------------------------------------------------------------ WL
    def _build_wirelength_ir(self):
        """Flatten nets into per-pin arrays for vectorized HPWL."""
        plc = self.plc
        pin_net = []        # net id per pin entry
        pin_owner = []      # tensor idx of owning macro, or -1 if fixed
        pin_off = []        # (ox, oy) offset (macro pin) or absolute pos (fixed)
        net_weight = []
        for net_id, driver_name in enumerate(plc.nets.keys()):
            driver_idx = plc.mod_name_to_indices[driver_name]
            net_weight.append(float(plc.modules_w_pins[driver_idx].get_weight()))
            members = [driver_idx] + [plc.mod_name_to_indices[s] for s in plc.nets[driver_name]]
            for pidx in members:
                owner, a, bb = _pin_position(plc, pidx)
                if owner is None:
                    pin_owner.append(-1); pin_off.append((a, bb))
                else:
                    pin_owner.append(self.plc_to_tensor[owner]); pin_off.append((a, bb))
                pin_net.append(net_id)
        self.num_nets = len(net_weight)
        self.pin_net = np.asarray(pin_net, dtype=np.int64)
        self.pin_owner = np.asarray(pin_owner, dtype=np.int64)
        self.pin_off = np.asarray(pin_off, dtype=np.float64)
        self.net_weight = np.asarray(net_weight, dtype=np.float64)
        # macro tensor idx -> set of nets it touches (for incremental updates)
        self.macro_nets = {}
        movable = self.pin_owner >= 0
        for owner, net in zip(self.pin_owner[movable], self.pin_net[movable]):
            self.macro_nets.setdefault(int(owner), set()).add(int(net))
        self.macro_nets = {k: np.array(sorted(v), dtype=np.int64)
                           for k, v in self.macro_nets.items()}

    def _pin_xy(self, pos):
        """Absolute pin coords given macro centers pos[N,2]."""
        owner = self.pin_owner
        x = np.where(owner >= 0, pos[np.clip(owner, 0, None), 0] + self.pin_off[:, 0],
                     self.pin_off[:, 0])
        y = np.where(owner >= 0, pos[np.clip(owner, 0, None), 1] + self.pin_off[:, 1],
                     self.pin_off[:, 1])
        return x, y

    def wirelength_sum(self, pos):
        """Total weighted HPWL (matches plc.get_wirelength())."""
        pos = np.asarray(pos, dtype=np.float64)
        x, y = self._pin_xy(pos)
        n = self.num_nets
        xmax = np.full(n, -np.inf); xmin = np.full(n, np.inf)
        ymax = np.full(n, -np.inf); ymin = np.full(n, np.inf)
        np.maximum.at(xmax, self.pin_net, x); np.minimum.at(xmin, self.pin_net, x)
        np.maximum.at(ymax, self.pin_net, y); np.minimum.at(ymin, self.pin_net, y)
        hpwl = (xmax - xmin) + (ymax - ymin)
        return float(np.sum(self.net_weight * hpwl))

    def wirelength_cost(self, pos):
        return self.wirelength_sum(pos) / ((self.W + self.H) * self.net_cnt)

    # ------------------------------------------------------------- DENSITY
    def _grid_params(self):
        gc, gr = self.plc.grid_col, self.plc.grid_row
        gw = self.W / gc; gh = self.H / gr
        return gc, gr, gw, gh

    def _macro_cell_area(self, cx, cy, w, h):
        """Return (rows, cols, area[r,c]) a macro block contributes to each covered
        cell — faithful to TILOS __add_module_to_grid_cells corner/OOB logic."""
        gc, gr, gw, gh = self._grid_params()
        import math
        xmin, xmax = cx - w / 2, cx + w / 2
        ymin, ymax = cy - h / 2, cy + h / 2
        ur_row = math.floor(ymax / gh); ur_col = math.floor(xmax / gw)
        bl_row = math.floor(ymin / gh); bl_col = math.floor(xmin / gw)
        if not (ur_row >= 0 and ur_col >= 0):
            return None
        bl_row = max(bl_row, 0); bl_col = max(bl_col, 0)
        if not (bl_row >= 0 and bl_col >= 0):
            return None
        ur_row = min(ur_row, gr - 1); ur_col = min(ur_col, gc - 1)
        rows = np.arange(bl_row, ur_row + 1)
        cols = np.arange(bl_col, ur_col + 1)
        xo = np.clip(np.minimum(xmax, (cols + 1) * gw) - np.maximum(xmin, cols * gw), 0, None)
        yo = np.clip(np.minimum(ymax, (rows + 1) * gh) - np.maximum(ymin, rows * gh), 0, None)
        return rows, cols, np.outer(yo, xo)  # area[r,c] = yo[r]*xo[c]

    def build_density(self, pos):
        """Full rasterization of all macros -> grid_occupied (area per cell)."""
        gc, gr, gw, gh = self._grid_params()
        occ = np.zeros(gr * gc, dtype=np.float64)
        sz = self.b.macro_sizes.numpy()
        for i in range(self.b.num_macros):
            r = self._macro_cell_area(pos[i, 0], pos[i, 1], sz[i, 0], sz[i, 1])
            if r is None:
                continue
            rows, cols, area = r
            occ[(rows[:, None] * gc + cols[None, :]).ravel()] += area.ravel()
        self.grid_occupied = occ
        self.grid_area = gw * gh
        return occ

    def density_cost_from_occ(self, occ):
        gc, gr, gw, gh = self._grid_params()
        cells = occ / (gw * gh)
        occupied = cells[cells != 0.0]
        ncells = gr * gc
        cnt = int(np.floor(ncells * 0.1))
        if ncells < 10:
            return 0.5 * float(occupied.mean())
        k = min(cnt, len(occupied))
        # top-k sum via partition (no full sort needed)
        top = occupied if k == len(occupied) else np.partition(occupied, len(occupied) - k)[len(occupied) - k:]
        return 0.5 * float(top.sum() / cnt)

    def density_cost(self, pos):
        return self.density_cost_from_occ(self.build_density(np.asarray(pos, np.float64)))

    # --------------------------------------------------------- CONGESTION
    def _build_congestion_ir(self):
        """One 'congestion net' per module that has sinks (PORT or MACRO_PIN),
        exactly as TILOS get_routing iterates. Each stores its member pins as
        (owner tensor idx or -1, offx/absx, offy/absy) and the source's position
        within the member list."""
        plc = self.plc
        nets = []          # list of dict(members=[(owner,ax,ay)...], src=0, weight)
        hard_set = set(self.b.hard_macro_indices)
        for pidx, m in enumerate(plc.modules_w_pins):
            t = m.get_type()
            if t not in ("PORT", "MACRO_PIN"):
                continue
            sink = m.get_sink()
            if not sink:
                continue
            weight = 1.0
            if t == "MACRO_PIN" and m.get_weight() > 1:
                weight = float(m.get_weight())
            members = []
            # source pin first
            o, ax, ay = _pin_position(plc, pidx)
            members.append((-1 if o is None else self.plc_to_tensor[o], ax, ay))
            # sinks
            if t == "PORT":
                for sname in sink:
                    for sp in sink[sname]:
                        si = plc.mod_name_to_indices[sp]
                        o, ax, ay = _pin_position(plc, si)
                        members.append((-1 if o is None else self.plc_to_tensor[o], ax, ay))
            else:
                for lst in sink.values():
                    for sname in lst:
                        si = plc.mod_name_to_indices[sname]
                        o, ax, ay = _pin_position(plc, si)
                        members.append((-1 if o is None else self.plc_to_tensor[o], ax, ay))
            nets.append({"members": members, "weight": weight})
        self.cong_nets = nets
        # macro (tensor idx) -> congestion nets it touches
        self.macro_cong_nets = {}
        for nid, net in enumerate(nets):
            for (owner, _, _) in net["members"]:
                if owner >= 0:
                    self.macro_cong_nets.setdefault(owner, set()).add(nid)
        # hard macros in tensor space (for shadows) = [0, num_hard)
        gc, gr, gw, gh = self._grid_params()
        self.grid_v_routes = gw * plc.vroutes_per_micron
        self.grid_h_routes = gh * plc.hroutes_per_micron
        self.vrouting_alloc = float(plc.vrouting_alloc)
        self.hrouting_alloc = float(plc.hrouting_alloc)
        self.smooth_range = int(plc.smooth_range)

    def _gcell(self, x, y):
        gc, gr, gw, gh = self._grid_params()
        import math
        col = int(math.floor(x / gw)); row = int(math.floor(y / gh))
        return max(0, min(row, gr - 1)), max(0, min(col, gc - 1))

    def _net_gcells(self, net, pos):
        gset = []
        seen = set()
        for (owner, ax, ay) in net["members"]:
            if owner >= 0:
                x = pos[owner, 0] + ax; y = pos[owner, 1] + ay
            else:
                x = ax; y = ay
            rc = self._gcell(x, y)
            if rc not in seen:
                seen.add(rc); gset.append(rc)
        src = self._gcell(*( (pos[net["members"][0][0], 0] + net["members"][0][1],
                              pos[net["members"][0][0], 1] + net["members"][0][2])
                             if net["members"][0][0] >= 0
                             else (net["members"][0][1], net["members"][0][2]) ))
        return gset, src

    def _route_net(self, gset, src, weight, Hraw, Vraw, sign=1.0):
        """Deposit (sign*weight) into Hraw/Vraw exactly as TILOS routes a net."""
        gc = self.plc.grid_col
        w = sign * weight
        n = len(gset)
        def H(r, c): Hraw[r * gc + c] += w
        def V(r, c): Vraw[r * gc + c] += w
        if n == 2:
            self._two_pin(src, gset, w, Hraw, Vraw)
        elif n == 3:
            self._three_pin(list(gset), w, Hraw, Vraw)
        elif n > 3:
            for node in gset:
                if node != src:
                    self._two_pin(src, [src, node], w, Hraw, Vraw)

    def _two_pin(self, src, gset, w, Hraw, Vraw):
        gc = self.plc.grid_col
        sink = gset[1] if gset[0] == src else gset[0]
        rmin, rmax = min(sink[0], src[0]), max(sink[0], src[0])
        cmin, cmax = min(sink[1], src[1]), max(sink[1], src[1])
        for c in range(cmin, cmax):
            Hraw[src[0] * gc + c] += w
        for r in range(rmin, rmax):
            Vraw[r * gc + sink[1]] += w

    def _three_pin(self, node_gcells, w, Hraw, Vraw):
        gc = self.plc.grid_col
        g = sorted(node_gcells, key=lambda p: (p[1], p[0]))
        (y1, x1), (y2, x2), (y3, x3) = g[0], g[1], g[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self._l_routing(g, w, Hraw, Vraw)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for c in range(x1, x2): Hraw[y1 * gc + c] += w
            for r in range(y1, max(y2, y3)): Vraw[r * gc + x2] += w
        elif y2 == y3:
            for c in range(x1, x2): Hraw[y1 * gc + c] += w
            for c in range(x2, x3): Hraw[y2 * gc + c] += w
            for r in range(min(y2, y1), max(y2, y1)): Vraw[r * gc + x2] += w
        else:
            self._t_routing(g, w, Hraw, Vraw)

    def _l_routing(self, g, w, Hraw, Vraw):
        gc = self.plc.grid_col
        gg = sorted(g, key=lambda p: (p[1], p[0]))
        (y1, x1), (y2, x2), (y3, x3) = gg[0], gg[1], gg[2]
        for c in range(x1, x2): Hraw[y1 * gc + c] += w
        for c in range(x2, x3): Hraw[y2 * gc + c] += w
        for r in range(min(y1, y2), max(y1, y2)): Vraw[r * gc + x2] += w
        for r in range(min(y2, y3), max(y2, y3)): Vraw[r * gc + x3] += w

    def _t_routing(self, g, w, Hraw, Vraw):
        gc = self.plc.grid_col
        gg = sorted(g)
        (y1, x1), (y2, x2), (y3, x3) = gg[0], gg[1], gg[2]
        xmin, xmax = min(x1, x2, x3), max(x1, x2, x3)
        for c in range(xmin, xmax): Hraw[y2 * gc + c] += w
        for r in range(min(y1, y2), max(y1, y2)): Vraw[r * gc + x1] += w
        for r in range(min(y2, y3), max(y2, y3)): Vraw[r * gc + x3] += w

    def _macro_shadow(self, cx, cy, w, h, Hmac, Vmac, sign=1.0):
        """TILOS __macro_route_over_grid_cell (with partial-overlap edge fix)."""
        gc, gr, gw, gh = self._grid_params()
        import math
        xmin, xmax = cx - w / 2, cx + w / 2
        ymin, ymax = cy - h / 2, cy + h / 2
        ur_row = math.floor(ymax / gh); ur_col = math.floor(xmax / gw)
        bl_row = math.floor(ymin / gh); bl_col = math.floor(xmin / gw)
        if not (ur_row >= 0 and ur_col >= 0):
            return
        bl_row = max(bl_row, 0); bl_col = max(bl_col, 0)
        if not (bl_row >= 0 and bl_col >= 0):
            return
        ur_row = min(ur_row, gr - 1); ur_col = min(ur_col, gc - 1)

        def xy_dist(r, c):
            # mirror TILOS __overlap_dist: BOTH zero unless both axes strictly overlap
            xd = min(xmax, (c + 1) * gw) - max(xmin, c * gw)
            yd = min(ymax, (r + 1) * gh) - max(ymin, r * gh)
            if xd > 0 and yd > 0:
                return xd, yd
            return 0.0, 0.0

        pv = ph = False
        for r in range(bl_row, ur_row + 1):
            for c in range(bl_col, ur_col + 1):
                xd, yd = xy_dist(r, c)
                if ur_row != bl_row and ((r == bl_row and abs(yd - gh) > 1e-5) or
                                         (r == ur_row and abs(yd - gh) > 1e-5)):
                    pv = True
                if ur_col != bl_col and ((c == bl_col and abs(xd - gw) > 1e-5) or
                                         (c == ur_col and abs(xd - gw) > 1e-5)):
                    ph = True
                Vmac[r * gc + c] += sign * xd * self.vrouting_alloc
                Hmac[r * gc + c] += sign * yd * self.hrouting_alloc
        if pv:
            r = ur_row
            for c in range(bl_col, ur_col + 1):
                xd, _ = xy_dist(r, c)
                Vmac[r * gc + c] -= sign * xd * self.vrouting_alloc
        if ph:
            c = ur_col
            for r in range(bl_row, ur_row + 1):
                _, yd = xy_dist(r, c)
                Hmac[r * gc + c] -= sign * yd * self.hrouting_alloc

    def _smooth(self, arr, axis):
        """TILOS __smooth_routing_cong, vectorized. Each source line spreads
        A/cnt to the sr-neighborhood (clipped), which is a sliding-window sum of
        the per-source-normalized array -> computed with one cumsum. axis='V'
        smooths across columns, 'H' across rows. arr is a [gr*gc] flat array."""
        gc, gr, gw, gh = self._grid_params()
        A = arr.reshape(gr, gc)
        sr = self.smooth_range

        def box(B):
            # out[:, j] = sum_i B[:, i] for |i-j|<=sr (window clipped at edges)
            n = B.shape[1]
            cs = np.zeros((B.shape[0], n + 1), dtype=B.dtype)
            np.cumsum(B, axis=1, out=cs[:, 1:])
            lo = np.maximum(np.arange(n) - sr, 0)
            hi = np.minimum(np.arange(n) + sr, n - 1)
            return cs[:, hi + 1] - cs[:, lo]

        if axis == "V":
            idx = np.arange(gc)
            cnt = np.minimum(idx + sr, gc - 1) - np.maximum(idx - sr, 0) + 1
            out = box(A / cnt[None, :])
        else:
            idx = np.arange(gr)
            cnt = np.minimum(idx + sr, gr - 1) - np.maximum(idx - sr, 0) + 1
            out = box((A.T / cnt[None, :])).T
        return out.ravel()

    def build_congestion(self, pos):
        if not hasattr(self, "cong_nets"):
            self._build_congestion_ir()
        gc, gr, gw, gh = self._grid_params()
        N = gr * gc
        self.Hraw = np.zeros(N); self.Vraw = np.zeros(N)
        self.Hmac = np.zeros(N); self.Vmac = np.zeros(N)
        for net in self.cong_nets:
            gset, src = self._net_gcells(net, pos)
            self._route_net(gset, src, net["weight"], self.Hraw, self.Vraw, 1.0)
        for i in range(self.b.num_hard_macros):
            sz = self.b.macro_sizes.numpy()
            self._macro_shadow(pos[i, 0], pos[i, 1], sz[i, 0], sz[i, 1],
                               self.Hmac, self.Vmac, 1.0)
        return self.congestion_cost_from_state()

    def congestion_cost_from_state(self):
        Vnet = self.Vraw / self.grid_v_routes
        Hnet = self.Hraw / self.grid_h_routes
        Vs = self._smooth(Vnet, "V"); Hs = self._smooth(Hnet, "H")
        V = Vs + self.Vmac / self.grid_v_routes
        H = Hs + self.Hmac / self.grid_h_routes
        tot = np.concatenate([V, H])
        cnt = int(np.floor(len(tot) * 0.05))
        if cnt == 0:
            return float(tot.max())
        top = np.partition(tot, len(tot) - cnt)[len(tot) - cnt:]
        return float(top.mean())

    def congestion_cost(self, pos):
        return self.build_congestion(np.asarray(pos, np.float64))

    # ======================================================================
    # INCREMENTAL single-move evaluation
    # ======================================================================
    # Keep per-net HPWL, the density grid, and the four congestion arrays as
    # live state. A single-macro move updates only the nets/cells that macro
    # touches; the final reductions (smooth + top-k) are recomputed (cheap).

    def _build_wl_slices(self):
        # pins are appended net-by-net in _build_wirelength_ir, so pin_net is
        # non-decreasing -> each net's pins are a contiguous slice.
        ids = np.arange(self.num_nets)
        self.net_start = np.searchsorted(self.pin_net, ids, side="left")
        self.net_end = np.searchsorted(self.pin_net, ids, side="right")

    def _net_hpwl(self, j, pos):
        s, e = self.net_start[j], self.net_end[j]
        owner = self.pin_owner[s:e]; off = self.pin_off[s:e]
        oc = np.clip(owner, 0, None)
        x = np.where(owner >= 0, pos[oc, 0] + off[:, 0], off[:, 0])
        y = np.where(owner >= 0, pos[oc, 1] + off[:, 1], off[:, 1])
        return (x.max() - x.min()) + (y.max() - y.min())

    def _macro_wl_ir(self, i):
        """Cached per-macro flat pin-index array + reduceat offsets covering all
        nets that macro touches (for one-shot vectorized HPWL recompute)."""
        cache = getattr(self, "_mwl", None)
        if cache is None:
            cache = self._mwl = {}
        r = cache.get(i)
        if r is None:
            nets = self.macro_nets.get(i, _EMPTY)
            idx = np.concatenate([np.arange(self.net_start[j], self.net_end[j]) for j in nets]) \
                if len(nets) else _EMPTY
            offs = np.concatenate([[0], np.cumsum(self.net_end[nets] - self.net_start[nets])[:-1]]) \
                if len(nets) else _EMPTY
            r = cache[i] = (nets, idx.astype(np.int64), offs.astype(np.int64))
        return r

    def optimal_xy(self, i, pos):
        """Closed-form wirelength-optimal position for macro i with all others
        fixed: the median of its nets' bounding intervals (excluding i's own
        pins), shifted into center coordinates via i's mean pin offset. The
        classic detailed-placement 'optimal region' -- used as a PROPOSAL; the
        exact evaluator still accepts/rejects on the full score."""
        nets, idx, offs = self._macro_wl_ir(i)
        if not len(nets):
            return None
        owner = self.pin_owner[idx]; off = self.pin_off[idx]
        oc = np.clip(owner, 0, None)
        x = np.where(owner >= 0, pos[oc, 0] + off[:, 0], off[:, 0])
        y = np.where(owner >= 0, pos[oc, 1] + off[:, 1], off[:, 1])
        own = owner == i
        # min/max of the OTHER pins per net (own pins masked out)
        xin = np.where(own, np.inf, x); xax = np.where(own, -np.inf, x)
        yin = np.where(own, np.inf, y); yax = np.where(own, -np.inf, y)
        mnx = np.minimum.reduceat(xin, offs); mxx = np.maximum.reduceat(xax, offs)
        mny = np.minimum.reduceat(yin, offs); mxy = np.maximum.reduceat(yax, offs)
        ok = np.isfinite(mnx)                        # nets with at least one other pin
        if not ok.any():
            return None
        ex = np.concatenate([mnx[ok], mxx[ok]]); ey = np.concatenate([mny[ok], mxy[ok]])
        oxm = off[own, 0].mean() if own.any() else 0.0
        oym = off[own, 1].mean() if own.any() else 0.0
        return float(np.median(ex) - oxm), float(np.median(ey) - oym)

    def _macro_nets_hpwl(self, i, pos):
        """Vectorized HPWL of every net touching macro i (== [_net_hpwl(j)...])."""
        nets, idx, offs = self._macro_wl_ir(i)
        if not len(nets):
            return nets, None
        owner = self.pin_owner[idx]; off = self.pin_off[idx]
        oc = np.clip(owner, 0, None)
        x = np.where(owner >= 0, pos[oc, 0] + off[:, 0], off[:, 0])
        y = np.where(owner >= 0, pos[oc, 1] + off[:, 1], off[:, 1])
        return nets, ((np.maximum.reduceat(x, offs) - np.minimum.reduceat(x, offs)) +
                      (np.maximum.reduceat(y, offs) - np.minimum.reduceat(y, offs)))

    # Klein-4 orientations: sign applied to pin offsets (N, FN=mirror-x,
    # FS=mirror-y, S=180). Footprint unchanged, so no overlap/density change.
    _ORIENT_SIGN = {0: (1.0, 1.0), 1: (-1.0, 1.0), 2: (1.0, -1.0), 3: (-1.0, -1.0)}

    def _init_orient(self):
        self.pin_base_off = self.pin_off.copy()
        self.macro_orient = np.zeros(self.b.num_macros, dtype=np.int64)
        mp = {}
        for pidx, o in enumerate(self.pin_owner):
            if o >= 0:
                mp.setdefault(int(o), []).append(pidx)
        self.macro_pins = {k: np.array(v) for k, v in mp.items()}

    def apply_flip(self, i, orient):
        """Set hard macro i to a Klein-4 orientation, updating WL + congestion
        incrementally (footprint/density unchanged). Returns undo token."""
        pins = self.macro_pins.get(i)
        old = int(self.macro_orient[i])
        if pins is None or len(pins) == 0 or orient == old:
            return dict(i=i, orient=old, noop=True)
        cong_nets = self._macro_cong_arr.get(i, _EMPTY)
        wl_nets = self.macro_nets.get(i, _EMPTY)
        undo = dict(i=i, orient=old, noop=False, wl_sum=self.wl_sum,
                    wl_nets=wl_nets,
                    wl_vals=self.wl_net[wl_nets].copy() if len(wl_nets) else None)
        for j in cong_nets:
            self._add_net_route(int(j), -1.0)
        sx, sy = self._ORIENT_SIGN[orient]
        self.pin_off[pins, 0] = self.pin_base_off[pins, 0] * sx
        self.pin_off[pins, 1] = self.pin_base_off[pins, 1] * sy
        self.macro_orient[i] = orient
        for j in cong_nets:
            self._add_net_route(int(j), +1.0)
        if len(wl_nets):
            new = np.array([self._net_hpwl(int(j), self.ipos) for j in wl_nets])
            self.wl_sum += float((self.net_weight[wl_nets] * (new - self.wl_net[wl_nets])).sum())
            self.wl_net[wl_nets] = new
        return undo

    def undo_flip(self, undo):
        if undo.get("noop"):
            return
        self.apply_flip(undo["i"], undo["orient"])
        self.wl_sum = undo["wl_sum"]
        if undo["wl_vals"] is not None:
            self.wl_net[undo["wl_nets"]] = undo["wl_vals"]

    def init_incremental(self, pos):
        """Build all live state for incremental evaluation from scratch."""
        if not hasattr(self, "cong_nets"):
            self._build_congestion_ir()
        self._build_wl_slices()
        self._init_orient()
        self._sz = self.b.macro_sizes.numpy()          # cached (hot path)
        self.ipos = np.asarray(pos, np.float64).copy()
        # wirelength: per-net hpwl + running weighted sum
        self.wl_net = np.array([self._net_hpwl(j, self.ipos)
                                for j in range(self.num_nets)])
        self.wl_sum = float((self.net_weight * self.wl_net).sum())
        # density grid + congestion arrays (full build)
        self.build_density(self.ipos)          # -> grid_occupied, grid_area
        self.build_congestion(self.ipos)       # -> Hraw, Vraw, Hmac, Vmac
        self._macro_cong_arr = {k: np.array(sorted(v)) for k, v
                                in self.macro_cong_nets.items()}
        return self.cost_current()

    def cost_current(self):
        wl = self.wl_sum / ((self.W + self.H) * self.net_cnt)
        dens = self.density_cost_from_occ(self.grid_occupied)
        cong = self.congestion_cost_from_state()
        return wl + 0.5 * dens + 0.5 * cong

    def _add_macro_density(self, i, sign):
        gc = self.plc.grid_col
        sz = self._sz
        r = self._macro_cell_area(self.ipos[i, 0], self.ipos[i, 1], sz[i, 0], sz[i, 1])
        if r is None:
            return
        rows, cols, area = r
        idx = (rows[:, None] * gc + cols[None, :]).ravel()
        self.grid_occupied[idx] += sign * area.ravel()

    def _add_net_route(self, j, sign):
        net = self.cong_nets[j]
        gset, src = self._net_gcells(net, self.ipos)
        self._route_net(gset, src, net["weight"], self.Hraw, self.Vraw, sign)

    def _add_shadow(self, i, sign):
        sz = self._sz
        self._macro_shadow(self.ipos[i, 0], self.ipos[i, 1], sz[i, 0], sz[i, 1],
                           self.Hmac, self.Vmac, sign)

    def apply_move(self, i, x, y):
        """Move macro i to (x,y), updating all live state incrementally.
        Returns an undo token for exact revert. Undo restores SNAPSHOTS of the
        (small) grid arrays taken here -- bitwise-exact and much cheaper than
        re-running the deposit routines in reverse."""
        nh = self.b.num_hard_macros
        is_hard = i < nh
        cong_nets = self._macro_cong_arr.get(i, _EMPTY)
        wl_nets = self.macro_nets.get(i, _EMPTY)
        undo = dict(i=i, old=self.ipos[i].copy(), wl_sum=self.wl_sum,
                    wl_nets=wl_nets, wl_vals=self.wl_net[wl_nets].copy()
                    if len(wl_nets) else None,
                    occ=self.grid_occupied.copy(),
                    Hraw=self.Hraw.copy(), Vraw=self.Vraw.copy(),
                    Hmac=self.Hmac.copy() if is_hard else None,
                    Vmac=self.Vmac.copy() if is_hard else None)
        # routing depends only on pin GRID CELLS: if no pin of this macro changes
        # cell (common for sub-cell jitters), every touched net's deposit is
        # identical and the re-route can be skipped entirely.
        pins = self.macro_pins.get(i)
        skip_route = False
        if pins is not None and len(pins) and not is_hard:
            gc, gr, gw, gh = self._grid_params()
            ox = self.pin_off[pins, 0]; oy = self.pin_off[pins, 1]
            oc_ = np.clip(np.floor((self.ipos[i, 0] + ox) / gw).astype(np.int64), 0, gc - 1)
            or_ = np.clip(np.floor((self.ipos[i, 1] + oy) / gh).astype(np.int64), 0, gr - 1)
            nc_ = np.clip(np.floor((x + ox) / gw).astype(np.int64), 0, gc - 1)
            nr_ = np.clip(np.floor((y + oy) / gh).astype(np.int64), 0, gr - 1)
            skip_route = bool(np.array_equal(oc_, nc_) and np.array_equal(or_, nr_))
        # --- remove old contributions (at current ipos[i]) ---
        self._add_macro_density(i, -1.0)
        if not skip_route:
            for j in cong_nets:
                self._add_net_route(int(j), -1.0)
        if is_hard:
            self._add_shadow(i, -1.0)
        # --- move ---
        self.ipos[i, 0] = x; self.ipos[i, 1] = y
        # --- add new contributions ---
        self._add_macro_density(i, +1.0)
        if not skip_route:
            for j in cong_nets:
                self._add_net_route(int(j), +1.0)
        if is_hard:
            self._add_shadow(i, +1.0)
        # --- wirelength: recompute affected nets (vectorized across the nets) ---
        if len(wl_nets):
            _, new = self._macro_nets_hpwl(i, self.ipos)
            self.wl_sum += float((self.net_weight[wl_nets] * (new - self.wl_net[wl_nets])).sum())
            self.wl_net[wl_nets] = new
        return undo

    def undo_move(self, undo):
        """Restore the pre-move snapshots -- bitwise-exact, no re-deposits."""
        i = undo["i"]
        self.ipos[i] = undo["old"]
        self.grid_occupied = undo["occ"]
        self.Hraw = undo["Hraw"]; self.Vraw = undo["Vraw"]
        if undo["Hmac"] is not None:
            self.Hmac = undo["Hmac"]; self.Vmac = undo["Vmac"]
        self.wl_sum = undo["wl_sum"]
        if undo["wl_vals"] is not None:
            self.wl_net[undo["wl_nets"]] = undo["wl_vals"]
