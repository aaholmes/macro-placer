"""TDD for the differentiable, non-gameable L-route congestion surrogate.

Run: python tests/test_congestion_surrogate.py
The decisive test is `nongameable_spreading_raises`: unlike RUDY (demand ~ 1/bbox
area, so spreading LOWERS it), a faithful L-route surrogate must RAISE congestion
when a net is spread apart, because a longer route crosses more grid cells.
"""
import sys, math, traceback
sys.path.insert(0, "/home/laz/partcl/my-macro-placer")
import torch
from placers.congestion import softrow, softseg, lroute_field, topk_mean, macro_blockage_field

T = torch.tensor
CASES = []
def case(fn): CASES.append(fn); return fn


@case
def softseg_wider_deposits_more():
    cxs = torch.arange(20).float() + 0.5
    narrow = softseg(cxs, T([5.0]), T([7.0]), 1.0)          # ~2 cells
    wide = softseg(cxs, T([5.0]), T([15.0]), 1.0)           # ~10 cells
    assert wide.sum() > 3 * narrow.sum(), (narrow.sum(), wide.sum())
    assert narrow.shape == (1, 20)


@case
def softrow_peaks_and_normalizes():
    cys = torch.arange(10).float() + 0.5
    r = softrow(cys, T([3.5]), 1.0)                         # [1,10], peak at row 3
    assert int(r.argmax()) == 3, r.argmax()
    assert abs(float(r.sum()) - 1.0) < 0.15, r.sum()        # ~one row's worth


@case
def lroute_shape_and_placement():
    cxs = torch.arange(30).float() + 0.5; cys = torch.arange(30).float() + 0.5
    H, V = lroute_field(T([10.]), T([20.]), T([10.]), T([20.]), T([1.0]), cxs, cys, 1., 1.)
    assert H.shape == (30, 30) and V.shape == (30, 30)
    # H-demand lives on the source rows (~ymin=10) and columns inside [10,20];
    # compare regions robustly (mass may split across the two nearest rows).
    assert float(H[9:11, 11:20].sum()) > 5 * float(H[9:11, 22:].sum())   # on-span >> past-span
    assert float(H[9:11, 11:20].sum()) > 5 * float(H[20:22, 11:20].sum())  # source rows >> other rows
    # V-demand lives on the sink columns (~xmax=20) and rows inside [10,20].
    assert float(V[11:20, 19:21].sum()) > 5 * float(V[11:20, 3:5].sum())   # sink col >> other col
    assert float(V[11:20, 19:21].sum()) > 5 * float(V[22:24, 19:21].sum())  # on-span rows >> past


@case
def nongameable_spreading_raises():
    """THE decisive property: spreading a net apart must RAISE congestion."""
    cxs = torch.arange(30).float() + 0.5; cys = torch.arange(30).float() + 0.5
    nw = T([1.0])
    Hc, Vc = lroute_field(T([13.]), T([15.]), T([13.]), T([15.]), nw, cxs, cys, 1., 1.)  # compact
    Hs, Vs = lroute_field(T([3.]), T([27.]), T([3.]), T([27.]), nw, cxs, cys, 1., 1.)    # spread
    compact = float(Hc.sum() + Vc.sum()); spread = float(Hs.sum() + Vs.sum())
    assert spread > 2 * compact, (compact, spread)


@case
def differentiable_finite():
    cxs = torch.arange(30).float() + 0.5; cys = torch.arange(30).float() + 0.5
    xmin = T([5.], requires_grad=True); xmax = T([20.], requires_grad=True)
    ymin = T([5.], requires_grad=True); ymax = T([20.], requires_grad=True)
    H, V = lroute_field(xmin, xmax, ymin, ymax, T([1.0]), cxs, cys, 1., 1.)
    loss = topk_mean(torch.cat([H.flatten(), V.flatten()]), 0.05, 0.5)
    loss.backward()
    for g in (xmin.grad, xmax.grad, ymin.grad, ymax.grad):
        assert g is not None and torch.isfinite(g).all(), g


@case
def topk_mean_favors_peak():
    """Concentrated demand (a peak) scores higher than the same total spread out."""
    peak = torch.zeros(100); peak[:5] = 10.0                # concentrated
    flat = torch.full((100,), 0.5)                          # same total, spread
    assert float(topk_mean(peak, 0.05, 0.3)) > float(topk_mean(flat, 0.05, 0.3))


@case
def macro_blockage_on_covered_cells():
    cxs = torch.arange(20).float() + 0.5; cys = torch.arange(20).float() + 0.5
    cx = T([10.0]); cy = T([10.0]); hw = T([3.0]); hh = T([2.0])
    M = macro_blockage_field(cx, cy, hw, hh, T([1.0]), cxs, cys, 1., 1.)
    assert M.shape == (20, 20)
    assert float(M[10, 10]) > 0.5                            # center covered
    assert float(M[10, 18]) < 0.2                            # far away not


if __name__ == "__main__":
    torch.manual_seed(0)
    npass = 0
    for fn in CASES:
        try:
            fn(); print(f"PASS {fn.__name__}"); npass += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{npass}/{len(CASES)} passed")
    sys.exit(0 if npass == len(CASES) else 1)
