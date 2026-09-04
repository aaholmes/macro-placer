"""Contest-tied gif: load an ICCAD04 benchmark through the challenge loader,
run the api pipeline with a TrajectoryRecorder, score frames with the full
contest proxy, save the gif. The rendering/recording itself is the reusable
placers.viz module; only this wrapper knows about the challenge repo.

Usage: python scripts/contest_gif.py ibm01 [out.gif] [budget_s]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CHAL = os.environ.get("CHAL", "/home/laz/partcl/macro-place-challenge-2026")


def main():
    nm = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    out = sys.argv[2] if len(sys.argv) > 2 else f"opt_{nm}_api.gif"
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0

    from macro_place.loader import load_benchmark_from_dir
    from placers.api import input_from_challenge, place, build_fast_eval
    from placers.viz import TrajectoryRecorder

    b, plc = load_benchmark_from_dir(
        f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    p = input_from_challenge(b, plc)
    fe = build_fast_eval(p)

    def contest_proxy(pos):
        return (fe.wirelength_cost(pos) + 0.5 * fe.density_cost(pos)
                + 0.5 * fe.congestion_cost(pos))

    rec = TrajectoryRecorder(p, score_fn=contest_proxy)
    r = place(p, budget_s=budget, seed=0, on_frame=rec, frame_every=200)
    rec.save_gif(out)
    print(f"{out}: {len(rec.frames)} frames, final proxy "
          f"{rec.frames[-1].score:.4f}, wall {r.meta['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
