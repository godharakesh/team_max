"""How each opponent looks through our opponent model (observer = pure bp2).
usage: python offline/opp_stats.py opp1,opp2,... [deals]"""
import os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena  # noqa: E402


def run(args):
    opp, deals, seed = args
    arena.E.build_tables()
    a, b = arena.make("bp2"), arena.make(opp)
    arena.play_half(a, b, seed, 1, deals)
    o = a.m["opp"]
    r = lambda p: p[0] / p[1] if p[1] else float("nan")
    sd = lambda p: p[0] / p[1] if p[1] else float("nan")
    return (opp, r(o["pre_aggr"]), r(o["post_aggr"]), r(o["fold_to_bet"]), sd(o["sd_pre"]), o["sd_pre"][1],
            sd(o["sd_post"]), o["sd_post"][1])


if __name__ == "__main__":
    opps = sys.argv[1].split(","); deals = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    with Pool(min(len(opps), os.cpu_count())) as p:
        rows = p.map(run, [(o, deals, "st1") for o in opps])
    print(f"{'opponent':<8} {'pre_aggr':>8} {'post_aggr':>9} {'fold2bet':>8} {'sd_pre':>7} {'n':>4} {'sd_post':>7} {'n':>4}")
    for r in rows:
        print(f"{r[0]:<8} {r[1]:8.3f} {r[2]:9.3f} {r[3]:8.3f} {r[4]:7.3f} {r[5]:4d} {r[6]:7.3f} {r[7]:4d}")
