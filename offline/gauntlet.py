"""Run every (variant x opponent x seed) duplicate match in parallel and print
a compact table: mean bb/deal per pair, worst-case CI low, verdict counts, and
max think time per half. Same seeds for every variant => paired comparison.

usage: python offline/gauntlet.py <variants comma> <opponents comma> <deals> <seed> [seed...]
"""
import os, sys
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena  # noqa: E402

if __name__ == "__main__":
    variants = sys.argv[1].split(","); opps = sys.argv[2].split(",")
    deals = int(sys.argv[3]); seeds = sys.argv[4:]
    jobs = [(v, o, deals, s) for v in variants for o in opps for s in seeds if v != o]
    res = defaultdict(list)
    with Pool(int(os.environ.get("ARENA_WORKERS", os.cpu_count()))) as p:
        for r in p.imap_unordered(arena.match, jobs):
            res[(r[0], r[1])].append(r)
    print(f"{'variant':>6} {'opponent':<8} {'mean bb':>8} {'min CI lo':>9}  W/C/L   max think")
    for v in variants:
        for o in opps:
            rs = res.get((v, o))
            if not rs:
                continue
            mean = sum(r[3] for r in rs) / len(rs)
            lo = min(r[4] for r in rs)
            w = sum(r[6] == "WIN" for r in rs); c = sum(r[6] == "CHOP" for r in rs); l = sum(r[6] == "LOSS" for r in rs)
            print(f"{v:>6} {o:<8} {mean:+8.3f} {lo:+9.3f}  {w}/{c}/{l}   {max(r[7] for r in rs):.2f}s", flush=True)
