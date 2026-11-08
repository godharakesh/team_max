"""Offline step A: sample deals and compute, for both seats, the per-street
equity-vs-random features that define the card abstraction (E[HS] buckets),
plus the showdown result. Uses the SAME estimator (plo_engine.equity) and
sample counts the bot uses at runtime, so bucket noise matches.

usage: python offline/build_deals.py <n_deals> <out.npz>
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plo_engine as E
from evaluator import omaha_score

N_FLOP, N_TURN, N_RIVER = 200, 200, 200   # keep in sync with strategy.BUCKET_SAMPLES


def work(args):
    seed, n = args
    E.build_tables()
    rng = np.random.default_rng(seed)
    feats = np.zeros((n, 2, 4), dtype=np.float32)
    res = np.zeros(n, dtype=np.int8)
    for i in range(n):
        cards = rng.permutation(36)[:15]
        holes = [sorted(cards[0:5].tolist()), sorted(cards[5:10].tolist())]
        board = cards[10:15].tolist()
        for s in range(2):
            h = holes[s]
            feats[i, s, 0] = E.PRE_EQ[E.colex5(np.array(h))]
            feats[i, s, 1] = E.equity(h, board[:3], N_FLOP, rng)
            feats[i, s, 2] = E.equity(h, board[:4], N_TURN, rng)
            feats[i, s, 3] = E.equity(h, board, N_RIVER, rng)
        a, b = omaha_score(holes[0], board), omaha_score(holes[1], board)
        res[i] = 1 if a > b else (-1 if b > a else 0)
    return feats, res


if __name__ == "__main__":
    n, out = int(sys.argv[1]), sys.argv[2]
    chunk = 2000
    jobs = [(7919 * k + 17, chunk) for k in range(n // chunk)]
    t = time.time()
    with Pool(os.cpu_count()) as p:
        parts = p.map(work, jobs)
    feats = np.concatenate([f for f, _ in parts]); res = np.concatenate([r for _, r in parts])
    np.savez_compressed(out, feats=feats, res=res)
    print("deals", len(res), "secs", round(time.time() - t, 1), "p(seat0 win)", (res == 1).mean())
