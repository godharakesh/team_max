"""Offline step A (v2, potential-aware): per deal and seat compute
  [pre_eq, flop EHS, flop HS_now, turn EHS, turn HS_now, river EHS]
with the SAME estimator the bot uses at runtime (plo_engine.sample_equity),
plus the showdown result (seat 0 perspective).

usage: python offline/build_deals2.py <n_deals> <out.npz> [samples] [workers]
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plo_engine as E  # noqa: E402
from evaluator import omaha_score  # noqa: E402

N = int(sys.argv[3]) if len(sys.argv) > 3 else 256


def work(args):
    seed, n = args
    E.build_tables()
    rng = np.random.default_rng(seed)
    feats = np.zeros((n, 2, 6), dtype=np.float32)
    res = np.zeros(n, dtype=np.int8)
    for i in range(n):
        cards = rng.permutation(36)[:15]
        holes = [sorted(cards[0:5].tolist()), sorted(cards[5:10].tolist())]
        board = cards[10:15].tolist()
        for s in range(2):
            h = holes[s]
            f = E.sample_equity(h, board[:3], N, rng)
            t = E.sample_equity(h, board[:4], N, rng)
            r = E.sample_equity(h, board, N, rng)
            feats[i, s] = (E.PRE_EQ[E.colex5(np.array(h))], f.raw, f.hs, t.raw, t.hs, r.raw)
        a, b = omaha_score(holes[0], board), omaha_score(holes[1], board)
        res[i] = 1 if a > b else (-1 if b > a else 0)
    return feats, res


if __name__ == "__main__":
    n, out = int(sys.argv[1]), sys.argv[2]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    chunk = 2500
    jobs = [(104729 * k + 3, chunk) for k in range(n // chunk)]
    t = time.time()
    with Pool(workers) as p:
        parts = p.map(work, jobs)
    feats = np.concatenate([f for f, _ in parts]); res = np.concatenate([r for _, r in parts])
    np.savez_compressed(out, feats=feats, res=res, samples=N)
    print("deals", len(res), "secs", round(time.time() - t, 1), "ties", float((res == 0).mean()))
