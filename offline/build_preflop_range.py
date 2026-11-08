"""Offline: for every suit-isomorphism class of 5-card hands, equity vs the
opponent's top (1-c) of hands by preflop strength, for c in PRE_RANGE_CUTS.
(The same rejection model the bot uses at runtime, tabulated.)
Writes preflop_range.npz: cls (376992 int16, by colex rank), eq (n_classes x n_cuts float16).
usage: python offline/build_preflop_range.py [samples_per_class] [workers]
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plo_engine as E

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()


def canon(h):
    by = {}
    for c in h:
        by.setdefault(c & 3, []).append(c >> 2)
    return tuple(sorted(tuple(sorted(v)) for v in by.values()))


def work(args):
    i, hand = args
    E.build_tables()
    rng = np.random.default_rng(50_000 + i)
    ws, pres = [], []
    for _ in range(SAMPLES // 2000):
        s = E.sample_equity(list(hand), [], 2000, rng)
        ws.append(s.w); pres.append(s.pre)
    w = np.concatenate(ws); pre = np.concatenate(pres)
    order = np.sort(pre)
    out = []
    for c in E.PRE_RANGE_CUTS:
        keep = pre >= order[min(len(pre) - 1, int(c * len(pre)))]
        out.append(float(w[keep].mean()))
    return out


if __name__ == "__main__":
    E.build_tables()
    combos = E.all_five_combos()
    ids, reps, cls = {}, [], np.zeros(len(combos), dtype=np.int16)
    ranks = E.colex5(combos)
    for idx, h in enumerate(combos):
        k = canon(h)
        if k not in ids:
            ids[k] = len(reps); reps.append(tuple(int(x) for x in h))
        cls[ranks[idx]] = ids[k]
    print("classes", len(reps), flush=True)
    t = time.time()
    with Pool(WORKERS) as p:
        res = p.map(work, list(enumerate(reps)), chunksize=32)
    eq = np.array(res, dtype=np.float16)
    np.savez_compressed(E.PRE_RANGE_PATH, cls=cls, eq=eq)
    print("saved", E.PRE_RANGE_PATH, "secs", round(time.time() - t), "mean eq by cut", eq.astype(np.float32).mean(axis=0).round(3))
