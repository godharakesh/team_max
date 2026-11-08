"""Offline: preflop equity (vs a uniformly random hand, full runout) for every
5-card short-deck hand, grouped by suit-isomorphism class.
Writes preflop_equity.npy (376,992 float16, indexed by plo_engine.colex5).
Run from the bundle root:  python offline/build_preflop.py [samples]
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plo_engine as E

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

def canon(h):
    by = {}
    for c in h:
        by.setdefault(c & 3, []).append(c >> 2)
    return tuple(sorted(tuple(sorted(v)) for v in by.values()))

def work(args):
    i, hand = args
    E.build_tables()
    rng = np.random.default_rng(1000 + i)
    eqs = [E.equity(list(hand), [], 1000, rng) for _ in range(SAMPLES // 1000)]
    return float(np.mean(eqs))

if __name__ == "__main__":
    combos = E.all_five_combos()
    classes = {}
    keys = [None] * len(combos)
    for idx, h in enumerate(combos):
        k = canon(h)
        keys[idx] = k
        classes.setdefault(k, tuple(int(x) for x in h))
    print("classes", len(classes), flush=True)
    items = list(classes.items())
    t = time.time()
    with Pool(os.cpu_count()) as p:
        res = p.map(work, [(i, h) for i, (k, h) in enumerate(items)], chunksize=64)
    print("mc done", time.time() - t, flush=True)
    ceq = {k: r for (k, _), r in zip(items, res)}
    out = np.zeros(len(combos), dtype=np.float16)
    ranks = E.colex5(combos)
    out[ranks] = np.array([ceq[k] for k in keys], dtype=np.float16)
    np.save(os.path.join(os.path.dirname(E.PREFLOP_PATH), "preflop_equity.npy"), out)
    print("saved; min/mean/max", out.min(), out.astype(np.float32).mean(), out.max())
