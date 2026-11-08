"""Offline step B: external-sampling MCCFR with Linear-CFR discounting (the
Pluribus blueprint recipe) over the exact betting tree with E[HS] card buckets.

usage: python offline/mccfr.py deals.npz <iterations> <out.pkl> [workers]
Each worker trains an independent chain; average strategies are merged
(weighted by accumulated strategy mass). Output: {key: [p_F,p_C,p_H,p_P]},
plus the bucket thresholds.
"""
import os, pickle, random, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from betting import NB, State, infoset_key, bucket  # noqa: E402


def thresholds_from(feats):
    th = []
    for st in range(4):
        v = feats[:, :, st].ravel()
        qs = np.quantile(v, [k / NB[st] for k in range(1, NB[st])])
        th.append([float(x) for x in qs])
    return th


def train(args):
    deals_path, iters, seed = args
    d = np.load(deals_path)
    feats, res = d["feats"], d["res"]
    th = thresholds_from(feats)
    # precompute buckets per deal/seat/street
    B = np.zeros(feats.shape, dtype=np.int16)
    for st in range(4):
        B[:, :, st] = np.searchsorted(np.array(th[st]), feats[:, :, st], side="right")
    B = B.tolist(); res = res.tolist()
    rng = random.Random(seed)
    regret, ssum = {}, {}

    def strat(key, legal):
        r = regret.get(key)
        if r is None:
            r = regret[key] = [0.0, 0.0, 0.0, 0.0]
        pos = [max(r[a], 0.0) for a in legal]
        tot = sum(pos)
        if tot <= 0:
            return [1.0 / len(legal)] * len(legal), r
        return [x / tot for x in pos], r

    def traverse(st, p, bk, sd):
        if st.terminal:
            return st.utility(p, sd)
        s = st.to_act
        legal = st.legal()
        key = infoset_key(st.street, bk[s][st.street], s == st.button, st.pot_start(), st.seq)
        sigma, r = strat(key, legal)
        if s == p:
            utils = []
            node = 0.0
            for a, pa in zip(legal, sigma):
                ch = st.copy(); ch.apply(a)
                u = traverse(ch, p, bk, sd)
                utils.append(u); node += pa * u
            for a, u in zip(legal, utils):
                r[a] += u - node
            return node
        # opponent node: accumulate average strategy, sample one action
        sv = ssum.get(key)
        if sv is None:
            sv = ssum[key] = [0.0, 0.0, 0.0, 0.0]
        for a, pa in zip(legal, sigma):
            sv[a] += pa
        x = rng.random(); acc = 0.0; pick = legal[-1]
        for a, pa in zip(legal, sigma):
            acc += pa
            if x < acc:
                pick = a; break
        ch = st.copy(); ch.apply(pick)
        return traverse(ch, p, bk, sd)

    n = len(res)
    disc_every = max(1, iters // 100)
    t0 = time.time()
    for it in range(1, iters + 1):
        i = rng.randrange(n)
        button = rng.randrange(2)
        for p in (0, 1):
            traverse(State(button), p, B[i], res[i])
        # Linear CFR discount for the first 40% of training (Pluribus)
        if it % disc_every == 0 and it <= 0.4 * iters:
            k = it // disc_every
            f = k / (k + 1)
            for v in regret.values():
                v[0] *= f; v[1] *= f; v[2] *= f; v[3] *= f
            for v in ssum.values():
                v[0] *= f; v[1] *= f; v[2] *= f; v[3] *= f
        if seed == 1 and it % (iters // 10) == 0:
            print(f"[w{seed}] {it}/{iters} infosets={len(regret)} {time.time()-t0:.0f}s", flush=True)
    return ssum, th


if __name__ == "__main__":
    deals, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    with Pool(workers) as pool:
        parts = pool.map(train, [(deals, iters, w + 1) for w in range(workers)])
    th = parts[0][1]
    merged = {}
    for ssum, _ in parts:
        for k, v in ssum.items():
            m = merged.setdefault(k, [0.0, 0.0, 0.0, 0.0])
            for a in range(4):
                m[a] += v[a]
    policy = {}
    for k, v in merged.items():
        tot = sum(v)
        if tot > 0:
            policy[k] = [round(x / tot, 4) for x in v]
    with open(out, "wb") as f:
        pickle.dump({"policy": policy, "thresholds": th, "nb": NB, "iters": iters * workers}, f)
    print("infosets", len(policy), "saved", out)
