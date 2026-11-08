"""Blueprint v2: JIT-compiled external-sampling MCCFR (Linear-CFR discounting)
over the flattened exact betting tree, with a potential-aware card abstraction.

Abstraction (per street):
  preflop  NB_PRE quantile bins of preflop equity (exact table, noise-free)
  flop     NE equity bins x NP potential bins (potential = EHS - current HS)
  turn     NE x NP likewise
  river    NB_RIV quantile bins of river equity
Runs in the training venv (numba):
  .venv-train/bin/python offline/mccfr_jit.py offline/deals2.npz <iters_per_chain> <chains> blueprint2.pkl
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree import build_tree  # noqa: E402

NB_PRE = int(os.environ.get("NB_PRE", "40"))
NE = int(os.environ.get("NB_E", "10"))
NP = int(os.environ.get("NB_P", "3"))
NB_RIV = int(os.environ.get("NB_RIV", "30"))
NB = (NB_PRE, NE * NP, NE * NP, NB_RIV)


# ---------------------------------------------------------------------------
# abstraction
# ---------------------------------------------------------------------------

def make_abstraction(feats):
    """feats (D, 2, 6) = pre, f_ehs, f_hs, t_ehs, t_hs, r_ehs -> thresholds dict."""
    q = lambda v, k: [float(x) for x in np.quantile(v, [i / k for i in range(1, k)])]
    ab = {"nb": NB, "ne": NE, "np": NP,
          "pre": q(feats[:, :, 0].ravel(), NB_PRE),
          "riv": q(feats[:, :, 5].ravel(), NB_RIV)}
    for name, ce, ch in (("flop", 1, 2), ("turn", 3, 4)):
        ehs = feats[:, :, ce].ravel(); pot = ehs - feats[:, :, ch].ravel()
        e_th = q(ehs, NE)
        e_bin = np.searchsorted(np.array(e_th), ehs, side="right")
        p_th = []
        for e in range(NE):
            sel = pot[e_bin == e]
            p_th.append(q(sel, NP) if sel.size > NP else [0.0] * (NP - 1))
        ab[name + "_e"] = e_th
        ab[name + "_p"] = p_th
    return ab


def bucketize(ab, feats):
    """Vectorised bucket ids (D, 2, 4) int16 using the abstraction thresholds."""
    D = feats.shape[0]
    B = np.zeros((D, 2, 4), dtype=np.int16)
    B[:, :, 0] = np.searchsorted(np.array(ab["pre"]), feats[:, :, 0], side="right")
    B[:, :, 3] = np.searchsorted(np.array(ab["riv"]), feats[:, :, 5], side="right")
    for st, name, ce, ch in ((1, "flop", 1, 2), (2, "turn", 3, 4)):
        ehs = feats[:, :, ce]; pot = ehs - feats[:, :, ch]
        e = np.searchsorted(np.array(ab[name + "_e"]), ehs, side="right")
        p_th = np.array(ab[name + "_p"])          # (NE, NP-1)
        p = (pot[..., None] > p_th[e]).sum(axis=-1)
        B[:, :, st] = e * NP + p
    return B


# ---------------------------------------------------------------------------
# JIT solver
# ---------------------------------------------------------------------------

@njit
def _seed(s):
    np.random.seed(s)


@njit
def traverse(n, p, b0, b1, res, node_type, node_player, node_street, node_key, node_nact,
             node_acts, node_child, term_c0, term_c1, term_folder, key_offset, regret, ssum):
    t = node_type[n]
    if t == 1:                                   # fold
        cp = term_c0[n] if p == 0 else term_c1[n]
        co = term_c1[n] if p == 0 else term_c0[n]
        if term_folder[n] == p:
            return -cp
        return co
    if t == 2:                                   # showdown (res: +1 role0 wins, -1 role1, 0 split)
        cp = term_c0[n] if p == 0 else term_c1[n]
        co = term_c1[n] if p == 0 else term_c0[n]
        if res == 0:
            return 0.5 * (co - cp)
        if (res > 0) == (p == 0):
            return co
        return -cp
    r = node_player[n]
    st = node_street[n]
    b = b0[st] if r == 0 else b1[st]
    I = key_offset[node_key[n]] + b
    na = node_nact[n]
    sigma = np.empty(4)
    tot = 0.0
    for j in range(na):
        x = regret[I, node_acts[n, j]]
        x = x if x > 0.0 else 0.0
        sigma[j] = x
        tot += x
    if tot > 0.0:
        for j in range(na):
            sigma[j] /= tot
    else:
        for j in range(na):
            sigma[j] = 1.0 / na
    if r == p:
        u = np.empty(4)
        node_u = 0.0
        for j in range(na):
            u[j] = traverse(node_child[n, j], p, b0, b1, res, node_type, node_player, node_street, node_key,
                            node_nact, node_acts, node_child, term_c0, term_c1, term_folder, key_offset,
                            regret, ssum)
            node_u += sigma[j] * u[j]
        for j in range(na):
            regret[I, node_acts[n, j]] += u[j] - node_u
        return node_u
    for j in range(na):
        ssum[I, node_acts[n, j]] += sigma[j]
    x = np.random.random()
    acc = 0.0
    pick = na - 1
    for j in range(na):
        acc += sigma[j]
        if x < acc:
            pick = j
            break
    return traverse(node_child[n, pick], p, b0, b1, res, node_type, node_player, node_street, node_key,
                    node_nact, node_acts, node_child, term_c0, term_c1, term_folder, key_offset, regret, ssum)


@njit
def train_chunk(it0, iters, total, B, R, node_type, node_player, node_street, node_key, node_nact,
                node_acts, node_child, term_c0, term_c1, term_folder, key_offset, regret, ssum, disc_every):
    D = B.shape[0]
    for k in range(iters):
        it = it0 + k + 1
        d = np.random.randint(D)
        bs = np.random.randint(2)
        b0 = B[d, bs]
        b1 = B[d, 1 - bs]
        res = R[d] if bs == 0 else -R[d]
        for p in range(2):
            traverse(0, p, b0, b1, res, node_type, node_player, node_street, node_key, node_nact,
                     node_acts, node_child, term_c0, term_c1, term_folder, key_offset, regret, ssum)
        # Linear CFR: discount regrets and the average strategy early in training
        if it % disc_every == 0 and it <= 0.4 * total:
            c = it // disc_every
            f = c / (c + 1.0)
            regret *= f
            ssum *= f


def chain(args):
    deals_path, iters, seed = args
    d = np.load(deals_path)
    feats, R = d["feats"], d["res"].astype(np.int8)
    ab = make_abstraction(feats)
    B = bucketize(ab, feats).astype(np.int64)
    T = build_tree()
    nb = np.array(NB, dtype=np.int64)
    key_offset = np.zeros(len(T["key_street"]), dtype=np.int64)
    off = 0
    for k, st in enumerate(T["key_street"]):
        key_offset[k] = off
        off += nb[st]
    regret = np.zeros((off, 4)); ssum = np.zeros((off, 4))
    _seed(seed)
    args_tree = (T["node_type"].astype(np.int64), T["node_player"].astype(np.int64),
                 T["node_street"].astype(np.int64), T["node_key"].astype(np.int64),
                 T["node_nact"].astype(np.int64), T["node_acts"].astype(np.int64),
                 T["node_child"].astype(np.int64), T["term_c0"], T["term_c1"],
                 T["term_folder"].astype(np.int64), key_offset)
    disc_every = max(1, iters // 200)
    t0 = time.time()
    step = max(1, iters // 10)
    done = 0
    while done < iters:
        k = min(step, iters - done)
        train_chunk(done, k, iters, B, R.astype(np.int64), *args_tree, regret, ssum, disc_every)
        done += k
        if seed == 1:
            print(f"[chain1] {done}/{iters} its  {time.time() - t0:.0f}s", flush=True)
    return ssum, ab, T["keys"], key_offset, off


if __name__ == "__main__":
    deals, iters, chains, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    t = time.time()
    # NOTE: no numba disk cache (cache=True): loading the cached RECURSIVE
    # traverse() segfaulted in every worker, which hung multiprocessing.Pool
    # silently. Each worker compiles for itself (~3 s).
    with Pool(chains) as pool:
        parts = pool.map(chain, [(deals, iters, s + 1) for s in range(chains)])
    ssum = sum(p[0] for p in parts)
    ab, keys, key_offset, n_inf = parts[0][1], parts[0][2], parts[0][3], parts[0][4]
    tot = ssum.sum(axis=1, keepdims=True)
    policy = np.where(tot > 0, ssum / np.maximum(tot, 1e-12), 0.0).astype(np.float32)
    visited = int((tot[:, 0] > 0).sum())
    with open(out, "wb") as f:
        pickle.dump({"version": 2, "keys": keys, "key_offset": key_offset.astype(np.int32),
                     "policy": policy.astype(np.float16), "abstraction": ab,
                     "iters": iters * chains, "visited": visited, "n_infosets": n_inf}, f)
    print(f"infosets {n_inf} visited {visited} iters {iters * chains} secs {time.time() - t:.0f} -> {out}")
