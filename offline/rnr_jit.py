"""Restricted Nash Responses (Johanson, Zinkevich & Bowling, NIPS 2007) on the
blueprint-v2 abstraction and the exact betting tree.

An RNR is a robust counter-strategy to a fixed opponent model ("archetype").
At the root of every hand a hidden coin decides whether the opponent plays the
archetype (probability P_FIX) or a learning, worst-case strategy. Our side
cannot tell which, so the solved strategy exploits the archetype while staying
hard to exploit itself. P_FIX = 0 gives an equilibrium; P_FIX = 1 a brittle
best response. The bot mixes a table in only when the opponent clearly looks
like that archetype (see strategy.py).

Same abstraction thresholds as blueprint2.pkl (deterministic from the same
deals file), so the runtime bucket of a hand indexes every table identically.

usage (training venv):
  .venv-train/bin/python offline/rnr_jit.py <deals2.npz> <archetype> <p_fix> <iters_per_chain> <chains> <out.pkl>
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mccfr_jit import NB, NE, NP, bucketize, make_abstraction  # noqa: E402
from tree import build_tree  # noqa: E402

F, C, H, P = 0, 1, 2, 3          # fold, check/call, raise half pot, raise pot
ARCHETYPES = ("station", "nit", "maniac", "passive")


def bucket_strength(street, b):
    """Equity percentile of a bucket (the abstraction is equity-quantile based)."""
    if street == 0 or street == 3:
        return (b + 0.5) / NB[street]
    return (b // NP + 0.5) / NE


def archetype_rule(name, s, facing):
    """Action probabilities (F, C, H, P) of an archetype holding strength s.
    Crude, rule-based styles that many equity-threshold bots fall into."""
    if name == "station":                      # calls almost everything, rarely raises
        if facing:
            if s < 0.10:
                return (1.0, 0.0, 0.0, 0.0)
            if s > 0.90:
                return (0.0, 0.6, 0.0, 0.4)
            return (0.0, 1.0, 0.0, 0.0)
        return (0.0, 0.5, 0.5, 0.0) if s > 0.85 else (0.0, 1.0, 0.0, 0.0)
    if name == "nit":                          # folds to pressure, bets only strong hands
        if facing:
            if s > 0.85:
                return (0.0, 0.5, 0.0, 0.5)
            if s > 0.60:
                return (0.0, 1.0, 0.0, 0.0)
            return (1.0, 0.0, 0.0, 0.0)
        return (0.0, 0.3, 0.7, 0.0) if s > 0.75 else (0.0, 1.0, 0.0, 0.0)
    if name == "maniac":                       # raises and bluffs far too often
        if facing:
            if s > 0.55:
                return (0.0, 0.3, 0.0, 0.7)
            if s > 0.25:
                return (0.0, 0.6, 0.4, 0.0)
            return (0.5, 0.25, 0.25, 0.0)
        return (0.0, 0.2, 0.0, 0.8) if s > 0.40 else (0.0, 0.5, 0.5, 0.0)
    if name == "passive":                      # rarely raises, folds weak hands at normal rates:
        if facing:                             # the profile most weak field bots show (dev1 logs)
            if s < 0.35:
                return (1.0, 0.0, 0.0, 0.0)
            if s > 0.90:
                return (0.0, 0.5, 0.0, 0.5)
            return (0.0, 1.0, 0.0, 0.0)
        return (0.0, 0.4, 0.6, 0.0) if s > 0.80 else (0.0, 1.0, 0.0, 0.0)
    raise ValueError(name)


def fixed_policy(name, T, key_offset, n_inf, eps=0.05):
    """Dense (n_infosets, 4) archetype policy over the same infoset layout as the solver."""
    fix = np.zeros((n_inf, 4))
    seen = set()
    for n in np.flatnonzero(T["node_type"] == 0):
        k = int(T["node_key"][n])
        if k in seen:
            continue
        seen.add(k)
        st = int(T["node_street"][n])
        legal = [int(a) for a in T["node_acts"][n][: int(T["node_nact"][n])]]
        facing = F in legal
        for b in range(NB[st]):
            pr = list(archetype_rule(name, bucket_strength(st, b), facing))
            for a in range(4):                     # illegal choices fall back to check/call
                if a not in legal and pr[a] > 0:
                    pr[C] += pr[a]
                    pr[a] = 0.0
            row = np.array([(1 - eps) * pr[a] + (eps / len(legal) if a in legal else 0.0) for a in range(4)])
            fix[key_offset[k] + b] = row / row.sum()
    return fix


# ---------------------------------------------------------------------------
# JIT solver
# ---------------------------------------------------------------------------

@njit
def _seed(s):
    np.random.seed(s)


@njit
def _regret_matching(reg, I, acts, na, out):
    tot = 0.0
    for j in range(na):
        x = reg[I, acts[j]]
        x = x if x > 0.0 else 0.0
        out[j] = x
        tot += x
    if tot > 0.0:
        for j in range(na):
            out[j] /= tot
    else:
        for j in range(na):
            out[j] = 1.0 / na


@njit
def _pick(sig, na):
    x = np.random.random()
    acc = 0.0
    for j in range(na):
        acc += sig[j]
        if x < acc:
            return j
    return na - 1


@njit
def traverse(n, p, e, fixed, reach, b0, b1, res, node_type, node_player, node_street, node_key, node_nact,
             node_acts, node_child, term_c0, term_c1, term_folder, key_offset, regE, ssE, regO, fixP):
    """External-sampling traversal for traverser p. Role e is the exploiter;
    role 1-e is the opponent: archetype if `fixed`, else its learning strategy.
    `reach` = exploiter's own reach probability (for its average strategy)."""
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
    acts = node_acts[n]
    sig = np.empty(4)
    if r == e:                                   # exploiter
        _regret_matching(regE, I, acts, na, sig)
        if p == e:
            u = np.empty(4)
            nu = 0.0
            for j in range(na):
                u[j] = traverse(node_child[n, j], p, e, fixed, reach * sig[j], b0, b1, res, node_type, node_player,
                                node_street, node_key, node_nact, node_acts, node_child, term_c0, term_c1,
                                term_folder, key_offset, regE, ssE, regO, fixP)
                nu += sig[j] * u[j]
            for j in range(na):
                regE[I, acts[j]] += u[j] - nu
                ssE[I, acts[j]] += reach * sig[j]
            return nu
        j = _pick(sig, na)
        return traverse(node_child[n, j], p, e, fixed, reach, b0, b1, res, node_type, node_player, node_street,
                        node_key, node_nact, node_acts, node_child, term_c0, term_c1, term_folder, key_offset,
                        regE, ssE, regO, fixP)
    if fixed:                                    # opponent plays the archetype
        for j in range(na):
            sig[j] = fixP[I, acts[j]]
        j = _pick(sig, na)
        return traverse(node_child[n, j], p, e, fixed, reach, b0, b1, res, node_type, node_player, node_street,
                        node_key, node_nact, node_acts, node_child, term_c0, term_c1, term_folder, key_offset,
                        regE, ssE, regO, fixP)
    _regret_matching(regO, I, acts, na, sig)     # opponent's learning (worst-case) part
    if p == r:
        u = np.empty(4)
        nu = 0.0
        for j in range(na):
            u[j] = traverse(node_child[n, j], p, e, fixed, reach, b0, b1, res, node_type, node_player, node_street,
                            node_key, node_nact, node_acts, node_child, term_c0, term_c1, term_folder, key_offset,
                            regE, ssE, regO, fixP)
            nu += sig[j] * u[j]
        for j in range(na):
            regO[I, acts[j]] += u[j] - nu
        return nu
    j = _pick(sig, na)
    return traverse(node_child[n, j], p, e, fixed, reach, b0, b1, res, node_type, node_player, node_street,
                    node_key, node_nact, node_acts, node_child, term_c0, term_c1, term_folder, key_offset,
                    regE, ssE, regO, fixP)


@njit
def train_chunk(it0, iters, total, pfix, B, R, node_type, node_player, node_street, node_key, node_nact,
                node_acts, node_child, term_c0, term_c1, term_folder, key_offset, regE, ssE, regO, fixP, disc_every):
    D = B.shape[0]
    for k in range(iters):
        it = it0 + k + 1
        d = np.random.randint(D)
        bs = np.random.randint(2)
        b0 = B[d, bs]
        b1 = B[d, 1 - bs]
        res = R[d] if bs == 0 else -R[d]
        for e in range(2):
            fixed = np.random.random() < pfix
            traverse(0, e, e, fixed, 1.0, b0, b1, res, node_type, node_player, node_street, node_key, node_nact,
                     node_acts, node_child, term_c0, term_c1, term_folder, key_offset, regE, ssE, regO, fixP)
            if not fixed:
                traverse(0, 1 - e, e, False, 1.0, b0, b1, res, node_type, node_player, node_street, node_key,
                         node_nact, node_acts, node_child, term_c0, term_c1, term_folder, key_offset,
                         regE, ssE, regO, fixP)
        if it % disc_every == 0 and it <= 0.4 * total:      # Linear CFR discounting early on
            c = it // disc_every
            f = c / (c + 1.0)
            regE *= f
            ssE *= f
            regO *= f


def chain(args):
    deals_path, name, pfix, iters, seed = args
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
    fixP = fixed_policy(name, T, key_offset, off)
    regE = np.zeros((off, 4)); ssE = np.zeros((off, 4)); regO = np.zeros((off, 4))
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
        train_chunk(done, k, iters, pfix, B, R.astype(np.int64), *args_tree, regE, ssE, regO, fixP, disc_every)
        done += k
        if seed == 1:
            print(f"[{name} chain1] {done}/{iters} its  {time.time() - t0:.0f}s", flush=True)
    return ssE, ab, T["keys"], key_offset, off


if __name__ == "__main__":
    deals, name, pfix, iters, chains, out = (sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]),
                                             int(sys.argv[5]), sys.argv[6])
    assert name in ARCHETYPES, name
    t = time.time()
    with Pool(chains) as pool:          # no numba disk cache: see mccfr_jit.py
        parts = pool.map(chain, [(deals, name, pfix, iters, s + 1) for s in range(chains)])
    ssum = sum(p[0] for p in parts)
    ab, keys, key_offset, n_inf = parts[0][1], parts[0][2], parts[0][3], parts[0][4]
    tot = ssum.sum(axis=1, keepdims=True)
    policy = np.where(tot > 0, ssum / np.maximum(tot, 1e-12), 0.0).astype(np.float32)
    visited = int((tot[:, 0] > 0).sum())
    with open(out, "wb") as f:
        pickle.dump({"version": 2, "keys": keys, "key_offset": key_offset.astype(np.int32),
                     "policy": policy.astype(np.float16), "abstraction": ab, "archetype": name, "p_fix": pfix,
                     "iters": iters * chains, "visited": visited, "n_infosets": n_inf}, f)
    print(f"{name} p_fix={pfix}: infosets {n_inf} visited {visited} iters {iters * chains} "
          f"secs {time.time() - t:.0f} -> {out}")
