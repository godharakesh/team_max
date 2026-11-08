"""How each RNR archetype looks through the bot's opponent tracker.

Plays our blueprint (blueprint2.pkl) against an opponent in the abstract game
(same deals, buckets and tree as training) and counts the opponent's actions
exactly like strategy._ingest_history / record_hand_end do:
  pre_aggr  = raises / actions preflop        post_aggr = raises / actions postflop
  fold      = folds / actions right after our raise
Opponent "blueprint" reproduces strategy.EQ_REF (a check of the simulation).

usage: .venv-train/bin/python offline/arch_profiles.py <deals2.npz> <blueprint2.pkl> [hands]
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mccfr_jit import NB, bucketize, make_abstraction  # noqa: E402
from rnr_jit import ARCHETYPES, fixed_policy  # noqa: E402
from tree import build_tree  # noqa: E402


def profile(T, key_offset, B, us_pol, opp_pol, hands, rng):
    nt, npl, nst, nkey = T["node_type"], T["node_player"], T["node_street"], T["node_key"]
    nact, acts, child = T["node_nact"], T["node_acts"], T["node_child"]
    c = {"pre": [0, 0], "post": [0, 0], "fold": [0, 0]}
    D = B.shape[0]
    for _ in range(hands):
        d = rng.integers(D)
        us = rng.integers(2)                      # our role this hand (0 = button)
        bk = {0: B[d, 0], 1: B[d, 1]}             # role -> bucket per street
        n, last = 0, None                         # last = (role, action)
        while nt[n] == 0:
            r, st = int(npl[n]), int(nst[n])
            I = key_offset[nkey[n]] + int(bk[r][st])
            legal = acts[n][: nact[n]]
            pol = us_pol if r == us else opp_pol
            w = np.array([max(0.0, float(pol[I, a])) for a in legal])
            w = w / w.sum() if w.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
            j = rng.choice(len(legal), p=w)
            a = int(legal[j])
            if r != us:
                k = "pre" if st == 0 else "post"
                c[k][1] += 1
                c[k][0] += a in (2, 3)
                if last is not None and last[0] == us and last[1] in (2, 3):
                    c["fold"][1] += 1
                    c["fold"][0] += a == 0
            last = (r, a)
            n = int(child[n, j])
    return {k: round(v[0] / max(1, v[1]), 3) for k, v in c.items()}


if __name__ == "__main__":
    deals, bp_path = sys.argv[1], sys.argv[2]
    hands = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
    d = np.load(deals)
    feats = d["feats"]
    ab = make_abstraction(feats)
    B = bucketize(ab, feats).astype(np.int64)
    T = build_tree()
    nb = np.array(NB, dtype=np.int64)
    key_offset = np.zeros(len(T["key_street"]), dtype=np.int64)
    off = 0
    for k, st in enumerate(T["key_street"]):
        key_offset[k] = off
        off += nb[st]
    bp = pickle.load(open(bp_path, "rb"))["policy"].astype(np.float64)
    rng = np.random.default_rng(7)
    print("blueprint", profile(T, key_offset, B, bp, bp, hands, rng), "(should match EQ_REF pre .49 post .415 fold .40)")
    for name in ARCHETYPES:
        print(name, profile(T, key_offset, B, bp, fixed_policy(name, T, key_offset, off), hands, rng))
