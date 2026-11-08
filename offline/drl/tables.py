"""Static tables for the GPU environment (numpy, built once, cached to .npz).

Tree arrays come from offline/tree.py (exact betting tree, role 0 = button).
Per decision node we also store betting features from the ACTOR's view and from
role 0's view, built with nn_policy.node_features (the runtime function), so the
GPU features and the runtime features are identical by construction.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "offline"))
from betting import State  # noqa: E402
from tree import build_tree, public_key  # noqa: E402
import nn_policy as NP  # noqa: E402


def _feat(st: State, hist, persp: int):
    """node_features for role `persp` at state st (st must be a decision node)."""
    r = persp
    legal = st.legal() if st.to_act == r else []
    half_amt = st.raise_to(r, 2) - st.bets[r] if 2 in legal else 0
    pot_amt = st.raise_to(r, 3) - st.bets[r] if 3 in legal else 0
    h = [(s, role == r, a) for (s, role, a) in hist]
    return NP.node_features(st.street, r == st.button, st.pot, st.to_call(r), st.bets[r], st.bets[1 - r],
                            st.stacks[r], st.stacks[1 - r], st.raises, legal, half_amt, pot_amt, h)


def build_tables():
    T = build_tree()
    n_nodes = len(T["node_type"])
    F_act = np.zeros((n_nodes, NP.NODE_DIM), dtype=np.float32)
    F_r0 = np.zeros((n_nodes, NP.NODE_DIM), dtype=np.float32)
    info = np.zeros((n_nodes, 8), dtype=np.int32)   # pot, to_call, my_bet, opp_bet, my_stack, opp_stack, raises, street
    counter = [0]
    keys = T["keys"]

    def visit(st: State, hist):
        n = counter[0]
        counter[0] += 1
        if st.terminal:
            return
        assert T["node_type"][n] == 0 and T["node_player"][n] == st.to_act
        assert T["node_key"][n] == keys[public_key(st)]
        r = st.to_act
        F_act[n] = _feat(st, hist, r)
        F_r0[n] = _feat(st, hist, 0) if r == 0 else _feat_view0(st, hist)
        info[n] = (st.pot, st.to_call(r), st.bets[r], st.bets[1 - r], st.stacks[r], st.stacks[1 - r],
                   st.raises, st.street)
        for a in st.legal():
            ch = st.copy()
            ch.apply(a)
            visit(ch, hist + [(st.street, r, a)])

    def _feat_view0(st, hist):
        # role 0's view when role 1 is to act: legal set / raise sizes are not role 0's (zeros)
        h = [(s, role == 0, a) for (s, role, a) in hist]
        return NP.node_features(st.street, True, st.pot, st.to_call(0), st.bets[0], st.bets[1],
                                st.stacks[0], st.stacks[1], st.raises, [], 0, 0, h)

    visit(State(button=0), [])
    assert counter[0] == n_nodes
    nact = T["node_nact"].astype(np.int64)
    acts = T["node_acts"].astype(np.int64)
    child = T["node_child"].astype(np.int64)
    legal = np.zeros((n_nodes, 4), dtype=bool)
    child_a = np.full((n_nodes, 4), -1, dtype=np.int64)
    for j in range(4):
        ok = j < nact
        a = acts[:, j]
        legal[np.where(ok)[0], a[ok]] = True
        child_a[np.where(ok)[0], a[ok]] = child[ok, j]
    key_strings = np.array(sorted(keys, key=keys.get))
    return {
        "node_type": T["node_type"].astype(np.int64), "node_player": T["node_player"].astype(np.int64),
        "node_street": T["node_street"].astype(np.int64), "node_key": T["node_key"].astype(np.int64),
        "legal": legal, "child_a": child_a,
        "term_c0": T["term_c0"].astype(np.float32), "term_c1": T["term_c1"].astype(np.float32),
        "term_folder": T["term_folder"].astype(np.int64),
        "F_act": F_act, "F_r0": F_r0, "info": info, "key_strings": key_strings,
    }


def load_tables(cache=os.path.join(HERE, "tables_cache.npz")):
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    t = build_tables()
    np.savez_compressed(cache, **t)
    return t


if __name__ == "__main__":
    import time
    t0 = time.time()
    t = build_tables()
    print("built in", round(time.time() - t0, 2), "s; nodes", len(t["node_type"]),
          "decision", int((t["node_type"] == 0).sum()), "keys", len(t["key_strings"]))
