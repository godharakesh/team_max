"""Flatten the (card-independent) betting tree into arrays for the JIT solver.

Roles: 0 = button (small blind, acts first preflop), 1 = big blind.
Every decision node maps to an abstract key (street, position, pot at street
start, street action sequence) = betting.infoset_key without the card bucket;
nodes that share a key share strategy rows (imperfect recall, as in Pluribus).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from betting import State  # noqa: E402

DECISION, FOLD, SHOWDOWN = 0, 1, 2


def public_key(st: State) -> str:
    """infoset key minus the bucket: street, b/n, pot_start, seq."""
    return f"{st.street}{'b' if st.to_act == st.button else 'n'}.{st.pot_start()}.{st.seq}"


def build_tree():
    node_type, node_player, node_street, node_key = [], [], [], []
    node_nact, node_acts, node_child = [], [], []
    term_c0, term_c1, term_folder = [], [], []
    keys: dict[str, int] = {}
    key_street: list[int] = []

    def new_node():
        node_type.append(0); node_player.append(-1); node_street.append(-1); node_key.append(-1)
        node_nact.append(0); node_acts.append([-1] * 4); node_child.append([-1] * 4)
        term_c0.append(0); term_c1.append(0); term_folder.append(-1)
        return len(node_type) - 1

    def visit(st: State) -> int:
        n = new_node()
        if st.terminal:
            node_type[n] = FOLD if st.folder >= 0 else SHOWDOWN
            term_c0[n], term_c1[n] = st.contrib[0], st.contrib[1]   # seat 0 == role 0 (button)
            term_folder[n] = st.folder
            return n
        k = public_key(st)
        if k not in keys:
            keys[k] = len(keys)
            key_street.append(st.street)
        node_type[n] = DECISION
        node_player[n] = st.to_act
        node_street[n] = st.street
        node_key[n] = keys[k]
        legal = st.legal()
        node_nact[n] = len(legal)
        for j, a in enumerate(legal):
            ch = st.copy()
            ch.apply(a)
            node_acts[n][j] = a
            node_child[n][j] = visit(ch)
        return n

    root = visit(State(button=0))
    assert root == 0
    arr = lambda x, t: np.array(x, dtype=t)
    return {
        "node_type": arr(node_type, np.int8), "node_player": arr(node_player, np.int8),
        "node_street": arr(node_street, np.int8), "node_key": arr(node_key, np.int32),
        "node_nact": arr(node_nact, np.int8), "node_acts": arr(node_acts, np.int8),
        "node_child": arr(node_child, np.int32),
        "term_c0": arr(term_c0, np.float64), "term_c1": arr(term_c1, np.float64),
        "term_folder": arr(term_folder, np.int8),
        "key_street": arr(key_street, np.int8), "keys": keys,
    }


if __name__ == "__main__":
    t = build_tree()
    nt = t["node_type"]
    print("nodes", len(nt), "decision", int((nt == 0).sum()), "fold", int((nt == 1).sum()),
          "showdown", int((nt == 2).sum()), "public keys", len(t["keys"]))
    ks = t["key_street"]
    print("public keys per street", [int((ks == s).sum()) for s in range(4)])
