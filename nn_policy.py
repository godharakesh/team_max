"""
nn_policy.py - runtime side of the deep-RL policy (BOT_MODE=nn), numpy only.

The SAME feature functions are used offline to build the GPU training tables
(offline/drl/), so runtime features match training features by construction:

  node_features()  betting state from the actor's view (pot, bets, stacks, raises,
                   legal raises, per-street history of both players)
  card_features()  hole + visible board multi-hot, rank/suit counts
  category_onehot  category of my current best made hand (exactly 2 + 3)
  strength         [preflop equity, equity vs random, made-hand strength, potential]
  public key id    street/position/pot-at-street-start/street sequence (as in the
                   CFR blueprint), fed through a learned embedding

Model file: nn_policy.npz (MLP weights + embedding + the public key strings).
"""

from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("BOT_NN_PATH", os.path.join(HERE, "nn_policy.npz"))

NODE_DIM = 43
CARD_DIM = 98
CAT_DIM = 9
STR_DIM = 4
BASE_DIM = CARD_DIM + CAT_DIM + STR_DIM + NODE_DIM   # 154 (+ key embedding)


def node_features(street, is_button, pot, to_call, my_bet, opp_bet, my_stack, opp_stack,
                  raises, legal, half_amt, pot_amt, hist):
    """Betting-state features from the acting player's view.
    legal: iterable of action ints (0 F, 1 C, 2 H, 3 P); half_amt / pot_amt: chips a
    RAISE_HALF / RAISE_POT would add (0 if illegal); hist: [(street, is_me, action), ...]."""
    f = np.zeros(NODE_DIM, dtype=np.float32)
    f[street] = 1.0
    f[4] = 1.0 if is_button else 0.0
    pot_start = pot - my_bet - opp_bet
    f[5] = pot / 160.0
    f[6] = pot_start / 160.0
    f[7] = to_call / 80.0
    f[8] = my_bet / 80.0
    f[9] = opp_bet / 80.0
    f[10] = my_stack / 80.0
    f[11] = opp_stack / 80.0
    f[12] = to_call / float(pot + to_call) if to_call > 0 else 0.0
    f[13 + min(int(raises), 4)] = 1.0
    leg = set(int(a) for a in legal)
    f[18] = 1.0 if 0 in leg else 0.0
    f[19] = 1.0 if 2 in leg else 0.0
    f[20] = half_amt / 80.0
    f[21] = pot_amt / 80.0
    # per street: my raises, opp raises, my check/calls, opp check/calls, last aggressor (+1 me, -1 opp)
    n_cur = 0
    for (st, is_me, a) in hist:
        base = 22 + 5 * int(st)
        if a >= 2:
            f[base + (0 if is_me else 1)] += 0.25
            f[base + 4] = 1.0 if is_me else -1.0
        elif a == 1:
            f[base + (2 if is_me else 3)] += 0.25
        if int(st) == street:
            n_cur += 1
    f[42] = n_cur / 8.0
    return f


def card_features(hole, board):
    v = np.zeros(CARD_DIM, dtype=np.float32)
    for c in hole:
        c = int(c)
        v[c] = 1.0
        v[72 + (c >> 2)] += 0.5
        v[81 + (c & 3)] += 0.2
    for c in board:
        c = int(c)
        v[36 + c] = 1.0
        v[85 + (c >> 2)] += 0.5
        v[94 + (c & 3)] += 0.2
    return v


def strength_features(pre, raw, hs):
    return np.array([pre, raw, hs, raw - hs], dtype=np.float32)


def public_key(obs) -> str:
    st = int(obs.street)
    pot_start = obs.pot - obs.my_bet - obs.opp_bet
    seq = "".join("FCHP"[int(a.action)] for a in obs.action_history if a.street == obs.street)
    return f"{st}{'b' if obs.is_button else 'n'}.{pot_start}.{seq}"


def obs_node_features(obs):
    legal = [int(a) for a in obs.legal_actions]
    half_amt = obs.raise_to.get(2, obs.my_bet) - obs.my_bet if 2 in legal else 0
    pot_amt = obs.raise_to.get(3, obs.my_bet) - obs.my_bet if 3 in legal else 0
    # raise_to is keyed by Action (an IntEnum, so int keys also hash equal)
    hist = [(int(a.street), bool(a.is_me), int(a.action)) for a in obs.action_history]
    return node_features(int(obs.street), bool(obs.is_button), obs.pot, obs.to_call, obs.my_bet, obs.opp_bet,
                         obs.my_stack, obs.opp_stack, obs.raises_this_street, legal, half_amt, pot_amt, hist)


def obs_features(obs, pre, raw, hs, cur_score):
    """Full actor feature vector (without the key embedding) for an Observation.
    cur_score: best made-hand score on the visible board (ignored preflop)."""
    x = np.empty(BASE_DIM, dtype=np.float32)
    x[:CARD_DIM] = card_features(obs.hole_cards, obs.board)
    cat = np.zeros(CAT_DIM, dtype=np.float32)
    if len(obs.board) >= 3:
        cat[min(8, int(cur_score) >> 20)] = 1.0
    x[CARD_DIM:CARD_DIM + CAT_DIM] = cat
    x[CARD_DIM + CAT_DIM:CARD_DIM + CAT_DIM + STR_DIM] = strength_features(pre, raw, hs)
    x[CARD_DIM + CAT_DIM + STR_DIM:] = obs_node_features(obs)
    return x


class NNPolicy:
    """MLP policy: [base features, key embedding] -> ReLU layers -> 4 logits (F, C, H, P).

    A file may hold several members (keys prefixed m0_, m1_, ...): a PSRO-style
    population whose deployed strategy is the uniform MIXTURE over members, one member
    drawn per hand by the caller (strategy.py), so a best response must beat all of them."""

    def __init__(self, path: str = MODEL_PATH):
        z = np.load(path, allow_pickle=False)
        idx = sorted(int(f[1:-4]) for f in z.files if f.endswith("_emb") and f[0] == "m" and f[1:-4].isdigit())
        prefixes = [""] if "emb" in z.files else [f"m{i}_" for i in idx]
        self.members = []
        for pf in prefixes:
            Ws, bs = [], []
            k = 0
            while f"{pf}W{k}" in z.files:
                Ws.append(z[f"{pf}W{k}"].astype(np.float32))
                bs.append(z[f"{pf}b{k}"].astype(np.float32))
                k += 1
            self.members.append((z[f"{pf}emb"].astype(np.float32), Ws, bs))
        self.emb, self.Ws, self.bs = self.members[0]
        self.n_members = len(self.members)
        self.keys = {str(s): i for i, s in enumerate(z["keys"].tolist())}
        self.meta = {k: z[k] for k in z.files if k.startswith("meta_")}
        # optional: share of hands played by the CFR blueprint itself (a population member)
        self.bp_weight = float(self.meta["meta_bp_weight"]) if "meta_bp_weight" in self.meta else 0.0

    def logits(self, x: np.ndarray, key_id: int, member: int = 0) -> np.ndarray:
        emb, Ws, bs = self.members[member]
        h = np.concatenate([x, emb[key_id]])
        n = len(Ws)
        for i in range(n):
            h = h @ Ws[i] + bs[i]
            if i < n - 1:
                np.maximum(h, 0.0, out=h)
        return h

    def probs(self, obs, pre, raw, hs, cur_score, member: int = 0) -> np.ndarray | None:
        """Action probabilities over [F, C, H, P] (zero on illegal actions), or None
        if this betting state is unknown (caller falls back to the blueprint)."""
        kid = self.keys.get(public_key(obs))
        if kid is None:
            return None
        lg = self.logits(obs_features(obs, pre, raw, hs, cur_score), kid, member).astype(np.float64)
        mask = np.zeros(4, dtype=bool)
        for a in obs.legal_actions:
            mask[int(a)] = True
        lg = np.where(mask, lg, -np.inf)
        lg -= lg.max()
        p = np.exp(lg)
        p /= p.sum()
        return p
