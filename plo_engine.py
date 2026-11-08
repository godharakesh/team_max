"""
plo_engine.py - fast vectorised equity engine for Short Deck PLO5.

Idea: an Omaha hand is "best of 10 hole-pairs x 10 board-triples". We build a
table T[triple_index, pair_index] -> 5-card score (int32, 7140 x 630 = 18 MB)
once at startup from evaluator.py's own lookup table, so scores are exactly
the referee's. A Monte Carlo batch of N samples then becomes a couple of
numpy fancy-index gathers instead of N*200 Python dict lookups.

Also exposes a preflop strength array PRE_EQ indexed by the colex rank of a
sorted 5-card hand (equity vs a random hand, precomputed offline by
tools/build_preflop.py and stored in preflop_equity.npy).
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np

from evaluator import get_table

NUM_CARDS = 36
HERE = os.path.dirname(os.path.abspath(__file__))
PREFLOP_PATH = os.path.join(HERE, "preflop_equity.npy")

_PAIRS5 = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
_TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)


def _binom_table(n: int, k: int) -> np.ndarray:
    from math import comb
    return np.array([[comb(i, j) for j in range(k + 1)] for i in range(n + 1)], dtype=np.int64)


BINOM = _binom_table(NUM_CARDS, 5)

# Symmetric index lookups (any card order gives the same index).
IDX2 = np.full((NUM_CARDS, NUM_CARDS), 0, dtype=np.int32)
for _i, (_a, _b) in enumerate(combinations(range(NUM_CARDS), 2)):
    IDX2[_a, _b] = IDX2[_b, _a] = _i
IDX3 = np.full((NUM_CARDS, NUM_CARDS, NUM_CARDS), 0, dtype=np.int32)
for _i, (_a, _b, _c) in enumerate(combinations(range(NUM_CARDS), 3)):
    for _p in ((_a, _b, _c), (_a, _c, _b), (_b, _a, _c), (_b, _c, _a), (_c, _a, _b), (_c, _b, _a)):
        IDX3[_p] = _i
N_PAIRS = 630
N_TRIPLES = 7140

T: np.ndarray | None = None          # (7140, 630) int32 scores
PRE_EQ: np.ndarray | None = None     # (376992,) float32 preflop equity by colex rank
PRE_CLASS: np.ndarray | None = None  # (376992,) suit-isomorphism class id by colex rank
PRE_RANGE: np.ndarray | None = None  # (n_classes, len(PRE_RANGE_CUTS)) equity vs top ranges
PRE_RANGE_CUTS = np.array([0.0, 0.2, 0.4, 0.55, 0.7, 0.8, 0.9], dtype=np.float64)
PRE_RANGE_PATH = os.path.join(HERE, "preflop_range.npz")
PRE_SORTED: np.ndarray | None = None  # sorted PRE_EQ, for percentile lookups


def all_five_combos() -> np.ndarray:
    return np.array(list(combinations(range(NUM_CARDS), 5)), dtype=np.int64)


def colex5(sorted_hands: np.ndarray) -> np.ndarray:
    """Colex rank of ascending-sorted 5-card hands, shape (..., 5) -> (...).
    Matches the order of itertools.combinations(range(36), 5)? No - colex order
    differs from lexicographic; we only need a bijection, and build_preflop
    uses this same function, so it is consistent."""
    return (BINOM[sorted_hands[..., 0], 1] + BINOM[sorted_hands[..., 1], 2] + BINOM[sorted_hands[..., 2], 3]
            + BINOM[sorted_hands[..., 3], 4] + BINOM[sorted_hands[..., 4], 5])


def build_tables() -> None:
    """Build T (and load PRE_EQ if present). ~1 s."""
    global T, PRE_EQ
    if T is not None:
        return
    table = get_table()  # dict built in combinations(range(36), 5) order
    scores = np.fromiter(table.values(), dtype=np.int32, count=len(table))
    combos = all_five_combos()
    t = np.zeros((N_TRIPLES, N_PAIRS), dtype=np.int32)
    for pi, (i, j) in enumerate(_PAIRS5):
        rest = [k for k in range(5) if k not in (i, j)]
        p_idx = IDX2[combos[:, i], combos[:, j]]
        t_idx = IDX3[combos[:, rest[0]], combos[:, rest[1]], combos[:, rest[2]]]
        t[t_idx, p_idx] = scores
    T = t
    if os.path.isfile(PREFLOP_PATH):
        PRE_EQ = np.load(PREFLOP_PATH).astype(np.float32)
        global PRE_SORTED
        PRE_SORTED = np.sort(PRE_EQ)
    global PRE_CLASS, PRE_RANGE
    if os.path.isfile(PRE_RANGE_PATH):
        z = np.load(PRE_RANGE_PATH)
        PRE_CLASS = z["cls"].astype(np.int64)
        PRE_RANGE = z["eq"].astype(np.float32)


def pair_indices(hands: np.ndarray) -> np.ndarray:
    """(..., 5) hole cards -> (..., 10) pair indices."""
    return IDX2[hands[..., _PAIRS5[:, 0]], hands[..., _PAIRS5[:, 1]]]


def triple_indices(boards: np.ndarray) -> np.ndarray:
    """(..., 5) boards -> (..., 10) triple indices."""
    return IDX3[boards[..., _TRIPLES5[:, 0]], boards[..., _TRIPLES5[:, 1]], boards[..., _TRIPLES5[:, 2]]]


def best_scores(tri: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """tri (N,10), pairs (N,10) or (10,) -> (N,) best Omaha score."""
    if pairs.ndim == 1:
        return T[tri[:, :, None], pairs[None, None, :]].reshape(tri.shape[0], -1).max(axis=1)
    return T[tri[:, :, None], pairs[:, None, :]].reshape(tri.shape[0], -1).max(axis=1)


def preflop_strength(hands_sorted: np.ndarray) -> np.ndarray:
    """Preflop equity vs random for ascending-sorted hands (N,5)."""
    if PRE_EQ is None:
        return np.full(hands_sorted.shape[0], 0.5, dtype=np.float32)
    return PRE_EQ[colex5(hands_sorted)]


_ALL = np.arange(NUM_CARDS, dtype=np.int64)


def sample_unknown(rng: np.random.Generator, dead, n: int, k: int) -> np.ndarray:
    """n samples of k distinct cards from the deck minus `dead` -> (n, k)."""
    mask = np.ones(NUM_CARDS, dtype=bool)
    mask[list(dead)] = False
    live = _ALL[mask]
    keys = rng.random((n, live.size))
    pick = np.argpartition(keys, k, axis=1)[:, :k]
    return live[pick]


def equity(hole, board, n: int, rng: np.random.Generator,
           opp_cut: float = 0.0, pre_cut: float = 0.0, return_both: bool = False):
    """Monte Carlo equity of `hole` vs one opponent given the visible `board`.

    Opponent range model (rejection weighting):
      pre_cut  - drop opponent hands whose preflop strength is in the bottom
                 `pre_cut` fraction of the sampled hands (opp raised preflop).
      opp_cut  - drop opponent hands whose current made-hand strength on the
                 visible board is in the bottom `opp_cut` fraction (opp bet
                 postflop). Ignored preflop.
    Returns win + 0.5*tie probability (or (raw, range-cut) if return_both).
    """
    hole = list(hole)
    board = list(board)
    nb = len(board)
    need = 5 - nb
    s = sample_unknown(rng, hole + board, n, 5 + need)
    opp = s[:, :5]
    opp_pairs = pair_indices(opp)                       # (n, 10)
    my_pairs = pair_indices(np.array(hole, dtype=np.int64))  # (10,)
    if need:
        full = np.empty((n, 5), dtype=np.int64)
        if nb:
            full[:, :nb] = board
        full[:, nb:] = s[:, 5:]
        tri = triple_indices(full)                      # (n, 10)
        mine = T[tri[:, :, None], my_pairs[None, None, :]].reshape(n, -1).max(axis=1)
        theirs = T[tri[:, :, None], opp_pairs[:, None, :]].reshape(n, -1).max(axis=1)
    else:
        # river: our score is fixed, only the opponent's hand varies
        tri_row = triple_indices(np.array(board, dtype=np.int64))  # (10,)
        mine = T[tri_row[:, None], my_pairs[None, :]].max()
        theirs = T[tri_row[None, :, None], opp_pairs[:, None, :]].reshape(n, -1).max(axis=1)
    keep = None
    if pre_cut > 0 and PRE_EQ is not None:
        ps = PRE_EQ[colex5(np.sort(opp, axis=1))]
        keep = ps >= np.quantile(ps, pre_cut)
    if opp_cut > 0 and nb >= 3:
        if nb == 5:
            cur = theirs
        elif nb == 3:
            cur = T[IDX3[board[0], board[1], board[2]], opp_pairs].max(axis=1)
        else:
            tri_vis = np.array([IDX3[a, b, c] for a, b, c in combinations(board, 3)], dtype=np.int64)
            cur = T[tri_vis[None, :, None], opp_pairs[:, None, :]].reshape(n, -1).max(axis=1)
        k2 = cur >= np.quantile(cur, opp_cut)
        keep = k2 if keep is None else (keep & k2)
    w = (mine > theirs) + 0.5 * (mine == theirs)
    raw = float(w.mean())
    if keep is None or keep.sum() < 20:
        cut = raw
    else:
        cut = float(w[keep].mean())
    return (raw, cut) if return_both else cut


# ---------------------------------------------------------------------------
# Sample-once-per-street API: draw one Monte Carlo batch, then answer any
# number of range-cut queries from it for (almost) free.
# ---------------------------------------------------------------------------

class EquitySamples:
    """w: win(1)/tie(.5)/loss(0) per sample; pre/cur: opponent strength keys
    (preflop equity, current made-hand score on the visible board), with
    sorted copies for O(1) quantile lookups."""
    __slots__ = ("w", "pre", "cur", "pre_sorted", "cur_sorted", "raw", "n", "hs_w")

    def __init__(self, w, pre, cur, hs_w=None):
        self.w, self.pre, self.cur = w, pre, cur
        self.hs_w = hs_w   # per-sample win/tie vs opp on the CURRENT board (no runout): made-hand strength
        self.n = w.shape[0]
        self.raw = float(w.mean())
        self.pre_sorted = np.sort(pre) if pre is not None else None
        self.cur_sorted = np.sort(cur) if cur is not None else None

    def merged(self, other: "EquitySamples") -> "EquitySamples":
        """Pool two batches drawn for the same situation."""
        cat = lambda a, b: None if a is None or b is None else np.concatenate([a, b])
        return EquitySamples(np.concatenate([self.w, other.w]), cat(self.pre, other.pre), cat(self.cur, other.cur),
                             cat(self.hs_w, other.hs_w))

    @property
    def hs(self) -> float:
        """Current made-hand strength vs a random hand (= equity on the river)."""
        return self.raw if self.hs_w is None else float(self.hs_w.mean())

    def cut(self, pre_cut: float = 0.0, opp_cut: float = 0.0) -> float:
        keep = None
        if pre_cut > 0 and self.pre is not None:
            keep = self.pre >= self.pre_sorted[min(self.n - 1, int(pre_cut * self.n))]
        if opp_cut > 0 and self.cur is not None:
            k2 = self.cur >= self.cur_sorted[min(self.n - 1, int(opp_cut * self.n))]
            keep = k2 if keep is None else (keep & k2)
        if keep is None:
            return self.raw
        cnt = int(keep.sum())
        if cnt < 16:
            return self.raw
        return float(self.w[keep].sum() / cnt)


def sample_equity(hole, board, n: int, rng: np.random.Generator) -> EquitySamples:
    hole = list(hole)
    board = list(board)
    nb = len(board)
    need = 5 - nb
    s = sample_unknown(rng, hole + board, n, 5 + need)
    opp = s[:, :5]
    opp_pairs = pair_indices(opp)
    my_pairs = pair_indices(np.array(hole, dtype=np.int64))
    if need:
        full = np.empty((n, 5), dtype=np.int64)
        if nb:
            full[:, :nb] = board
        full[:, nb:] = s[:, 5:]
        tri = triple_indices(full)
        mine = T[tri[:, :, None], my_pairs[None, None, :]].reshape(n, -1).max(axis=1)
        theirs = T[tri[:, :, None], opp_pairs[:, None, :]].reshape(n, -1).max(axis=1)
    else:
        tri_row = triple_indices(np.array(board, dtype=np.int64))
        mine = T[tri_row[:, None], my_pairs[None, :]].max()
        theirs = T[tri_row[None, :, None], opp_pairs[:, None, :]].reshape(n, -1).max(axis=1)
    w = (mine > theirs) + 0.5 * (mine == theirs)
    pre = PRE_EQ[colex5(np.sort(opp, axis=1))] if PRE_EQ is not None else None
    hs_w = None
    if nb == 5:
        cur = theirs
    elif nb == 3:
        t0 = IDX3[board[0], board[1], board[2]]
        cur = T[t0, opp_pairs].max(axis=1)
        my_cur = T[t0, my_pairs].max()
        hs_w = (my_cur > cur) + 0.5 * (my_cur == cur)
    elif nb == 4:
        tri_vis = np.array([IDX3[a, b, c] for a, b, c in combinations(board, 3)], dtype=np.int64)
        cur = T[tri_vis[None, :, None], opp_pairs[:, None, :]].reshape(n, -1).max(axis=1)
        my_cur = T[tri_vis[:, None], my_pairs[None, :]].max()
        hs_w = (my_cur > cur) + 0.5 * (my_cur == cur)
    else:
        cur = None
    return EquitySamples(w, pre, cur, hs_w)



def current_score(hole, board_vis) -> int:
    """Best Omaha score (exactly 2 hole + 3 board) on a 3, 4 or 5 card board."""
    pairs = pair_indices(np.array(list(hole), dtype=np.int64))
    tri = np.array([IDX3[a, b, c] for a, b, c in combinations(list(board_vis), 3)], dtype=np.int64)
    return int(T[tri[:, None], pairs[None, :]].max())


# ---------------------------------------------------------------------------
# Faster, lower-variance estimators exploiting Omaha's 2+3 structure
# (see BOT_DESIGN.md). On a COMPLETE board every hole pair's best score is a
# max over the board's 10 triples, computed once for all 630 pairs; each
# opponent then costs 10 lookups instead of 100.
# ---------------------------------------------------------------------------

_P5 = _PAIRS5
RIVER_MULT = 3           # river draws RIVER_MULT x the nominal sample count (biggest pots get the most precision)


def _river_fast(hole, board, n, rng) -> EquitySamples:
    tri_row = triple_indices(np.array(board, dtype=np.int64))       # (10,)
    best_p = T[tri_row].max(axis=0)                                  # (630,)
    mine = best_p[pair_indices(np.array(hole, dtype=np.int64))].max()
    opp = sample_unknown(rng, list(hole) + list(board), n * RIVER_MULT, 5)
    theirs = best_p[pair_indices(opp)].max(axis=1)
    w = (mine > theirs) + 0.5 * (mine == theirs)
    pre = PRE_EQ[colex5(np.sort(opp, axis=1))] if PRE_EQ is not None else None
    return EquitySamples(w, pre, theirs, None)


def _turn_stratified(hole, board, n, rng) -> EquitySamples:
    """Stratify over ALL possible river cards (equal weight), sampling M
    opponents per river: the runout variance disappears."""
    hole = list(hole); board = list(board)
    mask = np.ones(NUM_CARDS, dtype=bool)
    mask[hole + board] = False
    U = _ALL[mask]                                                   # 27 unseen cards
    R = U.size
    M = max(2, -(-n // R))
    boards5 = np.empty((R, 5), dtype=np.int64)
    boards5[:, :4] = board
    boards5[:, 4] = U
    best_p = T[triple_indices(boards5)].max(axis=1)                  # (R, 630)
    my_pairs = pair_indices(np.array(hole, dtype=np.int64))
    my_best = best_p[:, my_pairs].max(axis=1)                        # (R,)
    keys = rng.random((R, M, R))
    keys[np.arange(R), :, np.arange(R)] = 2.0                        # never deal the river card to the opponent
    opp = U[np.argpartition(keys, 5, axis=2)[:, :, :5]]              # (R, M, 5)
    opp_pairs = IDX2[opp[..., _P5[:, 0]], opp[..., _P5[:, 1]]]       # (R, M, 10)
    theirs = best_p[np.arange(R)[:, None, None], opp_pairs].max(axis=2)
    w = ((my_best[:, None] > theirs) + 0.5 * (my_best[:, None] == theirs)).ravel()
    tri_vis = np.array([IDX3[a, b, c] for a, b, c in combinations(board, 3)], dtype=np.int64)
    flat_pairs = opp_pairs.reshape(-1, 10)
    cur = T[tri_vis[None, :, None], flat_pairs[:, None, :]].reshape(flat_pairs.shape[0], -1).max(axis=1)
    my_cur = T[tri_vis[:, None], my_pairs[None, :]].max()
    hs_w = (my_cur > cur) + 0.5 * (my_cur == cur)
    flat_opp = opp.reshape(-1, 5)
    pre = PRE_EQ[colex5(np.sort(flat_opp, axis=1))] if PRE_EQ is not None else None
    return EquitySamples(w, pre, cur, hs_w)


def sample_equity_fast(hole, board, n: int, rng: np.random.Generator) -> EquitySamples:
    """Same estimand as sample_equity (unbiased), lower variance per unit time
    on the turn and river; the flop/preflop use the original estimator."""
    nb = len(board)
    if nb == 5:
        return _river_fast(hole, board, n, rng)
    if nb == 4:
        return _turn_stratified(hole, board, n, rng)
    return sample_equity(hole, board, n, rng)
