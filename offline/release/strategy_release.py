"""
strategy.py - decision logic for the Short Deck PLO5 bot (imported by agent.py).

Pipeline per decision:
  1. Opponent model (memory["opp"]): Bayesian-smoothed frequencies of how often
     the opponent bets/raises when given the chance, preflop and postflop, and
     how often it folds to our bets.
  2. Range estimate: if the opponent has been aggressive this hand, we discard
     the weakest part of their random-hand distribution, sized by how selective
     their aggression is (a maniac's raise means little, a nit's means a lot).
  3. Equity: preflop from the precomputed table (+ MC when a range cut applies),
     postflop by vectorised Monte Carlo (plo_engine), sample count scaled to the
     remaining time bank.
  4. Action: equity vs pot odds with value / semi-bluff / bluff thresholds.
All functions are pure w.r.t. `memory`; agent.py wraps them in try/except.
"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np

import plo_engine as E
from betting import infoset_key, bucket as _bucket
from game import Action, Street

# Our own compute budget per half. The judge also charges ~0.4-0.6 ms of
# server/JSON overhead per call (about 4 calls per hand, ~2-2.5 s a half), so
# the 8 s bank leaves us roughly 5 s; we aim for 2.5 s: measured judge charge
# is ~3.0 s on an idle machine and ~6 s with heavy CPU contention.
TIME_BUDGET = 2.5
PANIC_BUDGET = 3.2        # beyond this: tiny Monte Carlo only
N_MIN, N_MAX, N_PANIC = 48, 400, 24
BUCKET_SAMPLES = 200      # = offline/build_deals.py sample counts

# MODE: "mix" (blueprint + safe exploitation, default), "bp" (pure blueprint),
# "heur" (pure exploitative heuristic). Env var only for local experiments.
MODE = os.environ.get("BOT_MODE", "mix")
EXPLOIT_RAMP_HANDS = float(os.environ.get("BOT_RAMP", "60"))      # hands until the model is trusted
EXPLOIT_FULL_DEV = float(os.environ.get("BOT_FULLDEV", "0.9"))    # deviation that means "fully exploit"
EXPLOIT_FLOOR = float(os.environ.get("BOT_FLOOR", "0.3"))         # minimum exploit share once trusted

_BP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blueprint.pkl")
BLUEPRINT = None
THRESHOLDS = None
try:
    with open(_BP_PATH, "rb") as _f:
        _bp = pickle.load(_f)
    BLUEPRINT, THRESHOLDS = _bp["policy"], _bp["thresholds"]
except Exception:
    BLUEPRINT = None


# ---------------------------------------------------------------------------
# memory / opponent model
# ---------------------------------------------------------------------------

def init_memory(memory: dict, seed: int = 12345) -> None:
    memory["rng"] = np.random.default_rng(seed)
    memory["think"] = 0.0            # seconds spent in choose_action
    memory["n_decisions"] = 0
    memory["opp"] = {
        # [aggressive actions, opportunities]
        "pre_aggr": [0, 0], "post_aggr": [0, 0],
        # opponent folded when facing our bet/raise: [folds, times faced]
        "fold_to_bet": [0, 0],
        # opponent called/checked when facing our bet (for calling-station detection)
        "showdowns": 0, "hands": 0,
    }
    memory["cur_hand"] = -1
    memory["seen_hist"] = 0
    memory["we_bet_last"] = False    # our most recent action this hand was a raise
    memory["eq_cache"] = None        # (hand, street, cuts) -> (raw, cut)
    memory["bp_hits"] = 0
    memory["eq_calls"] = 0           # number of Monte Carlo equity calls this half
    memory["eq_time"] = 0.0          # seconds spent inside them
    memory["eq_samples"] = 0         # samples drawn in them
    memory["refines"] = 0            # extra batches drawn for close decisions
    memory["cpu0"] = time.process_time()  # process CPU at half start (server overhead + our compute)


def _freq(pair, prior_rate: float, prior_n: float) -> float:
    return (pair[0] + prior_rate * prior_n) / (pair[1] + prior_n)


def _ingest_history(obs, memory: dict) -> None:
    """Count opponent actions we have not processed yet in this hand."""
    if memory["cur_hand"] != obs.hand_index:
        memory["cur_hand"] = obs.hand_index
        memory["seen_hist"] = 0
        memory["we_bet_last"] = False
    opp = memory["opp"]
    hist = obs.action_history
    for i in range(memory["seen_hist"], len(hist)):
        a = hist[i]
        if a.is_me:
            continue
        # Did the opponent face a bet from us right before this action?
        faced_bet = i > 0 and hist[i - 1].is_me and hist[i - 1].action in (Action.RAISE_HALF, Action.RAISE_POT)
        key = "pre_aggr" if a.street == Street.PREFLOP else "post_aggr"
        opp[key][1] += 1
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            opp[key][0] += 1
        if faced_bet:
            opp["fold_to_bet"][1] += 1  # (fold itself ends the hand; counted in on_hand_end)
    memory["seen_hist"] = len(hist)


_DEBUG = bool(os.environ.get("BOT_DEBUG"))


def record_hand_end(result, memory: dict) -> None:
    if _DEBUG and result.hand_index % 250 == 249:
        import sys
        print(f"[bot] hand {result.hand_index}: think {memory['think']:.2f}s "
              f"cpu {time.process_time() - memory['cpu0']:.2f}s eq_calls {memory['eq_calls']} "
              f"avg_n {memory['eq_samples'] / max(1, memory['eq_calls']):.0f} decisions {memory['n_decisions']} "
              f"refines {memory['refines']}", file=sys.stderr, flush=True)
    opp = memory["opp"]
    opp["hands"] += 1
    if result.went_to_showdown:
        opp["showdowns"] += 1
    elif result.my_net > 0 and memory.get("we_bet_last") and memory.get("cur_hand") == result.hand_index:
        # we bet, they folded
        opp["fold_to_bet"][0] += 1
        opp["fold_to_bet"][1] += 1
    memory["we_bet_last"] = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _opp_aggr_this_hand(obs) -> tuple[int, int]:
    """(opponent raises preflop, opponent raises on the current street)."""
    pre = cur = 0
    for a in obs.action_history:
        if not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            if a.street == Street.PREFLOP:
                pre += 1
            if a.street == obs.street:
                cur += 1
    return pre, cur


def _opp_post_raises_total(obs) -> int:
    return sum(1 for a in obs.action_history
               if not a.is_me and a.street != Street.PREFLOP and a.action in (Action.RAISE_HALF, Action.RAISE_POT))


def _cut_from_rate(rate: float, n_raises: int, cap: float) -> float:
    """Range cut implied by aggression: raising with frequency r means the
    raising range is roughly the top r of hands; each extra raise narrows it."""
    if n_raises <= 0:
        return 0.0
    keep = rate ** (min(n_raises, 3) ** 0.5)
    keep = max(1.0 - cap, min(1.0, keep * 1.15))  # be a bit generous (they bluff too)
    return 1.0 - keep


def _n_samples(obs, memory: dict) -> int:
    used = memory["think"]
    if used > PANIC_BUDGET:
        return N_PANIC
    hi = obs.hand_index
    hands_left = max(1, 1000 - hi)
    # Self-calibrating: everything below is measured online on THIS machine.
    eq_calls_per_hand = (memory["eq_calls"] + 2.2 * 20) / (hi + 20)
    non_eq_per_hand = (used - memory["eq_time"] + 0.0004 * 20) / (hi + 20)
    sec_per_sample = (memory["eq_time"] + 0.0015 * 5) / (memory["eq_samples"] + 200 * 5)
    remaining = TIME_BUDGET - used - non_eq_per_hand * hands_left
    per_call = remaining / (hands_left * max(eq_calls_per_hand, 0.5))
    n = int(per_call / sec_per_sample)
    return max(N_MIN, min(n, N_MAX))


def _preflop_eq(hole) -> float:
    if E.PRE_EQ is None:
        return 0.5
    h = np.array(sorted(hole), dtype=np.int64)
    return float(E.PRE_EQ[E.colex5(h)])


def _legal(obs, a: Action) -> bool:
    return a in obs.legal_actions


def _raise_or(obs, prefer: Action) -> Action:
    if _legal(obs, prefer):
        return prefer
    other = Action.RAISE_HALF if prefer == Action.RAISE_POT else Action.RAISE_POT
    if _legal(obs, other):
        return other
    return Action.CHECK_CALL


# ---------------------------------------------------------------------------
# main decision
# ---------------------------------------------------------------------------

def decide(obs, memory: dict) -> Action:
    _ingest_history(obs, memory)
    opp = memory["opp"]
    rng = memory["rng"]

    pre_rate = _freq(opp["pre_aggr"], 0.35, 12)
    post_rate = _freq(opp["post_aggr"], 0.30, 12)
    fold_rate = _freq(opp["fold_to_bet"], 0.35, 10)

    pre_raises, cur_raises = _opp_aggr_this_hand(obs)
    post_raises = _opp_post_raises_total(obs)
    pre_cut = _cut_from_rate(pre_rate, pre_raises, 0.75)
    post_cut = _cut_from_rate(post_rate, post_raises, 0.7)

    pot, to_call = obs.pot, obs.to_call
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
    raw, eq = _equities(obs, memory, pre_cut, post_cut, post_raises)
    # Close call on a big pot: buy more precision if we are ahead of time pace.
    if (obs.street != Street.PREFLOP and to_call > 0 and pot >= 12 * obs.big_blind
            and memory["think"] < TIME_BUDGET * (obs.hand_index + 1) / 1000.0 * 0.85):
        c = memory["eq_cache"]
        se = 0.5 / max(1, c[1].n) ** 0.5
        if abs(eq - pot_odds) < 1.5 * se and c[1].n < 900:
            more = _timed_sample(memory, obs.hole_cards, obs.board, 300)
            memory["eq_cache"] = (c[0], c[1].merged(more))
            memory["refines"] += 1
            raw, eq = _equities(obs, memory, pre_cut, post_cut, post_raises)
    # ---------------- action ----------------
    mode = memory.get("mode", MODE)
    heur = _choose(obs, eq, pot_odds, fold_rate, post_rate, rng)
    action = heur
    if mode != "heur":
        bp = _blueprint_action(obs, raw, rng)
        if bp is not None:
            memory["bp_hits"] += 1
            w = 0.0 if mode == "bp" else _exploit_weight(memory, pre_rate, post_rate, fold_rate)
            action = heur if rng.random() < w else bp
    if action in (Action.RAISE_HALF, Action.RAISE_POT):
        memory["we_bet_last"] = True
    else:
        memory["we_bet_last"] = False
    return action


def _equities(obs, memory, pre_cut, post_cut, post_raises):
    """(raw equity vs random -> card bucket, range-adjusted equity -> exploit).

    One Monte Carlo batch per (hand, street); every later query on the same
    street (e.g. after the opponent re-raises and the range cut changes) is
    answered from the cached samples in microseconds."""
    key = (obs.hand_index, int(obs.street))
    c = memory.get("eq_cache")
    if c is None or c[0] != key:
        if obs.street == Street.PREFLOP:
            samples = None          # filled lazily: only needed when the opponent raised
        else:
            samples = _timed_sample(memory, obs.hole_cards, obs.board, _n_samples(obs, memory))
        c = (key, samples)
        memory["eq_cache"] = c
    samples = c[1]
    if obs.street == Street.PREFLOP:
        pre = _preflop_eq(obs.hole_cards)
        if pre_cut <= 0:
            return pre, pre
        tab = _preflop_range_eq(obs.hole_cards, pre_cut)
        if tab is not None:
            return pre, tab
        if samples is None:
            samples = _timed_sample(memory, obs.hole_cards, [], _n_samples(obs, memory))
            memory["eq_cache"] = (key, samples)
        return pre, samples.cut(pre_cut=pre_cut)
    return samples.raw, samples.cut(pre_cut=pre_cut, opp_cut=post_cut)


def _timed_sample(memory, hole, board, n):
    t0 = time.perf_counter()
    try:
        return E.sample_equity(hole, board, n, memory["rng"])
    finally:
        memory["eq_time"] += time.perf_counter() - t0
        memory["eq_samples"] += n
        memory["eq_calls"] += 1


def _preflop_range_eq(hole, pre_cut):
    """Equity vs the top (1-pre_cut) of hands from the offline table (linear
    interpolation between the tabulated cut levels), or None if absent."""
    if E.PRE_RANGE is None:
        return None
    row = E.PRE_RANGE[E.PRE_CLASS[E.colex5(np.array(sorted(hole), dtype=np.int64))]]
    cuts = E.PRE_RANGE_CUTS
    if pre_cut >= cuts[-1]:
        return float(row[-1])
    i = int(np.searchsorted(cuts, pre_cut, side="right")) - 1
    f = (pre_cut - cuts[i]) / (cuts[i + 1] - cuts[i])
    return float(row[i] * (1 - f) + row[i + 1] * f)


def _blueprint_action(obs, raw_eq, rng):
    if BLUEPRINT is None:
        return None
    st = int(obs.street)
    b = _bucket(st, raw_eq, THRESHOLDS)
    pot_start = obs.pot - obs.my_bet - obs.opp_bet
    seq = "".join("FCHP"[int(a.action)] for a in obs.action_history if a.street == obs.street)
    probs = BLUEPRINT.get(infoset_key(st, b, obs.is_button, pot_start, seq))
    if probs is None:
        return None
    legal = [int(a) for a in obs.legal_actions]
    ws = [probs[a] for a in legal]
    tot = sum(ws)
    if tot <= 0:
        return None
    x = rng.random() * tot
    for a, wv in zip(legal, ws):
        x -= wv
        if x <= 0:
            return Action(a)
    return Action(legal[-1])


def _exploit_weight(memory, pre_rate, post_rate, fold_rate) -> float:
    """Safe exploitation (RNR-style): trust the exploitative policy only as far
    as the opponent model is (a) confident and (b) far from balanced play."""
    hands = memory["opp"]["hands"]
    conf = min(1.0, hands / EXPLOIT_RAMP_HANDS)
    dev = abs(post_rate - 0.30) / 0.30 + abs(pre_rate - 0.35) / 0.35 + abs(fold_rate - 0.35) / 0.35
    # Floor: even vs balanced opponents the exploit policy keeps a share (it is
    # not exploitable by our own blueprint, see BOT_DESIGN.md); clearly
    # unbalanced opponents (dev >= EXPLOIT_FULL_DEV) get full exploitation.
    return conf * max(EXPLOIT_FLOOR, min(1.0, dev / EXPLOIT_FULL_DEV))


def _choose(obs, eq: float, pot_odds: float, fold_rate: float, opp_aggr: float, rng) -> Action:
    street = obs.street
    to_call = obs.to_call
    can_raise = _legal(obs, Action.RAISE_POT) or _legal(obs, Action.RAISE_HALF)
    r = rng.random()

    if street == Street.PREFLOP:
        # Preflop equities in PLO5 run close (roughly 0.30-0.70 vs random).
        if to_call > 0:
            if eq >= 0.62 and can_raise and obs.raises_this_street < 3:
                return _raise_or(obs, Action.RAISE_POT)
            if eq >= 0.56 and can_raise and obs.raises_this_street < 2:
                return _raise_or(obs, Action.RAISE_POT if obs.is_button else Action.RAISE_HALF)
            # position bonus: in position we realise more equity
            margin = 0.04 if obs.is_button or obs.raises_this_street == 0 else 0.0
            if obs.to_call <= obs.big_blind and eq >= 0.40:
                return Action.CHECK_CALL
            if eq + margin >= pot_odds + 0.06:
                return Action.CHECK_CALL
            return Action.FOLD if _legal(obs, Action.FOLD) else Action.CHECK_CALL
        # big blind option after a limp
        if eq >= 0.57 and can_raise:
            return _raise_or(obs, Action.RAISE_POT)
        return Action.CHECK_CALL

    # ---------------- postflop ----------------
    value_pot = {Street.FLOP: 0.70, Street.TURN: 0.72, Street.RIVER: 0.75}[street]
    value_half = {Street.FLOP: 0.60, Street.TURN: 0.62, Street.RIVER: 0.66}[street]

    if to_call == 0:
        if eq >= value_pot and can_raise:
            return _raise_or(obs, Action.RAISE_POT)
        if eq >= value_half and can_raise:
            return _raise_or(obs, Action.RAISE_HALF)
        # bluff / semi-bluff vs opponents that fold a lot
        if can_raise:
            bluff_p = max(0.0, (fold_rate - 0.30)) * 1.2
            if street != Street.RIVER and eq >= 0.40:
                bluff_p += 0.12
            if r < bluff_p:
                return _raise_or(obs, Action.RAISE_HALF)
        return Action.CHECK_CALL

    # facing a bet
    if eq >= value_pot + 0.05 and can_raise:
        return _raise_or(obs, Action.RAISE_POT)
    if eq >= value_pot - 0.02 and can_raise and obs.raises_this_street < 2 and r < 0.6:
        return _raise_or(obs, Action.RAISE_HALF)
    # call if equity beats pot odds (small implied-odds bonus before the river)
    bonus = 0.03 if street != Street.RIVER else 0.0
    if eq + bonus >= pot_odds:
        return Action.CHECK_CALL
    return Action.FOLD if _legal(obs, Action.FOLD) else Action.CHECK_CALL


def timed_decide(obs, memory: dict) -> Action:
    t0 = time.perf_counter()
    try:
        return decide(obs, memory)
    finally:
        memory["think"] += time.perf_counter() - t0
        memory["n_decisions"] += 1
