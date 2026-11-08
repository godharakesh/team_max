"""
strategy.py - decision logic for the Short Deck PLO5 bot (imported by agent.py).

Pipeline per decision:
  1. Opponent model (memory["opp"]): Bayesian-smoothed frequencies of the
     opponent's preflop / postflop aggression and fold-to-our-bets, plus
     showdown calibration: the true strength percentile of every hand the
     opponent raised with and later showed down.
  2. Range estimate: when the opponent raises, the weakest part of their
     possible hands is cut away. The kept fraction comes from their raise
     frequency, corrected by what their shown raising hands actually were.
  3. Equity: preflop from precomputed tables; postflop one vectorised Monte
     Carlo batch per street (plo_engine), sized by a self-calibrating time
     manager and refined for close calls on big pots.
  4. Action: CFR blueprint (potential-aware abstraction, MCCFR) mixed with an
     exploitative equity-vs-pot-odds policy, RNR-style: the exploit share
     grows with confidence in the opponent model and its imbalance.
All tunable numbers live in PARAMS (overridable per instance via
memory["params"] for offline tuning). agent.py wraps everything in try/except.
"""

from __future__ import annotations

import bisect
import math
import os
import pickle
import time

import numpy as np

import plo_engine as E
from betting import infoset_key, bucket as _bucket_v1
from game import Action, Street

HERE = os.path.dirname(os.path.abspath(__file__))

# Our own compute budget per half. The judge also charges HTTP/JSON overhead
# (measured: judge charge ~= our compute + 0.5 s on an idle machine, up to
# ~+4 s under heavy CPU contention), so we aim for 2.5 s of the 8 s bank.
TIME_BUDGET = 2.5
PANIC_BUDGET = 3.2        # beyond this: tiny Monte Carlo only
# Offline-only: arena runs 8 processes on 4 physical cores (~60% speed each),
# so experiments scale the budget to keep tournament-like sample counts.
# Never set in the tournament.
_TB_SCALE = float(os.environ.get("BOT_BUDGET_SCALE", "1.0"))
TIME_BUDGET *= _TB_SCALE
PANIC_BUDGET *= _TB_SCALE
N_MIN, N_MAX, N_PANIC = 48, 1500, 24   # the time manager still enforces TIME_BUDGET (the README says how this cap was set)
N_MAX = int(os.environ.get("BOT_NMAX", N_MAX))   # offline experiments only

# MODE: "mix" (blueprint + safe exploitation), "bp" (pure blueprint),
# "heur" (pure exploitative heuristic), "nn" (deep-RL policy, nn_policy.npz, the default; falls
# back to the blueprint on any problem), "nnmix" (mix with the deep-RL policy in place
# of the blueprint). Env var only for local experiments.
MODE = os.environ.get("BOT_MODE", "nn")

PARAMS = {
    # range model
    "gen": 1.15,            # generosity: kept range = raise-rate x gen (they bluff too)
    "cap_pre": 0.75, "cap_post": 0.70,   # max fraction of hands cut away
    "sd_prior": 8.0,        # pseudo-count of the frequency estimate vs showdown evidence
    # preflop (equity thresholds)
    "pf_raise_big": 0.62, "pf_raise": 0.56, "pf_limp": 0.40, "pf_call_margin": 0.06,
    "pf_pos_bonus": 0.04, "bb_raise": 0.57,
    # postflop value-bet thresholds (pot / half pot) per street
    "vp_f": 0.70, "vp_t": 0.72, "vp_r": 0.75, "vh_f": 0.60, "vh_t": 0.62, "vh_r": 0.66,
    # bluffing
    "bl_base": 0.30, "bl_slope": 1.2, "semi_p": 0.12, "semi_eq": 0.40,
    # facing bets
    "rr_margin": 0.05, "call_bonus": 0.03,
    # blueprint / exploit mixing: deviation from equilibrium behaviour below dev0
    # -> pure blueprint; above dev1 -> pure exploitation; linear in between
    "ramp": 60.0, "dev0": 1.5, "dev1": 4.0,
    # aggression gate: exploit passive opponents, respect aggressive ones. The exploit
    # weight fades from full at agg_a0 to zero at agg_a1, where aggression = how far the
    # opponent's raise rate (preflop or postflop, whichever is higher) sits above
    # equilibrium, in units of typical spread. The strong field bots are hyper-aggressive
    # and punish our exploitation; the leaky weak bots are passive (dev1 batch, Nov 7).
    "agg_gate": 0.0, "agg_a0": 0.0, "agg_a1": 1.0,
    # restricted Nash responses (mode "rnr", or rnr_on_mix with mode "mix"): blend in the
    # counter-strategy of the archetype the opponent most resembles, by how much closer it
    # is to that archetype than to equilibrium (rnr_m0 -> none, rnr_m1 -> rnr_wmax)
    "rnr_m0": 0.25, "rnr_m1": 1.0, "rnr_wmax": 1.0, "rnr_maniac": 0.0, "rnr_on_mix": 0.0,
    "wfix": -1.0,           # >= 0: fixed exploit share (experiments); < 0: deviation-based
    "range_v2": 0.0,        # 1: per-street, size-aware, per-street-calibrated opponent ranges
    "risk_lambda": 0.0,     # > 0: mean-variance penalty (per bb) on calls facing bets (CI-verdict aware)
    "fast_eq": 1.0,         # 1: pair-decomposed river + river-stratified turn estimators (unbiased, ~2.5x efficient)
    # implicit-modelling strategy selector (Bard et al. 2013): per hand play arm M (mix) or
    # H (pure heuristic); score BOTH arms on every hand by importance sampling
    "selector": 0.0, "sel_wmin": 0.03, "sel_prior_m": -0.1, "sel_prior_n": 30.0, "sel_gamma": 0.997,
    "sel_lo": 0.60, "sel_hi": 0.95,   # P(H better) below lo -> floor; above hi -> full H; linear between
}

# Equilibrium reference: how our own CFR blueprint (bp2) looks through this
# opponent model in self-play (offline/opp_stats.py), and the typical spread.
EQ_REF = {"pre": 0.49, "post": 0.415, "fold": 0.40, "sd_pre": 0.73, "sd_post": 0.665}
EQ_SCALE = {"pre": 0.25, "post": 0.20, "fold": 0.20, "sd_pre": 0.12, "sd_post": 0.10}
_TUNED_PATH = os.path.join(HERE, "params_tuned.json")
if os.path.isfile(_TUNED_PATH):          # written by offline/tune.py after validation
    try:
        import json
        with open(_TUNED_PATH) as _f:
            PARAMS.update({k: float(v) for k, v in json.load(_f).items() if k in PARAMS})
    except Exception:
        pass
# Offline experiments only: BOT_PARAMS='{"dev0": 2.0}' overrides PARAMS.
if os.environ.get("BOT_PARAMS"):
    try:
        import json as _json
        PARAMS.update({k: float(v) for k, v in _json.loads(os.environ["BOT_PARAMS"]).items() if k in PARAMS})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# blueprints
# ---------------------------------------------------------------------------

def _load_v1():
    with open(os.path.join(HERE, "blueprint.pkl"), "rb") as f:
        bp = pickle.load(f)
    return {"version": 1, "policy": bp["policy"], "thresholds": bp["thresholds"]}


def _load_v2(name="blueprint2.pkl"):
    with open(os.path.join(HERE, name), "rb") as f:
        bp = pickle.load(f)
    ab = bp["abstraction"]
    return {"version": 2, "keys": bp["keys"], "key_offset": bp["key_offset"].astype(np.int64),
            "policy": bp["policy"].astype(np.float32), "pre": list(ab["pre"]), "riv": list(ab["riv"]),
            "np": int(ab["np"]),
            "e": {1: list(ab["flop_e"]), 2: list(ab["turn_e"])},
            "p": {1: [list(x) for x in ab["flop_p"]], 2: [list(x) for x in ab["turn_p"]]}}


BLUEPRINTS = {}
for _v, _loader in ((1, _load_v1), (2, _load_v2), (3, lambda: _load_v2("blueprint3.pkl")),
                    (4, lambda: _load_v2("blueprint2L.pkl"))):
    try:
        BLUEPRINTS[_v] = _loader()
    except Exception:
        pass
# The shipped blueprint version (chosen by head-to-head evaluation; see BOT_DESIGN.md)
_SHIP_BPV = int(os.environ.get("BOT_BPV", "2"))
DEFAULT_BPV = _SHIP_BPV if _SHIP_BPV in BLUEPRINTS else max(BLUEPRINTS, default=0)

# Restricted Nash Responses (Johanson et al. 2007; offline/rnr_jit.py): robust
# counter-strategies to opponent archetypes, solved on blueprint v2's abstraction,
# so they share its infoset layout (one bucket lookup serves every table).
# ARCH_PROFILE: how each archetype looks through our tracker (offline/arch_profiles.py,
# scaled by EQ_REF / simulated-equilibrium so it matches real-game measurements).
ARCH_PROFILE = {
    "station": {"pre": 0.08, "post": 0.09, "fold": 0.09},
    "nit": {"pre": 0.14, "post": 0.20, "fold": 0.55},
    "maniac": {"pre": 0.61, "post": 0.73, "fold": 0.15},
    "passive": {"pre": 0.10, "post": 0.16, "fold": 0.27},
}
RNR = {}
for _name in ARCH_PROFILE:
    try:
        with open(os.path.join(HERE, f"rnr_{_name}.pkl"), "rb") as _f:
            _t = pickle.load(_f)
        if 2 in BLUEPRINTS and _t["keys"] == BLUEPRINTS[2]["keys"]:
            RNR[_name] = _t["policy"].astype(np.float32)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# memory / opponent model
# ---------------------------------------------------------------------------

def init_memory(memory: dict, seed: int = 12345) -> None:
    memory["rng"] = np.random.default_rng(seed)
    memory["think"] = 0.0            # seconds spent in choose_action
    memory["n_decisions"] = 0
    memory["opp"] = {
        "pre_aggr": [0, 0], "post_aggr": [0, 0],     # [aggressive actions, opportunities]
        "fold_to_bet": [0, 0],                       # [folds, times facing our bet]
        "showdowns": 0, "hands": 0,
        # showdown calibration: [sum of strength percentiles, count] of hands they raised with
        "sd_pre": [0.0, 0], "sd_post": [0.0, 0],
        # range model v2: per postflop street (1=flop, 2=turn, 3=river)
        "st_aggr": {1: [0, 0], 2: [0, 0], 3: [0, 0]},   # [raises, opportunities]
        "st_pot": {1: 0, 2: 0, 3: 0},                   # how many of those raises were pot-size
        "sd_st": {1: [0.0, 0], 2: [0.0, 0], 3: [0.0, 0]},  # per-street showdown calibration
    }
    memory["cur_hand"] = -1
    memory["seen_hist"] = 0
    memory["we_bet_last"] = False    # our most recent action this hand was a raise
    memory["eq_cache"] = None        # ((hand, street), EquitySamples)
    memory["hand_samples"] = {}      # street -> EquitySamples for the current hand (showdown calibration)
    memory["last_hist"] = ()         # the latest action history seen in the current hand
    memory["bp_hits"] = 0
    memory["eq_calls"] = 0           # number of Monte Carlo batches this half
    memory["eq_time"] = 0.0          # seconds spent inside them
    memory["eq_samples"] = 0         # samples drawn in them
    memory["refines"] = 0            # extra batches drawn for close decisions
    memory["cpu0"] = time.process_time()
    memory["sel_wH"] = None           # probability of playing arm H next hand (None = from prior)
    memory["sel_w_hand"] = 0.0        # the probability used for the current hand
    memory["sel_arm"] = "M"
    memory["sel_P"] = {"H": 1.0, "M": 1.0}   # prob. of our actions this hand under each arm
    memory["sel_stats"] = [0.0, 0.0, 0.0]    # discounted n, sum D, sum D^2  (D = G_H - G_M, bb)
    memory["bb"] = 2


def _P(memory) -> dict:
    return memory.get("params") or PARAMS


def _freq(pair, prior_rate: float, prior_n: float) -> float:
    return (pair[0] + prior_rate * prior_n) / (pair[1] + prior_n)


def _ingest_history(obs, memory: dict) -> None:
    """Count opponent actions we have not processed yet in this hand."""
    if memory["cur_hand"] != obs.hand_index:
        memory["cur_hand"] = obs.hand_index
        memory["seen_hist"] = 0
        memory["we_bet_last"] = False
        memory["hand_samples"] = {}
        memory["bb"] = obs.big_blind
        if _P(memory).get("selector", 0.0) >= 0.5:
            if memory["sel_wH"] is None:
                memory["sel_wH"] = _selector_weight(memory, _P(memory))
            memory["sel_w_hand"] = memory["sel_wH"]
            memory["sel_arm"] = "H" if memory["rng"].random() < memory["sel_wH"] else "M"
            memory["sel_P"] = {"H": 1.0, "M": 1.0}
    opp = memory["opp"]
    hist = obs.action_history
    memory["last_hist"] = hist
    for i in range(memory["seen_hist"], len(hist)):
        a = hist[i]
        if a.is_me:
            continue
        faced_bet = i > 0 and hist[i - 1].is_me and hist[i - 1].action in (Action.RAISE_HALF, Action.RAISE_POT)
        key = "pre_aggr" if a.street == Street.PREFLOP else "post_aggr"
        opp[key][1] += 1
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            opp[key][0] += 1
        if a.street != Street.PREFLOP:
            sa = opp["st_aggr"][int(a.street)]
            sa[1] += 1
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
                sa[0] += 1
                if a.action == Action.RAISE_POT:
                    opp["st_pot"][int(a.street)] += 1
        if faced_bet:
            opp["fold_to_bet"][1] += 1  # (a fold ends the hand; counted in record_hand_end)
    memory["seen_hist"] = len(hist)


_DEBUG = bool(os.environ.get("BOT_DEBUG"))


def record_hand_end(result, memory: dict) -> None:
    if _DEBUG and result.hand_index == 999:
        import json as _j
        import sys
        o = memory["opp"]
        rate = lambda p: round(p[0] / p[1], 4) if p[1] else None
        print("[oppstats] " + _j.dumps({
            "pre_aggr": rate(o["pre_aggr"]), "post_aggr": rate(o["post_aggr"]), "fold": rate(o["fold_to_bet"]),
            "sd_pre": rate(o["sd_pre"]), "n_sd_pre": o["sd_pre"][1], "sd_post": rate(o["sd_post"]),
            "n_sd_post": o["sd_post"][1], "showdown_rate": round(o["showdowns"] / max(1, o["hands"]), 4),
            "dev": round(_deviation(o), 3), "w": round(_exploit_weight(memory, _P(memory)), 3),
            "rnr": memory.get("rnr_last"),
            "sel_wH": None if memory["sel_wH"] is None else round(memory["sel_wH"], 3),
            "sel_D": round(memory["sel_stats"][1] / max(1e-9, memory["sel_stats"][0]), 3)}),
            file=sys.stderr, flush=True)
    if _DEBUG and result.hand_index % 250 == 249:
        import sys
        o = memory["opp"]
        print(f"[bot] hand {result.hand_index}: think {memory['think']:.2f}s "
              f"cpu {time.process_time() - memory['cpu0']:.2f}s eq_calls {memory['eq_calls']} "
              f"avg_n {memory['eq_samples'] / max(1, memory['eq_calls']):.0f} decisions {memory['n_decisions']} "
              f"refines {memory['refines']} sd_pre {o['sd_pre']} sd_post {o['sd_post']}",
              file=sys.stderr, flush=True)
    opp = memory["opp"]
    opp["hands"] += 1
    same_hand = memory.get("cur_hand") == result.hand_index
    if result.went_to_showdown:
        opp["showdowns"] += 1
        if same_hand and result.opp_hole is not None:
            _calibrate_from_showdown(result, memory)
    elif result.my_net > 0 and memory.get("we_bet_last") and same_hand:
        opp["fold_to_bet"][0] += 1      # we bet, they folded
        opp["fold_to_bet"][1] += 1
    memory["we_bet_last"] = False
    P = _P(memory)
    if same_hand and P.get("selector", 0.0) >= 0.5 and memory.get("mode", MODE) == "mix":
        _selector_update(result, memory, P)


def _selector_weight(memory, P) -> float:
    """Probability matching: P(arm H is better) from the posterior on D."""
    import math
    st = memory["sel_stats"]
    n, n0, m0 = st[0], P["sel_prior_n"], P["sel_prior_m"]
    mean = (st[1] + m0 * n0) / (n + n0)
    var = (st[2] / n - (st[1] / n) ** 2) if n > 20 else 25.0
    se = math.sqrt(max(var, 1.0) / (n + n0))
    p = 0.5 * (1.0 + math.erf(mean / se / math.sqrt(2.0)))       # P(arm H is better)
    # Conservative ramp (not plain probability matching): H can be evaluated
    # off-policy from the mix's own play, so explore only at the floor until
    # the evidence favours H. Any rule fixed BEFORE the hand keeps IS exact.
    ramp = min(1.0, max(0.0, (p - P["sel_lo"]) / max(1e-6, P["sel_hi"] - P["sel_lo"])))
    wmin = P["sel_wmin"]
    return wmin + (1.0 - 2.0 * wmin) * ramp


def _selector_update(result, memory, P) -> None:
    """Importance-sampling evaluation of BOTH arms on this hand. The arm was
    drawn at hand start with probability w (H) / 1-w (M), so the behaviour
    probability of our action sequence is w*P_H + (1-w)*P_M; opponent and
    chance terms cancel, so no opponent cards are needed."""
    pH, pM = memory["sel_P"]["H"], memory["sel_P"]["M"]
    w = memory["sel_w_hand"]
    denom = w * pH + (1.0 - w) * pM
    if denom <= 0.0:
        return
    u = result.my_net / float(memory.get("bb", 2) or 2)
    d = (pH - pM) / denom * u
    st, g = memory["sel_stats"], P["sel_gamma"]
    st[0] = g * st[0] + 1.0
    st[1] = g * st[1] + d
    st[2] = g * st[2] + d * d
    memory["sel_wH"] = _selector_weight(memory, P)


def _calibrate_from_showdown(result, memory) -> None:
    """For every street on which the opponent raised, record how strong their
    shown hand really was at that moment (percentile vs random hands)."""
    opp = memory["opp"]
    hole = list(result.opp_hole)
    raised = {int(a.street) for a in memory["last_hist"]
              if not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT)}
    if not raised:
        return
    if 0 in raised and E.PRE_SORTED is not None:
        v = E.PRE_EQ[E.colex5(np.array(sorted(hole), dtype=np.int64))]
        pct = float(np.searchsorted(E.PRE_SORTED, v)) / E.PRE_SORTED.size
        opp["sd_pre"][0] += pct
        opp["sd_pre"][1] += 1
    board = list(result.board)
    for st in (1, 2, 3):
        if st not in raised:
            continue
        smp = memory["hand_samples"].get(st)
        if smp is None or smp.cur is None or len(board) < 2 + st:
            continue
        their = E.current_score(hole, board[:2 + st])
        cur = smp.cur
        pct = float((cur < their).mean() + 0.5 * (cur == their).mean())
        opp["sd_post"][0] += pct
        opp["sd_post"][1] += 1
        opp["sd_st"][st][0] += pct
        opp["sd_st"][st][1] += 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _opp_raises(obs) -> tuple[int, int]:
    """(opponent raises preflop, opponent raises postflop) in this hand."""
    pre = post = 0
    for a in obs.action_history:
        if not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            if a.street == Street.PREFLOP:
                pre += 1
            else:
                post += 1
    return pre, post


def _keep1(rate: float, sd, P) -> float:
    """Fraction of hands an opponent raises with (one raise). Frequency model
    (raise rate x generosity), pulled toward the showdown evidence: if raising
    hands have mean strength percentile m, a 'top-k' range has m = 1 - k/2."""
    keep_freq = min(1.0, rate * P["gen"])
    if sd[1] <= 0:
        return keep_freq
    m = sd[0] / sd[1]
    keep_sd = min(1.0, max(0.05, 2.0 * (1.0 - m)))
    n0 = P["sd_prior"]
    return (sd[1] * keep_sd + n0 * keep_freq) / (sd[1] + n0)


def _post_cut_v2(obs, opp, P) -> float:
    """Range model v2: the range after the opponent's most recent postflop
    raise comes from THAT street's raise rate (size-adjusted) and THAT street's
    showdown evidence; each earlier raise in the hand narrows it by the square
    root of its own keep fraction."""
    raises = [a for a in obs.action_history
              if not a.is_me and a.street != Street.PREFLOP and a.action in (Action.RAISE_HALF, Action.RAISE_POT)]
    if not raises:
        return 0.0
    keep = 1.0
    for i, a in enumerate(reversed(raises)):
        st = int(a.street)
        sa = opp["st_aggr"][st]
        rate = _freq(sa, 0.30, 12)
        pot_share = (opp["st_pot"][st] + 3.0) / (sa[0] + 6.0)          # prior 0.5
        share = pot_share if a.action == Action.RAISE_POT else 1.0 - pot_share
        k1 = _keep1(rate * share ** 0.5, opp["sd_st"][st], P)
        keep *= k1 if i == 0 else k1 ** 0.5
    return 1.0 - max(1.0 - P["cap_post"], min(1.0, keep))


def _cut(keep1: float, n_raises: int, cap: float) -> float:
    """Range cut after n raises: each extra raise narrows the range."""
    if n_raises <= 0:
        return 0.0
    keep = keep1 ** (min(n_raises, 3) ** 0.5)
    return 1.0 - max(1.0 - cap, min(1.0, keep))


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
    return float(E.PRE_EQ[E.colex5(np.array(sorted(hole), dtype=np.int64))])


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
    if memory.get("mode", MODE) == "nn":
        a = _nn_decide(obs, memory)
        if a is not None:
            memory["we_bet_last"] = a in (Action.RAISE_HALF, Action.RAISE_POT)
            return a
    P = _P(memory)
    opp = memory["opp"]
    rng = memory["rng"]

    pre_rate = _freq(opp["pre_aggr"], 0.35, 12)
    post_rate = _freq(opp["post_aggr"], 0.30, 12)
    fold_rate = _freq(opp["fold_to_bet"], 0.35, 10)

    pre_raises, post_raises = _opp_raises(obs)
    pre_cut = _cut(_keep1(pre_rate, opp["sd_pre"], P), pre_raises, P["cap_pre"])
    if P.get("range_v2", 0.0) >= 0.5:
        post_cut = _post_cut_v2(obs, opp, P)
    else:
        post_cut = _cut(_keep1(post_rate, opp["sd_post"], P), post_raises, P["cap_post"])

    pot, to_call = obs.pot, obs.to_call
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
    raw, eq = _equities(obs, memory, pre_cut, post_cut)
    # Close call on a big pot: buy more precision if we are ahead of time pace.
    if (obs.street != Street.PREFLOP and to_call > 0 and pot >= 12 * obs.big_blind
            and memory["think"] < TIME_BUDGET * (obs.hand_index + 1) / 1000.0 * 0.85):
        c = memory["eq_cache"]
        se = 0.5 / max(1, c[1].n) ** 0.5
        if abs(eq - pot_odds) < 1.5 * se and c[1].n < 900:
            more = _timed_sample(memory, obs.hole_cards, obs.board, 300)
            merged = c[1].merged(more)
            memory["eq_cache"] = (c[0], merged)
            memory["hand_samples"][int(obs.street)] = merged
            memory["refines"] += 1
            raw, eq = _equities(obs, memory, pre_cut, post_cut)
    # ---------------- action ----------------
    mode = memory.get("mode", MODE)
    if mode == "nn":
        mode = "bp"                  # the deep-RL policy's fallback is the pure blueprint
    heur_d = _choose_dist(obs, eq, pot_odds, fold_rate, P)
    mix_d = heur_d
    if mode != "heur":
        bp_d = None
        if mode == "nnmix":          # the deep-RL policy replaces the blueprint as the anchor
            p = _nn_probs(obs, memory)
            if p is not None:
                bp_d = {a: float(p[int(a)]) for a in obs.legal_actions if p[int(a)] > 0.0}
        if not bp_d:
            bp_d = _bp_dist(obs, raw, memory)
        if bp_d is not None:
            memory["bp_hits"] += 1
            if mode == "rnr" or (mode == "mix" and P.get("rnr_on_mix", 0.0) >= 0.5):
                bp_d = _rnr_blend(obs, bp_d, memory, P)
            w = 0.0 if mode in ("bp", "rnr") else _exploit_weight(memory, P)
            mix_d = {a: w * heur_d.get(a, 0.0) + (1.0 - w) * bp_d.get(a, 0.0)
                     for a in set(heur_d) | set(bp_d)}
    sel = mode == "mix" and P.get("selector", 0.0) >= 0.5
    action = _sample(heur_d if (sel and memory["sel_arm"] == "H") else mix_d, rng)
    if sel:
        memory["sel_P"]["H"] *= heur_d.get(action, 0.0)
        memory["sel_P"]["M"] *= mix_d.get(action, 0.0)
    memory["we_bet_last"] = action in (Action.RAISE_HALF, Action.RAISE_POT)
    return action


def _equities(obs, memory, pre_cut, post_cut):
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
            memory["hand_samples"][int(obs.street)] = samples
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
        if _P(memory).get("fast_eq", 1.0) >= 0.5:
            return E.sample_equity_fast(hole, board, n, memory["rng"])
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


def _bp_probs(obs, raw_eq, memory):
    """Blueprint action probabilities [F, C, H, P] for this decision, or None."""
    memory["bp_idx"] = None
    bp = BLUEPRINTS.get(memory.get("bpv", DEFAULT_BPV))
    if bp is None:
        return None
    st = int(obs.street)
    pot_start = obs.pot - obs.my_bet - obs.opp_bet
    seq = "".join("FCHP"[int(a.action)] for a in obs.action_history if a.street == obs.street)
    if bp["version"] == 1:
        return bp["policy"].get(infoset_key(st, _bucket_v1(st, raw_eq, bp["thresholds"]), obs.is_button,
                                            pot_start, seq))
    kid = bp["keys"].get(f"{st}{'b' if obs.is_button else 'n'}.{pot_start}.{seq}")
    if kid is None:
        return None
    if st == 0:
        b = bisect.bisect_right(bp["pre"], raw_eq)
    elif st == 3:
        b = bisect.bisect_right(bp["riv"], raw_eq)
    else:
        smp = memory["eq_cache"][1]
        e = bisect.bisect_right(bp["e"][st], raw_eq)
        pot_feature = raw_eq - smp.hs
        b = e * bp["np"] + sum(1 for t in bp["p"][st][e] if pot_feature > t)
    idx = bp["key_offset"][kid] + b
    if bp is BLUEPRINTS.get(2):
        memory["bp_idx"] = idx               # same row in every RNR table
    return bp["policy"][idx]


def _rnr_choice(memory, P):
    """(archetype, weight): the archetype the opponent most resembles, weighted by how much
    closer it is to it than to equilibrium (3 frequency statistics, equilibrium priors)."""
    opp = memory["opp"]
    names = [n for n in RNR if n != "maniac" or P.get("rnr_maniac", 0.0) >= 0.5]
    if not names:
        return None, 0.0
    x = {"pre": _freq(opp["pre_aggr"], EQ_REF["pre"], 12),
         "post": _freq(opp["post_aggr"], EQ_REF["post"], 12),
         "fold": _freq(opp["fold_to_bet"], EQ_REF["fold"], 10)}
    dist = lambda prof: math.sqrt(sum(((x[k] - prof[k]) / EQ_SCALE[k]) ** 2 for k in x))
    best = min(names, key=lambda n: dist(ARCH_PROFILE[n]))
    gain = dist(EQ_REF) - dist(ARCH_PROFILE[best])
    conf = min(1.0, opp["hands"] / P["ramp"])
    span = max(1e-6, P["rnr_m1"] - P["rnr_m0"])
    return best, conf * P["rnr_wmax"] * min(1.0, max(0.0, (gain - P["rnr_m0"]) / span))


def _rnr_blend(obs, bp_d, memory, P):
    """Blueprint distribution blended with the matching archetype's counter-strategy."""
    idx = memory.get("bp_idx")
    if idx is None or not RNR:
        return bp_d
    name, w = _rnr_choice(memory, P)
    memory["rnr_last"] = (name, round(w, 3))
    if w <= 0.0:
        return bp_d
    row = RNR[name][idx]
    ws = {a: float(row[int(a)]) for a in obs.legal_actions}
    tot = sum(ws.values())
    if tot <= 0.0:
        return bp_d
    return {a: w * ws[a] / tot + (1.0 - w) * bp_d.get(a, 0.0) for a in obs.legal_actions}


def _bp_dist(obs, raw_eq, memory):
    """Blueprint policy over the legal actions as {Action: probability}, or None."""
    probs = _bp_probs(obs, raw_eq, memory)
    if probs is None:
        return None
    ws = {a: float(probs[int(a)]) for a in obs.legal_actions}
    tot = sum(ws.values())
    if tot <= 0.0:
        return None
    return {a: v / tot for a, v in ws.items() if v > 0.0}


def _blueprint_action(obs, raw_eq, memory, rng):
    probs = _bp_probs(obs, raw_eq, memory)
    if probs is None:
        return None
    legal = [int(a) for a in obs.legal_actions]
    ws = [float(probs[a]) for a in legal]
    tot = sum(ws)
    if tot <= 0:
        return None
    x = rng.random() * tot
    for a, wv in zip(legal, ws):
        x -= wv
        if x <= 0:
            return Action(a)
    return Action(legal[-1])


def _deviation(opp) -> float:
    """How far the opponent is from equilibrium play, in units of typical
    spread, summed over 5 statistics. Frequencies use equilibrium priors and
    showdown means are shrunk toward the reference, so an unknown opponent
    starts at ~0 (= play the blueprint)."""
    pre = _freq(opp["pre_aggr"], EQ_REF["pre"], 12)
    post = _freq(opp["post_aggr"], EQ_REF["post"], 12)
    fold = _freq(opp["fold_to_bet"], EQ_REF["fold"], 10)
    sdp = (opp["sd_pre"][0] + 10 * EQ_REF["sd_pre"]) / (opp["sd_pre"][1] + 10)
    sdq = (opp["sd_post"][0] + 10 * EQ_REF["sd_post"]) / (opp["sd_post"][1] + 10)
    return (abs(pre - EQ_REF["pre"]) / EQ_SCALE["pre"] + abs(post - EQ_REF["post"]) / EQ_SCALE["post"]
            + abs(fold - EQ_REF["fold"]) / EQ_SCALE["fold"] + abs(sdp - EQ_REF["sd_pre"]) / EQ_SCALE["sd_pre"]
            + abs(sdq - EQ_REF["sd_post"]) / EQ_SCALE["sd_post"])


def _exploit_weight(memory, P) -> float:
    """Safe exploitation (Restricted-Nash-Response spirit): play the CFR
    blueprint against balanced opponents, and shift to the exploitative policy
    only as far as the opponent is (a) observed long enough and (b) measurably
    far from equilibrium behaviour."""
    opp = memory["opp"]
    conf = min(1.0, opp["hands"] / P["ramp"])
    if P.get("wfix", -1.0) >= 0.0:
        return conf * min(1.0, P["wfix"])
    dev = _deviation(opp)
    span = max(1e-6, P["dev1"] - P["dev0"])
    w = conf * min(1.0, max(0.0, (dev - P["dev0"]) / span))
    if P.get("agg_gate", 0.0) >= 0.5 and w > 0.0:
        w *= _passivity(opp, P)
    return w


def _passivity(opp, P) -> float:
    """1 for opponents at or below equilibrium aggression, fading to 0 for aggressive ones."""
    pre = _freq(opp["pre_aggr"], EQ_REF["pre"], 12)
    post = _freq(opp["post_aggr"], EQ_REF["post"], 12)
    aggr = max((pre - EQ_REF["pre"]) / EQ_SCALE["pre"], (post - EQ_REF["post"]) / EQ_SCALE["post"])
    span = max(1e-6, P["agg_a1"] - P["agg_a0"])
    return min(1.0, max(0.0, (P["agg_a1"] - aggr) / span))


def _call_or_fold(obs, eq, pot_odds, P) -> Action:
    """Facing a bet, after the raise options were declined: call or fold."""
    street, to_call = obs.street, obs.to_call
    bonus = P["call_bonus"] if street != Street.RIVER else 0.0
    lam = P.get("risk_lambda", 0.0)
    if lam > 0.0:
        # Maximise EV - lambda*Var (in bb): P(win) of a 1000-deal match depends on mean/std,
        # so thin calls into big pots cost more in variance than they return in EV.
        bb = float(obs.big_blind)
        final = (obs.pot + to_call) / bb
        e = min(1.0, eq + bonus)
        ok = e * final - to_call / bb - lam * e * (1.0 - e) * final * final >= 0.0
    else:
        ok = eq + bonus >= pot_odds
    if ok:
        return Action.CHECK_CALL
    return Action.FOLD if _legal(obs, Action.FOLD) else Action.CHECK_CALL


def _choose_dist(obs, eq: float, pot_odds: float, fold_rate: float, P: dict) -> dict:
    """The exploitative heuristic as an explicit action distribution
    {Action: probability} (needed to evaluate it by importance sampling)."""
    street = obs.street
    to_call = obs.to_call
    can_raise = _legal(obs, Action.RAISE_POT) or _legal(obs, Action.RAISE_HALF)

    if street == Street.PREFLOP:
        # Preflop equities in PLO5 run close (roughly 0.30-0.70 vs random).
        if to_call > 0:
            if eq >= P["pf_raise_big"] and can_raise and obs.raises_this_street < 3:
                return {_raise_or(obs, Action.RAISE_POT): 1.0}
            if eq >= P["pf_raise"] and can_raise and obs.raises_this_street < 2:
                return {_raise_or(obs, Action.RAISE_POT if obs.is_button else Action.RAISE_HALF): 1.0}
            margin = P["pf_pos_bonus"] if obs.is_button or obs.raises_this_street == 0 else 0.0
            if obs.to_call <= obs.big_blind and eq >= P["pf_limp"]:
                return {Action.CHECK_CALL: 1.0}
            if eq + margin >= pot_odds + P["pf_call_margin"]:
                return {Action.CHECK_CALL: 1.0}
            return {Action.FOLD if _legal(obs, Action.FOLD) else Action.CHECK_CALL: 1.0}
        if eq >= P["bb_raise"] and can_raise:      # big blind option after a limp
            return {_raise_or(obs, Action.RAISE_POT): 1.0}
        return {Action.CHECK_CALL: 1.0}

    # ---------------- postflop ----------------
    if street == Street.FLOP:
        value_pot, value_half = P["vp_f"], P["vh_f"]
    elif street == Street.TURN:
        value_pot, value_half = P["vp_t"], P["vh_t"]
    else:
        value_pot, value_half = P["vp_r"], P["vh_r"]

    if to_call == 0:
        if eq >= value_pot and can_raise:
            return {_raise_or(obs, Action.RAISE_POT): 1.0}
        if eq >= value_half and can_raise:
            return {_raise_or(obs, Action.RAISE_HALF): 1.0}
        if can_raise:   # bluff / semi-bluff vs opponents that fold a lot
            bluff_p = max(0.0, (fold_rate - P["bl_base"])) * P["bl_slope"]
            if street != Street.RIVER and eq >= P["semi_eq"]:
                bluff_p += P["semi_p"]
            bluff_p = min(1.0, bluff_p)
            if bluff_p > 0.0:
                d = {_raise_or(obs, Action.RAISE_HALF): bluff_p}
                d[Action.CHECK_CALL] = d.get(Action.CHECK_CALL, 0.0) + 1.0 - bluff_p
                return d
        return {Action.CHECK_CALL: 1.0}

    # facing a bet
    if eq >= value_pot + P["rr_margin"] and can_raise:
        return {_raise_or(obs, Action.RAISE_POT): 1.0}
    cf = _call_or_fold(obs, eq, pot_odds, P)
    if eq >= value_pot - 0.02 and can_raise and obs.raises_this_street < 2:
        d = {_raise_or(obs, Action.RAISE_HALF): 0.6}
        d[cf] = d.get(cf, 0.0) + 0.4
        return d
    return {cf: 1.0}


def _sample(dist: dict, rng) -> Action:
    x = rng.random() * sum(dist.values())
    last = None
    for a, p in dist.items():
        last = a
        x -= p
        if x <= 0.0:
            return a
    return last


def _choose(obs, eq: float, pot_odds: float, fold_rate: float, P: dict, rng) -> Action:
    return _sample(_choose_dist(obs, eq, pot_odds, fold_rate, P), rng)


# ---------------------------------------------------------------------------
# deep-RL policy (mode "nn"): an MLP trained on the GPU (offline/drl/) over the
# exact betting state, card features and the SAME Monte Carlo equity features the
# blueprint uses (computed with the same time-managed estimator).
# ---------------------------------------------------------------------------

_NN_MODELS = {}


def _nn_model(path=None):
    import nn_policy
    p = path or nn_policy.MODEL_PATH
    m = _NN_MODELS.get(p)
    if m is None:
        m = nn_policy.NNPolicy(p)
        _NN_MODELS[p] = m
    return m


def _nn_probs(obs, memory):
    """Deep-RL policy probabilities over [F, C, H, P], or None on any problem.
    Reuses (or fills) the per-street Monte Carlo cache shared with the other modes."""
    try:
        model = _nn_model(memory.get("nn_path"))
        st = int(obs.street)
        pre = _preflop_eq(obs.hole_cards)
        if st == 0:
            raw = hs = pre
            cur = 0
        else:
            key = (obs.hand_index, st)
            c = memory.get("eq_cache")
            if c is None or c[0] != key or c[1] is None:
                c = (key, _timed_sample(memory, obs.hole_cards, obs.board, _n_samples(obs, memory)))
                memory["eq_cache"] = c
                memory["hand_samples"][st] = c[1]
            raw, hs = c[1].raw, c[1].hs
            cur = E.current_score(obs.hole_cards, obs.board)
        member = 0
        if model.n_members > 1 or model.bp_weight > 0.0:   # population: one member per hand (mixed strategy)
            if memory.get("nn_member_hand") != obs.hand_index:
                memory["nn_member_hand"] = obs.hand_index
                rng = memory["rng"]
                memory["nn_member"] = (-1 if rng.random() < model.bp_weight
                                       else int(rng.integers(model.n_members)))
            member = memory["nn_member"]
            if member < 0:               # this hand is played by the blueprint member
                memory["nn_bp_hands"] = memory.get("nn_bp_hands", 0) + 1
                return None
        p = model.probs(obs, pre, raw, hs, cur, member)
        if p is not None and np.isfinite(p).all() and sum(p[int(a)] for a in obs.legal_actions) > 0.0:
            return p
    except Exception:
        memory["nn_errors"] = memory.get("nn_errors", 0) + 1
    memory["nn_fallbacks"] = memory.get("nn_fallbacks", 0) + 1
    return None


def _nn_decide(obs, memory):
    """Action sampled from the deep-RL policy, or None (the caller then plays the blueprint)."""
    p = _nn_probs(obs, memory)
    if p is None:
        return None
    legal = [int(a) for a in obs.legal_actions]
    x = memory["rng"].random() * float(sum(p[a] for a in legal))
    for a in legal:
        x -= p[a]
        if x <= 0.0:
            return Action(a)
    return Action(max(legal, key=lambda a: p[a]))


if MODE in ("nn", "nnmix"):          # load the model at startup, not in the first decision
    try:
        _nn_model()
    except Exception:
        pass


def timed_decide(obs, memory: dict) -> Action:
    t0 = time.perf_counter()
    try:
        return decide(obs, memory)
    finally:
        memory["think"] += time.perf_counter() - t0
        memory["n_decisions"] += 1
