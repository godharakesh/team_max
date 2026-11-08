"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

from game import Action, AgentConfig, Game, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location("agent_under_test", os.path.join(BUNDLE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    module.TOKEN = TOKEN
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    _, client = bot
    start_half(client)
    play_hands(client, 200)


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------


from evaluator import cards_from_str  # noqa: E402
from game import ObservedAction, Observation, Street  # noqa: E402


def _obs(hole, board, street, to_call=0, pot=8, legal=None, hand_index=5, is_button=False):
    legal = legal or ((Action.FOLD,) if to_call else ()) + (Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)
    return Observation(hand_index=hand_index, street=street, hole_cards=tuple(sorted(cards_from_str(hole))),
                       board=tuple(cards_from_str(board)), pot=pot, to_call=to_call, my_bet=0, opp_bet=to_call,
                       my_stack=70, opp_stack=70 - to_call, is_button=is_button, legal_actions=legal,
                       raise_to={a: pot for a in legal if a in (Action.RAISE_HALF, Action.RAISE_POT)},
                       min_raise_to=2, raises_this_street=0, action_history=(), big_blind=2)


def test_never_folds_when_check_is_free():
    m = load_agent()
    m.new_half(AgentConfig(), m.MEMORY)
    for i in range(200):
        deal = make_deal("free", i)
        obs = _obs("", "", Street.FLOP)
        obs = Observation(**{**obs.__dict__, "hole_cards": deal.hole_cards[0], "board": deal.board[:3]})
        assert m.choose_action(obs, m.MEMORY) != Action.FOLD


def test_bets_the_nuts():
    # The blueprint is a mixed (equilibrium) strategy: out of position it may
    # check the nuts to trap, so test frequencies, not single decisions.
    m = load_agent()
    m.new_half(AgentConfig(), m.MEMORY)
    royal = ("As Ks 6c 7d 8h", "Qs Js Ts 6d 7c")   # royal flush in spades on the river
    bets = 0
    for i in range(100):
        obs = _obs(*royal, Street.RIVER, hand_index=1000 + i)
        if m.choose_action(obs, m.MEMORY) in (Action.RAISE_POT, Action.RAISE_HALF):
            bets += 1
    assert bets >= 40, f"bet the nuts only {bets}/100 times when first to act"
    for i in range(50):    # facing a bet: never fold the nuts
        obs = _obs(*royal, Street.RIVER, to_call=8, pot=16, hand_index=2000 + i)
        assert m.choose_action(obs, m.MEMORY) != Action.FOLD
    # in position after a check: always bet
    hist = (ObservedAction(Street.RIVER, False, Action.CHECK_CALL, 0, 0),)
    for i in range(50):
        obs = _obs(*royal, Street.RIVER, hand_index=3000 + i, is_button=True)
        obs = Observation(**{**obs.__dict__, "action_history": hist})
        assert m.choose_action(obs, m.MEMORY) in (Action.RAISE_POT, Action.RAISE_HALF)


def test_out_of_time_still_legal():
    m = load_agent()
    m.new_half(AgentConfig(), m.MEMORY)
    m.MEMORY["think"] = 99.0
    obs = _obs("As Ks 6c 7d 8h", "Qs Js Ts", Street.FLOP, to_call=6, pot=12, legal=(Action.FOLD, Action.CHECK_CALL))
    assert m.choose_action(obs, m.MEMORY) in obs.legal_actions


def test_engine_matches_referee_scores():
    import numpy as np
    import plo_engine as E
    from evaluator import omaha_score
    E.build_tables()
    rng = random.Random(9)
    for _ in range(500):
        c = rng.sample(range(36), 10)
        tri = E.triple_indices(np.array([c[5:]]))
        assert E.best_scores(tri, E.pair_indices(np.array(c[:5])))[0] == omaha_score(c[:5], c[5:])


def test_range_cut_lowers_equity_of_weak_hand_and_cache_reuse():
    import numpy as np
    import plo_engine as E
    E.build_tables()
    g = np.random.default_rng(0)
    s = E.sample_equity(cards_from_str("6c 7d 9h Js Kc"), cards_from_str("Ad Ah 8s"), 400, g)
    assert 0.0 <= s.raw <= 1.0
    assert s.cut(opp_cut=0.6) <= s.raw + 0.02          # vs a stronger range we do worse
    assert s.cut() == s.raw


def test_bot_respects_time_budget_in_process():
    """A full 1000-hand half vs an always-raise opponent must stay inside our own budget."""
    m = load_agent()
    m.new_half(AgentConfig(), m.MEMORY)
    rng = random.Random(3)
    for hand in range(1000):
        game = Game(make_deal("budget", hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                a = m.choose_action(obs, m.MEMORY)
                assert a in obs.legal_actions
            else:
                L = obs.legal_actions
                a = Action.RAISE_POT if Action.RAISE_POT in L else Action.CHECK_CALL
            game.apply(a)
        m.on_hand_end(game.hand_result(0), m.MEMORY)
    assert m.MEMORY["errors"] == 0
    assert m.MEMORY["think"] < 4.0, m.MEMORY["think"]


def test_blueprint_covers_live_decisions():
    """Nearly every live decision must find its infoset in the shipped blueprint."""
    import strategy as S
    if not S.BLUEPRINTS:
        pytest.skip("no blueprint file")
    if S.MODE in ("nn", "nnmix"):
        pytest.skip("the neural policy answers decisions itself, so blueprint coverage does not apply")
    m = load_agent()
    m.new_half(AgentConfig(), m.MEMORY)
    rng = random.Random(5)
    for hand in range(150):
        game = Game(make_deal("cover", hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            a = m.choose_action(obs, m.MEMORY) if seat == 0 else rng.choice(obs.legal_actions)
            game.apply(a)
        m.on_hand_end(game.hand_result(0), m.MEMORY)
    assert m.MEMORY["bp_hits"] >= 0.95 * m.MEMORY["n_decisions"], (m.MEMORY["bp_hits"], m.MEMORY["n_decisions"])


def test_tuned_params_file_is_valid():
    import json
    import strategy as S
    path = os.path.join(BUNDLE, "params_tuned.json")
    if not os.path.isfile(path):
        pytest.skip("no tuned params")
    data = json.load(open(path))
    assert set(data) <= set(S.PARAMS), set(data) - set(S.PARAMS)
    assert all(isinstance(v, (int, float)) and -1.0 <= v <= 200.0 for v in data.values())


def test_charge_guard_falls_back_cheaply():
    """If the estimated judge charge is near the bank, the bot must stop computing."""
    m = load_agent()
    m.new_half(AgentConfig(), m.MEMORY)
    m.MEMORY["cpu0"] = -100.0           # pretend 100 s of process CPU were used this half
    obs = _obs("As Ks 6c 7d 8h", "Qs Js Ts", Street.FLOP, to_call=6, pot=12)
    before = m.MEMORY["n_decisions"]
    assert m.choose_action(obs, m.MEMORY) in obs.legal_actions
    assert m.MEMORY["n_decisions"] == before, "strategy should not run when the charge guard trips"


def test_range_v2_reads_value_bettors_tighter_than_maniacs():
    import strategy as S
    P = {**S.PARAMS, "range_v2": 1.0}
    hist = (ObservedAction(Street.RIVER, False, Action.RAISE_POT, 8, 8),)
    obs = _obs("As Ks 6c 7d 8h", "Qs Js Ts 6d 7c", Street.RIVER, to_call=8, pot=16)
    obs = Observation(**{**obs.__dict__, "action_history": hist})

    def cut_for(raise_rate, sd_mean):
        m = {}
        S.init_memory(m)
        o = m["opp"]
        o["st_aggr"][3] = [int(200 * raise_rate), 200]
        o["st_pot"][3] = int(200 * raise_rate)
        o["sd_st"][3] = [sd_mean * 60, 60]
        return S._post_cut_v2(obs, o, P)

    nit = cut_for(0.10, 0.90)      # rarely bets the river, always strong when it does
    maniac = cut_for(0.90, 0.50)   # bets everything, random strength
    assert nit > maniac + 0.3, (nit, maniac)
    assert 0.0 <= maniac <= 0.2


def test_fast_estimators_are_unbiased():
    """River pair-decomposition and turn stratification must match brute force."""
    import numpy as np
    import plo_engine as E
    from evaluator import omaha_score
    E.build_tables()
    g = np.random.default_rng(3)
    rng = random.Random(3)
    hole = cards_from_str("6c 8d 9h Th Kc")
    for board in (cards_from_str("Ts Jd 7c 6h Ah"), cards_from_str("Ts Jd 7c 6h")):
        unknown = [c for c in range(36) if c not in hole + board]
        w, n = 0.0, 8000
        for _ in range(n):
            x = rng.sample(unknown, 5 + (5 - len(board)))
            full = board + x[5:]
            a, b = omaha_score(hole, full), omaha_score(x[:5], full)
            w += (a > b) + 0.5 * (a == b)
        fast = E.sample_equity_fast(hole, board, 6000, g).raw
        assert abs(fast - w / n) < 0.025, (len(board), fast, w / n)


def test_nn_mode_plays_legal_actions_without_fallback():
    """BOT_MODE=nn: the deep-RL policy answers every decision itself (no blueprint
    fallback), always legally, in both seats."""
    import nn_policy
    if not os.path.isfile(nn_policy.MODEL_PATH):
        pytest.skip("no nn_policy.npz")
    m = load_agent()
    m.new_half(AgentConfig(), m.MEMORY)
    m.MEMORY["mode"] = "nn"
    rng = random.Random(11)
    for hand in range(200):
        game = Game(make_deal("nn", hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                a = m.choose_action(obs, m.MEMORY)
                assert a in obs.legal_actions
            else:
                a = rng.choice(obs.legal_actions)
            game.apply(a)
        m.on_hand_end(game.hand_result(0), m.MEMORY)
    assert m.MEMORY["errors"] == 0
    assert m.MEMORY.get("nn_fallbacks", 0) == 0 and m.MEMORY.get("nn_errors", 0) == 0
