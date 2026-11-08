"""In-process duplicate-match arena (same cards/seats/scoring as judge_engine,
no HTTP), for fast A/B testing of strategy variants against an opponent pool.

usage: python offline/arena.py <variantA> <oppB>[,<oppC>...] [deals] [seed...]
  variants: mix | bp | heur | nn | nn:<checkpoint.npz>  (our strategy.py in that MODE)
  opponents: random half pot station nit tag maniac  or any variant name
"""
import math
import os
import random
import statistics
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plo_engine as E  # noqa: E402
import strategy as S  # noqa: E402
from game import Action, Game, Street, make_deal  # noqa: E402

R = (Action.RAISE_HALF, Action.RAISE_POT)


class Ours:
    def __init__(self, mode, bpv=None, params=None, nn_path=None):
        self.mode, self.bpv, self.params, self.nn_path = mode, bpv, params, nn_path

    def new_half(self, seed):
        self.m = {}
        S.init_memory(self.m, seed=seed)
        self.m["mode"] = self.mode
        if self.nn_path:
            self.m["nn_path"] = self.nn_path
        if self.bpv is not None:
            self.m["bpv"] = self.bpv
        if self.params is not None:
            self.m["params"] = {**S.PARAMS, **self.params}

    def act(self, obs):
        a = S.timed_decide(obs, self.m)
        return a if a in obs.legal_actions else Action.CHECK_CALL

    def hand_end(self, res):
        S.record_hand_end(res, self.m)

    @property
    def think(self):
        return self.m["think"]


class Simple:
    """Reference opponents."""
    def __init__(self, kind):
        self.kind = kind
        self.think = 0.0

    def new_half(self, seed):
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self.think = 0.0

    def _eq(self, obs):
        if obs.street == Street.PREFLOP:
            return float(E.PRE_EQ[E.colex5(np.array(sorted(obs.hole_cards)))])
        return E.equity(obs.hole_cards, obs.board, 120, self.nrng)

    def hand_end(self, res):
        pass

    def act(self, obs):
        t = time.perf_counter()
        try:
            return self._act(obs)
        finally:
            self.think += time.perf_counter() - t

    def _act(self, obs):
        L = obs.legal_actions
        k = self.kind
        if k == "random":
            return self.rng.choice(L)
        if k == "half":
            return Action.RAISE_HALF if Action.RAISE_HALF in L else Action.CHECK_CALL
        if k == "pot":
            return Action.RAISE_POT if Action.RAISE_POT in L else Action.CHECK_CALL
        if k == "station":
            return Action.CHECK_CALL
        if k == "maniac":
            if any(a in L for a in R) and self.rng.random() < 0.55:
                return self.rng.choice([a for a in R if a in L])
            return Action.CHECK_CALL
        eq = self._eq(obs)
        po = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
        can_r = Action.RAISE_POT in L
        if k == "tag":   # typical "equity vs pot odds" hackathon bot
            if eq > 0.62 and can_r:
                return Action.RAISE_POT
            if eq > 0.55 and can_r and obs.to_call == 0:
                return Action.RAISE_HALF
            if obs.to_call == 0 or eq >= po:
                return Action.CHECK_CALL
            return Action.FOLD
        if k == "barrel":  # exploiter: TAG preflop, then bets relentlessly when checked to
            if obs.to_call == 0 and can_r and obs.street != Street.PREFLOP:
                p = 0.6 if obs.street == Street.FLOP else 0.8
                if self.rng.random() < p or eq > 0.6:
                    return Action.RAISE_POT if obs.street != Street.FLOP else Action.RAISE_HALF
                return Action.CHECK_CALL
            if eq > 0.70 and can_r:
                return Action.RAISE_POT
            if obs.to_call == 0 or eq >= po:
                return Action.CHECK_CALL
            return Action.FOLD
        if k == "nit":
            if eq > 0.70 and can_r:
                return Action.RAISE_POT
            if obs.to_call == 0 or eq >= po + 0.12:
                return Action.CHECK_CALL
            return Action.FOLD
        raise ValueError(k)


class Release(Ours):
    """The shipped release (commit a759c01) frozen in offline/release/, so new
    versions are always measured against what is actually submitted."""
    _mod = None

    def __init__(self):
        super().__init__("mix")
        if Release._mod is None:
            import importlib.util
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release", "strategy_release.py")
            spec = importlib.util.spec_from_file_location("strategy_release", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            Release._mod = mod

    def new_half(self, seed):
        self.m = {}
        Release._mod.init_memory(self.m, seed=seed)

    def act(self, obs):
        a = Release._mod.timed_decide(obs, self.m)
        return a if a in obs.legal_actions else Action.CHECK_CALL

    def hand_end(self, res):
        Release._mod.record_hand_end(res, self.m)


OURS = {"mix": ("mix", None), "bp": ("bp", None), "heur": ("heur", None),
        "mix1": ("mix", 1), "mix2": ("mix", 2), "bp1": ("bp", 1), "bp2": ("bp", 2),
        "mix3": ("mix", 3), "bp3": ("bp", 3), "mix4": ("mix", 4), "bp4": ("bp", 4),
        "rnr": ("rnr", None)}
# candidate variants that need parameter overrides
CANDS = {"gate": ("mix", {"agg_gate": 1.0}),
         "gaternr": ("mix", {"agg_gate": 1.0, "rnr_on_mix": 1.0}),
         "rnrman": ("rnr", {"rnr_maniac": 1.0})}


def make(spec):
    """spec: a name ('mix', 'mix2', 'bp1', 'tag', ...) or a dict
    {"name": ..., "mode": ..., "bpv": ..., "params": {...}} for tuning candidates."""
    if isinstance(spec, dict):
        return Ours(spec.get("mode", "mix"), spec.get("bpv"), spec.get("params"))
    if spec == "rel":
        return Release()
    if spec == "nn" or spec.startswith("nn:"):      # deep-RL policy; "nn:<path to .npz>" picks a checkpoint
        return Ours("nn", None, None, spec[3:] or None)
    if spec == "nnmix" or spec.startswith("nnmix:"):  # mix with the deep-RL policy as the anchor
        return Ours("nnmix", None, None, spec[6:] or None)
    if spec in CANDS:
        mode, params = CANDS[spec]
        return Ours(mode, None, params)
    if spec in OURS:
        mode, bpv = OURS[spec]
        return Ours(mode, bpv)
    return Simple(spec)


def label(spec):
    if isinstance(spec, dict):
        return spec.get("name", "cand")
    for pre in ("nn:", "nnmix:"):
        if spec.startswith(pre):
            return pre + os.path.splitext(os.path.basename(spec[len(pre):]))[0]
    return spec


def play_half(a, b, seed, half, deals):
    """half 1: a in seat 0; half 2: a in seat 1. Returns a's nets per deal."""
    bots = (a, b) if half == 1 else (b, a)
    a.new_half(zlib.crc32(f"{seed}:{half}:a".encode()))   # stable across processes (CRN)
    b.new_half(zlib.crc32(f"{seed}:{half}:b".encode()))
    nets = []
    a_seat = 0 if half == 1 else 1
    for d in range(deals):
        g = Game(make_deal(seed, d), d % 2, hand_index=d)
        while not g.is_terminal:
            s = g.to_act
            g.apply(bots[s].act(g.observation(s)))
        nets.append(g.result().nets[a_seat])
        for s in (0, 1):
            bots[s].hand_end(g.hand_result(s))
    return nets, a.think, b.think


def match(args):
    va, vb, deals, seed = args
    E.build_tables()
    a, b = make(va), make(vb)
    va, vb = label(va), label(vb)
    n1, ta1, tb1 = play_half(a, b, seed, 1, deals)
    n2, ta2, tb2 = play_half(a, b, seed, 2, deals)
    paired = [(x + y) / 2.0 for x, y in zip(n1, n2)]   # chips -> bb (BB = 2)
    mean = sum(paired) / deals
    se = statistics.stdev(paired) / math.sqrt(deals)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    verdict = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "CHOP")
    return (va, vb, seed, mean, lo, hi, verdict, max(ta1, ta2), max(tb1, tb2))


if __name__ == "__main__":
    va = sys.argv[1]
    opps = sys.argv[2].split(",")
    deals = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seeds = sys.argv[4:] or ["101"]
    jobs = [(va, o, deals, s) for o in opps for s in seeds]
    with Pool(min(len(jobs), int(os.environ.get("ARENA_WORKERS", os.cpu_count())))) as p:
        for r in p.imap(match, jobs):
            va_, vb, seed, mean, lo, hi, v, ta, tb = r
            print(f"{va_:>5} vs {vb:<8} seed={seed:<5} {mean:+7.3f} bb/deal  CI[{lo:+.3f},{hi:+.3f}]  {v:<5}"
                  f"  think/half: ours {ta:.2f}s opp {tb:.2f}s", flush=True)
