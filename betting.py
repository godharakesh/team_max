"""
betting.py - shared betting abstraction for the CFR blueprint (offline trainer
and runtime bot use the same infoset keys and card buckets).

Infoset key = street | card bucket | position | pot at street start | this
street's action sequence. The pot at street start fully determines both
stacks (stacks reset to 80 every hand and contributions are equal when a
street begins), so it is a lossless summary of earlier betting except for
what it reveals about the opponent's range (imperfect-recall abstraction, as
used by Libratus/Pluribus-style blueprints).
"""

from __future__ import annotations

import bisect

ACT_CHARS = "FCHP"   # FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT  (== game.Action values)
NB = (16, 12, 12, 12)  # buckets per street (preflop is noise-free, so finer)
STACK, SB, BB, CAP = 80, 1, 2, 4


def bucket(street: int, equity: float, thresholds) -> int:
    return bisect.bisect_right(thresholds[street], equity)


def infoset_key(street: int, bkt: int, is_button: bool, pot_start: int, seq: str) -> str:
    return f"{street}{'b' if is_button else 'n'}{bkt}.{pot_start}.{seq}"


class State:
    """Exact replica of game.Game betting rules (seat based, no cards)."""
    __slots__ = ("street", "bets", "stacks", "contrib", "acted", "raises", "last_inc",
                 "to_act", "button", "seq", "terminal", "folder")

    def __init__(self, button: int):
        self.street = 0
        self.bets = [0, 0]; self.stacks = [STACK, STACK]; self.contrib = [0, 0]
        self.acted = [False, False]; self.raises = 0; self.last_inc = BB
        self.button = button; self.seq = ""; self.terminal = False; self.folder = -1
        self._put(button, SB); self._put(1 - button, BB)
        self.to_act = button

    def copy(self) -> "State":
        n = State.__new__(State)
        n.street = self.street; n.bets = self.bets[:]; n.stacks = self.stacks[:]
        n.contrib = self.contrib[:]; n.acted = self.acted[:]; n.raises = self.raises
        n.last_inc = self.last_inc; n.to_act = self.to_act; n.button = self.button
        n.seq = self.seq; n.terminal = self.terminal; n.folder = self.folder
        return n

    def _put(self, s, amt):
        self.stacks[s] -= amt; self.bets[s] += amt; self.contrib[s] += amt

    @property
    def pot(self):
        return self.contrib[0] + self.contrib[1]

    def pot_start(self):
        return self.pot - self.bets[0] - self.bets[1]

    def to_call(self, s):
        return max(0, self.bets[1 - s] - self.bets[s])

    def legal(self):
        s = self.to_act
        out = []
        if self.to_call(s) > 0:
            out.append(0)
        out.append(1)
        if self.raises < CAP and self.stacks[s] > self.to_call(s) and self.stacks[1 - s] > 0:
            out += [2, 3]
        return out

    def raise_to(self, s, a):
        my, opp = self.bets[s], self.bets[1 - s]
        tc = self.to_call(s); pot = self.pot
        all_in = my + self.stacks[s]
        pot_to = opp + pot + tc
        half_to = opp + (pot + tc) // 2
        tgt = half_to if a == 2 else pot_to
        return min(max(tgt, opp + max(self.last_inc, BB)), pot_to, all_in)

    def apply(self, a):
        s = self.to_act; o = 1 - s
        self.seq += ACT_CHARS[a]
        if a == 0:
            self.terminal = True; self.folder = s; return
        if a == 1:
            amt = min(self.to_call(s), self.stacks[s])
        else:
            new = self.raise_to(s, a)
            amt = new - self.bets[s]
            self.last_inc = new - self.bets[o]
            self.raises += 1
            self.acted[o] = False
        self._put(s, amt)
        self.acted[s] = True
        if self.acted[0] and self.acted[1] and self.bets[0] == self.bets[1]:
            if self.street == 3 or 0 in self.stacks:
                self.street = 3; self.terminal = True; return
            self.street += 1
            self.bets = [0, 0]; self.acted = [False, False]; self.raises = 0
            self.last_inc = BB; self.to_act = 1 - self.button; self.seq = ""
        else:
            self.to_act = o

    def utility(self, p, showdown_result):
        """Chips won by seat p. showdown_result: +1 seat0 wins, -1 seat1, 0 split."""
        pot = self.pot
        if self.folder >= 0:
            winner = 1 - self.folder
        elif showdown_result == 0:
            return pot // 2 - self.contrib[p]
        else:
            winner = 0 if showdown_result > 0 else 1
        return (pot if winner == p else 0) - self.contrib[p]
