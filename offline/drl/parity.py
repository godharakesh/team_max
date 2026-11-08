"""Parity of the GPU environment with the official rules engine (game.py).

Plays N random-action hands on the GPU, replays every one in game.Game with the
same cards, and checks at every decision: acting seat, legal actions, the full
actor feature vector (nn_policy.obs_features on the judge's Observation vs the
GPU's actor_x), the public key id, the blueprint distribution (strategy._bp_probs
vs GPU bp_probs); at the end: nets and showdown winner. Also compares the GPU
Monte Carlo equity estimates with plo_engine.sample_equity at high sample counts.

usage: python offline/drl/parity.py [hands]
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_env import Engine, LEARNER, BP, NIT, TAG, BARREL, MANIAC, STATION, cat_records  # noqa: E402
import nn_policy as NP  # noqa: E402
import plo_engine as PE  # noqa: E402
from game import Deal, Game, Action  # noqa: E402


class _S:
    def __init__(self, hs):
        self.hs = hs


def main(n_hands=12000):
    import strategy as S
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = Engine(dev, mc_samples=64, chunk=2048, seed=7)
    t0 = time.time()
    D = eng.gen_deals(n_hands)
    torch.cuda.synchronize() if dev == "cuda" else None
    print(f"gen_deals({n_hands}) at N=64: {time.time() - t0:.2f}s")
    # random policy for half the hands; simple bots/blueprint for the rest (covers their code paths too)
    ctrl = torch.zeros(n_hands, 2, dtype=torch.int64, device=dev)
    kinds = torch.tensor([LEARNER, BP, NIT, TAG, BARREL, MANIAC, STATION], device=dev)
    ctrl[:, 0] = kinds[torch.randint(0, len(kinds), (n_hands,), device=dev)]
    ctrl[:, 1] = kinds[torch.randint(0, len(kinds), (n_hands,), device=dev)]
    uniform = lambda x, kid: torch.zeros(x.shape[0], 4, device=x.device)
    node, pay0, recs = eng.rollout(D, ctrl, learner=uniform)
    H, N, R, C, A, PB = cat_records(recs)
    X, KID = eng.actor_x(D, H, R, N)
    cpu = lambda t: t.detach().cpu().numpy()
    H, N, R, C, A, PB, X, KID = map(cpu, (H, N, R, C, A, PB, X, KID))
    holes, board, strength, res = cpu(D["holes"]), cpu(D["board"]), cpu(D["strength"]), cpu(D["res"])
    pay0 = cpu(pay0)
    by_hand = {}
    for i in np.argsort(H, kind="stable"):
        by_hand.setdefault(int(H[i]), []).append(i)
    keys = list(eng.tb["key_strings"])
    n_dec = 0; max_fd = 0.0; bp_checked = 0; max_bpd = 0.0
    bad = []
    for hnd in range(n_hands):
        bseat = hnd % 2                      # role 0 (button) sits in seat bseat
        role_seat = (bseat, 1 - bseat)
        hc = [None, None]
        hc[role_seat[0]] = tuple(int(c) for c in holes[hnd, 0])
        hc[role_seat[1]] = tuple(int(c) for c in holes[hnd, 1])
        g = Game(Deal(hole_cards=(hc[0], hc[1]), board=tuple(int(c) for c in board[hnd])), bseat, hand_index=hnd)
        for i in by_hand.get(hnd, []):
            r = int(R[i]); seat = role_seat[r]
            if g.is_terminal or g.to_act != seat:
                bad.append((hnd, "seat")); break
            obs = g.observation(seat)
            leg = sorted(int(a) for a in obs.legal_actions)
            gl = sorted(int(a) for a in np.nonzero(cpu(eng.legal[int(N[i])]))[0])
            if leg != gl:
                bad.append((hnd, "legal", leg, gl)); break
            st = int(obs.street)
            pre, raw, hs = (float(v) for v in strength[hnd, r, st, :3])
            cur = PE.current_score(obs.hole_cards, obs.board) if st > 0 else 0
            x = NP.obs_features(obs, pre, raw, hs, cur)
            fd = float(np.abs(x - X[i]).max())
            max_fd = max(max_fd, fd)
            if fd > 1e-5:
                bad.append((hnd, "feat", int(np.argmax(np.abs(x - X[i]))), fd)); break
            if keys[int(KID[i])] != NP.public_key(obs):
                bad.append((hnd, "key")); break
            if C[i] == BP:
                mem = {"bpv": 2, "eq_cache": (None, _S(hs))}
                p = S._bp_probs(obs, raw if st > 0 else pre, mem)
                if p is not None:
                    p = np.array([float(p[a]) if a in leg else 0.0 for a in range(4)])
                    if p.sum() > 0:
                        p /= p.sum()
                        max_bpd = max(max_bpd, float(np.abs(p - PB[i]).max()))
                        bp_checked += 1
            g.apply(Action(int(A[i])))
            n_dec += 1
        if bad and bad[-1][0] == hnd:
            continue
        if not g.is_terminal:
            bad.append((hnd, "not terminal")); continue
        rr = g.result()
        if abs(rr.nets[role_seat[0]] - pay0[hnd]) > 1e-6:
            bad.append((hnd, "net", rr.nets, pay0[hnd]))
        if rr.went_to_showdown:
            w = 0 if rr.winner_seat is None else (1 if rr.winner_seat == role_seat[0] else -1)
            if w != res[hnd]:
                bad.append((hnd, "winner"))
    print(f"hands {n_hands} decisions {n_dec} mismatches {len(bad)} max feature diff {max_fd:.2e} "
          f"bp rows checked {bp_checked} max bp prob diff {max_bpd:.2e}")
    print("first mismatches:", bad[:5])
    # Monte Carlo estimand check vs the CPU estimator (high sample counts)
    eng2 = Engine(dev, mc_samples=4096, chunk=64, seed=11)
    D2 = eng2.gen_deals(150)
    rng = np.random.default_rng(3)
    errs = {1: [], 2: [], 3: []}
    hs_err = {1: [], 2: []}
    st2 = cpu(D2["strength"]); ho2 = cpu(D2["holes"]); bo2 = cpu(D2["board"])
    for i in range(150):
        for s in (1, 2, 3):
            smp = PE.sample_equity(list(ho2[i, 0]), list(bo2[i, :2 + s]), 4096, rng)
            errs[s].append(st2[i, 0, s, 1] - smp.raw)
            if s < 3:
                hs_err[s].append(st2[i, 0, s, 2] - smp.hs)
    for s in (1, 2, 3):
        e = np.array(errs[s])
        print(f"street {s}: GPU-CPU equity diff mean {e.mean():+.4f} sd {e.std():.4f} (expected sd ~{np.sqrt(2 * 0.2 / 4096):.4f})")
    for s in (1, 2):
        e = np.array(hs_err[s])
        print(f"street {s}: GPU-CPU made-hand strength diff mean {e.mean():+.4f} sd {e.std():.4f}")
    return len(bad)


if __name__ == "__main__":
    sys.exit(1 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 12000) else 0)
