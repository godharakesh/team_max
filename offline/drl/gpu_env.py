"""GPU-batched exact environment for heads-up Short Deck PLO5 (torch).

* Transitions: offline/tree.py's flattened betting tree (state = node index,
  child_a[node, action]); payoffs from term_c0/term_c1/term_folder + showdown.
* Showdown: plo_engine.T[triple, pair] (the referee's own scores), best of
  10 board triples x 10 hole pairs by gather + max.
* Card-strength features per (deal, seat, street), computed ONCE per deal with
  the same estimand the runtime bot uses (plo_engine.sample_equity):
  [preflop equity, equity vs random, current made-hand strength, potential],
  plus the made-hand category and the blueprint v2 bucket.
* Controllers: 0 learner (callable), 1 CFR blueprint v2 (bp mode), 2 station,
  3 nit, 4 maniac, 5 tag, 6 barrel (the offline/arena.py Simple bots), 7 frozen.
"""
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import plo_engine as PE  # noqa: E402
import nn_policy as NP  # noqa: E402
from tables import load_tables  # noqa: E402

LEARNER, BP, STATION, NIT, MANIAC, TAG, BARREL, FROZEN = range(8)
NCTRL = 8
CTRL_NAMES = ["learner", "bp", "station", "nit", "maniac", "tag", "barrel", "frozen"]
NB_VIS = (0, 3, 4, 5)


class Engine:
    def __init__(self, device="cuda", mc_samples=256, chunk=1024, seed=0):
        self.dev = torch.device(device)
        self.N = mc_samples
        self.chunk = chunk
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        PE.build_tables()
        d = self.dev
        self.T = torch.as_tensor(PE.T.reshape(-1), dtype=torch.int32, device=d)
        self.IDX2 = torch.as_tensor(PE.IDX2.reshape(-1).astype(np.int64), device=d)
        self.IDX3 = torch.as_tensor(PE.IDX3.reshape(-1).astype(np.int64), device=d)
        self.PRE = torch.as_tensor(PE.PRE_EQ, dtype=torch.float32, device=d)
        self.BINOM = torch.as_tensor(PE.BINOM, dtype=torch.int64, device=d)
        self.P5 = torch.as_tensor(list(combinations(range(5), 2)), dtype=torch.int64, device=d)
        self.T5 = torch.as_tensor(list(combinations(range(5), 3)), dtype=torch.int64, device=d)
        self.T4 = torch.as_tensor(list(combinations(range(4), 3)), dtype=torch.int64, device=d)
        tb = load_tables()
        self.tb = tb
        g = lambda k, dt=None: torch.as_tensor(tb[k], device=d) if dt is None else torch.as_tensor(tb[k], dtype=dt, device=d)
        self.node_type = g("node_type"); self.node_player = g("node_player"); self.node_street = g("node_street")
        self.node_key = g("node_key"); self.legal = g("legal"); self.child_a = g("child_a")
        self.term_c0 = g("term_c0"); self.term_c1 = g("term_c1"); self.term_folder = g("term_folder")
        self.F_act = g("F_act"); self.F_r0 = g("F_r0")
        info = g("info")
        self.pot = info[:, 0].float(); self.to_call = info[:, 1].float()
        self.n_keys = len(tb["key_strings"])
        # blueprint v2 on GPU
        with open(os.path.join(ROOT, "blueprint2.pkl"), "rb") as f:
            bp = pickle.load(f)
        assert all(bp["keys"][str(k)] == i for i, k in enumerate(tb["key_strings"]))
        ab = bp["abstraction"]
        self.bp_off = torch.as_tensor(bp["key_offset"].astype(np.int64), device=d)
        self.bp_pol = torch.as_tensor(bp["policy"].astype(np.float32), device=d)
        self.bp_pre = torch.as_tensor(ab["pre"], dtype=torch.float32, device=d)
        self.bp_riv = torch.as_tensor(ab["riv"], dtype=torch.float32, device=d)
        self.bp_np = int(ab["np"])
        self.bp_e = {1: torch.as_tensor(ab["flop_e"], dtype=torch.float32, device=d),
                     2: torch.as_tensor(ab["turn_e"], dtype=torch.float32, device=d)}
        self.bp_p = {1: torch.as_tensor(np.array(ab["flop_p"]), dtype=torch.float32, device=d),
                     2: torch.as_tensor(np.array(ab["turn_p"]), dtype=torch.float32, device=d)}

    # ------------------------------------------------------------------ cards
    def rand(self, *shape):
        return torch.rand(*shape, device=self.dev, generator=self.gen)

    def pair_idx(self, h):          # (...,5) -> (...,10)
        return self.IDX2[h[..., self.P5[:, 0]] * 36 + h[..., self.P5[:, 1]]]

    def tri_idx(self, b, tset):     # (...,k) board, tset (m,3) -> (...,m)
        return self.IDX3[b[..., tset[:, 0]] * 1296 + b[..., tset[:, 1]] * 36 + b[..., tset[:, 2]]]

    def best(self, tri, pairs):     # (...,m), (...,10) -> (...) best score
        idx = tri.unsqueeze(-1) * 630 + pairs.unsqueeze(-2)
        return self.T[idx].flatten(-2).amax(-1)

    def colex(self, hs):            # ascending (...,5)
        B = self.BINOM
        return B[hs[..., 0], 1] + B[hs[..., 1], 2] + B[hs[..., 2], 3] + B[hs[..., 3], 4] + B[hs[..., 4], 5]

    def _mc(self, hole, board, nb, N):
        """Equity vs a random hand (+ made-hand strength) for hole (n,5) on board[:, :nb]."""
        n = hole.shape[0]
        known = torch.cat([hole, board[:, :nb]], 1)
        keys = self.rand(n, N, 36)
        keys.scatter_(2, known.unsqueeze(1).expand(n, N, known.shape[1]), 2.0)
        k = 5 + (5 - nb)
        pick = keys.topk(k, dim=2, largest=False).indices
        opp = pick[..., :5]
        if nb < 5:
            full = torch.cat([board[:, None, :nb].expand(n, N, nb), pick[..., 5:]], 2)
        else:
            full = board[:, None, :].expand(n, N, 5)
        tri = self.tri_idx(full, self.T5)                        # (n,N,10)
        opp_p = self.pair_idx(opp)                               # (n,N,10)
        my_p = self.pair_idx(hole)                               # (n,10)
        if nb == 5:
            mine = self.best(self.tri_idx(board, self.T5), my_p).unsqueeze(1)
        else:
            mine = self.best(tri, my_p.unsqueeze(1).expand(n, N, 10))
        theirs = self.best(tri, opp_p)
        w = (mine > theirs).float() + 0.5 * (mine == theirs).float()
        raw = w.mean(1)
        if nb == 5:
            return raw, raw
        tset = self.T5[:1] if nb == 3 else self.T4
        vis = self.tri_idx(board[:, :nb], tset)                   # (n,m)
        my_cur = self.best(vis, my_p).unsqueeze(1)
        th_cur = self.best(vis.unsqueeze(1).expand(n, N, vis.shape[1]), opp_p)
        hs = ((my_cur > th_cur).float() + 0.5 * (my_cur == th_cur).float()).mean(1)
        return raw, hs

    @torch.no_grad()
    def gen_deals(self, B):
        """B uniform random deals with all per-(seat, street) features."""
        d = self.dev
        perm = self.rand(B, 36).argsort(1)[:, :15]
        holes = torch.stack([perm[:, 0:5].sort(1).values, perm[:, 5:10].sort(1).values], 1)   # (B,2,5)
        board = perm[:, 10:15].contiguous()
        strength = torch.zeros(B, 2, 4, 4, device=d)
        cat = torch.zeros(B, 2, 4, dtype=torch.int64, device=d)
        for p in range(2):
            pre = self.PRE[self.colex(holes[:, p])]
            strength[:, p, 0] = torch.stack([pre, pre, pre, torch.zeros_like(pre)], 1)
            for s in (1, 2, 3):
                nb = NB_VIS[s]
                raws, hss = [], []
                for i in range(0, B, self.chunk):
                    r, h = self._mc(holes[i:i + self.chunk, p], board[i:i + self.chunk], nb, self.N)
                    raws.append(r); hss.append(h)
                raw = torch.cat(raws); hs = torch.cat(hss)
                strength[:, p, s] = torch.stack([pre, raw, hs, raw - hs], 1)
                tset = self.T5[:1] if nb == 3 else (self.T4 if nb == 4 else self.T5)
                cur = self.best(self.tri_idx(board[:, :nb], tset), self.pair_idx(holes[:, p]))
                cat[:, p, s] = (cur >> 20).clamp(max=8)
        tri = self.tri_idx(board, self.T5)
        s0 = self.best(tri, self.pair_idx(holes[:, 0]))
        s1 = self.best(tri, self.pair_idx(holes[:, 1]))
        res = (s0 > s1).long() - (s1 > s0).long()
        D = {"holes": holes, "board": board, "strength": strength, "cat": cat, "res": res,
             "s0": s0, "s1": s1}
        D["bkt"] = self.bp_buckets(strength)
        return D

    def bp_buckets(self, strength):
        B = strength.shape[0]
        bkt = torch.zeros(B, 2, 4, dtype=torch.int64, device=self.dev)
        bkt[:, :, 0] = torch.searchsorted(self.bp_pre, strength[:, :, 0, 0].contiguous(), right=True)
        bkt[:, :, 3] = torch.searchsorted(self.bp_riv, strength[:, :, 3, 1].contiguous(), right=True)
        for s in (1, 2):
            raw = strength[:, :, s, 1].contiguous(); pot = raw - strength[:, :, s, 2]
            e = torch.searchsorted(self.bp_e[s], raw, right=True)
            pth = self.bp_p[s][e]                                   # (B,2,NP-1)
            bkt[:, :, s] = e * self.bp_np + (pot.unsqueeze(-1) > pth).sum(-1)
        return bkt

    # --------------------------------------------------------------- features
    def actor_x(self, D, h, r, node):
        """Actor features (n, BASE_DIM) + key ids (n,) for hands h, roles r, decision nodes."""
        n = h.shape[0]
        d = self.dev
        s = self.node_street[node]
        hole = D["holes"][h, r]                                     # (n,5)
        board = D["board"][h]
        nb = torch.as_tensor(NB_VIS, device=d)[s]
        bm = (torch.arange(5, device=d).unsqueeze(0) < nb.unsqueeze(1)).float()
        one = torch.ones(n, 5, device=d)
        z = lambda k: torch.zeros(n, k, device=d)
        cat = D["cat"][h, r, s]
        x = torch.cat([
            z(36).scatter_(1, hole, 1.0),
            z(36).scatter_add_(1, board, bm),
            z(9).scatter_add_(1, hole >> 2, 0.5 * one),
            z(4).scatter_add_(1, hole & 3, 0.2 * one),
            z(9).scatter_add_(1, board >> 2, 0.5 * bm),
            z(4).scatter_add_(1, board & 3, 0.2 * bm),
            z(9).scatter_(1, cat.unsqueeze(1), (s > 0).float().unsqueeze(1)),
            D["strength"][h, r, s],
            self.F_act[node]], 1)
        return x, self.node_key[node]

    CRITIC_DIM = NP.NODE_DIM + 2 + 8 + 18 + 12 + 3 + 2 * NCTRL

    def critic_x(self, D, h, node, ctrl):
        """Privileged critic features: everything about the deal (both hands, final result)."""
        d = self.dev
        n = h.shape[0]
        s = self.node_street[node]
        st = D["strength"][h]                                       # (n,2,4,4)
        ar = torch.arange(n, device=d)
        cur = st[ar, :, s]                                          # (n,2,4)
        cat = D["cat"][h][ar, :, s]                                 # (n,2)
        catoh = torch.zeros(n, 2, 9, device=d).scatter_(2, cat.unsqueeze(2), 1.0) * (s > 0).float().view(n, 1, 1)
        allst = torch.cat([st[:, :, 0, :1], st[:, :, 1, 1:3], st[:, :, 2, 1:3], st[:, :, 3, 1:2]], 2)  # (n,2,6)
        res = D["res"][h]
        resoh = torch.stack([(res > 0).float(), (res == 0).float(), (res < 0).float()], 1)
        ctl = torch.zeros(n, 2 * NCTRL, device=d)
        ctl.scatter_(1, ctrl[h, 0].unsqueeze(1), 1.0)
        ctl.scatter_(1, NCTRL + ctrl[h, 1].unsqueeze(1), 1.0)
        pl = torch.zeros(n, 2, device=d).scatter_(1, self.node_player[node].unsqueeze(1), 1.0)
        x = torch.cat([self.F_r0[node], pl, cur.reshape(n, 8), catoh.reshape(n, 18), allst.reshape(n, 12),
                       resoh, ctl], 1)
        return x, self.node_key[node]

    # ---------------------------------------------------------------- payoffs
    def payoff0(self, node, res):
        """Net chips of role 0 at terminal nodes."""
        c0, c1 = self.term_c0[node], self.term_c1[node]
        fold = self.node_type[node] == 1
        f = self.term_folder[node]
        pf = torch.where(f == 0, -c0, c1)
        ps = torch.where(res > 0, c1, torch.where(res < 0, -c0, 0.5 * (c1 - c0)))
        return torch.where(fold, pf, ps)

    # --------------------------------------------------------- fixed policies
    def bp_probs(self, D, h, r, node):
        s = self.node_street[node]
        idx = self.bp_off[self.node_key[node]] + D["bkt"][h, r, s]
        p = self.bp_pol[idx] * self.legal[node].float()
        tot = p.sum(1, keepdim=True)
        uni = self.legal[node].float()
        uni = uni / uni.sum(1, keepdim=True)
        return torch.where(tot > 0, p / tot.clamp_min(1e-12), uni)

    def simple_probs(self, kind, D, h, r, node):
        n = h.shape[0]
        d = self.dev
        L = self.legal[node]
        can_r = L[:, 3]
        s = self.node_street[node]
        eq = D["strength"][h, r, s, 1]
        tc = self.to_call[node]
        pot = self.pot[node]
        po = torch.where(tc > 0, tc / (pot + tc).clamp_min(1), torch.zeros_like(tc))
        P = torch.zeros(n, 4, device=d)
        F, C, H, Pt = 0, 1, 2, 3

        def put(mask, a):
            P[mask & (P.sum(1) == 0), a] = 1.0

        if kind == STATION:
            P[:, C] = 1.0
        elif kind == MANIAC:
            P[:, C] = torch.where(can_r, 0.45, 1.0)
            P[:, H] = torch.where(can_r, 0.275, 0.0)
            P[:, Pt] = torch.where(can_r, 0.275, 0.0)
        elif kind == NIT:
            put((eq > 0.70) & can_r, Pt)
            put((tc == 0) | (eq >= po + 0.12), C)
            put(torch.ones(n, dtype=torch.bool, device=d), F)
        elif kind == TAG:
            put((eq > 0.62) & can_r, Pt)
            put((eq > 0.55) & can_r & (tc == 0), H)
            put((tc == 0) | (eq >= po), C)
            put(torch.ones(n, dtype=torch.bool, device=d), F)
        elif kind == BARREL:
            post = s > 0
            lead = (tc == 0) & can_r & post
            pbet = torch.where(eq > 0.6, 1.0, torch.where(s == 1, 0.6, 0.8))
            ba = torch.where(s == 1, H, Pt)
            P[lead, C] = 1.0 - pbet[lead]
            P[lead.nonzero().squeeze(1), ba[lead]] = pbet[lead]
            rest = ~lead
            put(rest & (eq > 0.70) & can_r, Pt)
            put(rest & ((tc == 0) | (eq >= po)), C)
            put(rest, F)
        else:
            raise ValueError(kind)
        return P

    # ----------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, D, ctrl, learner=None, frozen=None, greedy=False):
        """Play all B hands to the end. ctrl (B,2): controller per role.
        learner/frozen: callables (x, kid) -> logits. Returns (final nodes, payoff0, records)."""
        d = self.dev
        B = ctrl.shape[0]
        node = torch.zeros(B, dtype=torch.int64, device=d)
        recs = []
        for _ in range(16):
            dec = self.node_type[node] == 0
            if not bool(dec.any()):
                break
            h = dec.nonzero().squeeze(1)
            nd = node[h]
            r = self.node_player[nd]
            c = ctrl[h, r]
            probs = torch.zeros(h.shape[0], 4, device=d)
            for k in range(NCTRL):
                m = c == k
                if not bool(m.any()):
                    continue
                hm, rm, nm = h[m], r[m], nd[m]
                if k in (LEARNER, FROZEN):
                    fn = learner if k == LEARNER else frozen
                    x, kid = self.actor_x(D, hm, rm, nm)
                    if isinstance(fn, (list, tuple)):      # league of frozen nets: one per hand (D["fz"])
                        lgs = torch.stack([f(x, kid).float() for f in fn], 0)
                        lg = lgs[D["fz"][hm], torch.arange(hm.shape[0], device=d)]
                    else:
                        lg = fn(x, kid).float()
                    lg = lg.masked_fill(~self.legal[nm], -1e9)
                    probs[m] = torch.softmax(lg, 1)
                elif k == BP:
                    probs[m] = self.bp_probs(D, hm, rm, nm)
                else:
                    probs[m] = self.simple_probs(k, D, hm, rm, nm)
            probs = probs * self.legal[nd].float()
            if greedy:
                a = probs.argmax(1)
            else:
                a = torch.multinomial(probs, 1, generator=self.gen).squeeze(1)
            recs.append((h, nd, r, c, a, probs))
            node[h] = self.child_a[nd, a]
        assert bool((self.node_type[node] > 0).all())
        return node, self.payoff0(node, D["res"]), recs


def cat_records(recs):
    return [torch.cat([rc[i] for rc in recs]) for i in range(6)]
