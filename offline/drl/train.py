"""Deep RL for heads-up Short Deck PLO5 on the GPU.

Phase 1 (optional, --bc-steps): behaviour cloning of the CFR blueprint (bp mode)
  on states visited by bp vs {bp, simple bots}; loss = CE(bp distribution, policy).
Phase 2: regularised self-play policy optimisation, Magnetic-Mirror-Descent style:
  maximise  E[sum_a pi(a|I) A(h,a)] - alpha * KL(pi || magnet)
  * all-actions advantages A(h,a) = Q(h,a) - sum_b mu(b) Q(h,b): every legal action's
    child is evaluated exactly (terminal payoff) or by a privileged critic V(h')
    that sees the whole deal (valid: chance outcomes do not depend on actions);
  * the critic regresses Monte Carlo returns of role 0 on every visited node;
  * the magnet starts as the initial policy (BC or uniform) and is reset to the
    current policy every --magnet-every steps (R-NaD / MMD-with-moving-magnet);
  * league: self-play plus fixed opponents (blueprint, station, nit, maniac, tag,
    barrel) per hand, the learner's seat random; only learner decisions train the actor.
Every --eval-every steps: duplicate evaluation (same deals, seats swapped) vs each
opponent on a fixed deal set; checkpoints are exported as runtime .npz files.

usage: python offline/drl/train.py --name run1 [--bc-steps 1500] [--steps 20000] ...
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_env import (Engine, cat_records, LEARNER, BP, STATION, NIT, MANIAC, TAG, BARREL,  # noqa: E402
                     FROZEN, CTRL_NAMES)
import nn_policy as NP  # noqa: E402

SCALE = 20.0          # chips per value unit


class MLP(nn.Module):
    def __init__(self, din, n_keys, emb, hid, nl, dout):
        super().__init__()
        self.emb = nn.Embedding(n_keys, emb)
        nn.init.normal_(self.emb.weight, std=0.1)
        layers, d = [], din + emb
        for _ in range(nl):
            layers += [nn.Linear(d, hid), nn.ReLU()]
            d = hid
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, kid):
        return self.net(torch.cat([x, self.emb(kid)], 1))


def export(actor, path, key_strings, meta):
    d = {"emb": actor.emb.weight.detach().float().cpu().numpy(), "keys": np.array(key_strings)}
    lin = [m for m in actor.net if isinstance(m, nn.Linear)]
    for i, l in enumerate(lin):
        d[f"W{i}"] = l.weight.detach().float().cpu().numpy().T.copy()
        d[f"b{i}"] = l.bias.detach().float().cpu().numpy().copy()
    for k, v in meta.items():
        d[f"meta_{k}"] = np.array(v)
    np.savez(path, **d)


def load_any(net, path, dev):
    """Load actor weights from a training .pt or an exported runtime .npz."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        sd = {"emb.weight": torch.as_tensor(z["emb"])}
        lin = [i for i, m in enumerate(net.net) if isinstance(m, nn.Linear)]
        for k, i in enumerate(lin):
            sd[f"net.{i}.weight"] = torch.as_tensor(z[f"W{k}"].T.copy())
            sd[f"net.{i}.bias"] = torch.as_tensor(z[f"b{k}"])
        net.load_state_dict(sd)
        return {}
    st = torch.load(path, map_location=dev)
    net.load_state_dict(st["actor"])
    return st


class Pool:
    def __init__(self, eng, size, chunk=65536):
        parts = [eng.gen_deals(min(chunk, size - i)) for i in range(0, size, chunk)]
        self.D = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        self.size, self.eng = size, eng

    def sample(self, B):
        idx = torch.randint(0, self.size, (B,), device=self.eng.dev, generator=self.eng.gen)
        return {k: v[idx] for k, v in self.D.items()}

    def refresh(self, k):
        new = self.eng.gen_deals(k)
        idx = torch.randint(0, self.size, (k,), device=self.eng.dev, generator=self.eng.gen)
        for key in self.D:
            self.D[key][idx] = new[key]


def make_ctrl(B, mix, eng):
    """mix: list of (opponent controller, weight); LEARNER = self-play."""
    dev = eng.dev
    w = torch.tensor([m[1] for m in mix], dtype=torch.float32, device=dev)
    kinds = torch.tensor([m[0] for m in mix], dtype=torch.int64, device=dev)
    opp = kinds[torch.multinomial(w, B, replacement=True, generator=eng.gen)]
    lr = torch.randint(0, 2, (B,), device=dev, generator=eng.gen)
    ar = torch.arange(B, device=dev)
    ctrl = torch.zeros(B, 2, dtype=torch.int64, device=dev)
    ctrl[ar, lr] = LEARNER
    ctrl[ar, 1 - lr] = opp
    return ctrl


def masked(lg, legal):
    return lg.float().masked_fill(~legal, -1e9)


@torch.no_grad()
def duplicate_eval(eng, Dev, actor, opp, frozen=None):
    """Learner vs opp on the same deals with seats swapped: mean and s.e. of
    (net_1 + net_2) / 2 chips = bb per deal (the arena's unit)."""
    B = Dev["res"].shape[0]
    c1 = torch.zeros(B, 2, dtype=torch.int64, device=eng.dev); c1[:, 0] = LEARNER; c1[:, 1] = opp
    c2 = torch.zeros(B, 2, dtype=torch.int64, device=eng.dev); c2[:, 0] = opp; c2[:, 1] = LEARNER
    f = lambda x, k: actor(x, k)
    _, p1, _ = eng.rollout(Dev, c1, learner=f, frozen=frozen)
    _, p2, _ = eng.rollout(Dev, c2, learner=f, frozen=frozen)
    paired = (p1 - p2) / 2.0
    return float(paired.mean()), float(paired.std() / np.sqrt(B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mc", type=int, default=256)
    ap.add_argument("--pool", type=int, default=1 << 20)
    ap.add_argument("--refresh", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--hid", type=int, default=384)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--bc-steps", type=int, default=0)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--init", default="", help="start from a checkpoint (.pt) instead of BC/random")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.0, help="entropy bonus")
    ap.add_argument("--magnet-every", type=int, default=500)
    ap.add_argument("--warm", type=int, default=300, help="critic-only steps before actor updates")
    ap.add_argument("--mb", type=int, default=2, help="minibatches per step")
    ap.add_argument("--mix", default="self:0.5,bp:0.25,station:0.05,nit:0.05,maniac:0.05,tag:0.05,barrel:0.05")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-deals", type=int, default=65536)
    ap.add_argument("--hours", type=float, default=99.0)
    ap.add_argument("--frozen", default="", help="fixed opponent checkpoint (.pt) for the 'frozen' controller")
    ap.add_argument("--mem-frac", type=float, default=0.15, help="cap on this process's share of GPU memory")
    ap.add_argument("--exploiter", type=int, default=0, help="hands per step for a co-trained exploiter (0 = off)")
    ap.add_argument("--exploiter-init", default="", help="exploiter start weights (.pt/.npz)")
    ap.add_argument("--exploiter-lr", type=float, default=2e-4)
    ap.add_argument("--exploiter-warm", type=int, default=100)
    ap.add_argument("--exploiter-reset", type=int, default=0,
                    help="every N steps: freeze a copy of the exploiter into the league, restart it from its init")
    ap.add_argument("--bp-eval-only", action="store_true", help="evaluate only vs bp (+frozen)")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)   # shared GPU: stay well under the 14 GB budget
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    out = os.path.join(HERE, "..", "..", "..", "runs", args.name)
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.jsonl"), "a")

    def log(d):
        d["t"] = round(time.time() - T0, 1)
        logf.write(json.dumps(d) + "\n"); logf.flush()
        print(json.dumps(d), flush=True)

    T0 = time.time()
    eng = Engine("cuda", mc_samples=args.mc, chunk=2048, seed=args.seed)
    keys = [str(k) for k in eng.tb["key_strings"]]
    pool = Pool(eng, args.pool)
    eval_eng_gen = torch.Generator(device=eng.dev); eval_eng_gen.manual_seed(12345)
    g_save = eng.gen; eng.gen = eval_eng_gen
    Dev = eng.gen_deals(args.eval_deals)
    eng.gen = g_save
    log({"phase": "init", "pool": args.pool, "secs": round(time.time() - T0, 1), "args": vars(args)})

    dev = eng.dev
    actor = MLP(NP.BASE_DIM, eng.n_keys, args.emb, args.hid, args.layers, 4).to(dev)
    critic = MLP(Engine.CRITIC_DIM, eng.n_keys, args.emb, 512, 3, 1).to(dev)
    names = {"self": LEARNER, "bp": BP, "station": STATION, "nit": NIT, "maniac": MANIAC, "tag": TAG,
             "barrel": BARREL, "frozen": FROZEN}
    mix = [(names[k], float(v)) for k, v in (x.split(":") for x in args.mix.split(","))]
    meta = {"hid": args.hid, "layers": args.layers, "emb": args.emb, "mc": args.mc, "name": args.name}

    init_actor = [None]
    fixed_opp = None
    fixed_list = []
    for fp in [x for x in args.frozen.split(",") if x]:
        net = MLP(NP.BASE_DIM, eng.n_keys, args.emb, args.hid, args.layers, 4).to(dev).eval()
        load_any(net, fp, dev)
        fixed_list.append(net)
    if len(fixed_list) == 1:
        fixed_opp = fixed_list[0]
    elif fixed_list:
        nets = fixed_list
        fixed_opp = nets                      # per-hand choice via D["fz"]

    def evaluate(step, label=""):
        actor.eval()
        res = {"phase": "eval", "step": step, "label": label}
        opps = (BP,) if args.bp_eval_only else (BP, STATION, NIT, MANIAC, TAG, BARREL)
        for o in opps:
            sub = Dev if o == BP else {k: v[:16384] for k, v in Dev.items()}
            m, se = duplicate_eval(eng, sub, actor, o)
            res[CTRL_NAMES[o]] = [round(m, 3), round(se, 3)]
        evs = [("init", init_actor[0])] + [(f"frozen{i}" if len(fixed_list) > 1 else "frozen", n_)
                                             for i, n_ in enumerate(fixed_list)]
        for nm, net in evs:
            if net is not None:
                m, se = duplicate_eval(eng, Dev, actor, FROZEN, frozen=lambda x, k, net=net: net(x, k))
                res[nm] = [round(m, 3), round(se, 3)]
        res["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        return res

    # ------------------------------------------------------------- phase 1: BC
    if args.init:
        st = load_any(actor, args.init, dev)
        if "critic" in st:
            critic.load_state_dict(st["critic"])
        log({"phase": "init_from", "path": args.init})
    if args.bc_steps > 0:
        opt = torch.optim.Adam(actor.parameters(), lr=args.bc_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.bc_steps, eta_min=args.bc_lr * 0.05)
        bc_mix = [(BP, 0.7), (STATION, 0.06), (NIT, 0.06), (MANIAC, 0.06), (TAG, 0.06), (BARREL, 0.06)]
        for step in range(args.bc_steps):
            D = pool.sample(args.batch)
            ctrl = make_ctrl(args.batch, bc_mix, eng)
            ctrl[ctrl == LEARNER] = BP                     # the "learner" seat is played by the blueprint
            _, _, recs = eng.rollout(D, ctrl)
            H, N, R, C, A, PB = cat_records(recs)
            m = C == BP
            H, N, R, PB = H[m], N[m], R[m], PB[m]
            x, kid = eng.actor_x(D, H, R, N)
            legal = eng.legal[N]
            actor.train()
            perm = torch.randperm(H.shape[0], device=dev, generator=eng.gen)
            tot = 0.0
            for mbi in perm.chunk(args.mb):
                logp = F.log_softmax(masked(actor(x[mbi], kid[mbi]), legal[mbi]), 1)
                loss = -(PB[mbi] * logp).sum(1).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt.step()
                tot += float(loss.detach()) / args.mb
            sched.step()
            if step % 4 == 0:
                pool.refresh(args.refresh)
            if step % 100 == 0 or step == args.bc_steps - 1:
                with torch.no_grad():
                    ent = float(-(PB * torch.log(PB.clamp_min(1e-12))).sum(1).mean())
                    agree = float((actor(x, kid).masked_fill(~legal, -1e9).argmax(1) == PB.argmax(1)).float().mean())
                log({"phase": "bc", "step": step, "ce": round(tot, 4), "kl": round(tot - ent, 4),
                     "agree": round(agree, 4), "n": int(H.shape[0])})
        r = evaluate(0, "bc")
        log(r)
        export(actor, os.path.join(out, "bc.npz"), keys, {**meta, "step": 0})
        torch.save({"actor": actor.state_dict()}, os.path.join(out, "bc.pt"))

    # ------------------------------------------------------------- phase 2: RL
    magnet = copy.deepcopy(actor).eval()
    for p in magnet.parameters():
        p.requires_grad_(False)
    init_actor[0] = copy.deepcopy(magnet)
    aopt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    copt = torch.optim.Adam(critic.parameters(), lr=args.clr)
    best = -1e9
    sign_of = lambda r: (1.0 - 2.0 * r.float())

    def q_values(D, h, n, r, ctrl, crit):
        Q = torch.zeros(h.shape[0], 4, device=dev)
        sg = sign_of(r)
        for a in range(4):
            c = eng.child_a[n, a]
            ok = c >= 0
            cc = c.clamp_min(0)
            ct = eng.node_type[cc]
            term = ok & (ct > 0)
            dec = ok & (ct == 0)
            if bool(term.any()):
                Q[term, a] = eng.payoff0(cc[term], D["res"][h[term]]) / SCALE * sg[term]
            if bool(dec.any()):
                xc, kc = eng.critic_x(D, h[dec], cc[dec], ctrl)
                Q[dec, a] = crit(xc, kc).squeeze(1) * sg[dec]
        return Q

    def update(D, ctrl, pay0, recs, net, crit, aopt_, copt_, mag, alpha, train_actor):
        """One critic step on every visited node, then (optionally) one all-actions
        actor step on the LEARNER-controlled decisions."""
        H, N, R, C, A, PB = cat_records(recs)
        xc, kc = eng.critic_x(D, H, N, ctrl)
        tgt = pay0[H] / SCALE
        crit.train()
        closs = 0.0
        for mbi in torch.randperm(H.shape[0], device=dev, generator=eng.gen).chunk(args.mb):
            v = crit(xc[mbi], kc[mbi]).squeeze(1)
            loss = F.smooth_l1_loss(v, tgt[mbi], beta=1.0)
            copt_.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(crit.parameters(), 1.0); copt_.step()
            closs += float(loss.detach()) / args.mb
        st = {"closs": round(closs, 4), "n": int(H.shape[0])}
        if not train_actor:
            return st
        m = C == LEARNER
        Hl, Nl, Rl, PBl = H[m], N[m], R[m], PB[m]
        crit.eval()
        with torch.no_grad():
            Q = q_values(D, Hl, Nl, Rl, ctrl, crit)
            Adv = Q - (PBl * Q).sum(1, keepdim=True)
            Adv = Adv * eng.legal[Nl].float()
            xa, ka = eng.actor_x(D, Hl, Rl, Nl)
            legal = eng.legal[Nl]
            lmag = F.log_softmax(masked(mag(xa, ka), legal), 1) if alpha > 0 else None
        net.train()
        pgs = kls = ents = 0.0
        for mbi in torch.randperm(Hl.shape[0], device=dev, generator=eng.gen).chunk(args.mb):
            logp = F.log_softmax(masked(net(xa[mbi], ka[mbi]), legal[mbi]), 1)
            p = logp.exp()
            pg = -(p * Adv[mbi]).sum(1).mean()
            kl = (p * (logp - lmag[mbi])).sum(1).mean() if lmag is not None else torch.zeros((), device=dev)
            ent = -(p * logp).sum(1).mean()
            loss = pg + alpha * kl - args.beta * ent
            aopt_.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); aopt_.step()
            pgs += float(pg.detach()) / args.mb; kls += float(kl.detach()) / args.mb
            ents += float(ent.detach()) / args.mb
        st.update({"pg": round(pgs, 5), "kl_mag": round(kls, 5), "ent": round(ents, 4),
                   "adv_abs": round(float(Adv.abs().sum(1).mean()), 4)})
        return st

    # co-trained exploiter (AlphaStar-style main exploiter): best-responds to the current
    # main policy every step; the main policy meets it as the 'frozen' controller
    expl = None
    if args.exploiter > 0:
        expl = MLP(NP.BASE_DIM, eng.n_keys, args.emb, args.hid, args.layers, 4).to(dev)
        load_any(expl, args.exploiter_init, dev)
        ecrit = MLP(Engine.CRITIC_DIM, eng.n_keys, args.emb, 512, 3, 1).to(dev)
        eaopt = torch.optim.Adam(expl.parameters(), lr=args.exploiter_lr)
        ecopt = torch.optim.Adam(ecrit.parameters(), lr=args.clr)
        fixed_list.append(expl)
        fixed_opp = expl if fixed_opp is None else (fixed_opp + [expl] if isinstance(fixed_opp, list)
                                                    else [fixed_opp, expl])

    for step in range(1, args.steps + 1):
        if time.time() - T0 > args.hours * 3600:
            break
        D = pool.sample(args.batch)
        ctrl = make_ctrl(args.batch, mix, eng)
        actor.eval()
        if expl is not None:
            expl.eval()
        if isinstance(fixed_opp, list):
            D["fz"] = torch.randint(0, len(fixed_opp), (args.batch,), device=dev, generator=eng.gen)
            fzf = fixed_opp
        else:
            fz = fixed_opp if fixed_opp is not None else magnet
            fzf = lambda x, k: fz(x, k)
        _, pay0, recs = eng.rollout(D, ctrl, learner=lambda x, k: actor(x, k), frozen=fzf)
        stats = {"phase": "rl", "step": step}
        stats.update(update(D, ctrl, pay0, recs, actor, critic, aopt, copt, magnet, args.alpha, step > args.warm))
        if step > args.warm and args.magnet_every > 0 and (step - args.warm) % args.magnet_every == 0:
            magnet.load_state_dict(actor.state_dict())
            stats["magnet_reset"] = True
        if expl is not None:
            De = pool.sample(args.exploiter)
            ce = make_ctrl(args.exploiter, [(FROZEN, 1.0)], eng)       # exploiter = LEARNER, main = FROZEN
            _, pe, re_ = eng.rollout(De, ce, learner=lambda x, k: expl(x, k), frozen=lambda x, k: actor(x, k))
            since = (step - 1) % args.exploiter_reset + 1 if args.exploiter_reset > 0 else step
            es = update(De, ce, pe, re_, expl, ecrit, eaopt, ecopt, None, 0.0, since > args.exploiter_warm)
            stats["expl"] = {k: es[k] for k in ("closs", "pg") if k in es}
            if args.exploiter_reset > 0 and step % args.exploiter_reset == 0:
                snap = copy.deepcopy(expl).eval()
                for p_ in snap.parameters():
                    p_.requires_grad_(False)
                fixed_list.insert(len(fixed_list) - 1, snap)        # the live exploiter stays last
                fixed_opp = list(fixed_list)
                torch.save({"actor": snap.state_dict()}, os.path.join(out, f"exploiter_{step}.pt"))
                load_any(expl, args.exploiter_init, dev)
                eaopt = torch.optim.Adam(expl.parameters(), lr=args.exploiter_lr)
                stats["exploiter_reset"] = len(fixed_list)
        if step % 4 == 0:
            pool.refresh(args.refresh)
        if step % 50 == 0:
            torch.cuda.empty_cache()
            stats["sp_value0"] = round(float(pay0[(ctrl[:, 0] == LEARNER) & (ctrl[:, 1] == LEARNER)].mean()), 3)
            log(stats)
        if step % args.eval_every == 0:
            r = evaluate(step)
            log(r)
            export(actor, os.path.join(out, f"ckpt_{step}.npz"), keys, {**meta, "step": step})
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, os.path.join(out, "last.pt"))
            torch.save({"actor": actor.state_dict()}, os.path.join(out, f"ckpt_{step}.pt"))
            if r["bp"][0] > best:
                best = r["bp"][0]
                export(actor, os.path.join(out, "best_bp.npz"), keys, {**meta, "step": step})
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, os.path.join(out, "last.pt"))
    log({"phase": "done", "best_bp": best})


if __name__ == "__main__":
    main()
