"""Action frequencies (self-play) of a checkpoint vs the CFR blueprint, by street and by
whether the actor faces a bet. usage: python offline/drl/style.py <ckpt.npz> [hands]"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_env import Engine, LEARNER, BP, cat_records  # noqa: E402
from train import MLP, load_any  # noqa: E402
import nn_policy as NP  # noqa: E402

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
eng = Engine("cuda", mc_samples=256, seed=99)
torch.cuda.set_per_process_memory_fraction(0.1)
D = eng.gen_deals(n)
pol = NP.NNPolicy(path)
actor = MLP(NP.BASE_DIM, eng.n_keys, int(pol.meta["meta_emb"]), int(pol.meta["meta_hid"]),
            int(pol.meta["meta_layers"]), 4).to(eng.dev).eval()
load_any(actor, path, eng.dev)
for name, c in (("nn self-play", LEARNER), ("bp self-play", BP)):
    ctrl = torch.full((n, 2), c, dtype=torch.int64, device=eng.dev)
    node, pay0, recs = eng.rollout(D, ctrl, learner=lambda x, k: actor(x, k))
    H, N, R, C, A, PB = cat_records(recs)
    st = eng.node_street[N]
    facing = eng.to_call[N] > 0
    sd = float((eng.node_type[node] == 2).float().mean())
    pot = float((eng.term_c0[node] + eng.term_c1[node]).mean())
    print(f"{name}: showdown rate {sd:.3f}, mean final pot {pot:.1f} chips, decisions/hand {H.shape[0] / n:.2f}")
    for s in range(4):
        for f in (False, True):
            m = (st == s) & (facing == f)
            if int(m.sum()) == 0:
                continue
            fr = torch.bincount(A[m], minlength=4).float() / float(m.sum())
            lab = "facing bet" if f else "no bet   "
            print(f"   street {s} {lab}  F {fr[0]:.3f}  C {fr[1]:.3f}  H {fr[2]:.3f}  P {fr[3]:.3f}   (n={int(m.sum())})")
