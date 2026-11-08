"""GPU duplicate evaluation of a FIXED controller (default: the CFR blueprint) vs the
simple bots, as a reference row for the learning curves.
usage: python offline/drl/ref_eval.py [deals]"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_env import Engine, BP, STATION, NIT, MANIAC, TAG, BARREL, CTRL_NAMES  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
eng = Engine("cuda", mc_samples=256, seed=12345)
D = eng.gen_deals(n)
for o in (BP, STATION, NIT, MANIAC, TAG, BARREL):
    c1 = torch.tensor([[BP, o]], device=eng.dev).repeat(n, 1)
    c2 = torch.tensor([[o, BP]], device=eng.dev).repeat(n, 1)
    _, p1, _ = eng.rollout(D, c1)
    _, p2, _ = eng.rollout(D, c2)
    x = (p1 - p2) / 2
    print(f"bp vs {CTRL_NAMES[o]:<8} {float(x.mean()):+.3f} +- {float(x.std()) / np.sqrt(n):.3f} bb/deal")
