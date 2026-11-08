"""Check that the exported .npz (numpy runtime, nn_policy.NNPolicy) reproduces the
torch actor's logits on real GPU-env states, and time the numpy forward pass.
usage: python offline/drl/check_export.py <ckpt.pt> <ckpt.npz>"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_env import Engine, LEARNER, BP, cat_records  # noqa: E402
from train import MLP  # noqa: E402
import nn_policy as NP  # noqa: E402

pt, npz = sys.argv[1], sys.argv[2]
eng = Engine("cuda", mc_samples=64, seed=5)
D = eng.gen_deals(2000)
ctrl = torch.full((2000, 2), BP, dtype=torch.int64, device=eng.dev)
_, _, recs = eng.rollout(D, ctrl)
H, N, R, C, A, PB = cat_records(recs)
x, kid = eng.actor_x(D, H, R, N)
pol = NP.NNPolicy(npz)
hid = int(pol.meta["meta_hid"]); nl = int(pol.meta["meta_layers"]); emb = int(pol.meta["meta_emb"])
actor = MLP(NP.BASE_DIM, eng.n_keys, emb, hid, nl, 4).to(eng.dev)
actor.load_state_dict(torch.load(pt, map_location=eng.dev)["actor"])
with torch.no_grad():
    lt = actor(x, kid).cpu().numpy()
xn, kn = x.cpu().numpy(), kid.cpu().numpy()
t = time.perf_counter()
ln = np.stack([pol.logits(xn[i], int(kn[i])) for i in range(len(kn))])
dt = (time.perf_counter() - t) / len(kn)
print(f"states {len(kn)}  max |logit diff| {np.abs(lt - ln).max():.2e}  numpy forward {dt * 1e6:.0f} us/decision")
