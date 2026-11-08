"""Combine several exported checkpoints into one population model file (keys m0_, m1_, ...).
The runtime plays the uniform mixture: one member drawn per hand.
usage: python offline/drl/make_mixture.py <out.npz> <ckpt1.npz> <ckpt2.npz> ... [bp=<share>] [--fp16]
bp=<share>: that share of hands is played by the CFR blueprint (also a population member)."""
import sys

import numpy as np

out, paths = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith(("bp=", "--"))]
bpw = [float(a[3:]) for a in sys.argv[2:] if a.startswith("bp=")]
f16 = "--fp16" in sys.argv          # store weights as float16 (half the size; runtime upcasts)
d = {}
keys = None
for i, p in enumerate(paths):
    z = np.load(p, allow_pickle=False)
    if keys is None:
        keys = z["keys"]
    assert (z["keys"] == keys).all()
    for f in z.files:
        if f == "emb" or f[0] in "Wb" and f[1:].isdigit():
            d[f"m{i}_{f}"] = z[f].astype(np.float16) if f16 else z[f]
d["keys"] = keys
d["meta_members"] = np.array(";".join(paths))
if bpw:
    d["meta_bp_weight"] = np.array(bpw[0])
for f in ("meta_hid", "meta_layers", "meta_emb", "meta_mc"):
    d[f] = np.load(paths[0], allow_pickle=False)[f]
np.savez(out, **d)
print("members", len(paths), "->", out)
