import json, sys, os
base = os.path.expanduser("~/poker-arena/gpu/drl/runs")
last = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for r in sys.argv[2:] or sorted(os.listdir(base)):
    p = os.path.join(base, r, "log.jsonl")
    if not os.path.isfile(p):
        continue
    ev = [json.loads(l) for l in open(p) if '"eval"' in l]
    rl = [json.loads(l) for l in open(p) if '"phase": "rl"' in l]
    print(f"== {r}  (last rl step {rl[-1]['step'] if rl else '-'}, t={rl[-1]['t'] if rl else '-'}s)")
    for d in ev[-last:]:
        ks = [k for k in ("station", "nit", "maniac", "tag", "barrel", "init") if k in d]
        ks += sorted([k for k in d if k.startswith("frozen")], key=lambda k: int(k[6:] or 0))
        o = " ".join(f"{(k[:5] if not k.startswith('frozen') else 'f' + k[6:])} {d[k][0]:+.2f}" for k in ks)
        print(f"  step {d['step']:>6}  bp {d['bp'][0]:+.3f}±{d['bp'][1]:.3f}  {o}")
