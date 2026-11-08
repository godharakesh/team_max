"""Export cumulative GPU learning curves (duplicate eval vs bp and the simple bots) of every
training line on beast to runs/curves.csv. usage: python offline/drl/export_curves.py"""
import json, os
base = os.path.expanduser("~/poker-arena/gpu/drl/runs")
# cumulative RL steps per training line
lines = {"bc_rl": [("bcrl", 0), ("bcrl_b", 3500), ("psro1", 6500), ("psro2", 8500), ("rob1", 9000)],
         "scratch": [("scratch", 0), ("scratch_b", 3500)]}
rows = ["line,run,cum_step,bp,bp_se,station,nit,maniac,tag,barrel"]
for line, parts in lines.items():
    for run, off in parts:
        p = os.path.join(base, run, "log.jsonl")
        if not os.path.isfile(p):
            continue
        for l in open(p):
            d = json.loads(l)
            if d.get("phase") != "eval":
                continue
            vals = ",".join(str(d[k][0]) for k in ("station", "nit", "maniac", "tag", "barrel"))
            rows.append(f"{line},{run},{d['step'] + off},{d['bp'][0]},{d['bp'][1]},{vals}")
open(os.path.join(base, "curves.csv"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
