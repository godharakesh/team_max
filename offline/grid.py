"""Targeted grid over a few parameters (CRN: same cards + stable seeds for
every config). Prints mean bb/deal per opponent and expected points.
usage: python offline/grid.py '<json list of param overrides>' opp1,opp2 deals seed...
"""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena  # noqa: E402
from tune import p_win  # noqa: E402

if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1]); opps = sys.argv[2].split(","); deals = int(sys.argv[3]); seeds = sys.argv[4:]
    bpv = int(os.environ.get("GRID_BPV", "2"))
    jobs = [({"name": f"g{i}", "mode": c.pop("_mode", "mix") if isinstance(c, dict) else "mix", "bpv": bpv,
              "params": {**arena.S.PARAMS, **c}}, o, deals, s)
            for i, c in enumerate([dict(c) for c in cfgs]) for o in opps for s in seeds]
    res = {}
    with Pool(os.cpu_count()) as p:
        for r in p.imap_unordered(arena.match, jobs, chunksize=1):
            res.setdefault(r[0], {}).setdefault(r[1], []).append(r)
    print("config".ljust(34) + "".join(o.rjust(9) for o in opps) + "   points")
    for i, c in enumerate(cfgs):
        row, pts = [], 0.0
        for o in opps:
            rs = res[f"g{i}"][o]; k = len(rs)
            m = sum(r[3] for r in rs) / k
            se1 = sum((r[5] - r[4]) / 3.92 for r in rs) / k
            pw, pl = p_win(m, se1), p_win(-m, se1)
            pts += pw + 0.5 * (1 - pw - pl)
            row.append(f"{m:+9.2f}")
        print(json.dumps(c)[:33].ljust(34) + "".join(row) + f"   {pts:.2f}/{len(opps)}", flush=True)
