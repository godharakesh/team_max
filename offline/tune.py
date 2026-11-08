"""Automated tuning of strategy.PARAMS: a (1+lambda) evolution strategy with
common random numbers (same cards and stable bot seeds for every candidate).

Objective = expected tournament points over the practice pool (P(win) +
0.5*P(chop) per opponent, from the pooled mean and the per-match standard
error of a 1000-deal match), tie-broken by the worst pooled CI lower bound.
The pool includes strong opponents: the old blueprint (bp1) and a frozen
copy of the shipped release (rel).
Winner's-curse guard: the generation's best two candidates are re-scored on
fresh seeds and accepted only if they beat the incumbent on both seed sets.
Final gate (holdout seeds + head-to-head vs the incumbent release) is run
separately before params_tuned.json is shipped.

usage: python offline/tune.py <generations> <lambda> <deals> [bpv]
"""
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arena  # noqa: E402
import strategy as S  # noqa: E402

POOL = ["random", "half", "pot", "station", "nit", "tag", "maniac", "barrel", "bp1", "rel"]
SEEDS_A = ["t11", "t12", "t13", "t14"]
SEEDS_B = ["t21", "t22", "t23", "t24"]

# name: (low, high, mutation sigma)
SPACE = {
    "gen": (0.9, 1.6, 0.10), "cap_pre": (0.5, 0.9, 0.05), "cap_post": (0.5, 0.9, 0.05),
    "sd_prior": (2.0, 30.0, 4.0),
    "pf_raise_big": (0.56, 0.70, 0.02), "pf_raise": (0.50, 0.62, 0.02), "pf_limp": (0.30, 0.48, 0.02),
    "pf_call_margin": (-0.02, 0.12, 0.02), "pf_pos_bonus": (0.0, 0.10, 0.02), "bb_raise": (0.50, 0.66, 0.02),
    "vp_f": (0.60, 0.80, 0.02), "vp_t": (0.62, 0.82, 0.02), "vp_r": (0.65, 0.85, 0.02),
    "vh_f": (0.50, 0.70, 0.02), "vh_t": (0.52, 0.72, 0.02), "vh_r": (0.55, 0.76, 0.02),
    "bl_base": (0.15, 0.45, 0.04), "bl_slope": (0.4, 2.0, 0.2), "semi_p": (0.0, 0.30, 0.04),
    "semi_eq": (0.30, 0.55, 0.03), "rr_margin": (0.0, 0.12, 0.02), "call_bonus": (-0.02, 0.08, 0.015),
    "ramp": (20.0, 150.0, 20.0), "dev0": (0.3, 3.5, 0.3), "dev1": (1.5, 9.0, 0.6),
}


def mutate(p, rng, k=3):
    q = dict(p)
    for name in rng.sample(sorted(SPACE), k):
        lo, hi, sg = SPACE[name]
        q[name] = min(hi, max(lo, q[name] + rng.gauss(0.0, sg)))
    if q["dev1"] < q["dev0"] + 0.5:
        q["dev1"] = q["dev0"] + 0.5
    return q


def p_win(mean, se1):
    """P(a single 1000-deal match ends with CI lower bound > 0), given the
    true edge ~ mean and per-match standard error se1: Phi(mean/se1 - 1.96)."""
    z = mean / max(se1, 1e-6) - 1.96
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def score_many(cands, seeds, deals, bpv, pool):
    """Evaluate several candidates in ONE parallel batch (keeps every core
    busy). Objective per candidate = expected tournament points over the pool:
    P(win) + 0.5*P(chop) per opponent from the pooled mean and the per-match
    standard error of a 1000-deal match; tie-break = worst pooled CI lower bound."""
    jobs = [({"name": f"c{i}", "mode": "mix", "bpv": bpv, "params": c}, o, deals, s)
            for i, c in enumerate(cands) for o in POOL for s in seeds]
    res = {}
    for r in pool.imap_unordered(arena.match, jobs, chunksize=1):
        res.setdefault(r[0], {}).setdefault(r[1], []).append(r)
    out = []
    for i in range(len(cands)):
        per, points, worst = {}, 0.0, 1e9
        for o, rs in res[f"c{i}"].items():
            k = len(rs)
            mean = sum(r[3] for r in rs) / k
            se1 = sum((r[5] - r[4]) / (2 * 1.96) for r in rs) / k      # per-match SE
            pw, pl = p_win(mean, se1), p_win(-mean, se1)
            points += pw + 0.5 * (1.0 - pw - pl)
            worst = min(worst, mean - 1.96 * se1 / math.sqrt(k))       # pooled CI lower bound
            per[o] = round(mean, 2)
        out.append((points + 0.01 * worst, per))
    return out


if __name__ == "__main__":
    gens, lam, deals = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    bpv = int(sys.argv[4]) if len(sys.argv) > 4 else S.DEFAULT_BPV
    rng = random.Random(2026)
    best = dict(S.PARAMS)
    log = []
    with Pool(os.cpu_count()) as pool:
        t0 = time.time()
        (best_a, per), = score_many([best], SEEDS_A, deals, bpv, pool)
        (best_b, _), = score_many([best], SEEDS_B, deals, bpv, pool)
        print(f"[gen 0] incumbent points A={best_a:.3f} B={best_b:.3f} {per} ({time.time()-t0:.0f}s)", flush=True)
        for g in range(1, gens + 1):
            cands = [mutate(best, rng) for _ in range(lam)]
            scored = [(sa, c, pa) for c, (sa, pa) in zip(cands, score_many(cands, SEEDS_A, deals, bpv, pool))]
            scored.sort(key=lambda x: -x[0])
            top = [x for x in scored[:2] if x[0] > best_a]
            accepted = False
            if top:
                rescored = score_many([c for _, c, _ in top], SEEDS_B, deals, bpv, pool)
                for (sa, c, pa), (sb, _) in zip(top, rescored):
                    if sb > best_b:
                        changed = {k: round(v, 3) for k, v in c.items() if abs(v - best[k]) > 1e-9}
                        best, best_a, best_b = c, sa, sb
                        accepted = True
                        print(f"[gen {g}] ACCEPT A={sa:.3f} B={sb:.3f} changed={changed} {pa} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                        break
            if not accepted:
                print(f"[gen {g}] no improvement (best A={scored[0][0]:.3f} vs {best_a:.3f}) "
                      f"({time.time()-t0:.0f}s)", flush=True)
            log.append({"gen": g, "A": best_a, "B": best_b})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_candidate.json")
    with open(out, "w") as f:
        json.dump({k: round(v, 4) for k, v in best.items()}, f, indent=1)
    print("wrote", out, "objective A/B", round(best_a, 3), round(best_b, 3))
