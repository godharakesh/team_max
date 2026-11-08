"""Pool offline/arena.py output lines (one per seed) into mean +- 95% CI per pairing.
usage: python offline/drl/summarize_arena.py <arena output files...>"""
import math
import re
import sys
from collections import defaultdict

pat = re.compile(r"^\s*(\S+) vs (\S+)\s+seed=(\S+)\s+([+-][\d.]+) bb/deal\s+CI\[([+-][\d.]+),([+-][\d.]+)\]\s+(\w+)"
                 r"\s+think/half: ours ([\d.]+)s opp ([\d.]+)s")
rows = defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path):
        m = pat.match(line)
        if m:
            a, b, seed, mean, lo, hi, verdict, ta, tb = m.groups()
            rows[(a, b)].append((float(mean), (float(hi) - float(lo)) / (2 * 1.96), verdict, float(ta), float(tb)))
print(f"{'variant':<22} {'opponent':<10} {'seeds':>5} {'bb/deal':>8} {'95% CI':>18}  W/C/L   think/half ours (max)")
for (a, b), r in sorted(rows.items()):
    n = len(r)
    mean = sum(x[0] for x in r) / n
    se = math.sqrt(sum(x[1] ** 2 for x in r)) / n
    w = sum(x[2] == "WIN" for x in r); c = sum(x[2] == "CHOP" for x in r); l = sum(x[2] == "LOSS" for x in r)
    print(f"{a:<22} {b:<10} {n:>5} {mean:>+8.3f} [{mean - 1.96 * se:+.3f},{mean + 1.96 * se:+.3f}]  {w}/{c}/{l}   "
          f"{sum(x[3] for x in r) / n:.2f}s ({max(x[3] for x in r):.2f}s)")
