import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root
RES = HERE / "measurements"
import random, sys, time
from pilot_anatomy import lcg_instance, all_deltas
for (n, m) in [(20, 5), (20, 20), (50, 10), (50, 20), (100, 10), (100, 20), (200, 10)]:
    p = lcg_instance(12345, n, m); pi = list(range(n)); random.Random(1).shuffle(pi)
    t0 = time.perf_counter(); all_deltas(pi, p); dt = time.perf_counter() - t0
    print(f"{n}x{m}: all-pairs deltas {dt*1e3:.0f} ms  (m*n^3/6 = {m*n**3/6:.2e} cell updates)")
