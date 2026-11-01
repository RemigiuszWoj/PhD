"""Safety and strength of the block-boundary filter (see articles/kbonacci/PLAN.md, section 14).

Compares the repo's boundary test (`compute_block_boundaries`, last two machines only)
with the machine-change columns of a true critical path (backtracking over Head).

  1. safety   : how many improving endpoint swaps (delta < 0) each filter would drop
  2. strength : share of candidate pairs (distance <= k-1) each filter keeps

Usage:
    python scripts/check_block_filter.py [--instances 40]
"""

import argparse
import os
import random
import sys
from typing import List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neighborhoods.accelerator import compute_block_boundaries  # noqa: E402
from src.neighborhoods.classical.kbonacci import kbonacci_neighborhood  # noqa: E402
from src.neighborhoods.common import (  # noqa: E402
    compute_endpoint_swap_delta,
    compute_head,
    compute_tail,
)


def true_boundaries(H: List[List[int]], pt: List[List[int]], pi: List[int]) -> Set[int]:
    """Columns where one critical path changes machine (|U| <= m - 1)."""
    m, n = len(pt), len(pi)
    r, t, U = m - 1, n - 1, set()
    while r > 0 or t > 0:
        if r > 0 and H[r][t] == H[r - 1][t] + pt[r][pi[t]]:
            U.add(t)
            r -= 1
        else:
            t -= 1
    return U


def tribonacci_local_optimum(pi: List[int], pt: List[List[int]]) -> List[int]:
    m, n = len(pt), len(pi)
    for _ in range(500):
        new_pi, new_c, _ = kbonacci_neighborhood(
            pi, pt, 3, use_block_accelerator=False, force_move_if_none=False
        )
        if new_c >= compute_head(pi, pt)[m - 1][n - 1]:
            break
        pi = new_pi
    return pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=40)
    a = ap.parse_args()

    rng = random.Random(7)
    print("1) safety: improving swaps dropped by each filter")
    for n, m in [(20, 5), (20, 20), (50, 5), (50, 10), (50, 20)]:
        for state in ("random", "local-opt(k=3)"):
            tot = lost_repo = lost_true = 0
            for _ in range(a.instances):
                pt = [[rng.randint(1, 99) for _ in range(n)] for _ in range(m)]
                pi = list(range(n))
                rng.shuffle(pi)
                if state != "random":
                    pi = tribonacci_local_optimum(pi, pt)
                H, T = compute_head(pi, pt), compute_tail(pi, pt)
                base = H[m - 1][n - 1]
                u_repo = set(compute_block_boundaries(H, pt, pi))
                u_true = true_boundaries(H, pt, pi)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if compute_endpoint_swap_delta(pi, i, j, H, T, pt, base) < 0:
                            tot += 1
                            lost_repo += not any(i <= u <= j for u in u_repo)
                            lost_true += not any(i <= u <= j for u in u_true)
            print(
                f"  {n}x{m:<3} {state:<15} improving={tot:6d}  "
                f"dropped repo={lost_repo:5d} ({100 * lost_repo / max(1, tot):4.1f}%)  "
                f"dropped true-CP={lost_true}"
            )

    rng = random.Random(11)
    print("2) strength: candidate pairs kept (tribonacci local optima)")
    for n, m in [(20, 5), (50, 5), (50, 20), (100, 10)]:
        for k in (4, n):
            kept_r = kept_t = tot = nb_r = nb_t = 0
            samples = max(1, a.instances // 2)
            for _ in range(samples):
                pt = [[rng.randint(1, 99) for _ in range(n)] for _ in range(m)]
                pi = list(range(n))
                rng.shuffle(pi)
                pi = tribonacci_local_optimum(pi, pt)
                H = compute_head(pi, pt)
                u_repo = set(compute_block_boundaries(H, pt, pi))
                u_true = true_boundaries(H, pt, pi)
                nb_r += len(u_repo)
                nb_t += len(u_true)
                for i in range(n - 1):
                    for j in range(i + 1, min(n - 1, i + k - 1) + 1):
                        tot += 1
                        kept_r += any(i <= u <= j for u in u_repo)
                        kept_t += any(i <= u <= j for u in u_true)
            print(
                f"  {n}x{m:<3} k={k:<4} kept repo={100 * kept_r / tot:5.1f}%  "
                f"kept true-CP={100 * kept_t / tot:5.1f}%  "
                f"#boundaries repo={nb_r / samples:5.1f} true={nb_t / samples:5.1f}"
            )


if __name__ == "__main__":
    main()
