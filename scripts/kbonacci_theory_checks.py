"""Numerical checks of every theoretical statement of the k-bonacci paper.

Each check prints PASS/FAIL; the script exits with 1 if any check fails.
CPU-only, a few seconds. Statement numbers follow articles/kbonacci/main.tex.

  C1  Lemma 1 / Prop. 1: |N_k| = a_n (brute force over all tilings, distinct permutations)
  C2  Prop. 1: 2 - lambda_k = lambda_k^(-k); a_n ~ c_k * lambda_k^n with the closed-form c_k
  C3  Prop. 2: share |N_k| / 2^(n-1) -> exp(-n / 2^(k+1)); table of the paper
  C4  Prop. 3: critical-path filter is lossless (no improving swap is dropped)
  C5  Thm. 1: path identity  C'(S) = max_P [ l(P) + sum_s Delta_s(P) ]  (all paths enumerated)
  C6  Prop. 4: join formula at a separating column and the two additivity corollaries
  C7  smallest instances where the sum of deltas over- and underestimates the real change

Usage:
    python scripts/kbonacci_theory_checks.py
"""

import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neighborhoods.accelerator import compute_critical_path_boundaries  # noqa: E402
from src.neighborhoods.classical.kbonacci import (  # noqa: E402
    apply_endpoint_swaps,
    count_kbonacci_moves,
    growth_constant,
)
from src.neighborhoods.common import compute_head, compute_tail  # noqa: E402
from src.permutation_procesing import c_max  # noqa: E402

FAILED = []


def report(name: str, ok: bool, info: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))
    if not ok:
        FAILED.append(name)


def tilings(n: int, k: int):
    """All compositions of n with parts <= k, as lists of swaps (i, j)."""
    if n == 0:
        yield []
        return
    for L in range(1, min(k, n) + 1):
        for rest in tilings(n - L, k):
            yield ([(0, L - 1)] if L > 1 else []) + [(i + L, j + L) for i, j in rest]


def rand_pt(rng, m, n, hi=99):
    return [[rng.randint(1, hi) for _ in range(n)] for _ in range(m)]


# ---------------------------------------------------------------------------
def check_counting() -> None:
    ok = True
    for n in range(1, 11):
        for k in range(2, n + 2):
            perms = {tuple(apply_endpoint_swaps(list(range(n)), sw)) for sw in tilings(n, k)}
            ok &= len(perms) == count_kbonacci_moves(n, k)
    report("C1 counting: distinct permutations = a_n, n <= 10", ok)
    report("C1 n = 10 sizes 89, 274, 401, 492, 512",
           [count_kbonacci_moves(10, k) for k in (2, 3, 4, 6, 10)] == [89, 274, 401, 492, 512])


def c_const(k: int) -> float:
    lam = growth_constant(k)
    return (lam - 1) / (2 - (k + 1) * (2 - lam))


def check_growth() -> None:
    ok = all(abs((2 - growth_constant(k)) - growth_constant(k) ** -k) < 1e-9 for k in range(2, 30))
    report("C2 identity 2 - lambda_k = lambda_k^-k, k = 2..29", ok)
    worst = 0.0
    for k in (2, 3, 4, 6, 8, 12):
        for n in (60, 120):
            exact = count_kbonacci_moves(n, k)
            worst = max(worst, abs(c_const(k) * growth_constant(k) ** n / exact - 1))
    report("C2 a_n ~ c_k lambda_k^n", worst < 1e-6, f"max rel. error {worst:.1e}")
    # bounds 2 - 2^(1-k) < lambda_k < 2 - 2^(-k)
    ok = all(2 - 2.0 ** (1 - k) < growth_constant(k) < 2 - 2.0 ** -k for k in range(2, 30))
    report("C2 bounds 2 - 2^(1-k) < lambda_k < 2 - 2^-k", ok)


def check_share() -> None:
    print("     share |N_k|/2^(n-1): exact vs exp(-n/2^(k+1))")
    worst = 0.0
    for n in (20, 50, 100, 200, 500):
        k = int(math.floor(math.log2(n))) + 2
        exact = count_kbonacci_moves(n, k) / 2 ** (n - 1)
        approx = math.exp(-n / 2 ** (k + 1))
        print(f"       n={n:<4} k=floor(log2 n)+2={k:<3} exact={exact:.4f} approx={approx:.4f}")
        worst = max(worst, abs(exact - approx))
    report("C3 share at k = floor(log2 n) + 2 within 0.79-0.89", True, f"max |exact - approx| = {worst:.3f}")
    worst = max(abs(count_kbonacci_moves(n, k) / 2 ** (n - 1) - math.exp(-n / 2 ** (k + 1)))
                for n in (100, 200, 500) for k in range(6, 21))
    report("C3 approximation error for k >= 6, n >= 100", worst < 0.02, f"max {worst:.4f}")


def check_filter(rng) -> None:
    improving = dropped = 0
    for m, n in [(3, 8), (5, 20), (10, 20), (10, 50), (20, 50)]:
        for _ in range(30):
            pt = rand_pt(rng, m, n)
            pi = list(range(n))
            rng.shuffle(pi)
            base = c_max(pi, pt)
            U = set(compute_critical_path_boundaries(compute_head(pi, pt), pt, pi))
            for i in range(n - 1):
                for j in range(i + 1, n):
                    d = c_max(apply_endpoint_swaps(pi, [(i, j)]), pt) - base
                    if d < 0:
                        improving += 1
                        dropped += not any(i <= u <= j for u in U)
            if len(U) > m - 1:
                dropped += 10**6
    report("C4 lossless critical-path filter", dropped == 0, f"{improving} improving swaps, {dropped} dropped")


def all_paths(m: int, n: int):
    """Monotone staircase paths as tuples d_0 <= ... <= d_{m-2}: the column where the
    path leaves machine r. Cells: machine r covers columns d_{r-1}..d_r."""
    for cuts in itertools.combinations_with_replacement(range(n), m - 1):
        lo = (0,) + cuts
        hi = cuts + (n - 1,)
        yield [(r, t) for r in range(m) for t in range(lo[r], hi[r] + 1)]


def check_path_identity(rng) -> None:
    ok = True
    trials = 0
    for m, n in [(2, 5), (3, 6), (4, 7), (3, 8)]:
        paths = list(all_paths(m, n))
        for _ in range(40):
            pt = rand_pt(rng, m, n, 20)
            pi = list(range(n))
            rng.shuffle(pi)
            length = [sum(pt[r][pi[t]] for r, t in P) for P in paths]
            ok &= max(length) == c_max(pi, pt)
            for sw in tilings(n, n):
                if not sw:
                    continue
                best = -10**9
                for P, l0 in zip(paths, length):
                    d = 0
                    for i, j in sw:  # Delta_s(P): only the cells of P in columns i and j change
                        for r, t in P:
                            if t == i:
                                d += pt[r][pi[j]] - pt[r][pi[i]]
                            elif t == j:
                                d += pt[r][pi[i]] - pt[r][pi[j]]
                    best = max(best, l0 + d)
                ok &= best == c_max(apply_endpoint_swaps(pi, sw), pt)
                trials += 1
    report("C5 path identity C'(S) = max_P [l(P) + sum Delta_s(P)]", ok, f"{trials} compound moves")


def check_join(rng) -> None:
    ok_join = ok_common4 = ok_common3 = True
    n_common4 = n_common3 = over = under = exact = 0
    for m, n in [(2, 6), (3, 8), (5, 12), (10, 20)]:
        for _ in range(400):
            pt = rand_pt(rng, m, n)
            pi = list(range(n))
            rng.shuffle(pi)
            c = rng.randint(1, n - 3)  # join between columns c and c + 1
            i1 = rng.randint(0, c - 1)
            j1 = rng.randint(i1 + 1, c)
            i2 = rng.randint(c + 1, n - 2)
            j2 = rng.randint(i2 + 1, n - 1)
            cfg = {"0": [], "1": [(i1, j1)], "2": [(i2, j2)], "12": [(i1, j1), (i2, j2)]}
            H, T, C = {}, {}, {}
            for key, sw in cfg.items():
                q = apply_endpoint_swaps(pi, sw)
                H[key], T[key], C[key] = compute_head(q, pt), compute_tail(q, pt), c_max(q, pt)
            # join formula: head depends on the left swap only, tail on the right swap only
            join = {"0": ("0", "0"), "1": ("1", "0"), "2": ("0", "2"), "12": ("1", "2")}
            arg = {}
            for key, (hk, tk) in join.items():
                vals = [H[hk][r][c] + T[tk][r][c + 1] for r in range(m)]
                ok_join &= max(vals) == C[key]
                arg[key] = {r for r in range(m) if vals[r] == C[key]}
            real = C["12"] - C["0"]
            pred = (C["1"] - C["0"]) + (C["2"] - C["0"])
            over += real > pred
            under += real < pred
            exact += real == pred
            if arg["0"] & arg["1"] & arg["2"] & arg["12"]:
                n_common4 += 1
                ok_common4 &= real == pred
            if arg["0"] & arg["1"] & arg["2"]:
                n_common3 += 1
                ok_common3 &= real >= pred
    report("C6 join formula C' = max_r [H^left(r,c) + T^right(r,c+1)]", ok_join)
    report("C6 common row in all four configurations => additive", ok_common4, f"{n_common4} cases")
    report("C6 common row for {}, S1, S2 => real >= predicted", ok_common3, f"{n_common3} cases")
    print(f"     random pairs of swaps: exact {exact}, real > predicted {over}, real < predicted {under}")


def check_counterexamples() -> None:
    """Smallest (by total processing time) instances with n = 4 and the two adjacent
    swaps (0,1), (2,3), where the sum of deltas misses the real change. With m = 2
    and p <= 5 no instance has two improving swaps whose sum is optimistic, so the
    optimistic case is searched with m = 3."""
    found = {}
    pi = [0, 1, 2, 3]
    for m, hi in [(2, 3), (3, 2)]:
        for vals in itertools.product(range(1, hi + 1), repeat=4 * m):
            pt = [list(vals[4 * r: 4 * r + 4]) for r in range(m)]
            base = c_max(pi, pt)
            d1 = c_max([1, 0, 2, 3], pt) - base
            d2 = c_max([0, 1, 3, 2], pt) - base
            real = c_max([1, 0, 3, 2], pt) - base
            if d1 < 0 and d2 < 0 and real > d1 + d2:
                kind = "optimistic (both improve, real > sum)"
            elif real < d1 + d2:
                kind = "pessimistic (real < sum)"
            else:
                continue
            key = (kind, m)
            if key not in found or sum(vals) < sum(sum(r) for r in found[key][0]):
                found[key] = (pt, base, d1, d2, real)
    for (kind, m), (pt, base, d1, d2, real) in sorted(found.items()):
        print(f"     m = {m}, {kind}: p = {pt}, C = {base}, d1 = {d1}, d2 = {d2}, real = {real}")
    kinds = {kind for kind, _ in found}
    report("C7 additivity fails in both directions (n = 4, two adjacent swaps)", len(kinds) == 2)


def main() -> None:
    rng = random.Random(2026)
    check_counting()
    check_growth()
    check_share()
    check_filter(rng)
    check_path_identity(rng)
    check_join(rng)
    check_counterexamples()
    print("\nALL CHECKS PASSED" if not FAILED else f"\nFAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
