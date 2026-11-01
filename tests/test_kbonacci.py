"""Tests for the k-bonacci neighborhood family."""

import random

import pytest

from src.neighborhoods.classical.dynasearch import dynasearch_full
from src.neighborhoods.classical.fibonacci import _compute_deltas_fast, _solve_dp_topk
from src.neighborhoods.classical.kbonacci import (
    apply_endpoint_swaps,
    compute_candidates,
    count_kbonacci_moves,
    growth_constant,
    kbonacci_neighborhood,
    parse_kbonacci_mode,
    solve_dp,
)
from src.permutation_procesing import c_max


def _random_instance(n: int, m: int, seed: int):
    rng = random.Random(seed)
    pt = [[rng.randint(1, 99) for _ in range(n)] for _ in range(m)]
    pi = list(range(n))
    rng.shuffle(pi)
    return pi, pt


def _all_moves(n: int, k: int):
    """All admissible swap sets (tilings with tiles of length 1..k)."""

    def rec(p):
        if p >= n:
            yield []
            return
        for rest in rec(p + 1):  # tile of length 1
            yield rest
        for L in range(2, k + 1):
            if p + L <= n:
                for rest in rec(p + L):
                    yield [(p, p + L - 1)] + rest

    return list(rec(0))


# ---------------------------------------------------------------- counting
@pytest.mark.parametrize("n", range(1, 10))
@pytest.mark.parametrize("k", [2, 3, 4, 5, 9])
def test_count_matches_distinct_permutations(n, k):
    base = list(range(n))
    perms = {tuple(apply_endpoint_swaps(base, mv)) for mv in _all_moves(n, k)}
    assert len(perms) == count_kbonacci_moves(n, k)


def test_known_sequences():
    fib = [count_kbonacci_moves(n, 2) for n in range(10)]
    assert fib == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    trib = [count_kbonacci_moves(n, 3) for n in range(10)]
    assert trib == [1, 1, 2, 4, 7, 13, 24, 44, 81, 149]
    assert count_kbonacci_moves(10, 3) == 274  # katalog_sasiedztw.md
    for n in range(1, 12):
        assert count_kbonacci_moves(n, max(2, n)) == 2 ** (n - 1)  # Dynasearch


def test_growth_constants():
    assert growth_constant(2) == pytest.approx(1.6180339887, abs=1e-8)
    assert growth_constant(3) == pytest.approx(1.8392867552, abs=1e-8)
    assert growth_constant(4) == pytest.approx(1.9275619754, abs=1e-8)


# ---------------------------------------------------------------- deltas / DP
@pytest.mark.parametrize("seed", range(5))
def test_candidate_deltas_are_exact(seed):
    pi, pt = _random_instance(9, 4, seed)
    base, cands = compute_candidates(pi, pt, k=5, use_block_accelerator=False)
    assert base == c_max(pi, pt)
    assert len(cands) == sum(min(4, 8 - i) for i in range(8))
    for i, j, d in cands:
        assert d == c_max(apply_endpoint_swaps(pi, [(i, j)]), pt) - base


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_dp_is_optimal_for_sum_of_deltas(seed, k):
    n = 8
    pi, pt = _random_instance(n, 3, seed)
    _, cands = compute_candidates(pi, pt, k, use_block_accelerator=False)
    delta = {(i, j): d for i, j, d in cands}
    brute = min(sum(delta[s] for s in mv) for mv in _all_moves(n, k))
    value, chosen = solve_dp(n, cands)
    assert value == brute
    assert value == sum(delta[s] for s in chosen)
    # chosen intervals are disjoint and of admissible length
    flat = [p for i, j in chosen for p in range(i, j + 1)]
    assert len(flat) == len(set(flat))
    assert all(1 <= j - i <= k - 1 for i, j in chosen)


@pytest.mark.parametrize("seed", range(5))
def test_k2_matches_fibonacci_dp(seed):
    pi, pt = _random_instance(15, 5, seed)
    _, fib_cands = _compute_deltas_fast(pi, pt, use_block_accelerator=False)
    fib_best = _solve_dp_topk(fib_cands, 1)[0][0]
    _, cands = compute_candidates(pi, pt, k=2, use_block_accelerator=False)
    assert solve_dp(len(pi), cands)[0] == fib_best


@pytest.mark.parametrize("seed", range(5))
def test_kn_matches_dynasearch(seed):
    pi, pt = _random_instance(12, 4, seed)
    _, _, dyn_moves = dynasearch_full(pi, pt)
    stats: dict = {}
    kbonacci_neighborhood(pi, pt, k=len(pi), stats=stats)
    _, cands = compute_candidates(pi, pt, k=len(pi))
    delta = {(i, j): d for i, j, d in cands}
    assert stats["predicted_delta"] == sum(delta[s] for s in dyn_moves)


def test_neighborhood_returns_valid_permutation_and_stats():
    pi, pt = _random_instance(30, 5, 0)
    stats: dict = {}
    new_pi, new_c, swaps = kbonacci_neighborhood(pi, pt, k=4, stats=stats)
    assert sorted(new_pi) == sorted(pi)
    assert new_c == c_max(new_pi, pt)
    assert stats["real_delta"] == new_c - c_max(pi, pt)
    assert stats["num_swaps"] == len(swaps) >= 1
    assert all(1 <= d <= 3 for d in stats["distances"])


def test_parse_mode():
    assert parse_kbonacci_mode("tribonacci", 20) == 3
    assert parse_kbonacci_mode("tetranacci", 20) == 4
    assert parse_kbonacci_mode("kbonacci_k6", 20) == 6
    assert parse_kbonacci_mode("kbonacci_kn", 20) == 20
    assert parse_kbonacci_mode("kbonacci_k1", 20) is None
    assert parse_kbonacci_mode("fibonacci", 20) is None


def test_ils_runs_with_kbonacci_mode():
    from src.algorithms.base import get_neighbor

    pi, pt = _random_instance(20, 5, 1)
    for mode in ("tribonacci", "kbonacci_k5", "kbonacci_kn"):
        new_pi, new_c, _, _ = get_neighbor(mode, pi, pt, len(pi))
        assert sorted(new_pi) == sorted(pi) and new_c == c_max(new_pi, pt)
