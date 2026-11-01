"""Insert neighborhood, exact critical-path filter, k-bonacci "_cp" ablation modes."""

import random

import pytest

from src.algorithms.base import get_neighbor
from src.neighborhoods.accelerator import compute_critical_path_boundaries
from src.neighborhoods.classical.insert import insert_neighborhood
from src.neighborhoods.classical.kbonacci import (
    compute_candidates,
    kbonacci_boundary_mode,
    parse_kbonacci_mode,
    solve_dp,
)
from src.neighborhoods.common import compute_endpoint_swap_delta, compute_head, compute_tail
from src.permutation_procesing import c_max


def _inst(n, m, seed):
    rng = random.Random(seed)
    pt = [[rng.randint(1, 99) for _ in range(n)] for _ in range(m)]
    pi = list(range(n))
    rng.shuffle(pi)
    return pi, pt


@pytest.mark.parametrize("seed", range(6))
def test_insert_matches_brute_force(seed):
    pi, pt = _inst(9, 4, seed)
    best = None
    for i in range(9):
        seq = pi[:i] + pi[i + 1 :]
        for t in range(9):
            if t == i:
                continue
            c = c_max(seq[:t] + [pi[i]] + seq[t:], pt)
            best = c if best is None else min(best, c)
    new_pi, new_c, _ = insert_neighborhood(pi, pt)
    assert new_c == best == c_max(new_pi, pt)
    assert sorted(new_pi) == sorted(pi)


@pytest.mark.parametrize("seed", range(15))
def test_critical_path_filter_is_lossless(seed):
    pi, pt = _inst(14, 5, seed)
    H, T = compute_head(pi, pt), compute_tail(pi, pt)
    base = H[-1][-1]
    U = compute_critical_path_boundaries(H, pt, pi)
    assert len(U) <= len(pt) - 1
    for i in range(13):
        for j in range(i + 1, 14):
            if compute_endpoint_swap_delta(pi, i, j, H, T, pt, base) < 0:
                assert any(i <= u <= j for u in U)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", [3, 8])
def test_cp_filter_keeps_dp_optimum(seed, k):
    pi, pt = _inst(16, 5, seed)
    _, full = compute_candidates(pi, pt, k, use_block_accelerator=False)
    _, cp = compute_candidates(pi, pt, k, boundary_mode="critical_path")
    assert solve_dp(16, cp)[0] == solve_dp(16, full)[0]
    assert len(cp) <= len(full)


def test_modes():
    assert parse_kbonacci_mode("kbonacci_k3_cp", 20) == 3
    assert kbonacci_boundary_mode("kbonacci_k3_cp") == "critical_path"
    assert kbonacci_boundary_mode("kbonacci_k3") == "last_machine"
    pi, pt = _inst(20, 5, 0)
    for mode in ("insert", "kbonacci_k3_cp", "kbonacci_k8"):
        new_pi, new_c, _, _ = get_neighbor(mode, pi, pt, 20)
        assert sorted(new_pi) == sorted(pi) and new_c == c_max(new_pi, pt)


def test_insert_reverse_move_has_same_tabu_attribute():
    pi, pt = _inst(12, 4, 3)
    new_pi, _, attr = insert_neighborhood(pi, pt)
    job, a, b = attr
    i, t = pi.index(job), new_pi.index(job)
    assert {i, t} == {a, b}
    # undo = take the job from t back to i: same job, same pair of positions
    seq = new_pi[:t] + new_pi[t + 1 :]
    assert seq[:i] + [job] + seq[i:] == pi
