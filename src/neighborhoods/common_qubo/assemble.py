"""QUBO assembly shared by every family.

Three assemblers, one per neighbourhood shape. Each takes a candidate list
and returns the exact ``Q`` dict the current code builds, so a D-Wave run
and a QAOA run receive an identical matrix.
"""
from typing import Callable, Dict, List, Tuple

from .candidates import AdjacentCand, IntervalCand

QUBO = Dict[Tuple[str, str], float]


def assemble_pairwise_qubo(
    candidates: List[IntervalCand],
    conflict_fn: Callable[[int, int, int, int], bool],
) -> QUBO:
    """Dynasearch / Motzkin: diagonal = delta, off-diagonal = penalty on conflict.

    ``conflict_fn`` is the only difference between neighbourhoods:
    interval-overlap for Dynasearch, crossing/shared-endpoint for Motzkin.
    """
    K = len(candidates)
    Q: QUBO = {}
    if K == 0:
        return Q
    penalty = sum(abs(d) for _, _, d in candidates) + 1
    for k in range(K):
        Q[(f"x{k}", f"x{k}")] = candidates[k][2]
    for k in range(K):
        i1, j1, _ = candidates[k]
        for l in range(k + 1, K):
            i2, j2, _ = candidates[l]
            if conflict_fn(i1, j1, i2, j2):
                Q[(f"x{k}", f"x{l}")] = penalty
    return Q


def assemble_onehot_qubo(candidates: List[AdjacentCand]) -> QUBO:
    """Adjacent: one-hot QUBO, Q[i,i] = delta_i - P, Q[i,j] = 2P, P = 2 max|delta| + 1."""
    delta_vals = [d for _, d in candidates]
    K = len(delta_vals)
    Q: QUBO = {}
    if K == 0:
        return Q
    penalty = 2 * max(abs(d) for d in delta_vals) + 1
    for i in range(K):
        Q[(f"x{i}", f"x{i}")] = delta_vals[i] - penalty
    for i in range(K):
        for j in range(i + 1, K):
            Q[(f"x{i}", f"x{j}")] = 2 * penalty
    return Q


def assemble_tridiagonal_qubo(candidates: List[AdjacentCand]) -> QUBO:
    """Fibonacci: diagonal = delta, penalty only between physically adjacent swaps."""
    positions = [p for p, _ in candidates]
    delta_vals = [d for _, d in candidates]
    K = len(delta_vals)
    Q: QUBO = {}
    if K == 0:
        return Q
    penalty = sum(abs(d) for d in delta_vals) + 1
    for i in range(K):
        Q[(f"x{i}", f"x{i}")] = delta_vals[i]
    for i in range(K - 1):
        if positions[i + 1] == positions[i] + 1:
            Q[(f"x{i}", f"x{i + 1}")] = penalty
    return Q
