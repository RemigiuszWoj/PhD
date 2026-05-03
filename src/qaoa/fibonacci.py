"""Fibonacci neighborhood QUBO → Ising mapping for QAOA.

QUBO formulation (tridiagonal, K = n-1 variables):
    H = Σ δ_i x_i + P Σ x_i x_{i+1}

where:
    x_i ∈ {0,1}  — whether to perform adjacent swap at position i
    δ_i          — Cmax change from swap i (Head+Tail)
    P            — penalty for overlapping swaps

Ising mapping via x_i = (1 - Z_i) / 2:
    h_i  = -δ_i/2 - P/4 * (number of penalty terms involving qubit i)
    J_ij = P/4            (only for |i-j| = 1)
    constant offset ignored (doesn't affect optimization)

The tridiagonal structure maps naturally to a nearest-neighbor
qubit chain on the Willow grid — minimal routing overhead.
"""

from typing import List, Tuple, Dict
import numpy as np

from src.neighborhoods.common import compute_deltas
from src.qaoa.circuit import circuit_stats


def fibonacci_qubo_to_ising(
    pi: List[int],
    processing_times: List[List[int]],
) -> Tuple[List[float], List[float], Dict[Tuple[int, int], float]]:
    """Compute Fibonacci QUBO and Ising coefficients for current permutation.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix

    Returns:
        (deltas, h, J):
            deltas: raw swap cost deltas δ_i
            h: Ising linear coefficients h_i
            J: Ising quadratic coefficients {(i,j): J_ij}
    """
    n = len(pi)
    if n < 2:
        return [], [], {}

    # Compute swap deltas via Head+Tail — O(m·n)
    deltas = compute_deltas(pi, processing_times)
    K = len(deltas)  # = n - 1

    # Penalty: dominates any subset of swap costs
    P = sum(abs(d) for d in deltas) + 1.0

    # QUBO → Ising mapping via x_i = (1 - Z_i) / 2
    # x_i x_{i+1} = (1 - Z_i)(1 - Z_{i+1}) / 4
    #             = (1 - Z_i - Z_{i+1} + Z_i Z_{i+1}) / 4
    #
    # h_i contributions:
    #   from cost term:    -δ_i / 2
    #   from each penalty term involving qubit i:  -P / 4

    h = [0.0] * K
    J: Dict[Tuple[int, int], float] = {}

    for i in range(K):
        h[i] -= deltas[i] / 2.0

    for i in range(K - 1):
        h[i]     -= P / 4.0
        h[i + 1] -= P / 4.0
        J[(i, i + 1)] = J.get((i, i + 1), 0.0) + P / 4.0

    return deltas, h, J


def fibonacci_qubo_energy(
    bitstring: List[int],
    deltas: List[float],
    P: float,
) -> float:
    """Evaluate QUBO energy H(x) for a given bitstring.

    Args:
        bitstring: Binary solution x ∈ {0,1}^K
        deltas: Swap cost deltas
        P: Penalty weight

    Returns:
        QUBO energy H(x)
    """
    energy = sum(deltas[i] * bitstring[i] for i in range(len(deltas)))
    for i in range(len(bitstring) - 1):
        energy += P * bitstring[i] * bitstring[i + 1]
    return energy


def fibonacci_circuit_info(n: int) -> Dict:
    """Return circuit resource summary for Fibonacci QAOA at given n and p=1,2,3.

    Args:
        n: Number of jobs

    Returns:
        Dict {p: circuit_stats_dict}
    """
    K = n - 1
    h_dummy = [1.0] * K
    J_dummy = {(i, i + 1): 1.0 for i in range(K - 1)}

    return {p: circuit_stats(h_dummy, J_dummy, p) for p in [1, 2, 3]}
