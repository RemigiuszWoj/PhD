"""Quantum Fibonahi neighborhood - select non-overlapping swaps using QUBO.

QUBO formulation (no-overlap):
    H = Σᵢ δᵢ·xᵢ + P·Σᵢ xᵢ·xᵢ₊₁

QUBO matrix:
    Q[i,i] = δᵢ            (cost only)
    Q[i,i+1] = P           (penalty for overlap)

Complexity: n-1 variables, O(n) coefficients (chain graph)
Number of valid swap sets = F_{n+1} (Fibonacci sequence)
"""

from typing import Dict, List, Tuple

from src.neighborhoods.common import (
    apply_swaps,
    compute_deltas,
    solve_qubo,
    validate_no_overlap,
)
from src.permutation_procesing import c_max


def quantum_fibonahi_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 100,
) -> Tuple[List[int], int, List[int]]:
    """Quantum fibonahi neighborhood - selects non-overlapping swaps via QUBO.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of samples for the solver

    Returns:
        (new_pi, new_cmax, swaps): New permutation, Cmax, list of swap positions
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    deltas = compute_deltas(pi, processing_times)
    num_vars = len(deltas)

    # Build QUBO no-overlap: non-overlapping swaps
    penalty = sum(abs(d) for d in deltas) + 1
    Q: Dict[Tuple[str, str], float] = {}
    for i in range(num_vars):
        Q[(f"x{i}", f"x{i}")] = deltas[i]
    for i in range(num_vars - 1):
        Q[(f"x{i}", f"x{i + 1}")] = penalty

    # Solve QUBO
    solution = solve_qubo(Q, num_reads)
    selected = sorted(int(v[1:]) for v, val in solution.items() if val == 1)
    valid_swaps = validate_no_overlap(selected)

    # Fallback: if nothing selected, pick best single swap
    if not valid_swaps:
        best_idx = min(range(num_vars), key=lambda i: deltas[i])
        if deltas[best_idx] < 0:
            valid_swaps = [best_idx]

    new_pi = apply_swaps(pi, valid_swaps)
    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, valid_swaps


def generate_neighbors_fibonahi_qubo(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 100,
) -> Tuple[List[int], int, List[int]]:
    """Alias for quantum_fibonahi_neighborhood (backward compatibility)."""
    return quantum_fibonahi_neighborhood(pi, processing_times, num_reads)
