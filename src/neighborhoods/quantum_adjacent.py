"""Quantum Adjacent neighborhood - select one swap using QUBO.

QUBO formulation (one-hot):
    H = Σᵢ δᵢ·xᵢ + P·(Σᵢ xᵢ - 1)²

QUBO matrix:
    Q[i,i] = δᵢ - P        (cost + penalty for not selecting)
    Q[i,j] = 2P            (penalty for selecting more than 1)

Complexity: n-1 variables, O(n²) coefficients
"""

from typing import Dict, List, Tuple

from src.neighborhoods.common import apply_swaps, compute_deltas, solve_qubo
from src.permutation_procesing import c_max


def quantum_adjacent_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 50,
) -> Tuple[List[int], int, Tuple[int, int]]:
    """Quantum adjacent neighborhood - selects exactly one swap via QUBO.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of samples for the solver

    Returns:
        (new_pi, new_cmax, move): New permutation, Cmax, move (i, i+1)
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)

    deltas = compute_deltas(pi, processing_times)
    num_vars = len(deltas)

    # Build QUBO one-hot: exactly 1 swap
    penalty = 2 * max(abs(d) for d in deltas) + 1
    Q: Dict[Tuple[str, str], float] = {}
    for i in range(num_vars):
        Q[(f"x{i}", f"x{i}")] = deltas[i] - penalty
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            Q[(f"x{i}", f"x{j}")] = 2 * penalty

    # Solve QUBO
    solution = solve_qubo(Q, num_reads)
    selected = sorted(int(v[1:]) for v, val in solution.items() if val == 1)

    # Fallback: if solver didn't select, pick minimum
    idx = selected[0] if selected else min(range(num_vars), key=lambda i: deltas[i])

    new_pi = apply_swaps(pi, [idx])
    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, (idx, idx + 1)


def generate_neighbors_adjacent_qubo(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 50,
) -> Tuple[List[int], Tuple[int, int]]:
    """Alias for quantum_adjacent_neighborhood (backward compatibility)."""
    new_pi, _, move = quantum_adjacent_neighborhood(pi, processing_times, num_reads)
    return new_pi, move
