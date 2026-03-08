"""Quantum Fibonahi neighborhood - select non-overlapping swaps using QUBO.

QUBO formulation (no-overlap constraint):
    H = Σᵢ δᵢ·xᵢ + P·Σᵢ xᵢ·xᵢ₊₁

Where:
    xᵢ ∈ {0,1} indicates whether to perform swap at position i
    δᵢ = ΔCₘₐₓ after swapping positions (i, i+1)
    P = penalty weight for overlapping swaps

Overlap definition:
    Swaps at positions i and i+1 are overlapping because they share
    the element at position i+1:
      - Swap i:   (posᵢ, posᵢ₊₁)
      - Swap i+1: (posᵢ₊₁, posᵢ₊₂)
    Therefore xᵢ·xᵢ₊₁ = 1 is forbidden.

QUBO matrix Q where H = Σᵢⱼ Q[i,j]·xᵢ·xⱼ:
    Q[i,i] = δᵢ                  (linear term: cost of selecting swap i)
    Q[i,i+1] = P  for i=0..n-3   (quadratic term: penalty for adjacent pair)
    Q[i,j] = 0    for |i-j|>1    (non-adjacent swaps don't overlap)

Penalty weight:
    P = Σᵢ |δᵢ| + 1              (ensures no-overlap constraint dominates)

Complexity:
    - Variables: n-1 binary variables
    - Coefficients: O(n) non-zero entries (chain graph structure)

Combinatorial structure:
    The number of valid non-overlapping swap sets from n-1 positions
    equals the Fibonacci number F_{n+1}:
        F₀=0, F₁=1, F₂=1, F₃=2, F₄=3, F₅=5, F₆=8, F₇=13, ...
    This is because:
        - If we don't select swap 0: F_n solutions from positions 1..n-2
        - If we select swap 0: can't select swap 1, so F_{n-1} solutions from 2..n-2
        - Total: F_{n+1} = F_n + F_{n-1}

Objective: Find binary assignment minimizing H, selecting non-overlapping
           improving swaps
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
    num_reads: int = 5,
    backend: str = "simulator",
    dwave_token: str | None = None,
) -> Tuple[List[int], int, List[int]]:
    """Quantum fibonahi neighborhood - selects non-overlapping swaps via QUBO.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of samples for the solver
        backend: "simulator" or "dwave"
        dwave_token: D-Wave API token (required when backend="dwave")

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
    solution = solve_qubo(Q, num_reads, backend=backend, dwave_token=dwave_token)
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
    num_reads: int = 5,
) -> Tuple[List[int], int, List[int]]:
    """Alias for quantum_fibonahi_neighborhood (backward compatibility)."""
    return quantum_fibonahi_neighborhood(pi, processing_times, num_reads)
