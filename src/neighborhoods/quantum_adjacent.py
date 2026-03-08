"""Quantum Adjacent neighborhood - select one swap using QUBO.

QUBO formulation (one-hot constraint):
    H = Σᵢ δᵢ·xᵢ + P·(Σᵢ xᵢ - 1)²

Expanding the constraint:
    (Σᵢ xᵢ - 1)² = (Σᵢ xᵢ)² - 2(Σᵢ xᵢ) + 1
                 = Σᵢ xᵢ² + 2Σᵢ<ⱼ xᵢ·xⱼ - 2Σᵢ xᵢ + 1
                 = Σᵢ xᵢ + 2Σᵢ<ⱼ xᵢ·xⱼ - 2Σᵢ xᵢ + 1  (since xᵢ² = xᵢ for binary)
                 = -Σᵢ xᵢ + 2Σᵢ<ⱼ xᵢ·xⱼ + 1

Full Hamiltonian:
    H = Σᵢ δᵢ·xᵢ + P·(-Σᵢ xᵢ + 2Σᵢ<ⱼ xᵢ·xⱼ + 1)
      = Σᵢ (δᵢ - P)·xᵢ + 2P·Σᵢ<ⱼ xᵢ·xⱼ + P

QUBO matrix Q where H = Σᵢⱼ Q[i,j]·xᵢ·xⱼ:
    Q[i,i] = δᵢ - P        (linear term: cost + penalty for not selecting)
    Q[i,j] = 2P  (i≠j)     (quadratic term: penalty for selecting multiple swaps)

Penalty weight:
    P = 2·max|δᵢ| + 1      (ensures one-hot constraint dominates)

Complexity: n-1 variables, O(n²) coefficients

Variables:
    xᵢ ∈ {0,1} for i=0..n-2 representing swap at positions (i, i+1)
    δᵢ = ΔCₘₐₓ after swapping positions (i, i+1)

Objective: Find binary assignment minimizing H, which enforces exactly one swap
"""

from typing import Dict, List, Tuple

from src.neighborhoods.common import apply_swaps, compute_deltas, solve_qubo
from src.permutation_procesing import c_max


def quantum_adjacent_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 5,
    backend: str = "simulator",
    dwave_token: str | None = None,
) -> Tuple[List[int], int, Tuple[int, int]]:
    """Quantum adjacent neighborhood - selects exactly one swap via QUBO.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of samples for the solver
        backend: "simulator" or "dwave"
        dwave_token: D-Wave API token (required when backend="dwave")

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
    solution = solve_qubo(Q, num_reads, backend=backend, dwave_token=dwave_token)
    selected = sorted(int(v[1:]) for v, val in solution.items() if val == 1)

    # Fallback: if solver didn't select, pick minimum
    idx = selected[0] if selected else min(range(num_vars), key=lambda i: deltas[i])

    new_pi = apply_swaps(pi, [idx])
    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, (idx, idx + 1)


def generate_neighbors_adjacent_qubo(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 5,
) -> Tuple[List[int], Tuple[int, int]]:
    """Alias for quantum_adjacent_neighborhood (backward compatibility)."""
    new_pi, _, move = quantum_adjacent_neighborhood(pi, processing_times, num_reads)
    return new_pi, move
