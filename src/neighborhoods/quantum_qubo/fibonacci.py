"""Quantum Fibonacci neighborhood - select non-overlapping swaps using QUBO.

QUBO formulation (no-overlap constraint, tridiagonal):
    H = Σᵢ δᵢ·xᵢ + P·Σᵢ xᵢ·xᵢ₊₁

QUBO matrix Q:
    Q[i,i]   = δᵢ       (linear cost)
    Q[i,i+1] = P        (penalty for adjacent overlapping swaps)

Penalty weight:
    P = Σᵢ |δᵢ| + 1

Accelerator: block property (Smutnicki 7.9) optionally filters blocked
swaps before QUBO construction. For Fibonacci the QUBO is already sparse
(tridiagonal), but filtering reduces variable count and may improve
QPU solution quality by removing irrelevant variables.
"""

from typing import Dict, List, Tuple

from src.neighborhoods.common import apply_swaps, solve_qubo, validate_no_overlap
from src.neighborhoods.common_qubo import (
    assemble_tridiagonal_qubo,
    enumerate_adjacent_candidates,
    selected_indices,
)
from src.permutation_procesing import c_max


def quantum_fibonacci_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 5,
    backend: str = "simulator",
    dwave_token: str | None = None,
    solver: str | None = None,
    annealing_time_us: int | None = None,
    chain_strength: float | None = None,
    num_spin_reversal_transforms: int | None = None,
    use_block_accelerator: bool = True,
) -> Tuple[List[int], int, List[int]]:
    """Quantum fibonacci neighborhood - selects non-overlapping swaps via QUBO.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of samples for the solver
        backend: "simulator" or "dwave"
        dwave_token: D-Wave API token (required when backend="dwave")
        use_block_accelerator: If True, apply block property filter before
            QUBO construction (Smutnicki 7.9).

    Returns:
        (new_pi, new_cmax, swaps): New permutation, Cmax, list of swap positions
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    # Candidate swaps + tridiagonal QUBO. Enumeration and assembly are shared
    # (common_qubo) with the enhanced and gate-model families: identical Q.
    candidates = enumerate_adjacent_candidates(
        pi, processing_times, use_block_accelerator=use_block_accelerator)
    positions = [pos for pos, _ in candidates]
    delta_vals = [d for _, d in candidates]
    num_vars = len(candidates)

    if num_vars == 0:
        return pi.copy(), c_max(pi, processing_times), []

    Q = assemble_tridiagonal_qubo(candidates)

    # Solve QUBO
    solution = solve_qubo(
        Q,
        num_reads,
        backend=backend,
        dwave_token=dwave_token,
        solver=solver,
        annealing_time_us=annealing_time_us,
        chain_strength=chain_strength,
        num_spin_reversal_transforms=num_spin_reversal_transforms,
    )
    selected_local = selected_indices(solution)

    # Map back to original positions and validate non-overlap
    selected_orig = [positions[i] for i in selected_local]
    valid_swaps = validate_no_overlap(selected_orig)

    # Fallback: best single improving swap
    if not valid_swaps:
        best_local = min(range(num_vars), key=lambda i: delta_vals[i])
        if delta_vals[best_local] < 0:
            valid_swaps = [positions[best_local]]

    new_pi = apply_swaps(pi, valid_swaps)
    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, valid_swaps


def generate_neighbors_fibonacci_qubo(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 5,
) -> Tuple[List[int], int, List[int]]:
    """Alias for backward compatibility."""
    return quantum_fibonacci_neighborhood(pi, processing_times, num_reads)
