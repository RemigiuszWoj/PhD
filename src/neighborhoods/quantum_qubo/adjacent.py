"""Quantum Adjacent neighborhood - select one swap using QUBO.

QUBO formulation (one-hot constraint):
    H = Σᵢ δᵢ·xᵢ + P·(Σᵢ xᵢ - 1)²

QUBO matrix Q:
    Q[i,i] = δᵢ - P        (linear term)
    Q[i,j] = 2P  (i≠j)     (quadratic penalty)

Penalty weight:
    P = 2·max|δᵢ| + 1

Accelerator: block property (Smutnicki 7.9) optionally filters blocked
swaps before QUBO construction, reducing variable count from n-1 to
at most 2(m-1). For dense one-hot QUBO this reduces quadratic terms
from O(n²) to O(m²) — significant speedup for large n.
"""

from typing import Dict, List, Optional, Tuple

from src.neighborhoods.common import (
    apply_swaps,
    compute_deltas,
    compute_head,
    solve_qubo,
)
from src.neighborhoods.accelerator import compute_block_boundaries, filter_blocked_swaps
from src.permutation_procesing import c_max


def quantum_adjacent_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 5,
    backend: str = "simulator",
    dwave_token: str | None = None,
    solver: str | None = None,
    annealing_time_us: int | None = None,
    chain_strength: float | None = None,
    num_spin_reversal_transforms: int | None = None,
    
) -> Tuple[List[int], int, Tuple[int, int]]:
    """Quantum adjacent neighborhood - selects exactly one swap via QUBO.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of samples for the solver
        backend: "simulator" or "dwave"
        dwave_token: D-Wave API token (required when backend="dwave")
            QUBO construction — reduces variable count to ~2(m-1),
            shrinking the dense one-hot QUBO from O(n²) to O(m²) terms.

    Returns:
        (new_pi, new_cmax, move): New permutation, Cmax, move (i, i+1)
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)

    deltas = compute_deltas(pi, processing_times)

    # Build candidate list (position, delta)
    candidates = list(enumerate(deltas))  # [(0, δ0), (1, δ1), ...]

    # Block accelerator: filter dominated swaps before QUBO construction
    Head = compute_head(pi, processing_times)
    boundaries = compute_block_boundaries(Head, processing_times, pi)
    candidates = filter_blocked_swaps(candidates, boundaries)

    num_vars = len(candidates)

    # Remap to contiguous variable indices
    positions = [pos for pos, _ in candidates]
    delta_vals = [d for _, d in candidates]

    # Build QUBO one-hot: exactly 1 swap selected
    penalty = 2 * max(abs(d) for d in delta_vals) + 1
    Q: Dict[Tuple[str, str], float] = {}
    for i in range(num_vars):
        Q[(f"x{i}", f"x{i}")] = delta_vals[i] - penalty
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            Q[(f"x{i}", f"x{j}")] = 2 * penalty

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
    selected = sorted(int(v[1:]) for v, val in solution.items() if val == 1)

    # Map back to original position
    local_idx = selected[0] if selected else min(range(num_vars), key=lambda i: delta_vals[i])
    idx = positions[local_idx]

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