"""Gate-model Adjacent neighborhood (one-hot QUBO solved by QAOA).

Same swap-selection QUBO as quantum_qubo/adjacent.py (built through the
shared common_qubo helpers, so the matrix Q is identical to the annealer's);
only the solver differs -- a fixed-angle QAOA circuit instead of an annealer.
"""
from typing import List, Optional, Sequence, Tuple

from src.neighborhoods.common import apply_swaps
from src.neighborhoods.common_qubo import (
    assemble_onehot_qubo,
    enumerate_adjacent_candidates,
    selected_indices,
)
from src.neighborhoods.gate_qaoa.solve import solve_qaoa
from src.permutation_procesing import c_max


def gate_adjacent_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    p: int = 1,
    backend: str = "statevector",
    angles: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    shots: int = 4096,
) -> Tuple[List[int], int, Tuple[int, int]]:
    """Select exactly one adjacent swap via a fixed-angle QAOA circuit.

    Returns (new_pi, new_cmax, move) with move = (i, i+1).
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)

    candidates = enumerate_adjacent_candidates(pi, processing_times)
    num_vars = len(candidates)
    if num_vars == 0:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)
    positions = [pos for pos, _ in candidates]
    delta_vals = [d for _, d in candidates]
    Q = assemble_onehot_qubo(candidates)

    solution = solve_qaoa(Q, "adjacent", p=p, backend=backend, angles=angles, shots=shots)
    selected = selected_indices(solution)

    local_idx = selected[0] if selected else min(range(num_vars), key=lambda i: delta_vals[i])
    idx = positions[local_idx]

    new_pi = apply_swaps(pi, [idx])
    return new_pi, c_max(new_pi, processing_times), (idx, idx + 1)
