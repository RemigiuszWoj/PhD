"""Gate-model Fibonacci neighborhood (tridiagonal QUBO solved by QAOA).

Same tridiagonal swap-selection QUBO as quantum_qubo/fibonacci.py (built
through the shared common_qubo helpers), solved by a fixed-angle QAOA
circuit instead of an annealer.
"""
from typing import List, Optional, Sequence, Tuple

from src.neighborhoods.common import apply_swaps, validate_no_overlap
from src.neighborhoods.common_qubo import (
    assemble_tridiagonal_qubo,
    enumerate_adjacent_candidates,
    selected_positions,
)
from src.neighborhoods.gate_qaoa.solve import solve_qaoa
from src.permutation_procesing import c_max


def gate_fibonacci_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    p: int = 1,
    backend: str = "ibm",
    angles: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    shots: int = 4096,
    use_block_accelerator: bool = True,
) -> Tuple[List[int], int, List[int]]:
    """Select non-overlapping adjacent swaps via a fixed-angle QAOA circuit.

    Returns (new_pi, new_cmax, swap_positions).
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    candidates = enumerate_adjacent_candidates(
        pi, processing_times, use_block_accelerator=use_block_accelerator)
    positions = [pos for pos, _ in candidates]
    delta_vals = [d for _, d in candidates]
    num_vars = len(candidates)
    if num_vars == 0:
        return pi.copy(), c_max(pi, processing_times), []

    Q = assemble_tridiagonal_qubo(candidates)

    solution = solve_qaoa(Q, "fibonacci", p=p, backend=backend, angles=angles, shots=shots)
    valid_swaps = validate_no_overlap(selected_positions(solution, positions))

    if not valid_swaps:                         # fallback: best single improving swap
        best_local = min(range(num_vars), key=lambda i: delta_vals[i])
        if delta_vals[best_local] < 0:
            valid_swaps = [positions[best_local]]

    new_pi = apply_swaps(pi, valid_swaps)
    return new_pi, c_max(new_pi, processing_times), valid_swaps
