"""Gate-model Motzkin neighborhood (crossing/shared-endpoint QUBO solved by QAOA).

Same endpoint-swap QUBO as quantum_qubo/motzkin.py (built through the shared
common_qubo helpers; nesting admitted, only crossings and shared endpoints
penalized), solved by a fixed-angle QAOA circuit.

Filter policy as in the gate Dynasearch neighborhood: delta<0 filtering only
on the noiseless simulator, never on real hardware. ``L_max`` caps K; a
windowed variant will handle full Taillard instances on hardware.
"""
from typing import List, Optional, Sequence, Tuple

from src.neighborhoods.common_qubo import (
    assemble_pairwise_qubo,
    enumerate_interval_candidates,
    selected_intervals,
)
from src.neighborhoods.accelerator import (
    motzkin_conflict as _motzkin_conflict,
    validate_motzkin_selection as _validate_motzkin_selection,
)
from src.neighborhoods.gate_qaoa.solve import solve_qaoa
from src.permutation_procesing import c_max


def gate_motzkin_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    p: int = 1,
    backend: str = "statevector",
    angles: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    shots: int = 4096,
    L_max: Optional[int] = None,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """Select non-crossing endpoint swaps via a fixed-angle QAOA circuit.

    Returns (new_pi, new_cmax, applied_swaps) with swaps as (i, j) pairs.
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    filter_delta = backend == "statevector"        # simulator may filter; hardware never
    all_candidates = enumerate_interval_candidates(
        pi, processing_times, L_max=L_max, filter_delta=False)
    if not all_candidates:
        return pi.copy(), c_max(pi, processing_times), []

    candidates = [c for c in all_candidates if c[2] < 0] if filter_delta else list(all_candidates)
    if not candidates:
        best = min(all_candidates, key=lambda x: x[2])
        if best[2] < 0:
            new_pi = pi.copy()
            new_pi[best[0]], new_pi[best[1]] = new_pi[best[1]], new_pi[best[0]]
            return new_pi, c_max(new_pi, processing_times), [(best[0], best[1])]
        return pi.copy(), c_max(pi, processing_times), []

    Q = assemble_pairwise_qubo(candidates, _motzkin_conflict)

    solution = solve_qaoa(Q, "motzkin", p=p, backend=backend, angles=angles, shots=shots)
    valid_swaps = _validate_motzkin_selection(selected_intervals(solution, candidates))

    if not valid_swaps:                             # fallback: best single improving swap
        best_k = min(range(len(candidates)), key=lambda k: candidates[k][2])
        if candidates[best_k][2] < 0:
            valid_swaps = [(candidates[best_k][0], candidates[best_k][1])]

    new_pi = pi.copy()
    for i, j in sorted(valid_swaps, key=lambda x: x[0]):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
    return new_pi, c_max(new_pi, processing_times), valid_swaps
