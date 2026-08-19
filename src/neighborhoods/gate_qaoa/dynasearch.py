"""Gate-model Dynasearch neighborhood (interval-overlap QUBO solved by QAOA).

Same endpoint-swap QUBO as quantum_qubo/dynasearch.py (built through the
shared common_qubo helpers), solved by a fixed-angle QAOA circuit.

Filter policy: the delta<0 improving-only filter is applied ONLY on the
noiseless simulator (backend='statevector'), never on real hardware -- there
the full candidate set is submitted, as on the annealer. ``L_max`` caps the
interval length to keep K within the reach of the simulator / device; for
full Taillard instances on hardware a windowed variant (analogous to
quantum_qubo_enhanced) will be used.
"""
from typing import List, Optional, Sequence, Tuple

from src.neighborhoods.common_qubo import (
    assemble_pairwise_qubo,
    enumerate_interval_candidates,
    selected_intervals,
)
from src.neighborhoods.accelerator import intervals_overlap, validate_no_overlap_intervals
from src.neighborhoods.gate_qaoa.solve import solve_qaoa
from src.permutation_procesing import c_max


def gate_dynasearch_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    p: int = 1,
    backend: str = "statevector",
    angles: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    shots: int = 4096,
    L_max: Optional[int] = None,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """Select non-overlapping endpoint swaps via a fixed-angle QAOA circuit.

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

    Q = assemble_pairwise_qubo(candidates, intervals_overlap)

    solution = solve_qaoa(Q, "dynasearch", p=p, backend=backend, angles=angles, shots=shots)
    valid_swaps = validate_no_overlap_intervals(selected_intervals(solution, candidates))

    if not valid_swaps:                             # fallback: best single improving swap
        best_k = min(range(len(candidates)), key=lambda k: candidates[k][2])
        if candidates[best_k][2] < 0:
            valid_swaps = [(candidates[best_k][0], candidates[best_k][1])]

    new_pi = pi.copy()
    for i, j in sorted(valid_swaps):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
    return new_pi, c_max(new_pi, processing_times), valid_swaps
