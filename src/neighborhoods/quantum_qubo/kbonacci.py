"""Quantum k-bonacci neighborhood: interval-overlap QUBO with swap distance <= k-1.

Same QUBO as ``quantum_qubo/dynasearch.py`` (shared common_qubo helpers, the
same NPI block filter, the same delta-filter policy and fallback), with one
difference: the interval-length cap ``L_max = k`` is honoured on EVERY backend.
``quantum_dynasearch_neighborhood`` drops ``L_max`` when ``backend == 'dwave'``
(full O(n^2) candidate space on the QPU), which is right for Dynasearch but
would silently turn k-bonacci into Dynasearch on the annealer.

    k = 2 -> quantum Fibonacci move set (adjacent swaps, as an interval QUBO)
    k = 3 -> quantum tribonacci
    k = n -> quantum Dynasearch
"""

from typing import List, Tuple

from src.neighborhoods.accelerator import validate_no_overlap_intervals
from src.neighborhoods.common import solve_qubo
from src.neighborhoods.common_qubo import (
    assemble_pairwise_qubo,
    enumerate_interval_candidates,
    selected_intervals,
)
from src.permutation_procesing import c_max


def _intervals_overlap(i1: int, j1: int, i2: int, j2: int) -> bool:
    return max(i1, i2) <= min(j1, j2)


def quantum_kbonacci_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    k: int,
    num_reads: int = 5,
    backend: str = "simulator",
    dwave_token: str | None = None,
    solver: str | None = None,
    annealing_time_us: int | None = None,
    chain_strength: float | None = None,
    num_spin_reversal_transforms: int | None = None,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """Select disjoint endpoint swaps of distance <= k-1 via QUBO (annealer or simulator)."""
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    all_candidates = enumerate_interval_candidates(
        pi, processing_times, L_max=k, filter_delta=False
    )
    if not all_candidates:
        return pi.copy(), c_max(pi, processing_times), []

    # Delta-filter policy identical to quantum_dynasearch: simulator-only.
    if backend != "dwave":
        candidates = [(i, j, d) for i, j, d in all_candidates if d < 0]
    else:
        candidates = list(all_candidates)

    if not candidates:
        best = min(all_candidates, key=lambda x: x[2])
        if best[2] < 0:
            new_pi = pi.copy()
            new_pi[best[0]], new_pi[best[1]] = new_pi[best[1]], new_pi[best[0]]
            return new_pi, c_max(new_pi, processing_times), [(best[0], best[1])]
        return pi.copy(), c_max(pi, processing_times), []

    Q = assemble_pairwise_qubo(candidates, _intervals_overlap)
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
    valid_swaps = validate_no_overlap_intervals(selected_intervals(solution, candidates))
    if not valid_swaps:
        best_k = min(range(len(candidates)), key=lambda t: candidates[t][2])
        if candidates[best_k][2] < 0:
            valid_swaps = [(candidates[best_k][0], candidates[best_k][1])]

    new_pi = pi.copy()
    for i, j in sorted(valid_swaps):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
    return new_pi, c_max(new_pi, processing_times), valid_swaps


def count_qubo_variables(pi: List[int], processing_times: List[List[int]], k: int) -> int:
    """Number of QUBO variables submitted to the QPU (full candidate set, no delta filter)."""
    return len(enumerate_interval_candidates(pi, processing_times, L_max=k, filter_delta=False))
