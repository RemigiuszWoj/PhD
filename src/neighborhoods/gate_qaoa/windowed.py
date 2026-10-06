"""Windowed decomposition for the interval gate neighborhoods (large n).

For full Taillard instances the interval QUBO (K = O(n^2)) is far too large for
a gate simulator or NISQ device. As in quantum_qubo_enhanced we split the
permutation into overlapping windows, solve each window's small QUBO with QAOA,
and merge the swaps. Head/Tail/boundaries are computed once and reused per
window (identical Q to the single-QUBO path restricted to that window).
"""
from typing import Callable, List, Sequence, Tuple

from src.neighborhoods.common import compute_head, compute_tail
from src.neighborhoods.accelerator import compute_block_boundaries
from src.neighborhoods.common_qubo import (
    assemble_pairwise_qubo,
    enumerate_interval_candidates,
    selected_intervals,
)
from src.neighborhoods.gate_qaoa.solve import solve_qaoa_batch


def windowed_interval_swaps(
    pi: List[int],
    processing_times: List[List[int]],
    neighborhood: str,
    conflict_fn: Callable[[int, int, int, int], bool],
    *,
    p: int,
    backend: str,
    angles,
    shots: int,
    window_size: int,
    overlap_ratio: float,
    filter_delta: bool,
) -> List[Tuple[int, int]]:
    """Collect endpoint swaps from all overlapping windows (raw, unmerged)."""
    n = len(pi)
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)
    boundaries = compute_block_boundaries(Head, processing_times, pi)

    w = min(window_size, n)
    step = max(1, int(w * (1 - overlap_ratio)))
    windows = []                      # (candidates, Q) for each non-empty window
    for start in range(0, n, step):
        end = min(start + w, n)
        if end - start < 2:
            break
        cands = enumerate_interval_candidates(
            pi, processing_times, window=(start, end), filter_delta=filter_delta,
            head=Head, tail=Tail, boundaries=boundaries,
        )
        if cands:
            windows.append((cands, assemble_pairwise_qubo(cands, conflict_fn)))
        if end == n:
            break
    if not windows:
        return []

    # All windows of this move go out as one batch: on hardware that is a single
    # job instead of one per window, which is what the queue actually charges for.
    solutions = solve_qaoa_batch([Q for _, Q in windows], neighborhood, p=p,
                                 backend=backend, angles=angles, shots=shots)
    swaps: List[Tuple[int, int]] = []
    for (cands, _), solution in zip(windows, solutions):
        swaps.extend(selected_intervals(solution, cands))
    return swaps
