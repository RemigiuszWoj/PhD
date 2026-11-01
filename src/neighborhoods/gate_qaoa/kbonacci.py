"""Gate-model k-bonacci neighborhood (windowed interval-overlap QUBO solved by QAOA).

The windowed decomposition of ``gate_dynasearch`` (window of w positions,
overlap ratio rho, all windows of one move in one job), with the swap distance
capped at k-1 inside every window (``L_max = k``). With w = 6 and k >= 6 the
move set equals windowed gate Dynasearch. The QAOA angles are the calibrated
``dynasearch`` angles: the QUBO family (pairwise interval overlap) is the same.
"""

from typing import List, Optional, Sequence, Tuple

from src.neighborhoods.accelerator import (
    compute_block_boundaries,
    intervals_overlap,
    validate_no_overlap_intervals,
)
from src.neighborhoods.common import compute_head, compute_tail
from src.neighborhoods.common_qubo import enumerate_interval_candidates
from src.neighborhoods.gate_qaoa.windowed import windowed_interval_swaps
from src.permutation_procesing import c_max


def gate_kbonacci_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    k: int,
    p: int = 1,
    backend: str = "iqm",
    angles: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    shots: int = 4096,
    window_size: int = 6,
    overlap_ratio: float = 0.5,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []
    raw = windowed_interval_swaps(
        pi, processing_times, "dynasearch", intervals_overlap,
        p=p, backend=backend, angles=angles, shots=shots,
        window_size=window_size, overlap_ratio=overlap_ratio, filter_delta=False, L_max=k,
    )
    valid_swaps = validate_no_overlap_intervals(raw)
    if not valid_swaps:  # fallback: best single improving swap (as gate_dynasearch)
        all_c = enumerate_interval_candidates(pi, processing_times, L_max=k, filter_delta=False)
        if all_c:
            best = min(all_c, key=lambda c: c[2])
            if best[2] < 0:
                valid_swaps = [(best[0], best[1])]
    new_pi = pi.copy()
    for i, j in sorted(valid_swaps):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
    return new_pi, c_max(new_pi, processing_times), valid_swaps


def window_plan(
    pi: List[int],
    processing_times: List[List[int]],
    k: int,
    window_size: int = 6,
    overlap_ratio: float = 0.5,
) -> List[int]:
    """Variables per circuit for one move (classical, free): one entry per non-empty window.

    Mirrors the window loop of ``windowed_interval_swaps``; used to price a move
    (credits = job fee + circuits x rate) and to check the qubit limit before
    anything is submitted.
    """
    n = len(pi)
    Head, Tail = compute_head(pi, processing_times), compute_tail(pi, processing_times)
    boundaries = compute_block_boundaries(Head, processing_times, pi)
    w = min(window_size, n)
    step = max(1, int(w * (1 - overlap_ratio)))
    sizes = []
    for start in range(0, n, step):
        end = min(start + w, n)
        if end - start < 2:
            break
        cands = enumerate_interval_candidates(
            pi, processing_times, window=(start, end), L_max=k, filter_delta=False,
            head=Head, tail=Tail, boundaries=boundaries,
        )
        if cands:
            sizes.append(len(cands))
        if end == n:
            break
    return sizes
