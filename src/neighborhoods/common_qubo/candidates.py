"""Candidate enumeration shared by every QUBO family.

The helpers here reproduce exactly the candidate lists that
``quantum_qubo``, ``quantum_qubo_enhanced`` and ``gate_qaoa`` build. They
are deliberately "dumb": each caller passes the flags that reproduce its
own behaviour (window, L_max, whether to drop non-improving swaps). This
is the single source of truth for *which swaps enter the QUBO*.
"""
from typing import List, Optional, Tuple

from src.neighborhoods.common import (
    compute_deltas,
    compute_endpoint_swap_delta,
    compute_head,
    compute_tail,
)
from src.neighborhoods.accelerator import (
    compute_block_boundaries,
    filter_blocked_pairs_npi,
    filter_blocked_swaps,
)

IntervalCand = Tuple[int, int, float]   # (i, j, delta) endpoint swap
AdjacentCand = Tuple[int, float]        # (position, delta) adjacent swap (i, i+1)


def enumerate_interval_candidates(
    pi: List[int],
    processing_times: List[List[int]],
    *,
    window: Optional[Tuple[int, int]] = None,   # [start, end); None = full permutation
    L_max: Optional[int] = None,                # cap on interval length j-i+1
    filter_delta: bool = False,                 # drop delta >= 0 (improving-only)
    head: Optional[List[List[int]]] = None,     # precomputed Head/Tail/boundaries:
    tail: Optional[List[List[int]]] = None,     #   pass them to avoid recomputing
    boundaries: Optional[List[int]] = None,     #   once per window (enhanced path)
) -> List[IntervalCand]:
    """Endpoint-swap candidates for Dynasearch / Motzkin.

    Order of operations is fixed and matches the current code: enumerate,
    then the NPI block filter (structural), then the optional delta filter.
    Head/Tail (and hence delta and the NPI boundaries) are over the *full*
    permutation, so a window only restricts the (i, j) range. Callers that
    already hold Head/Tail/boundaries (windowed decomposition) may pass them
    to skip recomputation; the result is identical.
    """
    n, m = len(pi), len(processing_times)
    Head = head if head is not None else compute_head(pi, processing_times)
    Tail = tail if tail is not None else compute_tail(pi, processing_times)
    base_c = Head[m - 1][n - 1]

    lo, hi = window if window is not None else (0, n)
    cands: List[IntervalCand] = []
    for i in range(lo, hi - 1):
        j_max = hi - 1 if L_max is None else min(hi - 1, i + L_max - 1)
        for j in range(i + 1, j_max + 1):
            d = compute_endpoint_swap_delta(pi, i, j, Head, Tail, processing_times, base_c)
            cands.append((i, j, d))

    bounds = boundaries if boundaries is not None else compute_block_boundaries(
        Head, processing_times, pi)
    cands = filter_blocked_pairs_npi(cands, bounds)
    if filter_delta:
        cands = [c for c in cands if c[2] < 0]
    return cands


def enumerate_adjacent_candidates(
    pi: List[int],
    processing_times: List[List[int]],
    *,
    use_block_accelerator: bool = True,
) -> List[AdjacentCand]:
    """Adjacent-swap candidates (position, delta) for Adjacent / Fibonacci."""
    deltas = compute_deltas(pi, processing_times)
    cands = list(enumerate(deltas))
    if use_block_accelerator:
        Head = compute_head(pi, processing_times)
        boundaries = compute_block_boundaries(Head, processing_times, pi)
        cands = filter_blocked_swaps(cands, boundaries)
    return cands
