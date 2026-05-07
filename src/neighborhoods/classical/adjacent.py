"""Adjacent neighborhood - single adjacent element swaps.

The simplest neighborhood: for a permutation π of n elements generates n-1 neighbors,
each by swapping elements at positions (i, i+1).

Complexity: O(n) neighbors, each computed in O(m) with Head+Tail → O(m·n) total

Accelerator: block property (Smutnicki 7.9) optionally filters swaps that
cannot improve Cmax, reducing effective neighborhood to ~2(m-1) candidates.
"""

from typing import Iterator, List, Tuple

from src.neighborhoods.common import compute_head, compute_head_and_tail, swap_jobs
from src.neighborhoods.accelerator import (
    compute_block_boundaries,
    is_blocked_adjacent_swap,
)
from src.permutation_procesing import c_max


def generate_neighbors_adjacent(
    pi: List[int],
) -> Iterator[Tuple[List[int], Tuple[int, int]]]:
    """Generate all neighbors by swapping adjacent elements.

    Lazy generator - neighbors are generated on demand.

    Args:
        pi: Current permutation

    Yields:
        (neighbor, move): Neighboring permutation and move (i, i+1)
    """
    n = len(pi)
    for i in range(n - 1):
        neighbor = swap_jobs(pi, i, i + 1)
        yield neighbor, (i, i + 1)


def best_adjacent_neighbor(
    pi: List[int],
    processing_times: List[List[int]],
    
) -> Tuple[List[int], int, Tuple[int, int]]:
    """Find the best neighbor in the adjacent neighborhood.

    Uses Head+Tail technique for O(m·n) instead of O(m·n²).
    Optionally applies block property accelerator to skip dominated swaps.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
            (Smutnicki block property). Reduces candidates from n-1 to
            at most 2*(m-1) on average. Safe: never skips improving swaps.

    Returns:
        (best_pi, best_cmax, move): Best permutation, its Cmax, move
    """
    n = len(pi)
    m = len(processing_times)

    if n < 2:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)

    Head, Tail = compute_head_and_tail(pi, processing_times)
    base_cmax = Head[m - 1][n - 1]

    boundaries = compute_block_boundaries(Head, processing_times, pi)

    best_pi = None
    best_cmax = float("inf")
    best_move = (-1, -1)

    for j in range(n - 1):
        # Block property: skip swap if both positions are inside same block
        if is_blocked_adjacent_swap(j, boundaries):
            continue

        job_a = pi[j]      # originally at j, will go to j+1
        job_b = pi[j + 1]  # originally at j+1, will go to j

        C_j  = [0] * m
        C_j1 = [0] * m

        for i in range(m):
            left = Head[i][j - 1] if j > 0 else 0
            top  = C_j[i - 1] if i > 0 else 0
            C_j[i] = max(top, left) + processing_times[i][job_b]

            top_j1 = C_j1[i - 1] if i > 0 else 0
            C_j1[i] = max(top_j1, C_j[i]) + processing_times[i][job_a]

        if j + 2 < n:
            new_cmax = max(C_j1[i] + Tail[i][j + 2] for i in range(m))
        else:
            new_cmax = C_j1[m - 1]

        if new_cmax < best_cmax:
            best_cmax = new_cmax
            best_move = (j, j + 1)

    if best_move[0] >= 0:
        best_pi = swap_jobs(pi, best_move[0], best_move[1])
    else:
        best_pi = pi.copy()
        best_cmax = base_cmax

    return best_pi, best_cmax, best_move