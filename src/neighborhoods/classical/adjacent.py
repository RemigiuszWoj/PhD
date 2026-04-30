"""Adjacent neighborhood - single adjacent element swaps.

The simplest neighborhood: for a permutation π of n elements generates n-1 neighbors,
each by swapping elements at positions (i, i+1).

Complexity: O(n) neighbors, each computed in O(m) with Head+Tail → O(m·n) total
"""

from typing import Iterator, List, Tuple

from src.neighborhoods.common import compute_head_and_tail, swap_jobs
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

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix

    Returns:
        (best_pi, best_cmax, move): Best permutation, its Cmax, move
    """
    n = len(pi)
    m = len(processing_times)

    if n < 2:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)

    Head, Tail = compute_head_and_tail(pi, processing_times)
    base_cmax = Head[m - 1][n - 1]

    best_pi = None
    best_cmax = float("inf")
    best_move = (-1, -1)

    for j in range(n - 1):
        # Swap positions j and j+1
        job_a = pi[j]  # originally at j, will go to j+1
        job_b = pi[j + 1]  # originally at j+1, will go to j

        # Compute completion times at swapped positions
        C_j = [0] * m  # completion at position j (now has job_b)
        C_j1 = [0] * m  # completion at position j+1 (now has job_a)

        for i in range(m):
            # Completion at position j with job_b
            left = Head[i][j - 1] if j > 0 else 0
            top = C_j[i - 1] if i > 0 else 0
            C_j[i] = max(top, left) + processing_times[i][job_b]

            # Completion at position j+1 with job_a
            top_j1 = C_j1[i - 1] if i > 0 else 0
            C_j1[i] = max(top_j1, C_j[i]) + processing_times[i][job_a]

        # Compute new Cmax using Tail
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
