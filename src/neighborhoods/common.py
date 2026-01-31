"""Common functions for Flow Shop neighborhoods.

This module contains basic operations used by all neighborhoods:
- swap_jobs: swap two elements in a permutation
- compute_head: compute Head matrix (forward completion times)
- compute_tail: compute Tail matrix (backward remaining times)
- compute_head_and_tail: compute both matrices together
- compute_deltas: compute Cmax changes for adjacent swaps (Head+Tail)
- apply_swaps: apply swaps to a permutation
- solve_qubo: solve QUBO using simulated annealing
"""

from typing import Dict, List, Tuple

from src.neighborhoods.boundaries import compute_prefix_boundaries
from src.permutation_procesing import c_max


def swap_jobs(pi: List[int], i: int, j: int) -> List[int]:
    """Return a new permutation with elements at positions i and j swapped.

    Args:
        pi: Input permutation
        i: First position
        j: Second position

    Returns:
        New permutation with swapped elements
    """
    neighbor = pi.copy()
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def compute_head(
    pi: List[int],
    processing_times: List[List[int]],
) -> List[List[int]]:
    """Compute Head matrix (forward completion times).

    Head[i][j] = completion time of job at position j on machine i.

    Complexity: O(m·n)

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix

    Returns:
        Head matrix of size m × n
    """
    m = len(processing_times)
    n = len(pi)

    Head = [[0] * n for _ in range(m)]
    Head[0][0] = processing_times[0][pi[0]]

    for j in range(1, n):
        Head[0][j] = Head[0][j - 1] + processing_times[0][pi[j]]

    for i in range(1, m):
        Head[i][0] = Head[i - 1][0] + processing_times[i][pi[0]]

    for i in range(1, m):
        for j in range(1, n):
            Head[i][j] = max(Head[i - 1][j], Head[i][j - 1]) + processing_times[i][pi[j]]

    return Head


def compute_tail(
    pi: List[int],
    processing_times: List[List[int]],
) -> List[List[int]]:
    """Compute Tail matrix (backward remaining times).

    Tail[i][j] = time needed from position j on machine i to complete all remaining work.
    Computed backwards from the end.

    Complexity: O(m·n)

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix

    Returns:
        Tail matrix of size m × n
    """
    m = len(processing_times)
    n = len(pi)

    Tail = [[0] * n for _ in range(m)]

    # Last position, last machine
    Tail[m - 1][n - 1] = processing_times[m - 1][pi[n - 1]]

    # Last position, remaining machines (bottom to top)
    for i in range(m - 2, -1, -1):
        Tail[i][n - 1] = Tail[i + 1][n - 1] + processing_times[i][pi[n - 1]]

    # Last machine, remaining positions (right to left)
    for j in range(n - 2, -1, -1):
        Tail[m - 1][j] = Tail[m - 1][j + 1] + processing_times[m - 1][pi[j]]

    # Remaining positions (right to left, bottom to top)
    for j in range(n - 2, -1, -1):
        for i in range(m - 2, -1, -1):
            Tail[i][j] = max(Tail[i + 1][j], Tail[i][j + 1]) + processing_times[i][pi[j]]

    return Tail


def compute_head_and_tail(
    pi: List[int],
    processing_times: List[List[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Compute both Head and Tail matrices.

    Convenience function that computes both matrices in one call.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix

    Returns:
        (Head, Tail) tuple of matrices
    """
    return compute_head(pi, processing_times), compute_tail(pi, processing_times)


def compute_deltas(
    pi: List[int],
    processing_times: List[List[int]],
) -> List[float]:
    """Compute Cmax delta for each possible adjacent swap.

    Uses Head+Tail technique for O(m·n) complexity.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix

    Returns:
        List of deltas: δᵢ = Cmax(π after swap i,i+1) - Cmax(π)
    """
    n = len(pi)
    if n < 2:
        return []

    m = len(processing_times)

    # Compute Head and Tail matrices
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)

    base_cmax = Head[m - 1][n - 1]
    deltas: List[float] = []

    for j in range(n - 1):
        # Swap positions j and j+1
        job_a = pi[j]  # originally at j, will go to j+1
        job_b = pi[j + 1]  # originally at j+1, will go to j

        # Local completion times for swapped positions
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

        # Combine with Tail
        if j + 2 < n:
            new_cmax = max(C_j1[i] + Tail[i][j + 2] for i in range(m))
        else:
            new_cmax = C_j1[m - 1]

        deltas.append(new_cmax - base_cmax)

    return deltas


def apply_swaps(pi: List[int], indices: List[int]) -> List[int]:
    """Apply selected adjacent swaps to a permutation.

    Each index i means swapping elements at positions i and i+1.
    Swaps are applied in ascending index order.

    Args:
        pi: Input permutation
        indices: List of positions to swap (each swap is (i, i+1))

    Returns:
        New permutation after swaps
    """
    new_pi = pi.copy()
    for idx in sorted(indices):
        new_pi[idx], new_pi[idx + 1] = new_pi[idx + 1], new_pi[idx]
    return new_pi


def validate_no_overlap(indices: List[int]) -> List[int]:
    """Remove overlapping swaps (keep only non-overlapping ones).

    Two swaps at positions i and i+1 overlap because they share
    position i+1. This function keeps only non-overlapping swaps.

    Args:
        indices: List of swap indices (sorted)

    Returns:
        List of indices without overlaps
    """
    valid: List[int] = []
    last_idx = -2

    for idx in sorted(indices):
        if idx > last_idx + 1:
            valid.append(idx)
            last_idx = idx

    return valid


def solve_qubo(
    Q: Dict[Tuple[str, str], float],
    num_reads: int = 50,
) -> Dict[str, int]:
    """Solve QUBO using Simulated Annealing.

    Can be replaced with DWaveSampler() or LeapHybridSampler() for real quantum.

    Args:
        Q: QUBO matrix as dict {(var_i, var_j): coefficient}
        num_reads: Number of samples for the solver

    Returns:
        Best solution as dict {variable_name: 0 or 1}
    """
    if not Q:
        return {}

    from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = SimulatedAnnealingSampler()
    result = sampler.sample(bqm, num_reads=num_reads)
    return dict(result.first.sample)
