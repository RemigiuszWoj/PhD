"""Wspólne funkcje obliczania granic (prefix boundaries) dla flow shop."""

from typing import List


def compute_prefix_boundaries(pi: List[int], processing_times: List[List[int]]) -> List[List[int]]:
    """
    Compute prefix boundary vectors F[k] for k = 0..n.
    F[k] is a length-m vector: completion times on machines after processing
    first k jobs (pi[0..k-1]).

    Args:
        pi: Permutacja zadań
        processing_times: Macierz czasów m × n

    Returns:
        F: Lista wektorów F[0]..F[n]; F[n][-1] == c_max(pi)
    """
    m = len(processing_times)
    n = len(pi)
    F = [[0] * m for _ in range(n + 1)]
    for k in range(1, n + 1):
        job = pi[k - 1]
        F[k][0] = F[k - 1][0] + processing_times[0][job]
        for r in range(1, m):
            F[k][r] = max(F[k][r - 1], F[k - 1][r]) + processing_times[r][job]
    return F
