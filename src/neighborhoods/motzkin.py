"""Motzkin neighborhood - non-crossing endpoint swaps.

Selects a set of non-crossing swaps where each swap exchanges
only the endpoints of a segment [i..j], keeping the middle unchanged.

Rules:
  * Move = pair (i, j), i < j, swap only endpoints of segment [i..j]
  * No crossing: pattern i1 < i2 < j1 < j2 is forbidden
  * No endpoint reuse: no two selected pairs share any endpoint
  * Strict nesting allowed: i1 < i2 < j2 < j1 (inner arc fully inside outer)

Goal: minimize final makespan (Cmax)
Method: Head+Tail for deltas + O(n³) DP selection
Complexity: O(m·n²) for deltas + O(n³) for DP = O(n³) total
"""

from typing import List, Optional, Tuple

from src.neighborhoods.common import compute_head, compute_tail
from src.permutation_procesing import c_max


def _compute_delta_with_tail(
    pi: List[int],
    i: int,
    j: int,
    Head: List[List[int]],
    Tail: List[List[int]],
    processing_times: List[List[int]],
    base_cmax: int,
) -> int:
    """Compute delta for swapping endpoints of segment [i..j] using Head+Tail.

    After swap: [pi[j]] + pi[i+1..j-1] + [pi[i]]

    Uses Head matrix for prefix and Tail matrix instead of propagating to end.
    Complexity: O(m·(j-i)) instead of O(m·(n-i))
    """
    m = len(processing_times)
    n = len(pi)

    # Start from Head at position i-1 (state after i-1 jobs)
    if i == 0:
        col_prev = [0] * m
    else:
        col_prev = [Head[r][i - 1] for r in range(m)]

    col = [0] * m

    # First job after swap: pi[j] (at position i)
    job = pi[j]
    col[0] = col_prev[0] + processing_times[0][job]
    for r in range(1, m):
        col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][job]
    col_prev, col = col, col_prev

    # Middle of segment (positions i+1 .. j-1) - unchanged
    for t in range(i + 1, j):
        job = pi[t]
        col[0] = col_prev[0] + processing_times[0][job]
        for r in range(1, m):
            col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][job]
        col_prev, col = col, col_prev

    # Last job of swapped segment: pi[i] (at position j)
    job = pi[i]
    col[0] = col_prev[0] + processing_times[0][job]
    for r in range(1, m):
        col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][job]
    col_prev, col = col, col_prev

    # Use Tail instead of propagating through remaining jobs
    if j + 1 < n:
        new_cmax = max(col_prev[r] + Tail[r][j + 1] for r in range(m))
    else:
        new_cmax = col_prev[m - 1]

    return new_cmax - base_cmax


def motzkin_neighborhood_full(
    pi: List[int],
    processing_times: List[List[int]],
    force_move_if_none: bool = True,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """Full composite Motzkin neighborhood move.

    Uses Head+Tail technique for faster delta computation.

    Steps:
      1. Compute Head and Tail matrices - O(m·n)
      2. Enumerate deltas for all pairs (i,j) using Head+Tail - O(m·n²)
      3. O(n³) DP picks minimal sum of deltas under constraints
      4. Reconstruct chosen pairs
      5. Apply swaps and recompute makespan

    Returns (new_pi, new_cmax, selected_pairs).
    """
    n = len(pi)
    if n < 2:
        return pi[:], c_max(pi, processing_times), []

    # Compute Head and Tail matrices
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)

    m = len(processing_times)
    base_c = Head[m - 1][n - 1]

    # Delta matrix for all pairs - using Head+Tail
    delta: List[List[float]] = [[float("inf")] * n for _ in range(n)]
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta[i][j] = _compute_delta_with_tail(pi, i, j, Head, Tail, processing_times, base_c)

    # Dynamic Programming tables
    dp: List[List[float]] = [[0.0] * n for _ in range(n)]
    choice: List[List[Optional[int]]] = [[None] * n for _ in range(n)]

    for length in range(2, n + 1):
        for L in range(0, n - length + 1):
            R = L + length - 1
            best_val = dp[L + 1][R]
            best_k: Optional[int] = None
            for k in range(L + 1, R + 1):
                left_val = dp[L + 1][k - 1] if (L + 1) <= (k - 1) else 0.0
                right_val = dp[k + 1][R] if (k + 1) <= R else 0.0
                cand = delta[L][k] + left_val + right_val
                if cand < best_val:
                    best_val = cand
                    best_k = k
            dp[L][R] = best_val
            choice[L][R] = best_k

    selected: List[Tuple[int, int]] = []

    def _reconstruct(L: int, R: int) -> None:
        if L >= R:
            return
        k = choice[L][R]
        if k is None:
            _reconstruct(L + 1, R)
        else:
            selected.append((L, k))
            _reconstruct(L + 1, k - 1)
            _reconstruct(k + 1, R)

    _reconstruct(0, n - 1)

    if not selected and force_move_if_none:
        # Fallback: choose the single pair with smallest delta
        best_pair: Optional[Tuple[int, int]] = None
        best_delta = float("inf")
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = delta[i][j]
                if d < best_delta:
                    best_delta = d
                    best_pair = (i, j)
        if best_pair:
            selected = [best_pair]

    new_pi = pi[:]
    for i, j in sorted(selected, key=lambda x: x[0]):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]

    new_c = c_max(new_pi, processing_times)
    return new_pi, new_c, selected
