from typing import List, Optional, Tuple

from src.permutation_procesing import c_max

# ---------------------------------------------------------------------------
# Motzkina Neighborhood (full): select a set of non-crossing (nested allowed) swaps
# ---------------------------------------------------------------------------
# Rules:
#  * Move = pair (i, j), i < j, swap only the endpoints of segment [i..j]
#  * No crossing: pattern i1 < i2 < j1 < j2 is forbidden
#  * No endpoint reuse: no two selected pairs share any endpoint
#  * Strict nesting allowed: i1 < i2 < j2 < j1 (inner arc fully inside outer)
# Goal: minimize final makespan (Cmax)
# Method: full enumeration of individual swap deltas + O(n^3) DP selection
# Complexity: ~ O(m * n^3) (m = number of machines)
# ---------------------------------------------------------------------------


def _compute_prefix_boundaries(pi: List[int], processing_times: List[List[int]]) -> List[List[int]]:
    """Compute prefix completion columns F[k] for k = 0..n.

    F[k][r] holds the completion time on machine r after the first k jobs
    (jobs pi[0..k-1]) are processed.
    """
    m = len(processing_times)
    n = len(pi)
    F: List[List[int]] = [[0] * m for _ in range(n + 1)]
    for k in range(1, n + 1):
        job = pi[k - 1]
        F[k][0] = F[k - 1][0] + processing_times[0][job]
        for r in range(1, m):
            F[k][r] = max(F[k][r - 1], F[k - 1][r]) + processing_times[r][job]
    return F


def _delta_swap(
    pi: List[int],
    i: int,
    j: int,
    boundary: List[int],
    processing_times: List[List[int]],
    base_cmax: int,
) -> int:
    """Return makespan delta for swapping endpoints of segment [i..j].

    New local segment order: [pi[j]] + pi[i+1..j-1] + [pi[i]]. Middle stays unchanged.
    We start from boundary column (state after i jobs), simulate swapped segment,
    then simulate tail jobs j+1..n-1. Delta = new_cmax - base_cmax.
    """
    m = len(processing_times)
    n = len(pi)
    col_prev = boundary[:]  # copy starting state
    col = [0] * m

    # First job after swap: pi[j]
    job = pi[j]
    col[0] = col_prev[0] + processing_times[0][job]
    for r in range(1, m):
        a = col[r - 1]
        b = col_prev[r]
        if a < b:
            a = b
        col[r] = a + processing_times[r][job]
    col_prev, col = col, col_prev

    # Middle of segment (i+1 .. j-1)
    for t in range(i + 1, j):
        job = pi[t]
        col[0] = col_prev[0] + processing_times[0][job]
        for r in range(1, m):
            a = col[r - 1]
            b = col_prev[r]
            if a < b:
                a = b
            col[r] = a + processing_times[r][job]
        col_prev, col = col, col_prev

    # Last job of segment after swap: pi[i]
    job = pi[i]
    col[0] = col_prev[0] + processing_times[0][job]
    for r in range(1, m):
        a = col[r - 1]
        b = col_prev[r]
        if a < b:
            a = b
        col[r] = a + processing_times[r][job]
    col_prev, col = col, col_prev

    # Tail (jobs j+1 .. n-1)
    for t in range(j + 1, n):
        job = pi[t]
        col[0] = col_prev[0] + processing_times[0][job]
        for r in range(1, m):
            a = col[r - 1]
            b = col_prev[r]
            if a < b:
                a = b
            col[r] = a + processing_times[r][job]
        col_prev, col = col, col_prev

    return col_prev[-1] - base_cmax


def motzkin_neighborhood_full(
    pi: List[int],
    processing_times: List[List[int]],
    force_move_if_none: bool = True,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """Full composite Motzkina neighborhood move.

    Steps:
      1. Compute base makespan.
      2. Prefix boundaries F.
      3. Enumerate deltas for all pairs (i,j), i < j.
      4. O(n^3) DP picks minimal sum of deltas under constraints:
         - no endpoint reuse
         - no crossing
         - strict nesting allowed.
      5. Reconstruct chosen pairs.
      6. If empty and force_move_if_none: pick best single pair.
      7. Apply swaps and recompute makespan.

    Returns (new_pi, new_cmax, selected_pairs).
    """
    n = len(pi)
    if n < 2:
        return pi[:], c_max(pi, processing_times), []

    base_c = c_max(pi, processing_times)
    F = _compute_prefix_boundaries(pi, processing_times)

    # Delta matrix for all pairs
    delta: List[List[float]] = [[float("inf")] * n for _ in range(n)]
    for i in range(n - 1):
        boundary = F[i]
        for j in range(i + 1, n):
            delta[i][j] = _delta_swap(pi, i, j, boundary, processing_times, base_c)

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
