from typing import Dict, List, Optional, Tuple

from src.neighborhoods.common import compute_endpoint_swap_delta, compute_head, compute_tail
from src.permutation_procesing import c_max


# --------------------------
# Core: Dynasearch (Head+Tail optimized)
# --------------------------
def dynasearch_full(
    pi: List[int],
    processing_times: List[List[int]],
    L_max: Optional[int] = None,
    force_move_if_none: bool = True,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """
    Dynasearch neighborhood faithful to article structure (practical implementation).

    Returns:
        new_pi, final_cmax, chosen_moves_list
    chosen_moves_list is list of (i,j) intervals applied (sorted by i).

    Parameters:
      - L_max: optional limit on maximum interval length (j - i + 1). If None, consider all pairs.
      - force_move_if_none: if True, when DP picks nothing, force smallest-delta move.
    """
    n = len(pi)
    if n < 2:
        return pi, c_max(pi, processing_times), []

    # Head+Tail matrices for O(m·(j-i)) delta computation per candidate
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)
    m = len(processing_times)
    base_c = Head[m - 1][n - 1]

    candidates: List[Tuple[int, int, int]] = []  # (i, j, delta)
    # enumerate candidate 2-exchanges (i,j) where i<j
    for i in range(0, n - 1):
        # apply L_max constraint on interval length if provided
        j_max = n - 1 if L_max is None else min(n - 1, i + L_max - 1)
        for j in range(i + 1, j_max + 1):
            delta = compute_endpoint_swap_delta(pi, i, j, Head, Tail, processing_times, base_c)
            candidates.append((i, j, delta))

    if not candidates:
        return pi, base_c, []

    # sort candidates by left index then by right index (stable)
    candidates.sort(key=lambda x: (x[0], x[1]))
    m = len(candidates)
    memo: Dict[int, Tuple[int, Tuple[Tuple[int, int], ...]]] = {}

    # DP: choose subset of non-overlapping intervals minimizing total delta
    def solve(pos: int) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
        if pos >= m:
            return 0, ()
        if pos in memo:
            return memo[pos]

        i, j, delta = candidates[pos]
        # option skip
        skip_val, skip_set = solve(pos + 1)
        best_val, best_set = skip_val, skip_set

        # option take -> find next pos that doesn't overlap with [i,j]
        next_pos = pos + 1
        while next_pos < m:
            ii, jj, _ = candidates[next_pos]
            # overlap if intervals share any index
            if not (jj < i or ii > j):
                next_pos += 1
            else:
                break
        take_rest_val, take_rest_set = solve(next_pos)
        take_val = delta + take_rest_val
        if take_val < best_val:
            best_val = take_val
            best_set = ((i, j),) + take_rest_set

        memo[pos] = (best_val, best_set)
        return memo[pos]

    total_delta, chosen = solve(0)

    # if DP picked nothing and we want to force a move => choose single minimal delta move
    chosen_intervals: List[Tuple[int, int]] = list(chosen)
    if not chosen_intervals and force_move_if_none:
        best_i, best_j, _ = min(candidates, key=lambda x: x[2])
        chosen_intervals = [(best_i, best_j)]

    # apply chosen swaps on a copy of pi (sorted by left index)
    new_pi = pi.copy()
    for i, j in sorted(chosen_intervals, key=lambda x: x[0]):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]

    final_c = c_max(new_pi, processing_times)
    return new_pi, final_c, chosen_intervals
