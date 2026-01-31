"""Fibonahi neighborhood - non-overlapping adjacent swaps.

Selects the optimal set of non-overlapping adjacent swaps
minimizing the total Cmax change.

The name comes from the Fibonacci sequence: the number of possible sets
of non-overlapping swaps from n-1 positions equals F_{n+1}.

Method: Dynamic programming O(n)
Complexity: O(m·n) for deltas + O(n·k) for DP = O(m·n) total
"""

from typing import Dict, List, Tuple

from src.neighborhoods.common import apply_swaps, compute_head, compute_tail
from src.permutation_procesing import c_max


def _compute_deltas_fast(
    pi: List[int],
    processing_times: List[List[int]],
) -> Tuple[int, List[Tuple[int, float]]]:
    """Compute deltas for all adjacent swaps using Head+Tail technique.

    Complexity: O(m·n) instead of O(n²·m).

    Returns:
        (base_cmax, candidates) where candidates = [(position, delta), ...]
    """
    m = len(processing_times)
    n = len(pi)

    if n < 2:
        return c_max(pi, processing_times), []

    # Compute Head (completion times) and Tail matrices
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)

    base_cmax = Head[m - 1][n - 1]
    candidates: List[Tuple[int, float]] = []

    for j in range(n - 1):
        # Swap positions j and j+1
        job_a = pi[j]  # originally at j, will go to j+1
        job_b = pi[j + 1]  # originally at j+1, will go to j

        # Compute new Cmax after swap using Head + local + Tail
        new_cmax = 0

        # Local completion times for swapped positions
        # C_new[i][j] and C_new[i][j+1]
        C_j = [0] * m  # completion at position j (now has job_b)
        C_j1 = [0] * m  # completion at position j+1 (now has job_a)

        for i in range(m):
            # Completion at position j with job_b
            if j == 0:
                left = 0
            else:
                left = Head[i][j - 1]

            if i == 0:
                top = 0
            else:
                top = C_j[i - 1]

            C_j[i] = max(top, left) + processing_times[i][job_b]

            # Completion at position j+1 with job_a
            top_j1 = C_j[i]
            if i == 0:
                C_j1[i] = C_j[i] + processing_times[i][job_a]
            else:
                C_j1[i] = max(C_j1[i - 1], C_j[i]) + processing_times[i][job_a]

        # Combine with tail from position j+2 using true Head+Tail formula
        # Cmax = max over all machines i of (C_j1[i] + Tail[i][j+2])
        if j + 2 < n:
            new_cmax = 0
            for i in range(m):
                # C_j1[i] = completion time at position j+1 after swap
                # Tail[i][j+2] = time to complete from position j+2 on machine i
                new_cmax = max(new_cmax, C_j1[i] + Tail[i][j + 2])
        else:
            # j+1 is the last position
            new_cmax = C_j1[m - 1]

        delta = new_cmax - base_cmax
        candidates.append((j, delta))

    return base_cmax, candidates


def _solve_dp_topk(
    candidates: List[Tuple[int, float]],
    k: int,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """Solve DP to find top-k best non-overlapping swap sets.

    Uses dynamic programming with memoization.

    Recurrence:
        dp[pos] = top-k of:
            - dp[pos+1]  (skip swap at pos)
            - delta[pos] + dp[pos+2]  (take swap, skip next)

    Complexity: O(n * k) time and space.

    Returns:
        List of (total_delta, chosen_positions) sorted by delta ascending
    """
    if not candidates:
        return [(0.0, ())]

    m = len(candidates)

    # dp[pos] = list of top-k (delta, chosen_set) from position pos onwards
    dp: Dict[int, List[Tuple[float, Tuple[int, ...]]]] = {}

    def solve(pos: int) -> List[Tuple[float, Tuple[int, ...]]]:
        """DP with memoization - returns top-k solutions from pos onwards."""
        if pos >= m:
            return [(0.0, ())]

        if pos in dp:
            return dp[pos]

        idx, delta = candidates[pos]

        # Option 1: skip this swap - take all solutions from pos+1
        skip_solutions = solve(pos + 1)

        # Option 2: take this swap - skip overlapping position
        next_pos = pos + 1
        while next_pos < m and candidates[next_pos][0] == idx + 1:
            next_pos += 1

        take_rest = solve(next_pos)
        take_solutions = [
            (delta + rest_delta, (idx,) + rest_set) for rest_delta, rest_set in take_rest
        ]

        # Merge and keep top-k unique
        all_solutions = skip_solutions + take_solutions
        all_solutions.sort(key=lambda x: x[0])

        seen = set()
        unique_topk = []
        for val, s in all_solutions:
            if s not in seen:
                seen.add(s)
                unique_topk.append((val, s))
                if len(unique_topk) >= k:
                    break

        dp[pos] = unique_topk
        return dp[pos]

    return solve(0)


def fibonahi_neighborhood_topk(
    pi: List[int],
    processing_times: List[List[int]],
    k: int,
) -> List[dict]:
    """Find top-k best sets of non-overlapping swaps.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        k: Number of top solutions to return

    Returns:
        List of dicts [{"pi": [...], "cmax": int, "move": (positions...)}, ...]
        sorted by cmax ascending
    """
    n = len(pi)
    if n < 2:
        c = c_max(pi, processing_times)
        return [{"pi": pi.copy(), "cmax": c, "move": ()}]

    base_c, candidates = _compute_deltas_fast(pi, processing_times)

    if not candidates:
        return [{"pi": pi.copy(), "cmax": base_c, "move": ()}]

    # Get top-k solutions from DP
    topk_solutions = _solve_dp_topk(candidates, k)

    results = []
    for total_delta, chosen in topk_solutions:
        # Skip empty set - means no swap is better, but we want actual moves
        if not chosen:
            continue

        new_pi = apply_swaps(pi, list(chosen))
        final_c = c_max(new_pi, processing_times)
        results.append({"pi": new_pi, "cmax": final_c, "move": chosen})

    # Sort by cmax to ensure correct ordering
    results.sort(key=lambda x: x["cmax"])

    return results[:k]
